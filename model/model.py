"""
GPT 语言模型的完整定义，全部集中在这一个文件里。
参考：
1) OpenAI 发布的官方 GPT-2 TensorFlow 实现：
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers 的 PyTorch 实现：
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
import dataclasses
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.nn import functional as F

# -----------------------------------------------------------------------------
# 对数放缩残差（log-scaled / log-sum-exp residual，简称 LSE 残差）
# 普通残差是「线性域相加」：x ← x + F(x)。这里换成「对数域 soft-max 合并」：
#     LSE(x, F) = log(exp(x) + exp(F)) = max(x, F) + log(1 + exp(-|x - F|))
# 三个性质让它直接对症「重复坍缩 / loss 骗低 / SwiGLU 数值尖刺」：
#   1) 数值稳定：logaddexp 对任意实数（含负）都不会溢出，梯度也干净。
#   2) 有界收缩：LSE ≤ max(x,F) + log2 —— 结果永远压在最强的分支附近，
#      不像加法那样随层数无限涨大（SwiGLU 的极端值就是线性相加层层放大的）。
#   3) 软选择偏置：某分支占优时输出≈该分支 + 一个 log 小项，而不是把两条
#      信号直接相加——模型被迫「选一个主路」而不是「叠加便宜的高频极端 token」。
# 零参数、纯数值连接改变，规模严格不变 —— 与 block_order / no_attn_layers 同类。
# 内置一个常数 lse_log2_offset 做「放缩」：=1 即纯 LSE；调大则向加法退化
# （当 offset 很大时 LSE≈x+F，等价于普通残差，可用作平滑消融）。
# -----------------------------------------------------------------------------

def logsumexp_residual(x, y):
    """对数域 soft-max 合并两个残差分支，替代线性相加。
    x, y: (B, T, C) 任意实数（靠 logaddexp 内部做 max-平移，负值也稳）。
    返回 LSE(x, y)，有界于 max(x,y)+log2。
    """
    m = torch.maximum(x, y)
    return m + torch.logaddexp(x - m, y - m)

class RMSNorm(nn.Module):
    """RMSNorm：LLaMA / DeepSeek 的标准归一化。
    LayerNorm 是 (x - mean) / std * weight；RMSNorm 省略了减均值，只按 RMS 缩放：
        x / sqrt(mean(x^2) + eps) * weight
    少算了 mean、也没有 bias 参数，更快且效果几乎一致。
    """

    def __init__(self, ndim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))

    def forward(self, x):
        # torch.rsqrt(x) = 1/sqrt(x)，数值上比 x.pow(-0.5) 更稳
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

# -----------------------------------------------------------------------------
# 旋转位置编码 RoPE（Rotary Position Embedding）
# DeepSeek / LLaMA 都用它替代可学习的位置编码。核心思想：
# 把每个 head_dim 的向量看成 head_dim/2 个二维平面，按 token 位置旋转每个平面。
# 这样 q·k 的点积里自动出现「相对位置」项（m-n），而不是绝对位置。
# -----------------------------------------------------------------------------

def rotate_half(x):
    """RoPE 的旋转操作：把后半段搬到前半段并取负。"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def precompute_rope_freqs(head_dim, seq_len, theta=10000.0):
    """预计算 cos/sin 表，形状 (seq_len, head_dim)。
    theta 越大频率越低 -> 旋转越慢 -> 能表达更长距离的位置关系。
    DeepSeek 系列专门研究过 theta 的取值，你可以自己试着调大（如 1e6）。
    """
    # 每个二维平面一个频率：i=0,2,4,... 对应 1/theta^(i/head_dim)
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)            # (seq_len, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)     # 复制成 head_dim 长度，和 rotate_half 配对
    return emb.cos(), emb.sin()                 # 各 (seq_len, head_dim)

def apply_rotary_pos_emb(q, k, cos, sin):
    """把旋转位置编码作用到 q 和 k 上（两者都要加，才能让点积带相对位置信息）。
    q, k: (B, T, n_head, head_dim)；cos, sin: (T, head_dim)
    公式：x_rotated = x * cos + rotate_half(x) * sin
    """
    cos = cos.unsqueeze(0).unsqueeze(2).to(q.dtype)  # (1, T, 1, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(2).to(q.dtype)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.n_embd % config.n_head == 0
        # 所有 head 的 key、query、value 投影，但放在同一个 batch 里计算
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # 输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # 正则化
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.use_rope = config.use_rope
        # MLA（DeepSeek-V2）：Q 独立投影；KV 先压缩到低秩潜在、再展开成 K 和 V。
        # RoPE 只作用于每个 head 的前 qk_rope_head_dim 维，其余是"无位置"的内容维。
        self.use_mla = config.use_mla
        self.use_csa = config.use_csa
        # QK-Norm：对 q/k 做 L2 归一化 + 每头可学习 scale。
        # scale 初始 = sqrt(head_dim)，forward 里再乘 1/sqrt(head_dim)，起点等价于原始 q·k/d。
        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.qk_scale = nn.Parameter(torch.full((config.n_head,), self.head_dim ** 0.5))
        self.rope_head_dim = config.qk_rope_head_dim if (config.use_mla or config.use_csa) else (config.n_embd // config.n_head)
        if config.use_csa:
            # CSA/HCA 的独立 Q/K/V 投影（与 MLA 的 KV 压缩路径分开，更直白）。
            # 输入直接投影出 K/V，再在 forward 里做块级压缩。
            self.q_proj_csa = nn.Linear(config.n_embd, config.n_embd, bias=False)
            self.k_proj_csa = nn.Linear(config.n_embd, config.n_embd, bias=False)
            self.v_proj_csa = nn.Linear(config.n_embd, config.n_embd, bias=False)
            if config.use_csa_learnable:
                # V4 可学习门控池化：把块内 csa_compress 个 token 压成 1 个潜在。
                # compress 线性压缩 + sigmoid gate；K/V 共享同一组权重（对应共享潜在）。
                # 门控初始化为 0 → sigmoid(0)=0.5，起步为中性门控，不扭曲初始行为。
                self.csa_compress_linear = nn.Linear(
                    config.csa_compress * self.head_dim, self.head_dim, bias=False)
                self.csa_gate_linear = nn.Linear(
                    config.csa_compress * self.head_dim, self.head_dim, bias=False)
                nn.init.zeros_(self.csa_gate_linear.weight)
            if config.use_lightning_indexer:
                # V4 Lightning Indexer（简版）：学习型块选择替代 raw top-k。
                # idx_q 把 query 投影到「选择空间」，idx_k 给每个压缩块打一个标量分。
                # 分数只决定「选哪几个块」，不参与注意力值；梯度经 KL 桥接到块选择。
                self.idx_q = nn.Linear(config.n_embd, config.n_head, bias=False)  # (n_embd→nh)
                self.idx_k = nn.Linear(self.head_dim, 1, bias=False)              # 块→标量分
                nn.init.zeros_(self.idx_k.weight)  # 起步打分≈0，中性选块
        if config.use_mla:
            assert config.qk_rope_head_dim % 2 == 0 and config.qk_rope_head_dim <= self.head_dim, \
                "qk_rope_head_dim 需为偶数且不超过 head_dim"
            self.q_proj  = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            self.kv_down = nn.Linear(config.n_embd, config.kv_lora_rank, bias=config.bias)
            self.kv_act  = nn.SiLU()
            self.k_up    = nn.Linear(config.kv_lora_rank, config.n_embd, bias=config.bias)
            self.v_up    = nn.Linear(config.kv_lora_rank, config.n_embd, bias=config.bias)
        # V4 Attention Sinks：每头一个可学习标量偏置，作为 softmax 的"垃圾桶"。
        # 追加一列 sink[h] 到分数末尾（对应零 value 向量），模型借此丢掉无关注意力预算。
        # flash attention 不支持追加 softmax 列，启用 sink 时回退到手动注意力。
        self.use_attn_sink = config.use_attn_sink
        if self.use_attn_sink:
            self.attn_sink = nn.Parameter(torch.zeros(config.n_head))
        # flash attention 能让 GPU 跑得飞快，但只在 PyTorch >= 2.0 才支持
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("警告：正在使用慢速 attention。Flash Attention 需要 PyTorch >= 2.0")
            # 因果掩码，确保 attention 只作用于输入序列左侧的位置。
            # 记忆 token 前缀 → 有效长度 block_size+n_memory_tokens，掩码同宽。
            max_len = config.block_size + config.n_memory_tokens
            self.register_buffer("bias", torch.tril(torch.ones(max_len, max_len))
                                        .view(1, 1, max_len, max_len))
        # 用 RoPE 的话，预计算 cos/sin 表并注册成 buffer（随模型移动设备、进 checkpoint）。
        # MLA 只旋转每头的前 qk_rope_head_dim 维，表长和 rope_head_dim 一致。
        # 记忆 token 前缀会让有效序列长到 block_size+K，表要扩到 block_size+n_memory_tokens。
        # n_memory_tokens=0 时表长不变 → 老 checkpoint 的 cos/sin buffer 形状零变化，兼容不破。
        if self.use_rope:
            max_len = config.block_size + config.n_memory_tokens
            cos, sin = precompute_rope_freqs(self.rope_head_dim, max_len, config.rope_theta)
            self.register_buffer("cos", cos)  # (max_len, rope_head_dim)
            self.register_buffer("sin", sin)  # (max_len, rope_head_dim)

        # QK-Norm：q/k 分别是 (B, T, nh, d)；返回归一化后的 q/k（保留 head 维）。
        # 只对参与点积的 q/k 做，v 不做。
        # 语义：q = normalize(q) * qk_scale，k = normalize(k)。qk_scale 初始 = sqrt(head_dim)，
        # 等价于原始 q·k/sqrt(d) 的按头缩放；开启后不再额外除 sqrt(d)。
    def _apply_qk_norm(self, q, k):
        if not self.use_qk_norm:
            return q, k
        # scale: (nh,)，按 head 广播到 (B,T,nh,d)
        scale = self.qk_scale.view(1, 1, -1, 1)
        q = F.normalize(q, dim=-1) * scale
        k = F.normalize(k, dim=-1)
        return q, k

    def forward(self, x):
        B, T, C = x.size() # batch 大小、序列长度、嵌入维度 (n_embd)

        if self.use_csa:
            # CSA/HCA 混合注意力（V4 简化版）：走独立的压缩稀疏路径
            y = self._csa_forward(x)
            y = self.resid_dropout(self.c_proj(y))
            return y

        if self.use_mla:
            # MLA 路径：Q 独立投影；KV 共享一个低秩潜在表示，再分别展开成 K 和 V
            q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
            kv_latent = self.kv_act(self.kv_down(x))          # (B, T, kv_lora_rank)
            k = self.k_up(kv_latent).view(B, T, self.n_head, self.head_dim)
            v = self.v_up(kv_latent).view(B, T, self.n_head, self.head_dim)
            if self.use_qk_norm:
                q, k = self._apply_qk_norm(q, k)  # RoPE 是范数保持的旋转，前后顺序等价
            if self.use_rope:
                # 部分 RoPE：每头只有前 rope_head_dim 维参与旋转，剩余是"无位置"内容维。
                # 让模型自己决定每个 head 需要多少位置信息（都能从同一 low-rank 潜在表达还原）
                q_rope, q_nope = q[..., :self.rope_head_dim], q[..., self.rope_head_dim:]
                k_rope, k_nope = k[..., :self.rope_head_dim], k[..., self.rope_head_dim:]
                q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, self.cos[:T], self.sin[:T])
                q = torch.cat([q_rope, q_nope], dim=-1)
                k = torch.cat([k_rope, k_nope], dim=-1)
        else:
            # 标准路径：在 batch 中计算所有 head 的 query、key、value，并把 head 维前移作为 batch 维
            q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
            # 先 reshape 成 (B, T, nh, hs)：RoPE 要在 head 维还没被换到前面时作用
            q = q.view(B, T, self.n_head, self.head_dim)
            k = k.view(B, T, self.n_head, self.head_dim)
            v = v.view(B, T, self.n_head, self.head_dim)
            if self.use_qk_norm:
                q, k = self._apply_qk_norm(q, k)  # RoPE 是范数保持的旋转，前后顺序等价

            if self.use_rope:
                # 旋转位置编码：只对 q 和 k 做（v 不旋转），这样 q·k 携带相对位置
                q, k = apply_rotary_pos_emb(q, k, self.cos[:T], self.sin[:T])

        # 把 head 维移到第 2 维：(B, nh, T, hs)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 因果自注意力；自注意力：(B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # QK-Norm 时 q 已自带 qk_scale（初始 = sqrt(d)），scale 必须传 1.0：
        # 若传 None/省略，SDPA 会按默认 1/sqrt(head_dim) 再除一次，与下方手动路径
        # （use_qk_norm 时不再除 sqrt(d)）不一致——flash 开/关会得到不同 logits。
        attn_scale = 1.0 if self.use_qk_norm else 1.0 / math.sqrt(k.size(-1))
        if self.flash:
            # 使用 Flash Attention CUDA 内核的高效 attention。
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True, scale=attn_scale)
        else:
            # 手动实现 attention
            att = q @ k.transpose(-2, -1)
            if not self.use_qk_norm:
                att = att * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            if self.use_attn_sink:
                # Attention Sink：追加一列 sink[h]（value 用零向量占位）。
                # 这列 softmax 后有 exp(sink) 的概率质量，但乘零向量 → 贡献为 0，
                # 等价于"把多余的注意力预算倒进垃圾桶"。
                sink = self.attn_sink.view(1, self.n_head, 1, 1).expand(B, self.n_head, T, 1)
                att = torch.cat([att, sink], dim=-1)                          # (B,nh,T,T+1)
                v = torch.cat([v, v.new_zeros(B, self.n_head, T, 1, v.size(-1))], dim=-2)  # (B,nh,T+1,hs)
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T+1) x (B, nh, T+1, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # 把所有 head 的输出并排重新组装

        # 输出投影
        y = self.resid_dropout(self.c_proj(y))
        return y

    def _csa_forward(self, x):
        """CSA + HCA 混合注意力（DeepSeek-V4 的简化教育版）。

        CSA（压缩稀疏注意力）：把 K/V 按 m 个 token 一块，平均池化成 1 个潜在向量。
        每个 query 只稀疏地选 top-k 个「它之前」的压缩块（长程信号用摘要传递），
        再保留一段滑窗的原始 token（近处信息用细节传递）。注意力开销从 O(T²)
        降到 O(T·(nb + win))——这是 V4 能跑 1M 上下文的核心原因。

        HCA（重度压缩注意力）：把所有「允许的」块再压成一个全局潜在（不做稀疏
        选择），每个 query 额外加上这份全局信号，补足长程上下文。

        注：V4 原版用可学习压缩和 lightning indexer，这里用平均池化 + top-k
        简化演示同一思想。压缩块的平均池化会丢失块内细节，靠滑窗补回近处信息。
        """
        B, T, C = x.shape
        nh, d = self.n_head, self.head_dim
        m = self.config.csa_compress
        win = self.config.csa_window
        # Lightning Indexer 的 KL 辅助损失（块路径跳过时兜底为 0）
        self.indexer_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        q = self.q_proj_csa(x).view(B, T, nh, d)
        k = self.k_proj_csa(x).view(B, T, nh, d)
        v = self.v_proj_csa(x).view(B, T, nh, d)
        if self.use_qk_norm:
            q, k = self._apply_qk_norm(q, k)  # RoPE 是范数保持的旋转，前后顺序等价

        if self.use_rope:
            # 部分 RoPE：只旋转前 rope_head_dim 维（与 MLA 一致）
            q_rope, q_nope = q[..., :self.rope_head_dim], q[..., self.rope_head_dim:]
            k_rope, k_nope = k[..., :self.rope_head_dim], k[..., self.rope_head_dim:]
            q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, self.cos[:T], self.sin[:T])
            q = torch.cat([q_rope, q_nope], dim=-1)
            k = torch.cat([k_rope, k_nope], dim=-1)

        # --- 1) 块级压缩：每 m 个连续 token 的 K/V 平均池化成 1 个潜在 ---
        # 短序列（块数 nb < topk）时 topk 会越界，用 k_eff = min(topk, nb) 兜底；
        # nb = 0（不足一个块）时块路径整个跳过，只靠滑窗。
        T_ok = (T // m) * m
        nb = T_ok // m
        k_eff = min(self.config.csa_topk, nb)
        y_comp = torch.zeros(B, T, nh, d, device=x.device, dtype=x.dtype)
        has_prior = torch.zeros(T, device=x.device)
        if nb > 0:
            if self.config.use_csa_learnable:
                # V4 可学习门控池化（K/V 共享权重）
                k_blocks = self._compress_block(k[:, :T_ok], B, nb, nh, d, m)
                v_blocks = self._compress_block(v[:, :T_ok], B, nb, nh, d, m)
            else:
                # 平均池化（简化版基线）
                k_blocks = k[:, :T_ok].view(B, nb, m, nh, d).mean(dim=2)   # (B, nb, nh, d)
                v_blocks = v[:, :T_ok].view(B, nb, m, nh, d).mean(dim=2)

            # --- 2) 稀疏块选择：query 只能看「它所在块之前」的块，再取 top-k ---
            bq = torch.arange(T, device=x.device) // m                # 每个 query 属于哪个块
            causal_block = bq.unsqueeze(-1) > torch.arange(nb, device=x.device)  # (T, nb)
            has_prior = causal_block.any(dim=-1)                      # (T,) 该 query 有没有历史块
            s_blk = torch.einsum('bthd,bnhd->bthn', q, k_blocks)
            if not self.use_qk_norm:
                s_blk = s_blk / math.sqrt(d)  # (B,T,nh,nb)
            s_blk = s_blk.masked_fill(~causal_block.unsqueeze(0).unsqueeze(2), float('-inf'))

            if self.config.use_lightning_indexer:
                # --- V4 Lightning Indexer（简版）：学习型块选择 ---
                # idx_q 把 query 投影到「选择空间」(B,T,nh)，idx_k 给每块打标量分 (B,nb,nh)。
                # 分数 = Σ_h q_idx[t,h]·k_idx[s,h]，是 (T,nb) 外积：query 和块的位置解耦。
                # 分数只决定「选哪几块」；s_blk 仍算真实注意力，选中的块才参与 softmax。
                q_idx = self.idx_q(x)                                        # (B,T,nh)
                k_idx = self.idx_k(k_blocks.reshape(-1, d)).view(B, nb, nh)  # (B,nb,nh)
                idx_scores = torch.einsum('bth,bnh->btn', q_idx, k_idx)     # (B,T,nb) 外积
                idx_scores = idx_scores.masked_fill(
                    ~causal_block.unsqueeze(0), float('-inf'))
                # 没有历史块的 query：整行置 0（否则 softmax 对全 -inf 产生 NaN）
                idx_scores = idx_scores.masked_fill(
                    ~has_prior.view(1, T, 1), 0.0)
                # 用 indexer 分数选 top-k 块（只决定 mask，不直接进注意力）
                idx_topk, _ = idx_scores.topk(k_eff, dim=-1)
                sel_mask = idx_scores >= idx_topk[..., [-1]]                # (B,T,nb)
                sel_mask = sel_mask.unsqueeze(2).expand(B, T, nh, nb)       # → (B,T,nh,nb)
                # 记录 KL 信号：让 indexer 的 softmax 逼近「真实注意力」的块分布。
                # p=0 的位置该项贡献 0（causal 不允许的块 p 和 q 都为 0）；
                # clamp 到 1e-12 避免 0×log(0)=NaN。
                # 没有历史块的 query：q 行整行置 0（softmax 后均匀分布，不与 p 比较）
                valid = has_prior.float()                                   # (T,)
                p = F.softmax(idx_scores.float(), dim=-1)                  # (B,T,nb)
                s_blk_mean = s_blk.detach().float().mean(dim=2)            # (B,T,nb)
                s_blk_mean = s_blk_mean.masked_fill(
                    ~has_prior.view(1, T, 1), 0.0)                          # 无历史块行置 0
                target_p = F.softmax(s_blk_mean, dim=-1)                   # (B,T,nb) KL 目标
                kl = (p * (p.clamp_min(1e-12).log() -
                           target_p.clamp_min(1e-12).log())).sum(dim=-1)  # (B,T)
                kl = (kl * valid.unsqueeze(0)).sum() / valid.sum().clamp(min=1)
                self.indexer_loss = kl
                # 用选中的块替换 s_blk 的稀疏 mask（注意：仍保留真实注意力分数）
                s_blk = s_blk.masked_fill(~sel_mask, float('-inf'))
            else:
                # 基线：直接对真实注意力分数取 top-k（raw 稀疏）
                self.indexer_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
                topk_vals, _ = s_blk.topk(k_eff, dim=-1)
                s_blk = s_blk.masked_fill(s_blk < topk_vals[..., [-1]], float('-inf'))
            # 没有历史块的 query：整行置 0（否则 softmax 对全 -inf 产生 NaN）
            s_blk = s_blk.masked_fill(~has_prior.view(1, T, 1, 1), 0.0)
            if self.use_attn_sink:
                # Attention Sink：块注意力也追加一列 sink[h]（v_blocks 补零块占位）
                sink = self.attn_sink.view(1, 1, self.n_head, 1).expand(B, T, self.n_head, 1)
                s_blk = torch.cat([s_blk, sink], dim=-1)                       # (B,T,nh,nb+1)
                v_blocks = torch.cat([v_blocks, v_blocks.new_zeros(B, 1, self.n_head, d)], dim=1)  # (B,nb+1,nh,d)
            a_blk = F.softmax(s_blk, dim=-1)                          # (B,T,nh,nb+1)
            y_comp = torch.einsum('bthn,bnhd->bthd', a_blk, v_blocks)  # (B,T,nh,d)
            y_comp = y_comp * has_prior.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).float()

        # --- 3) 滑窗：最近 win 个原始 token 的局部因果注意力 ---
        # 滑窗允许看自己（j ≤ i），保证每个位置至少有一个合法键，避免全 -inf。
        win = min(win, T)
        i = torch.arange(T, device=x.device)
        win_causal = (i.unsqueeze(-1) <= i.unsqueeze(0)) & (i.unsqueeze(0) - i.unsqueeze(-1) <= win)
        s_win = torch.einsum('bthd,bjhd->bthj', q, k)
        if not self.use_qk_norm:
            s_win = s_win / math.sqrt(d)  # (B,T,nh,T)
        s_win = s_win.masked_fill(~win_causal.unsqueeze(0).unsqueeze(2), float('-inf'))
        v_win = v
        if self.use_attn_sink:
            # Attention Sink：滑窗注意力同样追加一列（v 补零行占位）
            sink = self.attn_sink.view(1, 1, self.n_head, 1).expand(B, T, self.n_head, 1)
            s_win = torch.cat([s_win, sink], dim=-1)                        # (B,T,nh,T+1)
            v_win = torch.cat([v, v.new_zeros(B, 1, self.n_head, d)], dim=1)  # (B,T+1,nh,d)
        y_win = torch.einsum('bthj,bjhd->bthd', F.softmax(s_win, dim=-1), v_win)

        y = y_comp + y_win

        # --- 4) HCA：重度压缩的全局信号（可选）---
        # 把所有允许的压缩块再平均成一个全局潜在（不做稀疏选择 = 重度压缩），
        # 每个 query 加上它作为全局上下文。这是"全文一句话摘要"式的粗粒度信号。
        if self.config.use_hca and nb > 0:
            # 只用真实块：sink 模式下 v_blocks 末尾多了一个占位零块，切掉它
            v_blocks_real = v_blocks[:, :nb] if self.use_attn_sink else v_blocks
            n_allowed = causal_block.float().sum(dim=-1).clamp(min=1)  # (T,)
            k_glob = torch.einsum('tn,bnhd->bthd', causal_block.float(), k_blocks) / \
                     n_allowed.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            v_glob = torch.einsum('tn,bnhd->bthd', causal_block.float(), v_blocks_real) / \
                     n_allowed.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            # 单个全局 key 的 softmax 恒为 1，等价于直接加上这份全局摘要
            y = y + v_glob

        # 合并 head： (B, T, nh, d) → (B, T, C)
        return y.transpose(1, 2).contiguous().view(B, T, C)

    def get_indexer_loss(self):
        """取出本层 Lightning Indexer 的 KL 辅助损失（非 CSA 路径时无意义，返回 0）。"""
        return self.indexer_loss if hasattr(self, 'indexer_loss') else None

    def _compress_block(self, x_block, B, nb, nh, d, m):
        """V4 可学习门控池化：把块内 m 个 token 压成 1 个潜在（替代平均池化）。
        x_block: (B, T_ok, nh, d)，T_ok = nb*m。
        返回 (B, nb, nh, d)：compress 线性压缩 × sigmoid 门控，K/V 共享权重。
        """
        # (B, nb*m, nh, d) → (B, nb, nh, m, d) → 展平每个块 (B*nb*nh, m*d)
        flat = x_block.view(B, nb, m, nh, d).permute(0, 1, 3, 2, 4).reshape(B * nb * nh, m * d)
        h = self.csa_compress_linear(flat)                   # 线性压缩 (B*nb*nh, d)
        gate = torch.sigmoid(self.csa_gate_linear(flat))     # 门控 (0,1)
        return (h * gate).view(B, nb, nh, d)

class SwiGLU(nn.Module):
    """SwiGLU：门控前馈网络，DeepSeek/LLaMA 的标准 FFN。
    输出 = SiLU(x W1) ⊙ (x W2)，比单一路径的 GELU MLP 表达能力更强。
    参数量对比：
      MLP    : W(embd,4embd) + W(4embd,embd) = 8·embd²
      SwiGLU : W(embd,h) + W(embd,h) + W(h,embd) = 3·h·embd
    取 h = 8/3·embd 时两者参数量持平 —— 这就是 LLaMA 用 hidden = 8/3·n_embd 的来历。
    """

    def __init__(self, config, hidden_scale=8 / 3):
        super().__init__()
        self.config = config  # 保存 config，forward 里可能要用到钳制等技巧
        hidden = int(hidden_scale * config.n_embd)  # 默认 ≈ 2.67·n_embd；hidden_scale=3 时参数量≈注意力层（补注意力槽）
        self.c_fc   = nn.Linear(config.n_embd, hidden, bias=config.bias)  # 值分支
        self.c_fc2  = nn.Linear(config.n_embd, hidden, bias=config.bias)  # 门控分支
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = F.silu(self.c_fc(x)) * self.c_fc2(x)  # SiLU(xW1) ⊙ (xW2)
        if self.config.swiglu_clamp > 0:
            # V4 稳定性技巧：钳制门控输出，从源头压制异常值。
            # 注意是在 c_proj 之前钳——异常值就是在这个门控乘积里产生的。
            x = x.clamp(-self.config.swiglu_clamp, self.config.swiglu_clamp)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class MoE(nn.Module):
    """MoE：混合专家（DeepSeek-V3 的核心创新）。

    关键思想：每个 token 只路由到 top-k 个专家（而不是所有专家都算一遍）。
    于是参数量随专家数线性增长，但每个 token 的计算量保持不变——
    这就是「用更多参数、同一份算力」换更强的模型。

    还需要一个「负载均衡辅助损失」：迫使 token 均匀分布到各专家。
    否则路由器会陷入"赢家通吃"，少数专家承担绝大部分计算，
    其它专家几乎不被激活、学不到东西（类似专家"饿死"）。
    """

    def __init__(self, config, use_hash=False):
        super().__init__()
        self.n_experts = config.n_experts
        self.n_top_k = config.n_top_k
        self.moe_aux_weight = config.moe_aux_weight
        self.router_z_loss_weight = config.z_loss_weight
        self.use_shared_expert = config.use_shared_expert
        self.use_aux_free_balance = config.use_aux_free_balance
        self.use_sqrtsoftplus = config.use_sqrtsoftplus
        self.route_scale = config.route_scale
        # V4 Hash 路由：浅层用 hash(token_id)%n_experts 确定性分配，不学习。
        # 设计动机：浅层特征是简单语法/常见搭配，确定性路由稳定且零算力；
        # 深层语义复杂才需要学习型路由。use_hash 由 Block 按 layer_idx 传入。
        self.use_hash = use_hash
        # 路由器：给每个 token 在每个专家上打一个分（hash 模式下仍保留，作为后续层复用）
        self.router = nn.Linear(config.n_embd, config.n_experts, bias=False)
        # 专家：每个专家是一份完整的 SwiGLU FFN
        self.experts = nn.ModuleList([SwiGLU(config) for _ in range(config.n_experts)])
        # V4 共享专家：始终激活，捕获所有 token 的共性特征（语法、常见搭配）
        if self.use_shared_expert:
            self.shared_expert = SwiGLU(config)
        # 本次前向累积的辅助损失，forward 后由 GPT 取走并清零
        self.aux_loss = torch.tensor(0.0)
        self.z_loss = torch.tensor(0.0)
        if self.use_aux_free_balance:
            # aux-free 偏置修正：每个专家一个 bias，加到路由 logits 上影响 top-k 选择。
            # 不参与梯度（requires_grad=False），每步根据负载偏差用 balance_factor 更新，
            # 过载专家降 bias、欠载升 bias → 自然均衡。这是 V4 替代 Switch aux loss 的做法。
            self.register_buffer('router_bias', torch.zeros(config.n_experts))
            self.balance_factor = config.balance_factor

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)          # (B*T, C)，把 batch 和序列压平逐个 token 路由
        N = x_flat.shape[0]

        if self.use_hash:
            # --- V4 Hash 路由：确定性分配，不学习 ---
            # Knuth 乘法哈希：用 token 嵌入的第一个标量作签名，均匀映射到各专家。
            # 用 k 个不同的乘子生成 k 个不重复的专家槽位。
            hash_input = x_flat[:, 0].long()  # 用输入第一维当 token 签名（token 无关）
            # 生成 n_top_k 个不同的哈希：hash + i*prime 再 % n_experts
            top_k_indices = torch.stack([
                ((hash_input * (2654435761 + 97 * i)) >> 16) % self.n_experts
                for i in range(self.n_top_k)
            ], dim=-1)  # (N, k)
            top_k_probs = x_flat.new_full((N, self.n_top_k), 1.0 / self.n_top_k)  # 均匀权重
            self.aux_loss = torch.tensor(0.0, device=x_flat.device, dtype=x_flat.dtype)
            self.z_loss = torch.tensor(0.0, device=x_flat.device, dtype=x_flat.dtype)
            # 跳过负载均衡：哈希本身均匀分配，天然均衡
        else:
            # 路由打分：给每个 token 在每个专家上打一个分
            router_logits = self.router(x_flat)               # (N, n_experts)
            # aux-free 偏置修正：bias 加到 logits 上影响 top-k 选择（bias 不参与梯度）
            if self.use_aux_free_balance:
                router_logits = router_logits + self.router_bias
            # Router Z-Loss：惩罚 logsumexp(router_logits) 的平方，防止路由 logits 过大。
            if self.router_z_loss_weight > 0:
                self.z_loss = self.router_z_loss_weight * (
                    torch.logsumexp(router_logits, dim=-1).square().mean()
                )
            else:
                self.z_loss = torch.tensor(0.0, device=router_logits.device, dtype=router_logits.dtype)
            # 软概率：Switch aux loss 用（aux-free 模式下不参与损失）
            router_probs = F.softmax(router_logits, dim=-1)   # (N, n_experts)
            if self.use_sqrtsoftplus:
                # V4 打分：√softplus(logits) * route_scale，直接用于选择 + 归一化
                scores = torch.sqrt(F.softplus(router_logits)) * self.route_scale
                top_k_scores, top_k_indices = scores.topk(self.n_top_k, dim=-1)
                top_k_probs = top_k_scores / (top_k_scores.sum(dim=-1, keepdim=True) + 1e-6)
            else:
                # 基线：softmax 全专家 → top-k → 重归一化
                top_k_probs, top_k_indices = router_probs.topk(self.n_top_k, dim=-1)  # 各 (N, k)
                top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

            # 负载均衡：f_i = 第 i 个专家实际接到的 token 比例
            one_hot = F.one_hot(top_k_indices, self.n_experts).float()   # (N, k, n_experts)
            f_i = one_hot.sum(dim=(0, 1)) / (N * self.n_top_k)           # (n_experts,)
            if self.use_aux_free_balance:
                # V4 aux-free：无辅助损失，改为按负载偏差更新 bias（无梯度，决定性推动均衡）。
                # 过载专家（f_i > 均值）bias 下降 → 更难被选中；欠载专家 bias 上升 → 更容易被选中。
                self.aux_loss = torch.tensor(0.0)
                with torch.no_grad():
                    self.router_bias.add_((f_i - 1.0 / self.n_experts).sign() * self.balance_factor)
            else:
                # Switch Transformer 辅助损失：P_i = 路由器给第 i 个专家的平均概率。
                # 两者都高意味着该专家又热门又常被选，均衡时 sum(f_i * P_i) 取最小。
                P_i = router_probs.mean(dim=0)                           # (n_experts,)
                self.aux_loss = self.moe_aux_weight * self.n_experts * (f_i * P_i).sum()

        # V4 共享专家：所有 token 都过一遍共享专家（捕获共性），再叠加路由专家的差异化输出
        output = self.shared_expert(x_flat) if self.use_shared_expert else x_flat.new_zeros(N, C)
        # 逐个专家计算：把路由到它的 token 收集起来算 FFN，再按门控权重放回原处
        for i in range(self.n_experts):
            expert_mask = (top_k_indices == i)             # (N, k) 该专家被选中的位置
            token_ids, slot_ids = expert_mask.nonzero(as_tuple=True)
            if token_ids.numel() == 0:
                continue
            weights = top_k_probs[token_ids, slot_ids]     # 这些 token 在该专家上的门控权重
            expert_in = x_flat[token_ids]                  # (num, C)
            expert_out = self.experts[i](expert_in)        # (num, C)
            output[token_ids] += expert_out * weights.unsqueeze(-1)

        return output.view(B, T, C)

    def get_aux_loss(self):
        """取出本次前向累积的辅助损失（负载均衡 + Router Z-Loss）并清零。"""
        loss = self.aux_loss + self.z_loss
        self.aux_loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
        self.z_loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
        return loss

class Block(nn.Module):

    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.config = config
        self.use_mhc = config.use_mhc
        # 对数放缩残差 v1：与 mHC 互斥（mHC 已有自己的 4 流残差拓扑，v1 只服务标准路径）。
        assert not (config.use_mhc and config.use_lse_residual), \
            "use_lse_residual 与 use_mhc 互斥：LSE 对数域残差 v1 只作用在标准单流路径！"
        self.use_lse_residual = config.use_lse_residual
        # 对数放缩门控混合 v2：非 4 流拓扑，与 mHC 不互斥，但 v1 纯残差是它的退化特例。
        assert not (config.use_lse_residual and config.use_lse_gate), \
            "use_lse_residual（纯硬替换）与 use_lse_gate（门控混合）只能开一个——gate-mix 已含 v1"
        self.use_lse_gate = config.use_lse_gate
        # 稀疏注意力布线：本层是否跳过注意力子层（只跑 FFN）。
        self.skip_attn = layer_idx in config.no_attn_layers
        # 稀疏注意力布线：skip 层的注意力槽替换为 FFN（hidden_scale=3 → 57,600 参数 ≈
        # CausalSelfAttention 57,604，仅差 attn_sink 的 4 参数/层）→ 规模严格补回基线，
        # 只剩"无注意力"这一个拓扑变量，且所有参数真正参与训练（无冻结废参数）。
        # 布线从 A F 变 F F（FFN 变换 ×2）。
        self.attn = SwiGLU(config, hidden_scale=3) if self.skip_attn else CausalSelfAttention(config)
        # 固定架构：RMSNorm + 残差；FFN 用 MoE（可选）或 SwiGLU
        self.ln_1 = RMSNorm(config.n_embd)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MoE(config, use_hash=layer_idx < config.num_hash_layers) \
            if config.use_moe else SwiGLU(config)

        if self.use_mhc:
            # mHC 超连接：4 流并行残差。每流宽度仍为 n_embd，子层 F 只跑 1 次。
            # 两组 A/B/C（attn 子层 + FFN 子层），见 _mhc_forward。
            hc = config.hc_mult
            self.hc_mult = hc
            self.raw_A_attn = nn.Parameter(torch.zeros(1, hc))       # → sigmoid（有界非负）
            self.raw_B_attn = nn.Parameter(torch.zeros(hc, hc))      # → softplus → sinkhorn（双重随机）
            self.raw_C_attn = nn.Parameter(torch.zeros(hc, 1))       # → sigmoid
            self.raw_A_ffn = nn.Parameter(torch.zeros(1, hc))
            self.raw_B_ffn = nn.Parameter(torch.zeros(hc, hc))
            self.raw_C_ffn = nn.Parameter(torch.zeros(hc, 1))
            # 初始化：A/C 全 0 → sigmoid=0.5（各流等权）；B 全 0 → exp(0)=1 → sinkhorn
            # 后均匀混合（对称起点，不扭曲初始行为）。

        # v2 门控混合：可学习标量 α = sigmoid(raw_gate)，起点 sigmoid(0)=0.5（线性/LSE 均衡）。
        # 每层一个标量，attn/ffn 两个子层残差共享同一个 α（见 forward 的 res 定义）。
        if self.use_lse_gate:
            self.raw_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        if self.use_mhc:
            return self._mhc_forward(x)
        # skip_attn 层的 attn 槽已是宽 FFN（SwiGLU），走同样双子层路径，无特殊分支。
        # 计算图重排：默认 attn→ffn；block_order="ffn_attn" 时先 FFN 后注意力。
        # norm 与子层绑定不可拆：attn 永远用 ln_1，mlp 永远用 ln_2，只调换两段顺序。
        # 残差合并：三种模式
        #   0) 线性加法           : x + F(x)
        #   1) 纯 LSE (v1)        : LSE(x, F(x))
        #   2) 门控混合 (v2)      : α·x + (1-α)·LSE(x, F(x))，α = sigmoid(raw_gate) 可学习
        # 门控混合保留线性主通道（学力）∩ LSE 分支（抗尖刺/抗过度自信）。
        if self.use_lse_residual:
            res = lambda a, b: logsumexp_residual(a, b)
        elif self.use_lse_gate:
            alpha = torch.sigmoid(self.raw_gate)
            res = lambda a, b: alpha * a + (1.0 - alpha) * logsumexp_residual(a, b)
        else:
            res = lambda a, b: a + b
        if self.config.block_order == "ffn_attn":
            x = res(x, self.mlp(self.ln_2(x)))
            x = res(x, self.attn(self.ln_1(x)))
        else:
            x = res(x, self.attn(self.ln_1(x)))
            x = res(x, self.mlp(self.ln_2(x)))
        return x

    def _mhc_forward(self, x):
        """mHC 4-copy：X' = B·X + C·F(A·X)。

        x: (B, T, hc, d)  4 个并行残差流。
        每步：A（1×hc，sigmoid）把 4 流压成 1 流给子层 F；子层只算 1 次；
        C（hc×1，sigmoid）把子层输出展开回 4 流；B（hc×hc，双重随机）混合残差流。
        默认先注意力后 FFN；block_order="ffn_attn" 时对调。
        """
        hc = self.hc_mult
        # skip_attn 层的 attn 槽已是宽 FFN，is_attn=True 子层照样跑（SwiGLU 变换），
        # 只是不涉及注意力；两组 A/B/C 都参与训练。
        if self.config.block_order == "ffn_attn":
            x = self._mhc_sublayer(x, hc, is_attn=False)  # FFN 先
            x = self._mhc_sublayer(x, hc, is_attn=True)   # 注意力后
        else:
            x = self._mhc_sublayer(x, hc, is_attn=True)   # 注意力先
            x = self._mhc_sublayer(x, hc, is_attn=False)  # FFN 后
        return x

    def _mhc_sublayer(self, x, hc, is_attn):
        """mHC 单个子层：A 压流 → 子层 F 跑 1 次 → C 展开 → B 混合残差。
        is_attn=True 取 attn 组 A/B/C + ln_1/attn；False 取 ffn 组 + ln_2/mlp。
        两组权重各自跟随所属子层，重排顺序时无需 remap。
        """
        A = torch.sigmoid(self.raw_A_attn if is_attn else self.raw_A_ffn)
        h_in = (x * A.view(1, 1, hc, 1)).sum(dim=2)            # (B, T, d)
        h_out = (self.attn(self.ln_1(h_in)) if is_attn
                 else self.mlp(self.ln_2(h_in)))               # 子层只跑 1 次
        C = torch.sigmoid(self.raw_C_attn if is_attn else self.raw_C_ffn)
        delta = h_out.unsqueeze(2) * C.view(1, 1, hc, 1)       # (B, T, hc, d)
        B_ds = sinkhorn_knopp(F.softplus(
            self.raw_B_attn if is_attn else self.raw_B_ffn))   # (hc, hc) 双重随机
        return torch.einsum('bthd,hc->btcd', x, B_ds) + delta  # 残差混合 + 子层增量

    def get_moe_aux_loss(self):
        """如果本块是 MoE，返回其辅助损失；否则返回 None（GPT 据此累加）。"""
        if isinstance(self.mlp, MoE):
            return self.mlp.get_aux_loss()
        return None

    def get_indexer_loss(self):
        """本块注意力的 Lightning Indexer 辅助损失（未启用返回 None）。"""
        return self.attn.get_indexer_loss()

class MTPModule(nn.Module):
    """MTP：多 token 预测模块（DeepSeek-V3 核心）。

    普通 GPT 在位置 t 只预测 t+1；MTP 额外提供一个模块，用
    「位置 t 的隐藏状态 + 目标 token t+1 的嵌入」去预测 t+2。
    推理时这些模块不参与（生成没有 targets），训练信号却成倍增强——
    相当于几乎白赚一个"下一步预演"任务。
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.emb_proj    = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.norm = RMSNorm(config.n_embd)
        # 复用单层 Block 做特征融合（内部是因果 attention，正好合适）。
        # 强制关闭 mHC：MTP 的输入是单流 hidden + next_emb，不是 4 流残差。
        self.block = Block(dataclasses.replace(config, use_mhc=False))

    def forward(self, hidden, next_emb):
        # hidden:  (B, T, n_embd) 主模型在位置 t 的隐藏状态
        # next_emb: (B, T, n_embd) 目标 token t+1 的嵌入（提前剧透下一步）
        h = self.hidden_proj(hidden) + self.emb_proj(next_emb)
        h = self.norm(h)
        return self.block(h)

# -----------------------------------------------------------------------------
# Sinkhorn-Knopp 投影：把方阵投影到 Birkhoff 多胞体（双重随机矩阵）
# mHC 用它对 B 做流混合投影，保证 ‖B‖≤1 → 残差变换非扩张，深堆稳定。
# -----------------------------------------------------------------------------

def sinkhorn_knopp(log_alpha, n_iter=20):
    """把 log_alpha 投影成行列和均为 1 的双重随机矩阵（Birkhoff 多胞体）。

    Sinkhorn-Knopp 交替做行/列归一化。exp 保证元素正定（双重随机矩阵要求 ≥0）。
    双重随机矩阵的谱范数 ≤1 → 残差变换非扩张，梯度不会随层数指数爆炸。
    这是 V4 的 mHC 相比普通 Hyper-Connections 的核心稳定性保障。
    """
    M = torch.exp(log_alpha)  # 元素正定
    for _ in range(n_iter):
        M = M / M.sum(dim=-1, keepdim=True)   # 行归一化：每行和为 1
        M = M / M.sum(dim=-2, keepdim=True)   # 列归一化：每列和为 1
    return M


# -----------------------------------------------------------------------------
# Muon 优化器（DeepSeek-V4 用其替代 AdamW）
# 思路：对参数张量做「动量」后，矩阵参数会被正交化（Newton-Schulz 迭代）
# ——把梯度的「方向」拉到单位正交矩阵附近再更新。相比 AdamW 的自适应缩放，
# 正交化保留了梯度向量的几何结构，深层训练更稳、收敛更快。
# -----------------------------------------------------------------------------

def zeropower_via_newtonschulz(G, steps=10, eps=1e-7):
    """用 Newton-Schulz 迭代把矩阵 G 正交化（求「最近正交矩阵」）。
    数学上等于 SVD 里的 U V^T：对 G 做极分解的「旋转」部分，去掉缩放。

    用 DeepSeek-V4 的「经典系数」(2, -1.5, 0.5)：
        M ← 2M - 1.5(M·Mᵀ)M + 0.5(M·Mᵀ)²M
    实测比旧版 (1.5, -0.5) 收敛快约 3 个数量级（同步数、一般随机矩阵）。
    V4 报告里混相 8+2 的「激进系数」(3.4445, -4.7750, 2.0315) 经实测在一般矩阵上
    不满足 p(1)=1（正交矩阵不是不动点），正交化测试不过关，故未采用。

    先除以 Frobenius 范数：因为 ‖X‖_op ≤ ‖X‖_F，归一后算子范数 ≤ 1，
    特征值落在 (0,1]，迭代必然稳定收敛。
    """
    assert G.ndim == 2
    X = G.float()
    # 对非方阵：先在「窄」的一侧做正交化，再转置回去（省算力且更稳）。
    # 注意转置后矩阵方向变了，必须用转置前记录的 was_tall 判断要不要转回，
    # 不能再用 size(0) > size(1) 判断（否则高矩阵会漏转回、形状对不上）。
    was_tall = X.size(0) > X.size(1)
    if was_tall:
        X = X.T
    X = X / (X.norm() + eps)  # Frobenius 范数归一，保证算子范数 ≤ 1
    for _ in range(steps):
        XX = X @ X.T
        X = 2.0 * X - 1.5 * (XX @ X) + 0.5 * (XX @ XX @ X)
    if was_tall:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon 优化器（DeepSeek-V4 / Llama 4 的核心优化器）。

    更新规则（对每个矩阵参数 W）：
        1. 动量  m = 0.95·m + g
        2. 正交化 Q = NewtonSchulz(m)   ← 与 AdamW 的本质区别
        3. 权重衰减插值：g = (1-wd)·Q + wd·W
        4. W -= lr·g

    一维参数（bias、norm 权重）没有「方向」可言，退化为纯动量更新。
    使用标准的 state 机制，checkpoint 里能正常 save/load。
    """

    def __init__(self, params, lr, momentum=0.95, nesterov=True, ns_steps=10,
                 orthogonalization_fn=None):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)
        self.orthogonalization_fn = orthogonalization_fn or zeropower_via_newtonschulz

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            wd = group.get('weight_decay', 0.0)
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if nesterov:
                    # Nesterov 加速：在动量基础上再看一眼当前梯度
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                if g.ndim >= 2:
                    # Muon 的核心：只对矩阵参数做正交化
                    g = self.orthogonalization_fn(g, steps=ns_steps)
                    # 权重衰减：正交化的方向 + 掺一点原参数做收缩
                    g = (1 - wd) * g + wd * p.data
                p.data.add_(g, alpha=-lr)
        return loss


class MuonAdamW:
    """Muon + AdamW 组合优化器（V4 风格）。

    矩阵参数（除 embedding/lm_head）用 Muon 做正交化更新；embedding / lm_head /
    1D 参数（norm/bias）用 AdamW——它们没有矩阵结构，正交化没有意义。

    对 train.py 而言行为像单个优化器：支持 param_groups / step / zero_grad /
    state_dict / load_state_dict，checkpoint 里正常保存。
    """

    def __init__(self, muon, adamw):
        self.muon = muon
        self.adamw = adamw
        self._step_supports_amp_scaling = True  # 兼容 GradScaler（float16 训练）

    @property
    def param_groups(self):
        return self.muon.param_groups + self.adamw.param_groups

    def step(self, closure=None):
        self.muon.step(closure)
        self.adamw.step(closure)

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {'muon': self.muon.state_dict(), 'adamw': self.adamw.state_dict()}

    def load_state_dict(self, sd):
        self.muon.load_state_dict(sd['muon'])
        self.adamw.load_state_dict(sd['adamw'])


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 的 vocab_size 为 50257，向上填充到最近的 64 的倍数以提高效率
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True：在 Linear 里加 bias。False：效果略好且更快
    # 固定架构：RMSNorm + SwiGLU 硬编码。RoPE 与 wpe 二选一。
    use_rope: bool = False      # RoPE 旋转位置编码（True）或可学习位置编码 wpe（False）
    rope_theta: float = 10000.0 # RoPE 基频（可调，DeepSeek 系列对此做了很多文章）
    # --- V4 稳定性技巧：SwiGLU 输出钳制（DeepSeek-V4 报告）---
    # MoE 训练中，SwiGLU 的输出可能冒出极大的数值异常值（outlier），
    # 被路由机制逐层放大后触发 loss spike。V4 直接把门控输出钳制到 [-v, v]。
    swiglu_clamp: float = 0.0   # 钳制区间半宽；0.0 = 关闭（原始行为）
    # --- MoE 混合专家（DeepSeek-V3/V4 核心）。用 MoE 替换 FFN 层 ---
    use_moe: bool = False       # MoE 替换 MLP/SwiGLU
    n_experts: int = 8          # 路由专家总数
    n_top_k: int = 2            # 每个 token 激活的专家数
    moe_aux_weight: float = 0.01 # 负载均衡辅助损失的权重（Switch 式，use_aux_free_balance=False 时用）
    use_shared_expert: bool = False  # V4：始终激活的共享专家（output = shared(x) + Σ routed(x)）
    use_aux_free_balance: bool = False  # V4：aux-free 偏置修正替代 Switch aux loss（无需调 moe_aux_weight）
    balance_factor: float = 0.001   # aux-free 偏置每步更新幅度
    use_sqrtsoftplus: bool = False  # V4：路由打分用 √softplus(logits)*route_scale 替代 softmax
    route_scale: float = 2.5        # √softplus 打分的缩放系数（V4 默认）
    # --- MLA 多头潜在注意力（DeepSeek-V2 核心）。低秩压缩 KV + 部分 RoPE ---
    use_mla: bool = False       # 用 MLA 替换标准 KV 投影
    kv_lora_rank: int = 64      # KV 压缩后的潜在维度
    qk_rope_head_dim: int = 16  # 每头参与旋转的维数（其余维不带位置信息）
    # --- MTP 多 token 预测（DeepSeek-V3/V4 核心）。训练时额外预测未来的 token ---
    use_mtp: bool = False       # 开启多 token 预测
    n_mtp: int = 1              # 额外预测的 token 数（1 = 多预测 t+2）
    mtp_weight: float = 0.3     # MTP 损失在总 loss 中的权重（DeepSeek-V3 建议 0.3）
    # --- V4 优化器：Muon 替代 AdamW ---
    # 矩阵参数经 Newton-Schulz 正交化后更新，深层训练更稳、收敛更快。
    use_muon: bool = False      # True：矩阵参数用 Muon，embedding/lm_head/norm 用 AdamW
    muon_momentum: float = 0.95 # Muon 动量系数
    muon_ns_steps: int = 10     # Newton-Schulz 迭代次数（默认 8 激进 + 2 经典）
    # --- V4 核心：CSA/HCA 混合注意力（简化教育版）---
    # 块级 KV 压缩 + top-k 稀疏块选择 + 滑窗局部注意力 + HCA 重度压缩全局信号。
    # 核心收益：注意力开销从 O(T²) 降到 O(T·(nb + win))，这是 1M 上下文能跑起来的关键。
    use_csa: bool = False       # True：用 CSA 混合注意力（建议搭配 use_rope）
    csa_compress: int = 16      # 块大小 m：每 m 个 token 压成 1 个潜在 KV
    csa_topk: int = 4           # 每个 query 稀疏选几个压缩块（不看全部）
    csa_window: int = 64        # 滑窗：保留最近多少个原始 token（局部细节）
    use_hca: bool = False       # 加 HCA：重度压缩全局信号（无稀疏选择）
    use_csa_learnable: bool = True  # V4：可学习门控池化替代平均池化（压缩块内 m 个 token）
    # --- V4 结构设计升级（实验性，默认全关）---
    # Attention Sinks：每头一个可学习标量偏置，作为 softmax 的"垃圾桶"吸收无关注意力。
    use_attn_sink: bool = False
    # mHC 超连接：4 流并行残差（X_{l+1} = B·X_l + C·F(A·X_l)，A/C sigmoid 有界、B 双重随机）。
    use_mhc: bool = False
    hc_mult: int = 4            # 残差流数（V4 原版 = 4）
    # Lightning Indexer：学习型块选择替代 CSA 的 raw top-k（256 上下文收益有限，先搭框架）。
    use_lightning_indexer: bool = False
    # Hash 路由：前 num_hash_layers 层用 hash(token_id)%n_experts 确定性分配（0 = 禁用）。
    num_hash_layers: int = 0
    # 计算图重排（拓扑实验）：块内子层顺序。
    # "attn_ffn"（默认，标准 Pre-LN 布线）| "ffn_attn"（先 FFN 后注意力）。
    # 只交换执行顺序，不新增参数、不改权重名（attn 永远用 ln_1，mlp 永远用 ln_2）。
    block_order: str = "attn_ffn"
    # 稀疏注意力布线（拓扑实验）：这些层索引跳过注意力（只跑 FFN 子层）。
    # 参数保留但不参与前向 → 参数量严格不变，纯接线改变；空列表 = 所有层都有注意力。
    # 例：[0, 2, 4] → 6 层里第 1/3/5 层只跑 FFN，注意力稀疏布在第 2/4/6 层。
    no_attn_layers: list = field(default_factory=list)
    # 显式记忆 token（拓扑实验，Python-only）：序列前插入 K 个可学习嵌入，参与每一层
    # 注意力+FFN，作为跨 token 的长程全局工作区（直击 Sinks 训满失效/采样死循环的病根）。
    # 0 = 关闭（与老模型完全一致，checkpoint 兼容不破）。
    # 建议 K = csa_compress 的整数倍（如 16），使记忆块与压缩块边界对齐、真实 token 的
    # 块划分零偏移。参数量 = K×n_embd（K=16, d=80 → 1,280，全部参与训练）。
    n_memory_tokens: int = 0
    # --- 对数放缩残差（数值连接实验，零参数）---
    # 普通残差线性相加 x+F(x)；开启后换成对数域 soft-max 合并 LSE(x,F(x))。
    # 有界收缩（≤max+log2）+ 软选择偏置，直接压制 SwiGLU 极端值和「loss 骗低」。
    # 与 mHC 互斥（mHC 已有自己的 4 流残差拓扑，v1 只在标准路径应用）。
    use_lse_residual: bool = False
    # --- 对数放缩门控混合（LSE gate-mix，v2）---
    # 对上轮「纯 LSE 硬替换拖累学力」的修正：残差合并变成可学习标量 α 的插值
    #     x ← α·x + (1-α)·LSE(x, F(x))，α = sigmoid(raw_gate)，raw_gate 初始 0 → α=0.5
    # - α→1：偏向纯线性（保留 lin 臂的强学力 / 表征累加）
    # - α→0：偏向纯 LSE（偏 lse 臂的抗尖刺 / 抗过度自信 → 抗重复坍缩）
    # 模型自己权衡每条残差路径的软硬程度。每层 1 个标量，参数量极小（6 层 = 6 参数）。
    # 与 mHC 不互斥（gate-mix 是标准单流残差内部混合，非 mHC 那种 4 流拓扑）。
    use_lse_gate: bool = False
    # --- 注意力稳定性：QK-Norm（零/近零参数）---
    # 对 q/k 做 L2 归一化 + 每头可学习 scale，再乘 1/sqrt(head_dim)。
    # 目标：把注意力 logits 的尺度拉平，压制「过度自信 → 重复坍缩」，对标准注意力、
    # CSA、MLA 都生效。scale 初始 = sqrt(head_dim)，前向等价于原始 `q·k/sqrt(d)`
    # 的按头缩放，不会改变起点快照。
    use_qk_norm: bool = False
    # --- MoE 稳定性：Router Z-Loss ---
    # DeepSeek 系列用于稳定 MoE 路由的辅助正则：z = logsumexp(router_logits)，
    # 惩罚 z² 的平均，防止路由 logits 数值过大导致训练波动 / 专家崩溃。
    # 权重 0 = 关闭；建议从 1e-4 起步。
    z_loss_weight: float = 0.0

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config, layer_idx=i) for i in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))
        # 显式记忆 token（拓扑实验）：K×n_embd 可学习嵌入，forward 里拼到序列前、
        # 参与每一层注意力+FFN、过完所有层后剥离。是模型可写入/检索的跨 token 长程工作区。
        # 用 nn.Parameter 直接挂在 GPT 上（不进 ModuleDict，避免干扰 wte/lm_head 的 key 结构）。
        if config.n_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.zeros(config.n_memory_tokens, config.n_embd))
        # 用 RoPE 时，位置信息由 attention 内部注入，不需要可学习的位置编码表
        if not config.use_rope:
            self.transformer['wpe'] = nn.Embedding(config.block_size, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # MTP 模块（可选）：额外的"小 transformer 层"，训练时预测更远的 token
        if config.use_mtp:
            self.mtp_modules = nn.ModuleList([MTPModule(config) for _ in range(config.n_mtp)])
            self.mtp_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # 使用 torch.compile() 做权重共享（weight tying）时会产生一些警告：
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # 不完全确定这是什么原因，目前看是无害的。TODO 调查一下
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
        if config.use_mtp:
            # V4：MTP 输出头和主模型共享 lm_head 权重（节省参数，DeepSeek-V3/V4 都这么做）
            self.mtp_head.weight = self.lm_head.weight

        # 初始化所有权重
        self.apply(self._init_weights)
        # 按照 GPT-2 论文，对残差投影应用特殊缩放的初始化
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        # memory_tokens 是裸 Parameter，_init_weights（apply 遍历子模块）不会碰到它，手动初始化。
        # 同嵌入标准：N(0, 0.02)。K=16 时 16×80=1,280 参数，全参与训练。
        if config.n_memory_tokens > 0:
            torch.nn.init.normal_(self.memory_tokens, mean=0.0, std=0.02)

        # 报告参数量
        print("参数量：%.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        返回模型中的参数量。
        按非嵌入参数计数（默认）时，会减去位置嵌入。
        token 嵌入本来也应减去，但由于参数共享，这些参数实际上
        被用作最后一层的权重，所以我们要把它们包含进来。
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # 用 RoPE 时没有 wpe，跳过（cos/sin 是 buffer，本来就不算参数）
            if hasattr(self.transformer, 'wpe'):
                n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"无法前向传播长度为 {t} 的序列，block size 只有 {self.config.block_size}"

        # 前向传播 GPT 模型本身
        tok_emb = self.transformer.wte(idx) # 形状为 (b, t, n_embd) 的 token 嵌入
        if self.config.use_rope:
            # RoPE：位置信息在 attention 内部注入，这里只需要 token embedding
            x = self.transformer.drop(tok_emb)
        else:
            pos = torch.arange(0, t, dtype=torch.long, device=device) # 形状 (t)
            pos_emb = self.transformer.wpe(pos) # 形状为 (t, n_embd) 的位置嵌入
            x = self.transformer.drop(tok_emb + pos_emb)
        # 显式记忆 token：拼到序列前，作为每一层的全局工作区（真实 token 的相对位置不变）。
        # RoPE 表已扩到 block_size+K，前缀位置 0..K-1 正常旋转。
        if self.config.n_memory_tokens > 0:
            mem = self.memory_tokens.unsqueeze(0).expand(b, -1, -1)  # (b, K, n_embd)
            x = torch.cat([mem, x], dim=1)                            # (b, K+t, n_embd)
        if self.config.use_mhc:
            # mHC：4 个残差流从同一个嵌入出发（在流维扩展）
            x = x.unsqueeze(2).expand(b, x.size(1), self.config.hc_mult, self.config.n_embd)
        for block in self.transformer.h:
            x = block(x)
        if self.config.use_mhc:
            # 4 流均值回到 1 流，再给 ln_f / lm_head（V4 用可学习合并，这里用均值简化）
            x = x.mean(dim=2)
        if self.config.n_memory_tokens > 0:
            # 剥离记忆 token 前缀：它们不该喂给 lm_head/MTP（会生成无意义 token）
            x = x[:, self.config.n_memory_tokens:, :]
        x = self.transformer.ln_f(x)

        if targets is not None:
            # 如果给了目标 targets，就同时计算损失
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
            if self.config.use_moe:
                # 把各层 MoE 的负载均衡辅助损失加到总损失上
                moe_loss = torch.zeros(1, device=x.device, dtype=x.dtype)
                for block in self.transformer.h:
                    aux = block.get_moe_aux_loss()
                    if aux is not None:
                        moe_loss = moe_loss + aux
                loss = loss + moe_loss
            if self.config.use_mtp:
                # 多 token 预测：额外预测 t+2、t+3...，按权重加进总损失
                loss = loss + self.config.mtp_weight * self._compute_mtp_loss(x, targets)
            if self.config.use_lightning_indexer:
                # Lightning Indexer 辅助损失：让 indexer 的选块分布逼近真实注意力分布
                # （权重 0.01，作为辅助信号，不喧宾夺主）
                idx_loss = torch.zeros(1, device=x.device, dtype=x.dtype)
                for block in self.transformer.h:
                    aux = block.get_indexer_loss()
                    if aux is not None:
                        idx_loss = idx_loss + aux
                loss = loss + 0.01 * idx_loss
        else:
            # 推理时的小优化：只对最后一个位置前向传播 lm_head
            logits = self.lm_head(x[:, [-1], :]) # 注意：用列表 [-1] 来保留时间维度
            loss = None

        return logits, loss

    def _compute_mtp_loss(self, x, targets):
        """MTP 损失：第 k 个模块用「位置 t 的隐藏状态 + 目标 t+k+1 的嵌入」预测 t+k+2。
        x: (B, T, n_embd) 主模型 ln_f 的输出；targets: (B, T) 训练目标（即 t+1 的正确答案）。
        """
        B, T, C = x.shape
        if T < self.config.n_mtp + 2:
            return torch.zeros(1, device=x.device, dtype=x.dtype)
        mtp_loss = torch.zeros(1, device=x.device, dtype=x.dtype)
        h = x  # 当前"预备预测"的隐藏状态序列
        for k in range(self.config.n_mtp):
            # off：h[j] 当前对应要预测 targets[j+off]（主模型 off=1，预测 t+1）
            if k == 0:
                hidden = h[:, :-2]      # 主模型输出预测的是 t+1，要再前进两步到 t+2
                off = 1
            else:
                hidden = h[:, :-1]      # 上级模块输出预测的是 t+k+1，再前进一步即可
                off = k + 1
            length = hidden.shape[1]
            # 对 embedding 查找用 clamp 替换 -100（loss masking 屏蔽位），
            # 这些位置在 cross_entropy 中仍被 ignore_index=-100 忽略。
            safe_targets = targets.clamp(min=0)
            next_emb = self.transformer.wte(safe_targets[:, off : off+length])       # (B, len, C)
            mtp_targets = targets[:, off+1 : off+1+length]                      # (B, len)
            h = self.mtp_modules[k](hidden, next_emb)                           # (B, len, C)
            logits = self.mtp_head(h)
            mtp_loss = mtp_loss + F.cross_entropy(
                logits.view(-1, logits.size(-1)), mtp_targets.reshape(-1), ignore_index=-100)
        return mtp_loss / self.config.n_mtp

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # 从所有候选参数开始
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # 过滤掉那些不需要梯度的参数
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        if self.config.use_muon:
            # V4：矩阵参数（除 embedding/lm_head）走 Muon；嵌入/输出头/1D 参数用 AdamW 保护。
            # 注意 Muon 不依赖 beta2（没有二阶矩），所以 betas 参数被忽略。
            muon_params, adamw_decay, adamw_nodecay = [], [], []
            for n, p in param_dict.items():
                if n.startswith('transformer.wte') or n.startswith('lm_head'):
                    adamw_decay.append(p)   # 嵌入/输出头无矩阵结构，正交化无意义
                elif p.dim() < 2:
                    adamw_nodecay.append(p) # norm/bias
                else:
                    muon_params.append(p)   # 其余矩阵参数（attention/FFN/router）
            muon = Muon([{'params': muon_params, 'weight_decay': weight_decay}],
                        lr=learning_rate, momentum=self.config.muon_momentum,
                        ns_steps=self.config.muon_ns_steps)
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            extra_args = dict(fused=True) if fused_available and device_type == 'cuda' else {}
            adamw = torch.optim.AdamW(
                [{'params': adamw_decay, 'weight_decay': weight_decay},
                 {'params': adamw_nodecay, 'weight_decay': 0.0}],
                lr=learning_rate, betas=betas, **extra_args)
            return MuonAdamW(muon, adamw)

        # 创建优化器分组。任何 2D 参数都会做权重衰减，其余不做。
        # 即 matmul 和嵌入中的所有权重张量做衰减，所有 bias 和 layernorm 参数不做。
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        # 创建 AdamW 优化器，如果可用就使用 fused 版本
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ 估算模型算力利用率（MFU），以 A100 bfloat16 峰值 FLOPS 为单位 """
        # 首先估算每次迭代我们要做的 flops 数。
        # 参考 PaLM 论文附录 B：https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # 用 A100 bfloat16 峰值 flops 的比例来表示我们的 flops 吞吐量
        flops_achieved = flops_per_iter * (1.0/dt) # 每秒
        flops_promised = 312e12 # A100 GPU bfloat16 峰值 flops 是 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        输入一个条件序列 idx（形状为 (b,t) 的 LongTensor），并连续生成 max_new_tokens 次，
        每次把预测结果喂回模型。
        做这个之前，你很可能需要把模型切换到 model.eval() 模式。
        """
        for _ in range(max_new_tokens):
            # 如果序列上下文太长，必须在 block_size 处裁剪
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # 前向传播模型，得到序列中各位置对应的 logits
            logits, _ = self(idx_cond)
            # 取出最后一步的 logits，并按期望的温度缩放
            logits = logits[:, -1, :] / temperature
            # 可选：把 logits 裁剪到只保留 top k 个选项
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # 应用 softmax 把 logits 转成（归一化的）概率
            probs = F.softmax(logits, dim=-1)
            # 从分布中采样
            idx_next = torch.multinomial(probs, num_samples=1)
            # 把采样出的索引追加到序列末尾并继续
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
