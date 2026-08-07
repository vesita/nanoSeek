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
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm，但带可选的 bias。PyTorch 不支持直接 bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

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
        self.rope_head_dim = config.qk_rope_head_dim if (config.use_mla or config.use_csa) else (config.n_embd // config.n_head)
        if config.use_csa:
            # CSA/HCA 的独立 Q/K/V 投影（与 MLA 的 KV 压缩路径分开，更直白）。
            # 输入直接投影出 K/V，再在 forward 里做块级压缩。
            self.q_proj_csa = nn.Linear(config.n_embd, config.n_embd, bias=False)
            self.k_proj_csa = nn.Linear(config.n_embd, config.n_embd, bias=False)
            self.v_proj_csa = nn.Linear(config.n_embd, config.n_embd, bias=False)
        if config.use_mla:
            assert config.qk_rope_head_dim % 2 == 0 and config.qk_rope_head_dim <= self.head_dim, \
                "qk_rope_head_dim 需为偶数且不超过 head_dim"
            self.q_proj  = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            self.kv_down = nn.Linear(config.n_embd, config.kv_lora_rank, bias=config.bias)
            self.kv_act  = nn.SiLU()
            self.k_up    = nn.Linear(config.kv_lora_rank, config.n_embd, bias=config.bias)
            self.v_up    = nn.Linear(config.kv_lora_rank, config.n_embd, bias=config.bias)
        # flash attention 能让 GPU 跑得飞快，但只在 PyTorch >= 2.0 才支持
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("警告：正在使用慢速 attention。Flash Attention 需要 PyTorch >= 2.0")
            # 因果掩码，确保 attention 只作用于输入序列左侧的位置
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
        # 用 RoPE 的话，预计算 cos/sin 表并注册成 buffer（随模型移动设备、进 checkpoint）
        if self.use_rope:
            # MLA 只旋转每头的前 qk_rope_head_dim 维，表长和 rope_head_dim 一致
            cos, sin = precompute_rope_freqs(self.rope_head_dim, config.block_size, config.rope_theta)
            self.register_buffer("cos", cos)  # (block_size, rope_head_dim)
            self.register_buffer("sin", sin)

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

            if self.use_rope:
                # 旋转位置编码：只对 q 和 k 做（v 不旋转），这样 q·k 携带相对位置
                q, k = apply_rotary_pos_emb(q, k, self.cos[:T], self.sin[:T])

        # 把 head 维移到第 2 维：(B, nh, T, hs)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 因果自注意力；自注意力：(B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # 使用 Flash Attention CUDA 内核的高效 attention
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # 手动实现 attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
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

        q = self.q_proj_csa(x).view(B, T, nh, d)
        k = self.k_proj_csa(x).view(B, T, nh, d)
        v = self.v_proj_csa(x).view(B, T, nh, d)

        if self.use_rope:
            # 部分 RoPE：只旋转前 rope_head_dim 维（与 MLA 一致）
            q_rope, q_nope = q[..., :self.rope_head_dim], q[..., self.rope_head_dim:]
            k_rope, k_nope = k[..., :self.rope_head_dim], k[..., self.rope_head_dim:]
            q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, self.cos[:T], self.sin[:T])
            q = torch.cat([q_rope, q_nope], dim=-1)
            k = torch.cat([k_rope, k_nope], dim=-1)

        # --- 1) 块级压缩：每 m 个连续 token 的 K/V 平均池化成 1 个潜在 ---
        T_ok = (T // m) * m
        nb = T_ok // m
        k_blocks = k[:, :T_ok].view(B, nb, m, nh, d).mean(dim=2)   # (B, nb, nh, d)
        v_blocks = v[:, :T_ok].view(B, nb, m, nh, d).mean(dim=2)

        # --- 2) 稀疏块选择：query 只能看「它所在块之前」的块，再取 top-k ---
        bq = torch.arange(T, device=x.device) // m                  # 每个 query 属于哪个块
        causal_block = bq.unsqueeze(-1) > torch.arange(nb, device=x.device)  # (T, nb)
        has_prior = causal_block.any(dim=-1)                        # (T,) 该 query 有没有历史块
        s_blk = torch.einsum('bthd,bnhd->bthn', q, k_blocks) / math.sqrt(d)  # (B,T,nh,nb)
        s_blk = s_blk.masked_fill(~causal_block.unsqueeze(0).unsqueeze(2), float('-inf'))
        # 稀疏：只保留每个 query 得分最高的 topk 个块（其余 -inf，softmax 后为 0）
        topk_vals, _ = s_blk.topk(self.config.csa_topk, dim=-1)
        s_blk = s_blk.masked_fill(s_blk < topk_vals[..., [-1]], float('-inf'))
        # 没有历史块的 query：整行置 0（否则 softmax 对全 -inf 产生 NaN）
        s_blk = s_blk.masked_fill(~has_prior.view(1, T, 1, 1), 0.0)
        a_blk = F.softmax(s_blk, dim=-1)                            # (B,T,nh,nb)
        y_comp = torch.einsum('bthn,bnhd->bthd', a_blk, v_blocks)   # (B,T,nh,d)
        y_comp = y_comp * has_prior.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).float()

        # --- 3) 滑窗：最近 win 个原始 token 的局部因果注意力 ---
        # 滑窗允许看自己（j ≤ i），保证每个位置至少有一个合法键，避免全 -inf。
        win = min(win, T)
        i = torch.arange(T, device=x.device)
        win_causal = (i.unsqueeze(-1) <= i.unsqueeze(0)) & (i.unsqueeze(0) - i.unsqueeze(-1) <= win)
        s_win = torch.einsum('bthd,bjhd->bthj', q, k) / math.sqrt(d)  # (B,T,nh,T)
        s_win = s_win.masked_fill(~win_causal.unsqueeze(0).unsqueeze(2), float('-inf'))
        y_win = torch.einsum('bthj,bjhd->bthd', F.softmax(s_win, dim=-1), v)

        y = y_comp + y_win

        # --- 4) HCA：重度压缩的全局信号（可选）---
        # 把所有允许的压缩块再平均成一个全局潜在（不做稀疏选择 = 重度压缩），
        # 每个 query 加上它作为全局上下文。这是"全文一句话摘要"式的粗粒度信号。
        if self.config.use_hca:
            n_allowed = causal_block.float().sum(dim=-1).clamp(min=1)  # (T,)
            k_glob = torch.einsum('tn,bnhd->bthd', causal_block.float(), k_blocks) / \
                     n_allowed.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            v_glob = torch.einsum('tn,bnhd->bthd', causal_block.float(), v_blocks) / \
                     n_allowed.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            # 单个全局 key 的 softmax 恒为 1，等价于直接加上这份全局摘要
            y = y + v_glob

        # 合并 head： (B, T, nh, d) → (B, T, C)
        return y.transpose(1, 2).contiguous().view(B, T, C)

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class SwiGLU(nn.Module):
    """SwiGLU：门控前馈网络，DeepSeek/LLaMA 的标准 FFN。
    输出 = SiLU(x W1) ⊙ (x W2)，比单一路径的 GELU MLP 表达能力更强。
    参数量对比：
      MLP    : W(embd,4embd) + W(4embd,embd) = 8·embd²
      SwiGLU : W(embd,h) + W(embd,h) + W(h,embd) = 3·h·embd
    取 h = 8/3·embd 时两者参数量持平 —— 这就是 LLaMA 用 hidden = 8/3·n_embd 的来历。
    """

    def __init__(self, config):
        super().__init__()
        self.config = config  # 保存 config，forward 里可能要用到钳制等技巧
        hidden = int(8 * config.n_embd / 3)  # ≈ 2.67·n_embd
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

    def __init__(self, config):
        super().__init__()
        self.n_experts = config.n_experts
        self.n_top_k = config.n_top_k
        self.moe_aux_weight = config.moe_aux_weight
        self.use_anticipatory_routing = config.use_anticipatory_routing
        self.ar_momentum = config.ar_momentum
        # 路由器：给每个 token 在每个专家上打一个分
        self.router = nn.Linear(config.n_embd, config.n_experts, bias=False)
        # 专家：复用 SwiGLU/MLP，每个专家是一份完整的 FFN
        ExpertType = SwiGLU if config.use_swiglu else MLP
        self.experts = nn.ModuleList([ExpertType(config) for _ in range(config.n_experts)])
        # 本次前向累积的辅助损失，forward 后由 GPT 取走并清零
        self.aux_loss = torch.tensor(0.0)
        if self.use_anticipatory_routing:
            # 慢路由：当前路由器的 EMA 平滑副本（buffer，不参与优化）。
            # 与骨干网络的更新解耦，负责"离散地选哪些专家"。
            self.register_buffer('router_slow', self.router.weight.detach().clone())

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)          # (B*T, C)，把 batch 和序列压平逐个 token 路由
        N = x_flat.shape[0]

        # 路由打分：softmax 得到每个 token 在每个专家上的概率
        if self.use_anticipatory_routing:
            # —— 预判路由（V4）：让路由决策与骨干更新解耦 ——
            # V4 用"预判器"预测路由参数的未来状态来路由；这里用 EMA 平滑副本演示同原理：
            #   · 离散选择（选哪些专家）：用旧参数 router_slow → 切断"路由-骨干"反馈回路，
            #     异常值不会被当前梯度逐层放大（这正是 V4 防 loss spike 的机制之一）。
            #   · 连续门控权重：仍用当前 router → 主损失梯度能流回路由器，专家偏好照常学习。
            with torch.no_grad():
                self.router_slow.mul_(1 - self.ar_momentum).add_(
                    self.router.weight, alpha=self.ar_momentum)
            slow_probs = F.softmax(F.linear(x_flat, self.router_slow), dim=-1)
            _, top_k_indices = slow_probs.topk(self.n_top_k, dim=-1)      # 离散选择：旧参数
            router_probs = F.softmax(self.router(x_flat), dim=-1)         # 门控：当前参数
            top_k_probs = router_probs.gather(1, top_k_indices)           # (N, k)
            top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-6)
        else:
            router_logits = self.router(x_flat)               # (N, n_experts)
            router_probs = F.softmax(router_logits, dim=-1)   # (N, n_experts)
            # 选 top-k 个专家，门控权重在选中的 k 个之间重新归一化
            top_k_probs, top_k_indices = router_probs.topk(self.n_top_k, dim=-1)  # 各 (N, k)
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # 负载均衡辅助损失（Switch Transformer 论文的做法）：
        #   f_i = 第 i 个专家实际接到的 token 比例
        #   P_i = 路由器给第 i 个专家的平均概率
        # 两者都高意味着该专家又热门又常被选，均衡时 sum(f_i * P_i) 取最小
        one_hot = F.one_hot(top_k_indices, self.n_experts).float()   # (N, k, n_experts)
        f_i = one_hot.sum(dim=(0, 1)) / (N * self.n_top_k)           # (n_experts,)
        P_i = router_probs.mean(dim=0)                               # (n_experts,)
        self.aux_loss = self.moe_aux_weight * self.n_experts * (f_i * P_i).sum()

        # 逐个专家计算：把路由到它的 token 收集起来算 FFN，再按门控权重放回原处
        output = x_flat.new_zeros(N, C)
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
        """取出本次前向累积的辅助损失并清零。"""
        loss = self.aux_loss
        self.aux_loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
        return loss

def sinkhorn_knopp(logits, iters=10):
    """mHC 的流形约束：把 2×2 矩阵投影到「双随机矩阵」（每行、每列和都为 1，元素非负）。
    交替做行归一、列归一——这就是最优传输里的 Sinkhorn 迭代。
    双随机矩阵的谱范数恒为 1：信号经过每个超连接最多不被放大，
    这是 V4 在 1.6T 参数量下训练稳定（不 loss spike）的关键保证。
    """
    m = F.softplus(logits)                       # softplus：非负且处处可导
    for _ in range(iters):
        m = m / m.sum(dim=1, keepdim=True)       # 行归一：每行和为 1
        m = m / m.sum(dim=0, keepdim=True)       # 列归一：每列和为 1
    return m


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        # 归一化：RMSNorm（modern）或 LayerNorm（原始 GPT-2）
        self.ln_1 = RMSNorm(config.n_embd) if config.use_rmsnorm else LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd) if config.use_rmsnorm else LayerNorm(config.n_embd, bias=config.bias)
        # 前馈：MoE > SwiGLU > 标准 GELU MLP（优先级由配置开关决定）
        self.mlp = MoE(config) if config.use_moe else (SwiGLU(config) if config.use_swiglu else MLP(config))
        # mHC 超连接：2×2 混合矩阵的 logits，forward 时经 Sinkhorn 投影成双随机矩阵。
        # logits 全 0 → softplus(0)=ln2 → 投影后 ≈ [[0.5,0.5],[0.5,0.5]]，两条流均衡起步。
        if config.use_mhc:
            self.mix = nn.Parameter(torch.zeros(2, 2))

    def forward(self, x, z=None):
        if self.config.use_mhc:
            # 两条流：x 是「工作流」（喂给 attention/FFN），z 是「记忆流」（跨层累积）。
            # 每层算出的块输出 z_block，同时按双随机矩阵混合进两条流。
            # 注意这里不再有 x + f(x) 的残差——混合矩阵取代了残差连接。
            z = x if z is None else z
            z_block = self.mlp(self.ln_2(self.attn(self.ln_1(x))))
            m = sinkhorn_knopp(self.mix)               # (2,2) 双随机矩阵
            x_new = m[0, 0] * x + m[0, 1] * z_block    # 工作流：旧 x 和块输出混合
            r_new = m[1, 0] * z + m[1, 1] * z_block    # 记忆流：旧 z 和块输出混合
            return x_new, r_new
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def get_moe_aux_loss(self):
        """如果本块是 MoE，返回其辅助损失；否则返回 None（GPT 据此累加）。"""
        if isinstance(self.mlp, MoE):
            return self.mlp.get_aux_loss()
        return None

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
        self.norm = RMSNorm(config.n_embd) if config.use_rmsnorm else LayerNorm(config.n_embd, bias=config.bias)
        # 复用单层 Block 做特征融合（内部是因果 attention，正好合适）
        self.block = Block(config)

    def forward(self, hidden, next_emb):
        # hidden:  (B, T, n_embd) 主模型在位置 t 的隐藏状态
        # next_emb: (B, T, n_embd) 目标 token t+1 的嵌入（提前剧透下一步）
        h = self.hidden_proj(hidden) + self.emb_proj(next_emb)
        h = self.norm(h)
        out = self.block(h)
        if isinstance(out, tuple):
            # mHC 模式：Block 返回 (工作流, 记忆流)，MTP 只取工作流做预测
            out = out[0]
        return out

# -----------------------------------------------------------------------------
# Muon 优化器（DeepSeek-V4 用其替代 AdamW）
# 思路：对参数张量做「动量」后，矩阵参数会被正交化（Newton-Schulz 迭代）
# ——把梯度的「方向」拉到单位正交矩阵附近再更新。相比 AdamW 的自适应缩放，
# 正交化保留了梯度向量的几何结构，深层训练更稳、收敛更快。
# -----------------------------------------------------------------------------

def zeropower_via_newtonschulz(G, steps=10, eps=1e-7):
    """用 Newton-Schulz 迭代把矩阵 G 正交化（求「最近正交矩阵」）。
    数学上等于 SVD 里的 U V^T：对 G 做极分解的「旋转」部分，去掉缩放。

    经典迭代：X ← (3X − X·Xᵀ·X)/2，等价于对 XᵀX 的特征值 s 做 p(s)=(3−s)/2。
    p(1)=1（正交阵是不动点），s<1 被拉大、s>1 被压小 → 奇异值全部趋于 1。

    先除以 Frobenius 范数：因为 ‖X‖_op ≤ ‖X‖_F，归一后算子范数 ≤ 1，
    特征值落在 (0,1]，迭代必然稳定收敛（这是它和记忆里那组高阶常数不同、
    但被验证可靠的写法）。经典法收敛慢，所以步数要给足。
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
        X = 1.5 * X - 0.5 * X @ X.T @ X
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


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 的 vocab_size 为 50257，向上填充到最近的 64 的倍数以提高效率
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True：在 Linear 和 LayerNorm 里加 bias，和 GPT-2 一样。False：效果略好且更快
    # --- 现代化架构开关（DeepSeek / LLaMA 风格）。默认全关 = 原始 GPT-2 结构，保持兼容 ---
    use_rmsnorm: bool = False   # RMSNorm 替换 LayerNorm
    use_rope: bool = False      # RoPE 替换可学习位置编码 wpe
    use_swiglu: bool = False    # SwiGLU 替换 GELU MLP
    rope_theta: float = 10000.0 # RoPE 基频（可调，DeepSeek 系列对此做了很多文章）
    # --- V4 稳定性技巧：SwiGLU 输出钳制（DeepSeek-V4 报告）---
    # MoE 训练中，SwiGLU 的输出可能冒出极大的数值异常值（outlier），
    # 被路由机制逐层放大后触发 loss spike。V4 直接把门控输出钳制到 [-v, v]。
    swiglu_clamp: float = 0.0   # 钳制区间半宽；0.0 = 关闭（原始行为）
    # --- MoE 混合专家（DeepSeek-V3 核心）。用 MoE 替换 FFN 层 ---
    use_moe: bool = False       # MoE 替换 MLP/SwiGLU
    n_experts: int = 8          # 专家总数
    n_top_k: int = 2            # 每个 token 激活的专家数
    moe_aux_weight: float = 0.01 # 负载均衡辅助损失的权重
    # --- V4 稳定性技巧：预判路由（Anticipatory Routing）---
    # 离散路由选择用 EMA 旧参数副本，与骨干更新解耦，防异常值被路由逐层放大。
    use_anticipatory_routing: bool = False  # True：路由决策用旧参数；False：当前参数
    ar_momentum: float = 0.99   # 慢路由 EMA 系数（越大，路由用的参数越"旧"）
    # --- MLA 多头潜在注意力（DeepSeek-V2 核心）。低秩压缩 KV + 部分 RoPE ---
    use_mla: bool = False       # 用 MLA 替换标准 KV 投影
    kv_lora_rank: int = 64      # KV 压缩后的潜在维度
    qk_rope_head_dim: int = 16  # 每头参与旋转的维数（其余维不带位置信息）
    # --- MTP 多 token 预测（DeepSeek-V3 核心）。训练时额外预测未来的 token ---
    use_mtp: bool = False       # 开启多 token 预测
    n_mtp: int = 1              # 额外预测的 token 数（1 = 多预测 t+2）
    mtp_weight: float = 1.0     # MTP 损失在总 loss 中的权重
    # --- V4 优化器：Muon 替代 AdamW ---
    # 矩阵参数经 Newton-Schulz 正交化后更新，深层训练更稳、收敛更快。
    use_muon: bool = False      # True：configure_optimizers 返回 Muon；False：AdamW
    # --- V4 架构：mHC 流形约束超连接（替换残差连接）---
    # 双流混合（工作流 + 记忆流），混合矩阵约束为双随机矩阵 → 谱范数 = 1。
    use_mhc: bool = False       # True：Block 用 mHC 双流；False：普通 x + f(x) 残差
    # --- V4 核心：CSA/HCA 混合注意力（简化教育版）---
    # 块级 KV 压缩 + top-k 稀疏块选择 + 滑窗局部注意力 + HCA 重度压缩全局信号。
    # 核心收益：注意力开销从 O(T²) 降到 O(T·(nb + win))，这是 1M 上下文能跑起来的关键。
    use_csa: bool = False       # True：用 CSA 混合注意力（建议搭配 use_rope）
    csa_compress: int = 16      # 块大小 m：每 m 个 token 压成 1 个潜在 KV
    csa_topk: int = 4           # 每个 query 稀疏选几个压缩块（不看全部）
    csa_window: int = 64        # 滑窗：保留最近多少个原始 token（局部细节）
    use_hca: bool = False       # 加 HCA：重度压缩全局信号（无稀疏选择）

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        if config.use_rmsnorm:
            self.transformer['ln_f'] = RMSNorm(config.n_embd)
        else:
            self.transformer['ln_f'] = LayerNorm(config.n_embd, bias=config.bias)
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

        # 初始化所有权重
        self.apply(self._init_weights)
        # 按照 GPT-2 论文，对残差投影应用特殊缩放的初始化
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

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
        z = None  # mHC 的记忆流：None 时第一层 Block 内部会用 x 初始化
        for block in self.transformer.h:
            if self.config.use_mhc:
                x, z = block(x, z)   # 两条流都在块间传递
            else:
                x = block(x)
        if self.config.use_mhc:
            # 最终用「记忆流」解码：它累积了所有层的块输出
            x = z
        x = self.transformer.ln_f(x)

        if targets is not None:
            # 如果给了目标 targets，就同时计算损失
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
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
            next_emb = self.transformer.wte(targets[:, off : off+length])       # (B, len, C)
            mtp_targets = targets[:, off+1 : off+1+length]                      # (B, len)
            h = self.mtp_modules[k](hidden, next_emb)                           # (B, len, C)
            logits = self.mtp_head(h)
            mtp_loss = mtp_loss + F.cross_entropy(
                logits.view(-1, logits.size(-1)), mtp_targets.reshape(-1), ignore_index=-1)
        return mtp_loss / self.config.n_mtp

    def crop_block_size(self, block_size):
        # 模型“手术”：必要时减小 block size
        # 例如我们可能加载 GPT2 预训练模型 checkpoint（block size 1024）
        # 但想用更小的 block size 训练一个更小、更简单的模型
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if self.config.use_rope:
            # RoPE：裁剪预计算的 cos/sin 表
            for block in self.transformer.h:
                block.attn.cos = block.attn.cos[:block_size]
                block.attn.sin = block.attn.sin[:block_size]
        else:
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # 默认为空字典
        # 只有 dropout 可以被覆盖，更多说明见下文
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("正在从预训练的 gpt 加载权重：%s" % model_type)

        # n_layer、n_head 和 n_embd 由 model_type 决定
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 1.24 亿参数量
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 3.5 亿参数量
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 7.74 亿参数量
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 15.58 亿参数量
        }[model_type]
        print("强制设置 vocab_size=50257、block_size=1024、bias=True")
        config_args['vocab_size'] = 50257 # GPT 模型 checkpoint 中始终是 50257
        config_args['block_size'] = 1024 # GPT 模型 checkpoint 中始终是 1024
        config_args['bias'] = True # GPT 模型 checkpoint 中始终是 True
        # 加载 GPT-2 预训练权重时必须是原始结构（参数名/形状才能对齐），强行关掉所有开关
        config_args['use_rmsnorm'] = False
        config_args['use_rope'] = False
        config_args['use_swiglu'] = False
        config_args['use_moe'] = False
        config_args['use_mla'] = False
        config_args['use_mtp'] = False
        # 如果需要，可以覆盖 dropout 比率
        if 'dropout' in override_args:
            print(f"正在把 dropout 比率覆盖为 {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # 创建从零初始化的 minGPT 模型
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # 丢弃这个掩码/buffer，它不是参数

        # 初始化一个 huggingface/transformers 模型
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # 拷贝参数，并确保所有参数在名称和形状上对齐匹配
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # 忽略这些，只是个 buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # 同上，只是掩码（buffer）
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # 基本上 OpenAI 的 checkpoint 用的是 "Conv1D" 模块，但我们只想用普通的 Linear
        # 这意味着导入时我们必须转置这些权重
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # 对需要转置的 Conv1D 权重做特殊处理
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # 其它参数直接普通拷贝
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # 从所有候选参数开始
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # 过滤掉那些不需要梯度的参数
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # 创建优化器分组。任何 2D 参数都会做权重衰减，其余不做。
        # 即 matmul 和嵌入中的所有权重张量做衰减，所有 bias 和 layernorm 参数不做。
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        if self.config.use_muon:
            # V4：Muon 替代 AdamW。matrix 组由 Muon 做正交化+权重衰减；
            # 一维参数组（bias/norm）纯动量更新（Muon 里 ndim<2 的分支）。
            # 注意 Muon 不依赖 beta2（没有二阶矩），所以 betas 参数被忽略。
            return Muon(optim_groups, lr=learning_rate)

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
