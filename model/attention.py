"""CausalSelfAttention：标准注意力 + CSA/HCA 压缩稀疏注意力 + MLA 多头潜在注意力。"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from .utils import RMSNorm, precompute_rope_freqs, apply_rotary_pos_emb


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
            # CSA/HCA 的 Q/K/V 投影（与 MLA 的 KV 压缩路径分开，更直白）。
            # 默认三个独立 Linear；use_csa_fused_qkv=True 时合三为一（省 kernel launch，
            # 数学等价：拼一个 [Wq; Wk; Wv] 大矩阵一次 matmul 再 split）。
            if config.use_csa_fused_qkv:
                self.c_qkv_csa = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
            else:
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
            # Attention Sink 与 is_causal 的 SDPA 不兼容（flash 内核不支持追加 softmax 列），
            # 旧代码在 flash 可用时会走这里、把 sink 静默丢弃（与注释"回退手动"不符）。
            # 这里用 attn_mask 注入 sink：k/v 各补一列零，掩码末尾列加 sink[h]——数学上与
            # 手动补列完全等价（softmax 多一项 exp(sink)，value 为零 → 贡献 0）。
            if self.use_attn_sink:
                Bn, nh, Tq, d = q.shape
                k_pad = torch.cat([k, k.new_zeros(Bn, nh, 1, d)], dim=2)   # (B,nh,T+1,d)
                v_pad = torch.cat([v, v.new_zeros(Bn, nh, 1, d)], dim=2)
                mask = torch.zeros(Bn, nh, Tq, Tq + 1, device=q.device, dtype=q.dtype)
                causal = torch.triu(torch.ones(Tq, Tq, device=q.device, dtype=torch.bool), diagonal=1)
                mask[:, :, :, :Tq].masked_fill_(causal.view(1, 1, Tq, Tq), float('-inf'))
                mask[:, :, :, Tq] = self.attn_sink.view(1, nh, 1)
                y = torch.nn.functional.scaled_dot_product_attention(
                    q, k_pad, v_pad, attn_mask=mask,
                    dropout_p=self.dropout if self.training else 0,
                    is_causal=False, scale=attn_scale)
            else:
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
                v = torch.cat([v, v.new_zeros(B, self.n_head, 1, v.size(-1))], dim=-2)  # (B,nh,T+1,hs)
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

        if getattr(self.config, 'use_csa_fused_qkv', False):
            q, k, v = self.c_qkv_csa(x).split(self.n_embd, dim=2)
            q = q.view(B, T, nh, d)
            k = k.view(B, T, nh, d)
            v = v.view(B, T, nh, d)
        else:
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
            if getattr(self.config, 'use_csa_bmm', False):
                # 块打分：显式批量 matmul（等价 einsum 'bthd,bnhd->bthn'，走 cuBLAS gemm）
                s_blk = torch.bmm(
                    q.permute(0, 2, 1, 3).reshape(B * nh, T, d),
                    k_blocks.permute(0, 2, 3, 1).reshape(B * nh, d, nb),
                ).view(B, nh, T, nb).permute(0, 2, 1, 3)  # (B,T,nh,nb)
            else:
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
                if getattr(self.config, 'use_csa_bmm', False):
                    # 外积批量 matmul（等价 einsum 'bth,bnh->btn'）
                    idx_scores = torch.bmm(q_idx, k_idx.permute(0, 2, 1))  # (B,T,nh)@(B,nh,nb)→(B,T,nb)
                else:
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
            a_blk = F.softmax(s_blk, dim=-1)                          # (B,T,nh,nb')
            if getattr(self.config, 'use_csa_bmm', False):
                # 块聚合：显式批量 matmul（等价 einsum 'bthn,bnhd->bthd'）。
                # 注意 v_blocks 是 (B,nb,nh,d)，要按 (B,nh,nb,d) 折叠 → permute(0,2,1,3)。
                nb_p = v_blocks.shape[1]
                y_comp = torch.bmm(
                    a_blk.permute(0, 2, 1, 3).reshape(B * nh, T, nb_p),
                    v_blocks.permute(0, 2, 1, 3).reshape(B * nh, nb_p, d),
                ).view(B, nh, T, d).permute(0, 2, 1, 3)               # (B,T,nh,d)
            else:
                y_comp = torch.einsum('bthn,bnhd->bthd', a_blk, v_blocks)  # (B,T,nh,d)
            y_comp = y_comp * has_prior.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).float()

        # --- 3) 滑窗：最近 win 个原始 token 的局部因果注意力 ---
        # 滑窗允许看自己（j ≤ i），保证每个位置至少有一个合法键，避免全 -inf。
        # 有 SDPA 时走 fused kernel；sink 经 attn_mask 注入（k/v 补零列），与手动补列
        # 数学等价。无 SDPA 时退回手动 einsum（含 sink 补列）。
        win = min(win, T)
        i = torch.arange(T, device=x.device)
        win_causal = (i.unsqueeze(-1) <= i.unsqueeze(0)) & (i.unsqueeze(0) - i.unsqueeze(-1) <= win)
        scale = 1.0 if self.use_qk_norm else 1.0 / math.sqrt(d)
        if self.flash:
            qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # (B,nh,T,d)
            if self.use_attn_sink:
                k_pad = torch.cat([kt, kt.new_zeros(B, nh, 1, d)], dim=2)   # (B,nh,T+1,d)
                v_pad = torch.cat([vt, vt.new_zeros(B, nh, 1, d)], dim=2)
                mask = torch.zeros(B, nh, T, T + 1, device=x.device, dtype=x.dtype)
                mask[:, :, :, :T].masked_fill_(~win_causal.unsqueeze(0).unsqueeze(1), float('-inf'))
                mask[:, :, :, T] = self.attn_sink.view(1, nh, 1)
                y_win = torch.nn.functional.scaled_dot_product_attention(
                    qt, k_pad, v_pad, attn_mask=mask,
                    dropout_p=0.0, is_causal=False, scale=scale)
            else:
                mask = torch.zeros(B, nh, T, T, device=x.device, dtype=x.dtype)
                mask.masked_fill_(~win_causal.unsqueeze(0).unsqueeze(1), float('-inf'))
                y_win = torch.nn.functional.scaled_dot_product_attention(
                    qt, kt, vt, attn_mask=mask,
                    dropout_p=0.0, is_causal=False, scale=scale)
            y_win = y_win.transpose(1, 2)                                    # (B,T,nh,d)
        else:
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
            if getattr(self.config, 'use_csa_bmm', False):
                # HCA 全局聚合：显式批量 matmul（等价 einsum 'tn,bnhd->bthd'）
                cb = causal_block.float().unsqueeze(0).expand(B, -1, -1)      # (B,T,nb)
                k_glob = torch.bmm(cb, k_blocks.reshape(B, nb, nh * d)).view(B, T, nh, d) / \
                         n_allowed.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                v_glob = torch.bmm(cb, v_blocks_real.reshape(B, nb, nh * d)).view(B, T, nh, d) / \
                         n_allowed.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            else:
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