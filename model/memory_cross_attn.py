# -*- coding: utf-8 -*-
"""RETRO-lite 记忆读取模块（dev-notes/79 B1）。

把 no_grad 的 chunk 记忆库读进残差流：

  查询：每个 chunk（默认 64 token）用块内 hidden 均值做一次检索
  检索：q · storeᵀ → top-k（可屏蔽与查询位置过近的条目，防训练集自匹配）
  读取：chunk 内所有 token 对 top-k 记忆做多头 cross-attention
  注入：h ← h + gate · wo(CrossAttn(wq(h), wk(mem), wv(mem)))

store 是 buffer（no_grad，不参与优化器）；可训的只有 wq/wk/wv/wo + gate。
"""
import torch
import torch.nn as nn


class MemoryCrossAttention(nn.Module):
    def __init__(self, n_embd, store_keys, store_pos=None, chunk=64, top_k=4,
                 n_head=8, gate_init=0.1, exclude_radius=0):
        super().__init__()
        self.n_embd = int(n_embd)
        self.chunk = int(chunk)
        self.top_k = int(top_k)
        self.n_head = int(n_head)
        self.head_dim = self.n_embd // self.n_head
        self.exclude_radius = int(exclude_radius)
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)
        nn.init.zeros_(self.wo.weight)  # 零初始化输出：记忆起点是 no-op，只在有用时才被学起来
        # 逐 token 门控：g_t = σ(w·h_t + b)，让模型自己决定哪些位置注入
        self.gate_proj = nn.Linear(n_embd, 1, bias=False)
        nn.init.zeros_(self.gate_proj.weight)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        # store：no_grad，永不被优化器触碰
        store = store_keys.detach().clone()
        self.register_buffer("store", store, persistent=False)
        if store_pos is not None:
            self.register_buffer("store_pos", store_pos.detach().to(torch.int32),
                                 persistent=False)
        else:
            self.store_pos = None

    def _retrieve(self, q, q_pos=None):
        """q: (B,C,D) → idx (B,C,k)。检索本身不可微，走 no_grad。"""
        if getattr(self, "random_retrieve", False):  # 对照：随机检索（内容无意义）
            B, C, _ = q.shape
            return torch.randint(0, self.store.shape[0], (B, C, self.top_k), device=q.device)
        with torch.no_grad():
            sim = q.to(self.store.dtype) @ self.store.t()  # (B,C,N)
            if self.exclude_radius > 0 and q_pos is not None and self.store_pos is not None:
                d = (self.store_pos[None, None, :] - q_pos[:, :, None]).abs()
                sim = sim.masked_fill(d < self.exclude_radius, float("-inf"))
            idx = sim.topk(self.top_k, dim=-1).indices  # (B,C,k)
        return idx

    def forward(self, h, q_pos=None):
        """h: (B,T,D) → delta (B,T,D)，加到 h 上。"""
        B, T, D = h.shape
        C = T // self.chunk
        Tc = C * self.chunk
        if Tc == 0:
            return torch.zeros_like(h)
        hc = h[:, :Tc].reshape(B, C, self.chunk, D).float()
        q = hc.mean(dim=2)  # (B,C,D)
        idx = self._retrieve(q, q_pos)  # (B,C,k)
        mem = self.store[idx].float()  # (B,C,k,D)
        K = self.wk(mem)  # (B,C,k,D)
        V = self.wv(mem)
        Q = self.wq(hc)   # (B,C,chunk,D)
        # 多头
        Q = Q.reshape(B, C, self.chunk, self.n_head, self.head_dim).permute(0, 1, 3, 2, 4)
        K = K.reshape(B, C, self.top_k, self.n_head, self.head_dim).permute(0, 1, 3, 4, 2)
        V = V.reshape(B, C, self.top_k, self.n_head, self.head_dim).permute(0, 1, 3, 2, 4)
        att = (Q @ K) / (self.head_dim ** 0.5)   # (B,C,H,chunk,k)
        att = att.softmax(dim=-1)
        out = (att @ V).permute(0, 1, 3, 2, 4).reshape(B, C, self.chunk, D)
        out = self.wo(out)  # (B,C,chunk,D)
        g = torch.sigmoid(self.gate_proj(hc) + self.gate)  # (B,C,chunk,1) 逐 token 门控
        out = (g * out).reshape(B, Tc, D)
        delta = torch.zeros(B, T, D, dtype=h.dtype, device=h.device)
        delta[:, :Tc] = out.to(h.dtype)
        return delta
