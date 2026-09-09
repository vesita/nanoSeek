# -*- coding: utf-8 -*-
"""神经元级神经数据库 (NeuronDB) —— v6。

在 v5（有序无梯度值矩阵 + 精确后缀键 + 神经元向量 value）基础上按杠杆顺序优化：

  1. 多长度后缀 backoff（覆盖率 ↑）
     同时建多级表（默认 L=12/8/6，槽位分别 2^21/2^22/2^21）；读出取**最长命中**：
     长 L 精度高、短 L 兜底覆盖。不同长度 = 不同的键，等价于扩大有效键空间。
  2. 收缩估计（SNR ↑）
     v̂ = n/(n+κ)·mean，小 count 的槽向 0 收缩，避免噪声注入。
  3. count 门控（SNR ↑）
     λ = σ( gate(h) + α_lvl + β·log1p(n) )，让可靠槽多注入；每级一个可学习偏置 α_lvl。

value 仍是 E[−∂CE/∂h | suffix]（512 维神经元向量，固定投影到 value_dim）。
存储全为 no_grad：hash_tab / count / value 永不被优化器触碰。
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_MIX = np.uint64(0x9E3779B97F4A7C15)
_BASE = np.uint64(1000003)


def suffix_hash_np(seq_np, L, bits):
    """精确后缀哈希（CPU numpy uint64）。

    seq_np: (B, T) 整数序列。位置 t 的键 = seq[t-L+1 .. t]（含当前 token）。
    返回 (全哈希 uint64 (B,T), 槽索引 int64 (B,T))。
    """
    B, T = seq_np.shape
    acc = np.zeros((B, T), dtype=np.uint64)
    seq_u = seq_np.astype(np.uint64)
    for j in range(L):
        if j == 0:
            sh = seq_u
        else:
            sh = np.zeros((B, T), dtype=np.uint64)
            sh[:, j:] = seq_u[:, :T - j]
        acc = acc * _BASE + sh
    mixed = acc * _MIX
    return acc, (mixed >> np.uint64(64 - bits)).astype(np.int64)


def parse_levels(s):
    """'12:21,8:22,6:21' → ((12,21),(8,22),(6,21))。"""
    out = []
    for part in s.split(","):
        L, bits = part.split(":")
        out.append((int(L), int(bits)))
    return tuple(out)


class NeuronDB(nn.Module):
    """多长度后缀 backoff 的神经元值矩阵。"""

    def __init__(self, n_embd=512, levels=((12, 21), (8, 22), (6, 21)), value_dim=128,
                 device=None, dtype=torch.bfloat16, proj_seed=0,
                 decoder_init=1.0, gate_bias=-2.0, kappa=3.0, val_proj=None):
        super().__init__()
        self.levels = tuple(tuple(x) for x in levels)
        self.n_levels = len(self.levels)
        self.n_embd = n_embd
        self.value_dim = int(value_dim) or n_embd
        self.dtype = dtype
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.kappa = float(kappa)

        # 固定值投影（不训练）
        g = torch.Generator().manual_seed(proj_seed)
        if val_proj is not None:
            proj = val_proj.clone().float()
        elif self.value_dim == n_embd:
            proj = torch.eye(n_embd)
        else:
            proj = torch.randn(n_embd, self.value_dim, generator=g) / math.sqrt(n_embd)
        self.register_buffer("val_proj", proj)

        # 每级有序无梯度存储
        for i, (L, bits) in enumerate(self.levels):
            n = 1 << bits
            self.register_buffer(f"hash_tab_{i}", torch.zeros(n, dtype=torch.int64))
            self.register_buffer(f"count_{i}", torch.zeros(n, dtype=torch.float32))
            self.register_buffer(f"value_{i}", torch.zeros(n, self.value_dim, dtype=dtype))

        # 接口（可训练）
        self.decoder = nn.Linear(self.value_dim, n_embd, bias=False)
        with torch.no_grad():
            if decoder_init > 0:
                pinv = torch.eye(n_embd) if self.value_dim == n_embd \
                    else torch.linalg.pinv(proj).t()
                self.decoder.weight.copy_(pinv * decoder_init)
            else:
                nn.init.zeros_(self.decoder.weight)
        self.read_gate = nn.Linear(n_embd, 1, bias=True)
        nn.init.constant_(self.read_gate.bias, gate_bias)
        self.alpha = nn.Parameter(torch.zeros(self.n_levels))   # 每级门控偏置
        self.beta = nn.Parameter(torch.tensor(0.0))             # count 权重

        self.to(self.device)

    # ------------------------------------------------------------------ 写
    @torch.no_grad()
    def _write_level(self, i, idx, hh, proj):
        ht = getattr(self, f"hash_tab_{i}")
        ct = getattr(self, f"count_{i}")
        vl = getattr(self, f"value_{i}")
        cnt = ct[idx]
        match = (cnt == 0) | (ht[idx] == hh)
        if not bool(match.any()):
            return 0
        i2, h2, p2 = idx[match], hh[match], proj[match]
        empty = cnt[match] == 0
        if bool(empty.any()):
            ht[i2[empty]] = h2[empty]
        u, inv = torch.unique(i2, return_inverse=True)
        num = torch.zeros(u.numel(), self.value_dim, device=self.device, dtype=torch.float32)
        num.index_add_(0, inv, p2.float())
        den = torch.zeros(u.numel(), device=self.device, dtype=torch.float32)
        den.index_add_(0, inv, torch.ones(i2.numel(), device=self.device))
        old_cnt = ct[u]
        old_sum = vl[u].float() * old_cnt.unsqueeze(1)
        new_cnt = old_cnt + den
        vl[u] = ((old_sum + num) / new_cnt.clamp_min(1e-9).unsqueeze(1)).to(self.dtype)
        ct[u] = new_cnt
        return int(i2.numel())

    @torch.no_grad()
    def write(self, seq_np, delta):
        """seq_np: (B,T)；delta: (N, n_embd)。写所有级，返回成功写入条数。"""
        d = delta.to(self.device).float()
        proj = (d @ self.val_proj.float()).to(self.dtype)
        total = 0
        for i, (L, bits) in enumerate(self.levels):
            acc_np, idx_np = suffix_hash_np(seq_np, L, bits)
            idx = torch.from_numpy(idx_np.reshape(-1)).to(self.device)
            hh = torch.from_numpy(acc_np.view(np.int64).reshape(-1)).to(self.device)
            total += self._write_level(i, idx, hh, proj)
        return total

    # ------------------------------------------------------------------ 读（最长命中 backoff）
    def read(self, seq_np, h):
        """返回 (dec (N,n_embd) f32, lam (N,1), matched (N,), lvls (N,), cnts (N,))。"""
        B, T, _ = h.shape
        N = B * T
        dev = h.device
        vals = torch.zeros(N, self.value_dim, device=dev, dtype=torch.float32)
        cnts = torch.zeros(N, device=dev, dtype=torch.float32)
        lvls = torch.zeros(N, dtype=torch.long, device=dev)
        matched = torch.zeros(N, dtype=torch.bool, device=dev)
        for i, (L, bits) in enumerate(self.levels):
            acc_np, idx_np = suffix_hash_np(seq_np, L, bits)
            idx = torch.from_numpy(idx_np.reshape(-1)).to(dev)
            hh = torch.from_numpy(acc_np.view(np.int64).reshape(-1)).to(dev)
            ht = getattr(self, f"hash_tab_{i}")
            ct = getattr(self, f"count_{i}")
            vl = getattr(self, f"value_{i}")
            cnt = ct[idx]
            m = (cnt > 0) & (ht[idx] == hh) & (~matched)
            if bool(m.any()):
                vals[m] = vl[idx[m]].float()
                cnts[m] = cnt[m]
                lvls[m] = i
                matched |= m
            if bool(matched.all()):
                break
        # 收缩估计：小 count 向 0 收缩
        sh = (cnts / (cnts + self.kappa)).unsqueeze(-1)
        dec = F.linear(vals * sh, self.decoder.weight.float())
        a = self.alpha[lvls] if self.n_levels > 0 else torch.zeros(N, device=dev)
        lam = torch.sigmoid(self.read_gate(h.reshape(N, self.n_embd).float()).squeeze(-1)
                            + a + self.beta * torch.log1p(cnts))
        lam = (lam * matched.float()).unsqueeze(1)
        return dec, lam, matched, lvls, cnts

    # ------------------------------------------------------------------ 统计
    @torch.no_grad()
    def stats(self):
        out = {"levels": [], "bytes": self.val_proj.numel() * 4}
        for i, (L, bits) in enumerate(self.levels):
            ct = getattr(self, f"count_{i}")
            filled = int((ct > 0).sum())
            n = ct.numel()
            out["levels"].append({
                "L": L, "slots": n, "filled": filled, "fill_ratio": filled / n,
                "mean_count": float(ct[ct > 0].mean()) if filled else 0.0})
            out["bytes"] += n * (self.value_dim * 2 + 8 + 4)
        return out

    @torch.no_grad()
    def save(self, path, meta=None):
        ck = {"format": "nanoseek_neuron_db_v2", "meta": meta or {},
              "val_proj": self.val_proj.cpu(),
              "decoder": self.decoder.state_dict(),
              "read_gate": self.read_gate.state_dict(),
              "alpha": self.alpha.detach().cpu(), "beta": self.beta.detach().cpu(),
              "config": {"n_embd": self.n_embd, "levels": self.levels,
                         "value_dim": self.value_dim, "kappa": self.kappa}}
        for i in range(self.n_levels):
            ck[f"hash_tab_{i}"] = getattr(self, f"hash_tab_{i}").cpu()
            ck[f"count_{i}"] = getattr(self, f"count_{i}").cpu()
            ck[f"value_{i}"] = getattr(self, f"value_{i}").cpu()
        torch.save(ck, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.val_proj.copy_(ck["val_proj"].to(self.device))
        for i in range(self.n_levels):
            getattr(self, f"hash_tab_{i}").copy_(ck[f"hash_tab_{i}"].to(self.device))
            getattr(self, f"count_{i}").copy_(ck[f"count_{i}"].to(self.device))
            getattr(self, f"value_{i}").copy_(ck[f"value_{i}"].to(self.device))
        self.decoder.load_state_dict(ck["decoder"])
        self.read_gate.load_state_dict(ck["read_gate"])
        self.alpha.data.copy_(ck["alpha"].to(self.device))
        self.beta.data.copy_(ck["beta"].to(self.device))
        return ck.get("meta", {})
