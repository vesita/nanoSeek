# -*- coding: utf-8 -*-
"""神经元级神经数据库 —— 多向量槽 (mixture) 版 (P2)。

动机：单槽只存条件均值 E[−∂CE/∂h | k] 会把**多峰续写**抹平（§14 实测槽内方差占 97%）。
改成每槽 K 个模态：

  写：把新 value 归到最相似的模态（在线 K-means），该模态做 running-mean；
  读：用学习查询 q(h) 对各模态打分，softmax 加权求和 → 取当前上下文最匹配的模态。

内存预算与 v5 对齐：K=2 × value_dim=64 ≈ 1 × value_dim=128。
其余（多长度 backoff、收缩估计、count 门控）与 v6 一致。
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.neuron_db import suffix_hash_np, parse_levels


class MixtureNeuronDB(nn.Module):
    def __init__(self, n_embd=512, levels=((8, 22),), value_dim=64, modes=2,
                 device=None, dtype=torch.bfloat16, proj_seed=0,
                 decoder_init=1.0, gate_bias=-2.0, kappa=3.0, val_proj=None):
        super().__init__()
        self.levels = tuple(tuple(x) for x in levels)
        self.n_levels = len(self.levels)
        self.n_embd = n_embd
        self.value_dim = int(value_dim)
        self.modes = int(modes)
        self.dtype = dtype
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.kappa = float(kappa)

        g = torch.Generator().manual_seed(proj_seed)
        if val_proj is not None:
            proj = val_proj.clone().float()
        else:
            proj = torch.randn(n_embd, self.value_dim, generator=g) / math.sqrt(n_embd)
        self.register_buffer("val_proj", proj)

        for i, (L, bits) in enumerate(self.levels):
            n = 1 << bits
            self.register_buffer(f"hash_tab_{i}", torch.zeros(n, dtype=torch.int64))
            self.register_buffer(f"value_{i}",
                                 torch.zeros(n, self.modes, self.value_dim, dtype=dtype))
            self.register_buffer(f"mcnt_{i}", torch.zeros(n, self.modes, dtype=torch.float32))

        self.mix_q = nn.Linear(n_embd, self.value_dim, bias=False)
        nn.init.normal_(self.mix_q.weight, std=0.02)
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
        self.alpha = nn.Parameter(torch.zeros(self.n_levels))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.to(self.device)

    # ------------------------------------------------------------------ 写
    @torch.no_grad()
    def _write_level(self, i, idx, hh, proj):
        ht = getattr(self, f"hash_tab_{i}")
        vl = getattr(self, f"value_{i}")
        mc = getattr(self, f"mcnt_{i}")
        tot = mc[idx].sum(-1)
        match = (tot == 0) | (ht[idx] == hh)
        if not bool(match.any()):
            return 0
        i2, h2, p2 = idx[match], hh[match], proj[match]
        empty = tot[match] == 0
        if bool(empty.any()):
            ht[i2[empty]] = h2[empty]
        cur = vl[i2].float()                                  # (n,K,D)
        sim = (p2.unsqueeze(1) * cur).sum(-1)                 # (n,K)
        best = sim.argmax(-1)
        best = torch.where(empty, torch.zeros_like(best), best)
        key = i2 * self.modes + best
        u, inv = torch.unique(key, return_inverse=True)
        us, um = u // self.modes, u % self.modes
        num = torch.zeros(u.numel(), self.value_dim, device=self.device)
        num.index_add_(0, inv, p2.float())
        den = torch.zeros(u.numel(), device=self.device)
        den.index_add_(0, inv, torch.ones(i2.numel(), device=self.device))
        old_c = mc[us, um]
        old_v = vl[us, um].float()
        new_c = old_c + den
        vl[us, um] = ((old_v * old_c.unsqueeze(1) + num) /
                      new_c.clamp_min(1e-9).unsqueeze(1)).to(self.dtype)
        mc[us, um] = new_c
        return int(i2.numel())

    @torch.no_grad()
    def write(self, seq_np, delta):
        d = delta.to(self.device).float()
        proj = (d @ self.val_proj.float()).to(self.dtype)
        total = 0
        for i, (L, bits) in enumerate(self.levels):
            acc_np, idx_np = suffix_hash_np(seq_np, L, bits)
            idx = torch.from_numpy(idx_np.reshape(-1)).to(self.device)
            hh = torch.from_numpy(acc_np.view(np.int64).reshape(-1)).to(self.device)
            total += self._write_level(i, idx, hh, proj)
        return total

    # ------------------------------------------------------------------ 读
    def read(self, seq_np, h):
        B, T, _ = h.shape
        N = B * T
        dev = h.device
        vals = torch.zeros(N, self.modes, self.value_dim, device=dev)
        mcnt = torch.zeros(N, self.modes, device=dev)
        lvls = torch.zeros(N, dtype=torch.long, device=dev)
        matched = torch.zeros(N, dtype=torch.bool, device=dev)
        for i, (L, bits) in enumerate(self.levels):
            acc_np, idx_np = suffix_hash_np(seq_np, L, bits)
            idx = torch.from_numpy(idx_np.reshape(-1)).to(dev)
            hh = torch.from_numpy(acc_np.view(np.int64).reshape(-1)).to(dev)
            ht = getattr(self, f"hash_tab_{i}")
            vl = getattr(self, f"value_{i}")
            mc = getattr(self, f"mcnt_{i}")
            cnt = mc[idx].sum(-1)
            m = (cnt > 0) & (ht[idx] == hh) & (~matched)
            if bool(m.any()):
                vals[m] = vl[idx[m]].float()
                mcnt[m] = mc[idx[m]]
                lvls[m] = i
                matched |= m
            if bool(matched.all()):
                break
        q = self.mix_q(h.reshape(N, self.n_embd).float())          # (N,D)
        scores = (q.unsqueeze(1) * vals).sum(-1) + torch.log1p(mcnt)   # (N,K)
        a = torch.softmax(scores, dim=-1)
        v = (a.unsqueeze(-1) * vals).sum(1)                        # (N,D)
        cnt_tot = mcnt.sum(-1)
        sh = (cnt_tot / (cnt_tot + self.kappa)).unsqueeze(-1)
        dec = F.linear(v * sh, self.decoder.weight.float())
        lam = torch.sigmoid(self.read_gate(h.reshape(N, self.n_embd).float()).squeeze(-1)
                            + self.alpha[lvls] + self.beta * torch.log1p(cnt_tot))
        lam = (lam * matched.float()).unsqueeze(1)
        return dec, lam, matched, lvls, cnt_tot

    # ------------------------------------------------------------------ 统计 / 存档
    @torch.no_grad()
    def stats(self):
        out = {"levels": [], "bytes": self.val_proj.numel() * 4}
        for i, (L, bits) in enumerate(self.levels):
            mc = getattr(self, f"mcnt_{i}")
            tot = mc.sum(-1)
            filled = int((tot > 0).sum())
            n = mc.shape[0]
            out["levels"].append({"L": L, "slots": n, "filled": filled,
                                  "fill_ratio": filled / n,
                                  "mean_count": float(tot[tot > 0].mean()) if filled else 0.0})
            out["bytes"] += n * (self.modes * self.value_dim * 2 + 8 + self.modes * 4)
        return out

    @torch.no_grad()
    def save(self, path, meta=None):
        ck = {"format": "nanoseek_neuron_db_mix_v1", "meta": meta or {},
              "val_proj": self.val_proj.cpu(),
              "decoder": self.decoder.state_dict(),
              "read_gate": self.read_gate.state_dict(),
              "mix_q": self.mix_q.state_dict(),
              "alpha": self.alpha.detach().cpu(), "beta": self.beta.detach().cpu(),
              "config": {"n_embd": self.n_embd, "levels": self.levels,
                         "value_dim": self.value_dim, "modes": self.modes,
                         "kappa": self.kappa}}
        for i in range(self.n_levels):
            ck[f"hash_tab_{i}"] = getattr(self, f"hash_tab_{i}").cpu()
            ck[f"value_{i}"] = getattr(self, f"value_{i}").cpu()
            ck[f"mcnt_{i}"] = getattr(self, f"mcnt_{i}").cpu()
        torch.save(ck, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.val_proj.copy_(ck["val_proj"].to(self.device))
        for i in range(self.n_levels):
            getattr(self, f"hash_tab_{i}").copy_(ck[f"hash_tab_{i}"].to(self.device))
            getattr(self, f"value_{i}").copy_(ck[f"value_{i}"].to(self.device))
            getattr(self, f"mcnt_{i}").copy_(ck[f"mcnt_{i}"].to(self.device))
        self.decoder.load_state_dict(ck["decoder"])
        self.read_gate.load_state_dict(ck["read_gate"])
        self.mix_q.load_state_dict(ck["mix_q"])
        self.alpha.data.copy_(ck["alpha"].to(self.device))
        self.beta.data.copy_(ck["beta"].to(self.device))
        return ck.get("meta", {})
