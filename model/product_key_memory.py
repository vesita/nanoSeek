# -*- coding: utf-8 -*-
"""Product-Key 神经记忆库 (ProductKeyMemory)：no_grad 双子键稀疏检索。

设计（继承旧 ProductKeyNeuralDB 已验证的检索机制，重构为 no_grad 外部库）：

  数据结构:
    keys1: (sub_keys, half_dim)  子空间1 码本
    keys2: (sub_keys, half_dim)  子空间2 码本
    values: (sub_keys*sub_keys, value_dim)  槽位内容
    总槽位 M = sub_keys^2；检索复杂度 O(sub_keys) 而非 O(M)

  检索（读）:
    query(N,key_dim) 切半 → 各子空间打分 top-k_sub → 笛卡尔积 → 全局 top-k
    → 温度软化加权聚合 values

  写入（规则，no_grad）:
    query 命中 top-1 槽 → EMA 覆盖（write_lr），写门控缩放强度

  遗忘（GC）:
    slot_usage EMA → 分位数死槽判定 → 热门槽变异分裂复活

  全部参数 requires_grad=False（零优化器状态，显存=纯权重）。
  存储/IO bf16（与模型基础类型同构），检索打分 f32。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProductKeyMemory(nn.Module):
    def __init__(self, key_dim, sub_keys=1024, top_k=32, value_dim=None,
                 write_lr=0.2, temperature=0.1, device=None,
                 usage_decay=0.99, dead_threshold=0.05, dtype=torch.bfloat16):
        super().__init__()
        assert key_dim % 2 == 0, "key_dim 必须为偶数以切分双子空间"
        self.key_dim = key_dim
        self.half_dim = key_dim // 2
        self.sub_keys = sub_keys
        self.total_slots = sub_keys * sub_keys
        self.top_k = min(top_k, self.total_slots)
        self.k_sub = min(self.top_k, sub_keys)
        self.value_dim = value_dim or key_dim
        self.write_lr = write_lr
        self.temperature = temperature
        self.dtype = dtype
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        scale = 1.0 / math.sqrt(self.half_dim)
        # 子键码本 + 槽位值库：全部 no_grad（零优化器状态）
        self.keys1 = nn.Parameter(torch.randn(sub_keys, self.half_dim, dtype=dtype, device=self.device) * scale)
        self.keys2 = nn.Parameter(torch.randn(sub_keys, self.half_dim, dtype=dtype, device=self.device) * scale)
        self.values = nn.Parameter(torch.randn(self.total_slots, self.value_dim, dtype=dtype, device=self.device) * scale)
        for p in (self.keys1, self.keys2, self.values):
            p.requires_grad_(False)

        self.register_buffer("slot_usage", torch.ones(self.total_slots, device=self.device))
        self.register_buffer("steps_since_purge", torch.zeros(1, dtype=torch.long, device=self.device))
        # 非 top-1 槽位的 usage 权重（制造命中方差，供 GC 识别死槽）
        self.usage_aux_scale = 0.1
        self.usage_decay = usage_decay
        self.dead_threshold = dead_threshold

    # ------------------------------------------------------------------ 检索
    def retrieve(self, query_f32, top_k=None):
        """双子键稀疏检索。query_f32: (N, key_dim) f32。

        返回: (out (N,value_dim) bf16, slot_ids (N,top_k), weights (N,top_k))
        复杂度 O(N × sub_keys × half_dim)——对 262k 槽仅 512 子键点积。
        """
        k = top_k or self.top_k
        q = query_f32.to(self.device)
        N = q.size(0)
        with torch.no_grad():
            q1, q2 = q[:, :self.half_dim], q[:, self.half_dim:]
            # 各子空间打分: (N, sub_keys) —— O(sub_keys) 而非 O(M)
            s1 = q1 @ self.keys1.float().t()
            s2 = q2 @ self.keys2.float().t()
            # 各取 top-k_sub 候选
            t1s, t1i = s1.topk(self.k_sub, dim=-1)
            t2s, t2i = s2.topk(self.k_sub, dim=-1)
            # 笛卡尔积组合评分: (N, k_sub, k_sub) → 展平
            cart = (t1s.unsqueeze(-1) + t2s.unsqueeze(-2)).reshape(N, -1)
            k_eff = min(self.top_k, cart.size(-1))
            fs, fi = cart.topk(k_eff, dim=-1)
            # 展平索引 → 子空间候选索引 → 全局槽位 id
            i1 = fi // self.k_sub
            i2 = fi % self.k_sub
            id1 = t1i.gather(1, i1)                      # (N, k_eff)
            id2 = t2i.gather(1, i2)
            slot_ids = id1 * self.sub_keys + id2         # (N, k_eff)
            # 温度软化加权聚合（温度小→检索锐利，避免 top-k 平均池化）
            w = F.softmax(fs / self.temperature, dim=-1)  # (N, k_eff)
            vals = self.values[slot_ids]                  # (N, k_eff, value_dim) bf16
            out = (vals.float() * w.unsqueeze(-1)).sum(dim=1).to(self.dtype)

            # usage EMA：用命中指示（hit indicator）而非原始频次，保证大槽位下也能分化。
            # 命中=1/未命中=0 的 EMA 天然落在 [0,1] 且随命中分布分化，不受 N/M 规模影响。
            if self.training:
                hit = torch.zeros(self.total_slots, device=slot_ids.device)
                hit.scatter_add_(0, slot_ids.reshape(-1),
                                 torch.ones(slot_ids.numel(), device=slot_ids.device))
                hit = (hit > 0).float()                       # 命中指示（0/1）
                bu = hit * (1 + 0.05 * torch.randn_like(hit))  # 轻微探索扰动
                self.slot_usage.mul_(self.usage_decay).add_(bu, alpha=1 - self.usage_decay)
                self.steps_since_purge.add_(1)
        return out.to(query_f32.device), slot_ids, w

    # ------------------------------------------------------------------ 写入
    def write(self, query_f32, value, weight=None, top_m=1):
        """规则写入：EMA 覆盖 query 命中的 top_m 个槽（no_grad）。

        query_f32: (N, key_dim) f32；value: (N, value_dim) bf16/f32；
        weight: (N,) 写门控（0-1，WriteInterface.gate）；top_m: 写入前 m 个命中槽。
        """
        q = query_f32.to(self.device)
        v = value.to(self.device).to(self.dtype)
        with torch.no_grad():
            q1, q2 = q[:, :self.half_dim], q[:, self.half_dim:]
            s1 = q1 @ self.keys1.float().t()
            s2 = q2 @ self.keys2.float().t()
            t1s, t1i = s1.topk(self.k_sub, dim=-1)
            t2s, t2i = s2.topk(self.k_sub, dim=-1)
            N = q.size(0)
            cart = (t1s.unsqueeze(-1) + t2s.unsqueeze(-2)).reshape(N, -1)
            m = min(top_m, cart.size(-1))
            _, fi = cart.topk(m, dim=-1)                 # (N, m)
            i1, i2 = fi // self.k_sub, fi % self.k_sub
            id1 = t1i.gather(1, i1); id2 = t2i.gather(1, i2)
            slot_ids = (id1 * self.sub_keys + id2).reshape(-1)   # (N*m,)
            v_flat = v.unsqueeze(1).expand(N, m, self.value_dim).reshape(-1, self.value_dim)
            if weight is not None:
                w_flat = weight.to(self.device).reshape(-1, 1).expand(N, m).reshape(-1, 1)
                v_flat = v_flat * w_flat.to(self.dtype)
            old = self.values.data[slot_ids]
            self.values.data[slot_ids] = (1 - self.write_lr) * old + self.write_lr * v_flat

    # ------------------------------------------------------------------ GC
    def gc(self):
        avg = self.slot_usage.mean().item()
        std = self.slot_usage.std().item()
        # 死槽判定：绝对阈值（未命中槽的 usage 随 EMA 衰减趋向 0，命中槽保持高位）。
        # 低于 (1-dead_threshold) 分位且明显低于均值 → 冷槽。
        qt = float(torch.quantile(self.slot_usage, 1.0 - self.dead_threshold))
        cutoff = max(qt, avg * self.dead_threshold)
        # 额外保护：cutoff 必须显著低于均值（差距<50% 时说明未分化，跳过清洗）
        if cutoff > avg * 0.5:
            self.steps_since_purge.zero_()
            return {"dead_slots": 0, "total": self.total_slots,
                    "dead_ratio": 0.0, "avg": avg, "skipped": "usage 分化不足"}
        dead_mask = self.slot_usage < cutoff
        nd = int(dead_mask.sum().item())
        stats = {"dead_slots": nd, "total": self.total_slots,
                 "dead_ratio": nd / self.total_slots, "avg": avg}
        if nd == 0:
            self.steps_since_purge.zero_()
            return stats
        di = torch.nonzero(dead_mask).squeeze(1)
        hi = torch.nonzero(~dead_mask).squeeze(1)
        scale = 1.0 / math.sqrt(self.value_dim)
        if hi.numel() == 0:
            self.values.data[dead_mask] = torch.randn_like(self.values.data[dead_mask]) * scale
            self.slot_usage.fill_(1.0)
            self.steps_since_purge.zero_()
            return stats
        donor = hi[torch.multinomial(self.slot_usage[hi], nd, replacement=True)]
        noise = torch.randn_like(self.values.data[di]) * (self.values.data.std().item() * 0.05)
        self.values.data[di] = self.values.data[donor] + noise
        self.slot_usage[dead_mask] = avg
        self.steps_since_purge.zero_()
        return stats

    # ------------------------------------------------------------- 可移植
    def export_state(self):
        st = {}
        for n, b in self.named_buffers(): st[n] = b.detach().clone()
        for n, p in self.named_parameters(): st[n] = p.detach().clone()
        return st

    def import_state(self, state):
        norm = {k.split("neural_db.", 1)[-1]: v for k, v in state.items()}
        valid = set(self.state_dict().keys())
        subset = {k: v for k, v in norm.items() if k in valid}
        if subset: self.load_state_dict(subset, strict=False)
        return [k for k in valid if k not in subset]

    def save_db(self, path):
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        torch.save({"format": "nanoseek_pkm_v1", "key_dim": self.key_dim,
                    "sub_keys": self.sub_keys, "top_k": self.top_k,
                    "value_dim": self.value_dim, "state": self.export_state()}, path)

    @classmethod
    def load_db(cls, key_dim, path, sub_keys=None, top_k=None):
        meta = torch.load(path, map_location="cpu", weights_only=False)
        m = cls(key_dim, sub_keys or meta["sub_keys"], top_k or meta["top_k"],
                value_dim=meta.get("value_dim"))
        m.import_state(meta["state"])
        return m
