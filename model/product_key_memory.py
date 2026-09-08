# -*- coding: utf-8 -*-
"""Product-Key 神经记忆库 (ProductKeyMemory)：no_grad 外部记忆 + 可微寻址。

数据结构（沿用）:
  keys1/keys2: (sub_keys, half_dim)  双子键码本
  values:      (sub_keys^2, value_dim) 槽位内容
  检索复杂度 O(sub_keys)，总槽位 M = sub_keys^2

框架改进（全部可开关，默认等价旧版）:

  P8'  values_init="zeros"   未写入槽返回 0，而不是 i.i.d. 噪声。
                              旧版 randn 初始化让「稀疏 top-k 检索」退化成
                              「高维噪声平均」：未写槽贡献纯噪声，门控只能学成 0。

  P9'  write_mode="delta"    软 top-m delta 规则（Widrow-Hoff / 核记忆）：
                              p = Σ_j w_j v_j ;  v_i ← v_i + η·w_i·(target − p)
                              与「每个槽各自 EMA 覆盖」不同：邻近槽按检索权重
                              共同承担同一个误差 → 值场平滑、可泛化。

  P10' diff_addr=True        top-k 索引仍离散(no_grad)，但聚合权重
                              w = softmax(s_i/τ) 对 query 可微。
                              旧版 retrieve 整体 no_grad → key_proj 永远随机，
                              寻址空间是一组固定随机投影（top-1 几乎全不同）。

  P12' 支持度/使用度修复      slot_support 用 index_add_（重复槽安全），
                              slot_usage 可零初始化，活跃槽判据 = support>0。

  另: temperature / write_top_m 可调；token_ids 供 kNN-LM 模式。
"""
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProductKeyMemory(nn.Module):
    def __init__(self, key_dim, sub_keys=1024, top_k=32, value_dim=None,
                 write_lr=0.2, temperature=0.1, device=None,
                 usage_decay=0.99, dead_threshold=0.05, dtype=torch.bfloat16,
                 use_support=False, support_floor=1.0, support_scale=2.0, support_cap=20.0,
                 # --- 框架改进开关（默认等价旧版）---
                 values_init="randn",     # P8': randn | zeros
                 write_mode="ema",        # P9': ema | delta
                 write_top_m=1,           # P9': 写入扩散到的近邻槽数
                 err_clip=0.0,            # P9': 单条误差范数上限（0=关）
                 value_clip=0.0,          # P9': 槽值范数上限（0=关，安全网）
                 trainable_codebook=False, # P15': 码本可训练（随机冻结码本 → 在线 VQ 质心）
                 usage_init="ones"):      # P12': ones | zeros
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

        self.values_init = values_init
        self.write_mode = write_mode
        self.write_top_m = int(write_top_m)
        self.err_clip = float(err_clip)
        self.value_clip = float(value_clip)
        self.usage_init = usage_init

        scale = 1.0 / math.sqrt(self.half_dim)
        # 子键码本 + 槽位值库：全部 no_grad（零优化器状态）
        self.keys1 = nn.Parameter(torch.randn(sub_keys, self.half_dim, dtype=dtype, device=self.device) * scale)
        self.keys2 = nn.Parameter(torch.randn(sub_keys, self.half_dim, dtype=dtype, device=self.device) * scale)
        if values_init == "zeros":
            # P8': 未写入槽 = 0 → 「无信息」而不是噪声
            init_vals = torch.zeros(self.total_slots, self.value_dim, dtype=dtype, device=self.device)
        else:
            init_vals = torch.randn(self.total_slots, self.value_dim, dtype=dtype, device=self.device) * scale
        self.values = nn.Parameter(init_vals)
        for p in (self.keys1, self.keys2, self.values):
            p.requires_grad_(False)
        # P15': 码本可训练——旧版码本是随机冻结的，1M 个槽的格子是随机撒的。
        # 配合 diff_addr，keys 会沿读端梯度漂移成在线 VQ 质心（仅 2×1024×64 参数）。
        if trainable_codebook:
            self.keys1.requires_grad_(True)
            self.keys2.requires_grad_(True)
        self.trainable_codebook = trainable_codebook

        usage0 = torch.zeros(self.total_slots, device=self.device) if usage_init == "zeros" \
            else torch.ones(self.total_slots, device=self.device)
        self.register_buffer("slot_usage", usage0)
        self.register_buffer("steps_since_purge", torch.zeros(1, dtype=torch.long, device=self.device))
        # 非 top-1 槽位的 usage 权重（制造命中方差，供 GC 识别死槽）
        self.usage_aux_scale = 0.1
        self.usage_decay = usage_decay
        self.dead_threshold = dead_threshold

        # P5: 支持度（可复制性）——写时累加、读时低支持槽置信缩压，压制一次性/随机写入
        self.use_support = use_support
        self.support_floor = support_floor
        self.support_scale = support_scale
        self.support_cap = support_cap
        self.register_buffer("slot_support", torch.zeros(self.total_slots, device=self.device))
        # P7: kNN-LM—槽位最后写入的 token id（未写=-1），读时据此构建 p_kNN
        self.register_buffer("token_ids", torch.full((self.total_slots,), -1, dtype=torch.long, device=self.device))

    # ------------------------------------------------------------------ 地址检索（无副作用）
    def _lookup(self, q, top_k=None, temperature=None):
        """双子键稀疏检索，只算地址与权重，不读值、不更新统计。

        返回 (slot_ids (N,k_eff) long, weights (N,k_eff) f32, scores (N,k_eff) f32)。
        """
        k = top_k or self.top_k
        tau = self.temperature if temperature is None else temperature
        q1, q2 = q[:, :self.half_dim], q[:, self.half_dim:]
        s1 = q1 @ self.keys1.float().t()          # (N, sub_keys)
        s2 = q2 @ self.keys2.float().t()
        t1s, t1i = s1.topk(self.k_sub, dim=-1)
        t2s, t2i = s2.topk(self.k_sub, dim=-1)
        N = q.size(0)
        cart = (t1s.unsqueeze(-1) + t2s.unsqueeze(-2)).reshape(N, -1)   # (N, k_sub^2)
        k_eff = min(k, cart.size(-1))
        fs, fi = cart.topk(k_eff, dim=-1)
        i1 = fi // self.k_sub
        i2 = fi % self.k_sub
        id1 = t1i.gather(1, i1)
        id2 = t2i.gather(1, i2)
        slot_ids = id1 * self.sub_keys + id2          # (N, k_eff)
        w = F.softmax(fs / max(tau, 1e-4), dim=-1)
        return slot_ids, w, fs

    # ------------------------------------------------------------------ 检索（读）
    def retrieve(self, query_f32, top_k=None, temperature=None,
                 return_support=False, diff_addr=True, renorm_written=False):
        """双子键稀疏检索。query_f32: (N, key_dim) f32。

        diff_addr=True (P10'): 聚合权重对 query 可微（索引仍离散）。
        renorm_written=True (P8''): 零初始化下未写槽 v=0，把权重只在已写槽上
        重归一化，避免「读出被未写槽稀释 × 门控再乘覆盖率」的二次衰减。

        返回:
          return_support=False → (out, slot_ids, weights)
          return_support=True  → (out, slot_ids, weights, support, coverage)
        """
        q = query_f32.to(self.device)
        tau = self.temperature if temperature is None else temperature
        with torch.no_grad():
            slot_ids, w_sel, _ = self._lookup(q.detach(), top_k=top_k, temperature=tau)
        # P10': 用带梯度的 query 重算所选槽位分数（gather 很小，开销可忽略）
        if diff_addr and q.requires_grad:
            q1, q2 = q[:, :self.half_dim].float(), q[:, self.half_dim:].float()
            k1 = self.keys1.float()[slot_ids // self.sub_keys]     # (N,k,half)
            k2 = self.keys2.float()[slot_ids % self.sub_keys]
            fs = (q1.unsqueeze(1) * k1).sum(-1) + (q2.unsqueeze(1) * k2).sum(-1)
            w = F.softmax(fs / max(tau, 1e-4), dim=-1)
        else:
            w = w_sel
        vals = self.values[slot_ids]                    # (N,k,value_dim) bf16，无梯度
        sup_slots = None
        if self.use_support or return_support or renorm_written:
            sup_slots = self.slot_support[slot_ids]     # (N,k)
        if self.use_support:
            # P5: 低支持（一次性/随机写入）槽被置信缩压，再归一化
            sup_conf = torch.sigmoid((sup_slots - self.support_floor) / self.support_scale)
            w = w * sup_conf
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-9)
        # 覆盖率 = 已写槽占的权重质量（重归一化之前算，才是有信息的信任信号）
        coverage = None
        if sup_slots is not None:
            wmask = (sup_slots > 0).to(w.dtype)
            mass = (w * wmask).sum(dim=-1, keepdim=True)
            if renorm_written:
                w = w * wmask / mass.clamp_min(1e-9)
            coverage = mass.clamp(max=1.0)
        out = (vals.float() * w.unsqueeze(-1)).sum(dim=1).to(self.dtype)

        # usage EMA：命中指示 EMA（0/1），保证大槽位下也能分化
        if self.training:
            with torch.no_grad():
                hit = torch.zeros(self.total_slots, device=slot_ids.device)
                hit.scatter_add_(0, slot_ids.reshape(-1),
                                 torch.ones(slot_ids.numel(), device=slot_ids.device))
                hit = (hit > 0).float()
                bu = hit * (1 + 0.05 * torch.randn_like(hit))  # 轻微探索扰动
                self.slot_usage.mul_(self.usage_decay).add_(bu, alpha=1 - self.usage_decay)
                self.steps_since_purge.add_(1)

        if return_support:
            return out.to(query_f32.device), slot_ids, w, sup_slots, coverage
        return out.to(query_f32.device), slot_ids, w

    # ------------------------------------------------------------------ 写入
    def write(self, query_f32, value, weight=None, top_m=None, mode=None,
              token_ids=None, temperature=None):
        """规则写入（no_grad）。

        mode="ema"   : 旧版——命中 top_m 槽各自 EMA 覆盖 v ← (1−η)v + η·target。
        mode="delta" : P9'——软 top-m delta 规则 v_i ← v_i + η·w_i·(target − Σ_j w_j v_j)。

        query_f32: (N,key_dim) f32；value: (N,value_dim)；weight: (N,) 写门控。
        """
        q = query_f32.to(self.device)
        v = value.to(self.device).to(self.dtype)
        m = self.write_top_m if top_m is None else int(top_m)
        md = self.write_mode if mode is None else mode
        tau = self.temperature if temperature is None else temperature
        with torch.no_grad():
            slot_ids, w, _ = self._lookup(q.detach(), top_k=m, temperature=tau)   # (N,m)
            if weight is not None:
                ww = w * weight.to(self.device).reshape(-1, 1).float()
            else:
                ww = w
            N = q.size(0)
            flat = slot_ids.reshape(-1)
            if md == "delta":
                cur = self.values.data[slot_ids].float()              # (N,m,d)
                pred = (cur * w.unsqueeze(-1)).sum(dim=1)             # (N,d) 邻域当前预测
                err = v.float() - pred
                # 误差范数上限：防单条极端目标把槽值推飞
                if self.err_clip > 0:
                    en = err.norm(dim=-1, keepdim=True)
                    err = err * (self.err_clip / en.clamp_min(self.err_clip))
                raw = (ww.unsqueeze(-1) * err.unsqueeze(1)).reshape(-1, self.value_dim)   # (N*m,d)
                # 归一化雅可比聚合：同一批内多个 query 命中同一热槽时，
                #   v_s += η·Σ_q w(t−p) / (1 + η·Σ_q w²)
                # 直接 index_add_ 会让特征值 1−η·Σw² 变负而发散（真实文本热槽必现）。
                uniq, inv = flat.unique(return_inverse=True)
                num = torch.zeros(uniq.numel(), self.value_dim, device=self.device, dtype=torch.float32)
                num.index_add_(0, inv, raw)
                den = torch.zeros(uniq.numel(), device=self.device, dtype=torch.float32)
                den.index_add_(0, inv, ww.reshape(-1) ** 2)
                stepv = (self.write_lr * num / (1.0 + self.write_lr * den).unsqueeze(-1)).to(self.dtype)
                self.values.data.index_add_(0, uniq, stepv)
                # 槽值范数上限（安全网）
                if self.value_clip > 0:
                    vv = self.values.data[uniq].float()
                    vn = vv.norm(dim=-1, keepdim=True)
                    self.values.data[uniq] = (vv * (self.value_clip / vn.clamp_min(self.value_clip))).to(self.dtype)
                if os.environ.get("DB_DEBUG"):
                    vv = self.values.data[uniq].float()
                    print(f"    [DB] N={N} m={m} 热槽={uniq.numel()} |target|max={v.float().norm(dim=-1).max():.3e} "
                          f"|pred|max={pred.norm(dim=-1).max():.3e} |err|max={err.norm(dim=-1).max():.3e} "
                          f"vmax(本轮)={vv.norm(dim=-1).max():.3e}", flush=True)
            else:
                v_flat = v.unsqueeze(1).expand(N, m, self.value_dim).reshape(-1, self.value_dim)
                w_flat = ww.reshape(-1, 1).to(self.dtype)
                old = self.values.data[flat]
                self.values.data[flat] = (1 - self.write_lr) * old + self.write_lr * (v_flat * w_flat)

            # P5/P12': 支持度累加（index_add_ 对重复槽安全；源 dtype 跟随缓冲，防父模块 .to(bf16) 后不匹配）
            if self.use_support:
                self.slot_support.index_add_(0, flat, ww.reshape(-1).to(self.slot_support.dtype))
                self.slot_support.clamp_(max=self.support_cap)
            # P7: 存 token id（kNN-LM 用）
            if token_ids is not None:
                tok_flat = token_ids.to(self.device).unsqueeze(1).expand(N, m).reshape(-1)
                self.token_ids[flat] = tok_flat

    # ------------------------------------------------------------------ GC
    def gc(self):
        avg = self.slot_usage.mean().item()
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
