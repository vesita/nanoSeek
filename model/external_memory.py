# -*- coding: utf-8 -*-
"""外部神经记忆库 (ExternalMemory)：no_grad 键值存储。

定位（对齐用户架构）：
  这是"外部记忆主体"——纯存储 + 检索 + 规则写入 + 自动淘汰 (GC)。
  不参与梯度（requires_grad=False），不占优化器状态，可移植（save/load），
  理想形态后期用 Rust 重写，初期 Python 实现。

核心特性：
  1. no_grad：数据库参数不参与反向传播，零 AdamW 优化器状态 → 显存可控。
  2. 与模型基础类型同构：存储/IO 用 bf16（模型 dtype），检索打分用 f32（提升检索质量）。
  3. 规则写入：surprise 门控触发时，把 value EMA 写入检索命中的槽位。
  4. 自动淘汰 (GC)：槽位活跃度 EMA + 死槽检测 + 热门槽变异分裂复活。
  5. 数据可移植：export/import/save_db/load_db。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExternalMemory(nn.Module):
    def __init__(self, key_dim, n_slots, top_k=8, write_lr=0.2, device=None,
                 usage_decay=0.99, dead_threshold=0.05, dtype=torch.bfloat16):
        super().__init__()
        self.key_dim = key_dim
        self.n_slots = n_slots
        self.top_k = min(top_k, n_slots)
        self.write_lr = write_lr
        self.usage_decay = usage_decay
        self.dead_threshold = dead_threshold
        self.dtype = dtype
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._mem_device = self.device  # 记忆库实际运行的设备（可为 cpu 或 cuda）

        # keys/values：与模型基础类型同构存储（bf16）。no_grad。
        scale = 1.0 / math.sqrt(key_dim)
        self.keys = nn.Parameter(torch.randn(n_slots, key_dim, dtype=dtype, device=self.device) * scale)
        self.values = nn.Parameter(torch.randn(n_slots, key_dim, dtype=dtype, device=self.device) * scale)
        self.keys.requires_grad_(False)
        self.values.requires_grad_(False)
        self.register_buffer("slot_usage", torch.ones(n_slots, device=self.device))
        self.register_buffer("steps_since_purge", torch.zeros(1, dtype=torch.long, device=self.device))

    @property
    def n_params(self):
        """记忆库的"存储容量"（非可训练参数）。"""
        return self.keys.numel() + self.values.numel() + self.slot_usage.numel()

    def retrieve(self, query_f32):
        """检索 top-k 槽位，返回加权聚合 value（bf16）。

        query_f32: (N, key_dim) float32 —— 接口网络编码，检索打分用 f32 提升精度。
        返回: (N, key_dim) bf16 加权 value + top-k 索引 + 权重
        """
        q = query_f32.to(self._mem_device)
        with torch.no_grad():
            qn = F.normalize(q, dim=-1)
            kn = F.normalize(self.keys.float(), dim=-1)
            scores = qn @ kn.t()  # (N, n_slots) f32
            topk_s, topk_i = scores.topk(self.top_k, dim=-1)
            w = F.softmax(topk_s / 0.5, dim=-1)
            vals = self.values[topk_i]  # (N, top_k, key_dim) bf16
            out = (vals.float() * w.unsqueeze(-1)).sum(dim=1).to(self.dtype)

            # 更新槽位活跃度 EMA（top-1 高权重，制造命中方差，供 GC 识别死槽）
            if self.training:
                flat = topk_i.reshape(-1, self.top_k)
                top1 = flat[:, 0]
                c1 = torch.bincount(top1, minlength=self.n_slots).float()
                co = torch.bincount(flat.reshape(-1), minlength=self.n_slots).float() - c1
                bu = (c1 + 0.1 * co) / max(q.size(0), 1) * (1 + 0.05 * torch.randn_like(c1))
                self.slot_usage.mul_(self.usage_decay).add_(bu, alpha=1 - self.usage_decay)
                self.steps_since_purge.add_(1)
        return out.to(query_f32.device), topk_i, w

    def write(self, query_f32, value_bf16, weight=None):
        """规则写入：把 value 写进 query 命中的槽位（EMA 覆盖，no_grad）。

        query_f32: (N, key_dim) f32；value_bf16: (N, key_dim) bf16；weight: (N,) 可选写入强度。
        """
        q = query_f32.to(self._mem_device)
        v = value_bf16.to(self._mem_device).to(self.dtype)
        with torch.no_grad():
            qn = F.normalize(q, dim=-1)
            kn = F.normalize(self.keys.float(), dim=-1)
            scores = qn @ kn.t()
            top1_i = scores.argmax(dim=-1)
            if weight is not None:
                v = v * weight.to(self._mem_device).to(self.dtype).view(-1, 1)
            old = self.values.data[top1_i]
            self.values.data[top1_i] = (1 - self.write_lr) * old + self.write_lr * v

    def gc(self):
        """自动淘汰：找冷槽 → 从热门槽变异分裂复活。"""
        avg = self.slot_usage.mean().item()
        q = float(torch.quantile(self.slot_usage, 1.0 - self.dead_threshold))
        cutoff = max(q, avg * self.dead_threshold)
        dead_mask = self.slot_usage < cutoff
        nd = dead_mask.sum().item()
        stats = {"dead_slots": nd, "total": self.n_slots,
                 "dead_ratio": nd / self.n_slots, "avg": avg}
        if nd == 0:
            self.steps_since_purge.zero_()
            return stats
        di = torch.nonzero(dead_mask).squeeze(1)
        hi = torch.nonzero(~dead_mask).squeeze(1)
        scale = 1.0 / math.sqrt(self.key_dim)
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

    # ---- 数据可移植 ----
    def export_state(self):
        state = {}
        for n, b in self.named_buffers(): state[n] = b.detach().clone()
        for n, p in self.named_parameters(): state[n] = p.detach().clone()
        return state

    def import_state(self, state):
        norm = {}
        for k, v in state.items():
            b = k.split("neural_db.", 1)[-1]
            norm[b] = v
        valid = set(self.state_dict().keys())
        subset = {k: v for k, v in norm.items() if k in valid}
        if subset: self.load_state_dict(subset, strict=False)
        return [k for k in valid if k not in subset]

    def save_db(self, path):
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"format": "nanoseek_external_memory_v1", "key_dim": self.key_dim,
                    "n_slots": self.n_slots, "top_k": self.top_k,
                    "state": self.export_state()}, path)

    @classmethod
    def load_db(cls, key_dim, path, n_slots=None, top_k=None):
        meta = torch.load(path, map_location="cpu", weights_only=False)
        n_slots = n_slots or meta["n_slots"]; top_k = top_k or meta["top_k"]
        m = cls(key_dim, n_slots, top_k)
        m.import_state(meta["state"])
        return m
