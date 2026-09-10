# -*- coding: utf-8 -*-
"""NDB v7 原型：**训练时可读可写**的后缀 n-gram 数据库（no_grad 表 + 可学习门控）。

## 与 v5/v6 的根本区别（三处，都针对已被证伪的旧范式）

```
              v5/v6（残差式 / 神经元级）           v7（本文件）
  键          隐藏态 h                             离散 token 后缀哈希
  值          e_y − E_p[e]（残差向量）              token 计数的 top-K 分布
  寻址        可微（PKM 的 key_proj 可学）          离散 + no_grad（hash 直查）
  写          建库一次 / 写残差（被 E_p 过滤）      在线增量计数（数据统计量）
```

为什么必须这样改（证据见 dev-notes/78 与 PROJECT_STATE §6）：
- 残差式写的目标被 `E_p` 过滤 ⇒ 校准定理给出 `E[r|h]=0` ⇒ Δ −0.0003
- 隐藏态键与当前 h 的 cos = 0.824（82% 冗余）⇒ 注入有害
- 可微寻址让"外部容量"退化成"另一种参数" ⇒ co-train 后被吸收（−0.0046 → −0.0002）
- 换成离散 token 键 + 计数：实测 Δ = **−0.0723**（残差式的 240 倍）

## 「模型自己决定读写」是怎么落的

```
【表】离散、no_grad、**不进优化器**（是数据统计量，不是参数）
  每级 L：slot = hash(tokens[t-L+1 : t+1]) % M
          存 top-K 的 (token, count) + total（该槽**全部**观测的加权和）

【写】模型决定「写多重」
  w_t = σ(W_w · h_t)                     ← 可学习写门控
  counts[slot][y_{t+1}] += w_t            ← 加权计数

【读】模型决定「信多少、信哪一级」
  查表 → p_ng^(L) = counts/total         ← 离散查表，no_grad
  g_t = σ(W_r · [h_t ; 槽统计量 ; w_t])   ← 可学习读门控
  α   = softmax(level_weight)            ← 可学习多级混合
  p   = (1−g_t)·p_model + g_t·Σ_L α_L·p_ng^(L)
```

### 梯度路径（必须说清楚，别假装是严格端到端）

```
  ✓ ∂L/∂W_r            ≠ 0   读门控直接可微，路径干净
  ✓ ∂L/∂level_weight   ≠ 0   多级混合直接可微
  △ ∂L/∂W_w            ≠ 0   **但不是通过"写"来的**：
        离散表阻断了跨 batch 的梯度链（写影响未来读，不可微）。
        本实现把 w_t 同时喂进 read_gate 的输入 → W_w 从读侧拿到真实梯度。
        效果：写门控学成"我会想读的地方就多写"，是合理的归纳偏置，
        但**不是**"写得好不好"的严格信用分配。严格版需要辅助损失或 REINFORCE。
```

## ★ 安全设计：写必须显式开启（防 val 泄漏）

`train.py` 的同一个 forward 同时服务训练 / `estimate_loss` / 健康体检。若把写挂在
forward 上，**验证集会写进训练库** —— 本项目被 val 泄漏坑过最惨的一次
（旧 val 切分双标准，导致所有指标失真、据此做过多次错误决策）。

所以写路径**默认关闭**，必须显式：

```python
with ndb.write_enabled():
    ...        # 只包训练 micro-batch，绝不能包 estimate_loss / 健康体检
```

## 内存

表 = Σ_levels M×(K×4 + K×4 + 4) 字节：

```
    M=67M,  K=4  → 2.41GB      M=67M,  K=1 → 0.80GB
    M=268M, K=1  → 3.22GB      M=268M, K=2 → 6.44GB
```

构造时按 `max_table_gb` 直接拦（跑之前就报错，不跑到一半 OOM）。
"""
from __future__ import annotations

import contextlib
import math

import numpy as np
import torch
import torch.nn as nn

__all__ = ['NgramNDB', 'NDBStats', 'suffix_hashes']

MULT = np.uint64(1000003)


def suffix_hashes(tokens, L):
    """多项式滚动哈希：`h[i]` 对应后缀 `tokens[i : i+L]`，用于预测 `tokens[i+L]`。

    **对齐约定（本仓库历史 bug 高发区）**：把 `h` 放回"逐位置"数组时必须落在
    下标 `i + L - 1`（上下文结尾），不是 `i + L`。
    见 tests/test_ngram_hash.py::test_lookup_alignment_toy_table。
    """
    n = len(tokens) - L
    h = np.zeros(max(n, 0), dtype=np.uint64)
    for j in range(L):
        h = h * MULT + tokens[j:j + n].astype(np.uint64)
    return h


class NDBStats:
    """一次 `read()` 的附带统计（监控用，不参与计算图）。"""

    __slots__ = ('covered', 'top1_hit', 'mean_total')

    def __init__(self, covered, top1_hit, mean_total):
        self.covered = covered        # 槽非空的比例
        self.top1_hit = top1_hit      # 覆盖内 top-1 恰好等于真值的比例（需传 targets）
        self.mean_total = mean_total  # 非空槽的平均观测数（置信度的直接代理）

    def __repr__(self):
        return (f"NDBStats(covered={self.covered:.4f} top1_hit={self.top1_hit:.4f} "
                f"mean_total={self.mean_total:.1f})")


class NgramNDB(nn.Module):
    """后缀 n-gram 外部数据库：no_grad 表 + 可学习读写门控。

    参数
    ----
    n_embd       : 门控网络的输入维度（= 模型隐藏维）
    levels       : 后缀长度列表，如 `(8,)` 或 `(8, 6)`
    slots        : 每级槽位数（int 或与 levels 等长）
    top_k        : 每槽保留的续写个数（1 = 只存 top-1，退化为硬检索）
    vocab_size   : 词表大小
    max_table_gb : 表内存上限，超过直接报错
    """

    def __init__(self, n_embd, levels=(8,), slots=67_108_864, top_k=4,
                 vocab_size=8192, max_table_gb=6.0):
        super().__init__()
        self.levels = tuple(int(x) for x in levels)
        if isinstance(slots, int):
            slots = [slots] * len(self.levels)
        self.slots = [int(s) for s in slots]
        if len(self.slots) != len(self.levels):
            raise ValueError("slots 与 levels 长度必须一致")
        if top_k < 1:
            raise ValueError("top_k 必须 >= 1")
        self.top_k = int(top_k)
        self.vocab_size = int(vocab_size)
        self.max_L = max(self.levels)

        # ---- 内存估算：跑之前就拦住，别跑到一半 OOM ----
        per_slot = self.top_k * 4 * 2 + 4        # toks + cnts (int32) + totals (int32)
        self.table_bytes = sum(M * per_slot for M in self.slots)
        if self.table_bytes / 2**30 > max_table_gb:
            raise ValueError(
                f"NDB 表需要 {self.table_bytes/2**30:.2f}GB，超过上限 {max_table_gb}GB。"
                f"缩小 slots / top_k / levels，或提高 max_table_gb。")

        # ---- 表：**故意不注册成 buffer** ----
        # 理由：2.4GB 起、不进优化器、也不该进 state_dict（否则 checkpoint 爆炸）。
        # 用 numpy 而不是 torch 张量：写路径是"随机下标累加"，numpy 侧更快更省。
        # ⚠ 必须是 **-1（空）** 而不是 0：token 0 是合法词表 id，用 0 当"空"
        #   会让第一次 flush 时每个槽凭空多出 K 个 (token=0, count=0) 的幽灵候选，
        #   并被当成合法续写排进 top-K。这个 bug 由"增量写 vs 离线建表逐位对照"抓到。
        self._toks = [np.full((M, self.top_k), -1, dtype=np.int32) for M in self.slots]
        # 计数用 **float32** 而不是 int32：写权重 w_t ∈ (0,1) 是小数，
        #   用 int32 就得每次 flush 四舍五入，多次 flush 累积会丢精度
        #   （实测 5.0 vs 5.848）。float32 占同样的 4 字节，但精确累加。
        self._cnts = [np.zeros((M, self.top_k), dtype=np.float32) for M in self.slots]
        self._totals = [np.zeros(M, dtype=np.float32) for M in self.slots]

        # ---- 热缓冲：本窗口累积的 (level, slot, token) → 加权计数 ----
        self._buf = [{} for _ in self.levels]
        self._write_on = False
        self.n_observed = 0        # 累计观测位置数
        self.n_flushed = 0

        # ---- 可学习门控（**这些才是模型参数**）----
        self.n_stats = 3           # [log(total), top1 占比, 槽是否非空]
        self.write_gate = nn.Linear(n_embd, 1)
        nn.init.zeros_(self.write_gate.weight)
        nn.init.constant_(self.write_gate.bias, 1.0)     # 起点 σ(1)≈0.73：先平均地写
        # ★ 读门控**不能**全零初始化：`∂g/∂w_t = g(1−g)·W_r[:,−1]`，若整行权重为 0，
        #   写门控在 init 时梯度恒为零（要等 W_r 自己先动起来才"复活"）—— 那是个死启动。
        #   保留 Linear 的默认初始化，只把 bias 设成 -2，使初始门控 ≈0.12（先轻信）。
        self.read_gate = nn.Linear(n_embd + self.n_stats + 1, 1)
        nn.init.constant_(self.read_gate.bias, -2.0)
        self.level_weight = nn.Parameter(torch.zeros(len(self.levels)))

    # ==================================================================
    # 写
    # ==================================================================
    @contextlib.contextmanager
    def write_enabled(self):
        """只有包在这里面的 `observe()` 才真正写入。

        ★ 训练循环必须只在**训练 micro-batch** 外套它，绝不能包住
        `estimate_loss` / 健康体检 —— 否则验证集写进训练库（val 泄漏）。
        """
        prev, self._write_on = self._write_on, True
        try:
            yield self
        finally:
            self._write_on = prev

    @property
    def write_active(self):
        return self._write_on

    @torch.no_grad()
    def observe(self, h, inputs, targets):
        """观测一个 micro-batch，把 (后缀 → 续写) 加权累积进热缓冲。

        h       : (B, T, C) 隐藏态 —— **只用来算写门控**，不作为检索键
        inputs  : (B, T) token id（后缀从这里取）
        targets : (B, T) 下一个 token（-100 表示不计）

        返回本批的平均写权重（监控用）；写未开启时返回 None。
        """
        if not self._write_on:
            return None
        w = torch.sigmoid(self.write_gate(h)).squeeze(-1)          # (B,T)
        w_np = w.detach().float().cpu().numpy().reshape(-1)
        tok = inputs.detach().cpu().numpy().astype(np.int64).reshape(-1)
        tgt = targets.detach().cpu().numpy().astype(np.int64).reshape(-1)
        self._accumulate(tok, tgt, w_np, inputs.shape[1])
        return float(w.mean())

    @torch.no_grad()
    def observe_tokens(self, inputs, targets, weight=1.0):
        """**离线预热**：不带模型 / 隐藏态，用常数权重把计数累加进热缓冲。

        为什么要它（混合方案「底座离线建 + 在线增量」）：
        小预算的在线实验里，300 步 × 1024 token = 30 万次观测，
        相对 6700 万槽位的覆盖率只有 ~0.5% —— 表几乎是空的，测不出任何东西。
        所以先用大量语料把表填到有覆盖率，再在训练中增量写。

        `weight` 用常数（默认 1.0）：预热阶段还没有"模型想写多重"这个信号，
        用写门控的初值反而引入一个无意义的偏置。
        """
        tok = inputs.detach().cpu().numpy().astype(np.int64).reshape(-1)
        tgt = targets.detach().cpu().numpy().astype(np.int64).reshape(-1)
        T = inputs.shape[-1]
        self._accumulate(tok, tgt, np.full(tok.size, float(weight), dtype=np.float32), T)

    def _accumulate(self, tok, tgt, w_np, T):
        """把 (tok, tgt, w) 三个扁平数组按槽累加进热缓冲。observe / observe_tokens 共用。"""
        for li, L in enumerate(self.levels):
            if T <= L:
                continue
            hh = suffix_hashes(tok, L)
            if hh.size == 0:
                continue
            slot = (hh % np.uint64(self.slots[li])).astype(np.int64)
            nxt = tgt[L:L + len(hh)]
            ww = w_np[L:L + len(hh)]
            keep = (nxt != -100) & (ww > 1e-4)
            if not keep.any():
                continue
            self.n_observed += int(keep.sum())
            d = self._buf[li]
            for s, y, wv in zip(slot[keep], nxt[keep], ww[keep]):
                k = (int(s), int(y))
                d[k] = d.get(k, 0.0) + float(wv)

    def flush(self):
        """把热缓冲合并进 top-K 表。返回本次合并的 (槽, 词) 对数。

        两步：
          1. `_totals[slot] += 本窗口该槽的**全部**权重`（含之后会被挤出 top-K 的词）
             —— total 必须是"该槽被观测了多少次"，而不是"留下的 top-K 之和"，
             否则读门控的"top-1 占比"会被系统性高估。
          2. 把 旧 top-K 与 新观测 合并，按计数取前 K。

        复杂度 O(更新量)，与表大小无关 —— 所以可以每 N 步做一次，不必每步全表扫。
        """
        n_merged = 0
        V = self.vocab_size
        for li in range(len(self.levels)):
            buf = self._buf[li]
            if not buf:
                continue
            n = len(buf)
            # dict 的 key 是 (slot, token) 二元组；np.fromiter 不会自动展平元组，
            # 直接转成 (n, 2) 数组最省事也最不容易错。
            keys = np.array(list(buf.keys()), dtype=np.int64).reshape(-1, 2)
            vals = np.fromiter(buf.values(), dtype=np.float64, count=n)
            self._buf[li] = {}
            n_merged += n

            new_slot, new_tok, new_cnt = keys[:, 0], keys[:, 1], vals

            # ① 先把该槽的全部观测计入 totals
            tot_inc = np.zeros(new_slot.max() + 1, dtype=np.float64)
            np.add.at(tot_inc, new_slot, new_cnt)
            nz = np.nonzero(tot_inc)[0]
            self._totals[li][nz] += tot_inc[nz].astype(np.float32)

            # ② 合并 top-K：旧 K 行 + 新观测 → 按 (slot, token) 聚合 → 每槽取前 K
            slots_u = np.unique(new_slot)
            old_t = self._toks[li][slots_u].astype(np.int64)          # (n_u, K)
            old_c = self._cnts[li][slots_u].astype(np.float64)
            K = self.top_k
            comb_slot = np.concatenate([np.repeat(slots_u, K), new_slot])
            comb_tok = np.concatenate([old_t.ravel(), new_tok])
            comb_cnt = np.concatenate([old_c.ravel(), new_cnt])
            ok = comb_tok >= 0
            comb_slot, comb_tok, comb_cnt = comb_slot[ok], comb_tok[ok], comb_cnt[ok]

            uk, inv = np.unique(comb_slot * V + comb_tok, return_inverse=True)
            inv = inv.reshape(-1)
            agg = np.zeros(len(uk), dtype=np.float64)
            np.add.at(agg, inv, comb_cnt)
            us, ut = uk // V, uk % V

            # 按 slot 升序、计数降序 → 每个 slot 取前 K
            order = np.lexsort((-agg, us))
            us_s, agg_s, ut_s = us[order], agg[order], ut[order]
            bounds = np.searchsorted(us_s, slots_u, side='left')
            rank = np.arange(len(us_s)) - np.repeat(bounds, np.diff(np.append(bounds, len(us_s))))
            sel = rank < K

            # 先整体清空这些槽，再写回入选的 (token, rank)。
            # ⚠ 这里**只能**清一次，且 toks/cnts 必须同时清：
            #   第一版多加了一行 `self._toks[li][us_s[sel]] = 0`（整行置 0），
            #   而 _cnts 没有对应操作 → 表里出现 (token=0, count=0) 的幽灵条目，
            #   它们会被当成合法候选排进 top-K，污染检索结果。
            #   这个 bug 只有"增量写 vs 离线建表逐位对照"才抓得到。
            self._toks[li][slots_u] = -1
            self._cnts[li][slots_u] = 0
            self._toks[li][us_s[sel], rank[sel]] = ut_s[sel].astype(np.int32)
            self._cnts[li][us_s[sel], rank[sel]] = agg_s[sel].astype(np.float32)
            self.n_flushed += 1
        return n_merged

    # ==================================================================
    # 读
    # ==================================================================
    @torch.no_grad()
    def _lookup(self, inputs):
        """查表。返回每级的 `(tok, cnt, total, covered)`，全部是 numpy。

        tok/cnt: (B, T, K) int64（-1 = 空）  total: (B, T) int64  covered: (B, T) bool
        """
        flat = inputs.detach().cpu().numpy().astype(np.int64).reshape(-1)
        B, T = inputs.shape
        out = []
        for li, L in enumerate(self.levels):
            hh = suffix_hashes(flat, L)
            slot = np.full(flat.size, -1, dtype=np.int64)
            if hh.size:
                # ★ 对齐：h[i] 的上下文结尾在下标 i+L-1
                slot[L - 1: L - 1 + len(hh)] = (hh % np.uint64(self.slots[li])).astype(np.int64)
            has = slot >= 0
            safe = np.where(has, slot, 0)
            tok = np.full((flat.size, self.top_k), -1, dtype=np.int64)
            cnt = np.zeros((flat.size, self.top_k), dtype=np.float32)
            tok[has] = self._toks[li][safe[has]]
            cnt[has] = self._cnts[li][safe[has]]
            tot = self._totals[li][safe].astype(np.float32)
            covered = has & (tot > 0)
            tok[~covered] = -1
            cnt[~covered] = 0
            tot = np.where(covered, tot, 0)
            out.append((tok.reshape(B, T, -1), cnt.reshape(B, T, -1),
                        tot.reshape(B, T), covered.reshape(B, T)))
        return out

    def _stat_features(self, total, cnt, device):
        """读门控的输入统计量：全部是**推理时可得**的量（不需要标签）。"""
        t = torch.from_numpy(total).to(device).float()
        c = torch.from_numpy(cnt).to(device).float()
        denom = t.clamp(min=1.0)
        f_total = torch.log1p(t) / math.log1p(100.0)                  # ∈ [0, ~1]
        f_share = (c.max(dim=-1).values / denom)                      # top-1 占比
        f_cov = (t > 0).float()
        return torch.stack([f_total, f_share, f_cov], dim=-1)

    def read(self, h, inputs, p_model, targets=None):
        """读 + 注入。返回 `(p_new, stats)`。

        p_new = (1−g)·p_model + g·p_ng，`g` 与多级混合 `α` 都由模型决定。
        `targets` 只用于监控 top-1 命中率，传 None 就不统计。
        """
        B, T, V = p_model.shape
        dev = p_model.device
        lk = self._lookup(inputs)

        # 写门控值：既决定"写多重"，也作为读门控的输入 → W_w 才拿得到梯度
        w_t = torch.sigmoid(self.write_gate(h)).squeeze(-1)            # (B,T)
        alpha = torch.softmax(self.level_weight, dim=0)                # (n_levels,)

        p_ng = p_model.new_zeros(B, T, V)
        covered_any = torch.zeros(B, T, dtype=torch.bool, device=dev)
        n_cov = 0
        tot_sum = 0.0
        hit_ok = 0
        n_hit_tot = 0
        feats = []
        for li, (tok, cnt, total, covered) in enumerate(lk):
            cov_t = torch.from_numpy(covered).to(dev)
            covered_any |= cov_t
            if cov_t.any():
                n = int(cov_t.sum())
                n_cov += n
                tot_sum += float(total[covered].sum())
                denom = torch.from_numpy(total).to(dev).float().clamp(min=1.0)
                idx = torch.from_numpy(tok).to(dev).clamp(min=0)
                c = torch.from_numpy(cnt).to(dev).float()
                valid = (torch.from_numpy(tok).to(dev) >= 0).float()
                p_l = torch.zeros(B, T, V, device=dev)
                p_l.scatter_add_(2, idx, (c / denom.unsqueeze(-1)) * valid)
                # ★ 必须在级内重新归一：top-K 的计数之和 < total（有被挤出 / 未进 top-K 的词），
                #   所以 `cnt/total` 本身**不是**概率分布。不归一就混进凸组合，`p_new` 会
                #   整体小于 1（实测 0.88），直接破坏"恒合法的凸组合"这个核心性质。
                #   K=1 时结果恰好是 one-hot(tgt) —— 与探针 `nll_under_retrieval` 口径
                #   **完全一致**，离线测出的 Δ 才能在在线实现里兑现。
                p_l = p_l / p_l.sum(-1, keepdim=True).clamp(min=1e-9)
                p_ng = p_ng + alpha[li] * p_l
                if targets is not None:
                    y = targets.clamp(min=0).unsqueeze(-1)
                    top1 = idx[:, :, :1]
                    hit_ok += int(((top1 == y) & cov_t.unsqueeze(-1)).sum())
                    n_hit_tot += n
            feats.append(self._stat_features(total, cnt, dev))

        feat = torch.stack(feats, 0).mean(0)
        g = torch.sigmoid(self.read_gate(torch.cat([h, feat, w_t.unsqueeze(-1)], dim=-1)))
        # ★ 门控必须在**槽为空**的位置归零：那里 p_ng ≡ 0，若还注入 g·p_ng = 0 就直接
        #   把 p_model 缩成 (1−g) 倍 → p_new 总和 < 1（实测 0.89），凸组合性质被破坏。
        #   探针里由 `okf = ok.float()` 做这件事，搬到模块时漏了一次。
        g = g * covered_any.float().unsqueeze(-1)
        p_new = (1 - g) * p_model + g * p_ng

        stats = NDBStats(
            covered=float(covered_any.float().mean()),
            top1_hit=(hit_ok / n_hit_tot) if n_hit_tot else float('nan'),
            mean_total=(tot_sum / n_cov) if n_cov else 0.0,
        )
        self._last_gate = float(g.detach().mean())
        return p_new, stats

    # ==================================================================
    # 监控 / 杂项
    # ==================================================================
    def table_gb(self):
        return self.table_bytes / 2**30

    def n_filled_slots(self):
        return [int((t >= 0).any(axis=1).sum()) for t in self._toks]

    def gate_summary(self):
        return {
            'write_gate_bias': float(self.write_gate.bias.detach()),
            'read_gate_bias': float(self.read_gate.bias.detach()),
            'level_alpha': torch.softmax(self.level_weight, 0).detach().cpu().tolist(),
            'n_observed': self.n_observed,
            'n_flushed': self.n_flushed,
        }

    def extra_repr(self):
        return (f"levels={self.levels} slots={self.slots} top_k={self.top_k} "
                f"table={self.table_gb():.2f}GB")
