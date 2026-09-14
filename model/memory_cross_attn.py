# -*- coding: utf-8 -*-
"""RETRO-lite 记忆读取模块（dev-notes/79 B1；B4 低秩 per-token）。

把 no_grad 的 chunk 记忆库读进残差流：

  查询：每个 chunk（默认 64 token）用块内 hidden 均值做一次检索
  检索：q · storeᵀ → top-k（可屏蔽与查询位置过近的条目，防训练集自匹配）
  读取：
    mean  模式：chunk 内 token 对 top-k 个「chunk 均值」做 cross-attention（每 chunk 1 个键）
    token 模式：chunk 内 token 对 top-k×chunk 个「低秩 per-token 向量」做 cross-attention
                （每 chunk 64 个键，K/V 由 rank→n_embd 的可训升维给出）
  注入：h ← h + gate_t · wo(CrossAttn(wq(h), wk(mem), wv(mem)))

store 是 buffer（no_grad，不参与优化器）；可训的只有 wq/wk/wv/wo + 逐 token 门控。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryCrossAttention(nn.Module):
    def __init__(self, n_embd, store_keys, store_pos=None, store_vals=None,
                 chunk=64, top_k=4, n_head=8, gate_init=0.1, exclude_radius=0,
                 retr_noise=0.0, att_sim_gain=0.0, live=None):
        super().__init__()
        # 训练期检索噪声：每个槽以 retr_noise 的概率被换成随机条目。默认 0 = 旧行为逐位一致。
        # 显式属性（不是 self.training），因为本模块挂在 forward hook 上、不是 model 的子模块，
        # model.eval() 管不到它；训练脚本会在监控 eval 期间临时置 0。
        self.retr_noise = float(retr_noise)
        self.n_embd = int(n_embd)
        self.chunk = int(chunk)
        self.top_k = int(top_k)
        self.n_head = int(n_head)
        self.head_dim = self.n_embd // self.n_head
        self.exclude_radius = int(exclude_radius)
        # store_vals 非空 → per-token 低秩模式：记忆内容是 (N, chunk, rank) 的逐 token 低秩向量
        self.per_token = store_vals is not None
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)
        nn.init.zeros_(self.wo.weight)  # 零初始化输出：记忆起点是 no-op，只在有用时才被学起来
        if self.per_token:
            self.rank = int(store_vals.shape[-1])
            # K/V 从 rank 空间升回 n_embd（低秩投影的（可训）伪逆）
            self.wk = nn.Linear(self.rank, n_embd, bias=False)
            self.wv = nn.Linear(self.rank, n_embd, bias=False)
            self.register_buffer("vals", store_vals.detach().clone(), persistent=False)
        else:
            self.rank = None
            self.wk = nn.Linear(n_embd, n_embd, bias=False)
            self.wv = nn.Linear(n_embd, n_embd, bias=False)
        # 逐 token 门控：g_t = σ(w·h_t + b)，让模型自己决定哪些位置注入
        self.gate_proj = nn.Linear(n_embd, 1, bias=False)
        nn.init.zeros_(self.gate_proj.weight)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        # 检索相关性增益 λ：把「条目与查询的余弦相似度 z 分数」加进注意力 logit。
        # 初值 0 = 起点等价于旧行为（逐位一致），由训练自己决定是否使用这条免费信息。
        self.sim_gain = nn.Parameter(torch.tensor(float(att_sim_gain)))
        # ★ 本模块**没有**可学习写门控：`write_online` 是 `@torch.no_grad()` 的**规则式**写入
        #   （人给的惊讶分位），写什么由规则定，不由模型定。
        #   ⇒ **本项目采用的、由模型自己决定读写的 NDB 是 `model/ngram_ndb.py`**：
        #     `w_t = σ(W_w·h)` 在 `read()` 里带梯度重算并进 `read_gate` 的输入
        #     （`ngram_ndb.py:384/:422/:427`）⇒ `∂L/∂W_w ≠ 0`，
        #     `tests/test_ngram_ndb.py:292` 就钉着这条。要迁移的正是那套接线。
        # store：no_grad，永不被优化器触碰
        self.register_buffer("store", store_keys.detach().clone(), persistent=False)
        if store_pos is not None:
            self.register_buffer("store_pos", store_pos.detach().to(torch.int32),
                                 persistent=False)
        else:
            self.store_pos = None
        # --- 在线写（2026-09-15 新增）：`live` = 已写入的活跃槽数，`ptr` = 环形写指针 ---
        #   ★ 向后兼容的判据：**预构建库**（ndb_store 非空）走 live=整库 ⇒ 检索范围、
        #     行为与旧版**逐位一致**；`live=0` 只在"新建空库、准备在线写"时出现。
        #   两者都是 Python int（不是 buffer）：`store` 本身是 buffer，会跟着 .to(device) 走，
        #   而 int 不需要搬运，也不该进 state_dict。
        self.live = int(store_keys.shape[0]) if live is None else int(live)
        self.ptr = 0 if live is not None else 0
        self.written_total = 0
        self.write_full_events = 0

    def _retrieve(self, q, q_pos=None):
        """q: (B,C,D) → idx (B,C,k)。检索本身不可微，走 no_grad。"""
        n = self._n_live()
        k = min(self.top_k, n)
        if getattr(self, "random_retrieve", False):  # 对照：随机检索（内容无意义）
            B, C, _ = q.shape
            return torch.randint(0, n, (B, C, k), device=q.device)
        with torch.no_grad():
            sim = q.to(self.store.dtype) @ self.store[:n].t()  # (B,C,n)
            if (self.exclude_radius > 0 and q_pos is not None
                    and self.store_pos is not None):
                d = (self.store_pos[:n][None, None, :] - q_pos[:, :, None]).abs()
                sim = sim.masked_fill(d < self.exclude_radius, float("-inf"))
            idx = sim.topk(k, dim=-1).indices  # (B,C,k)，k = min(top_k, 活跃槽数)
        # 训练期检索噪声（DeepSeek-V4.1 §2.5「train-aware adaptation」的同构做法）：
        # 训练时永远给正确记忆 → 与推理期分布不一致（推理期检索必然有错），
        # 模型因此学不会「记忆不可靠时降权」，实测 10% 错误就废掉 65% 收益。
        # 按概率把槽换成随机条目，让模型在该分布下被优化。
        rn = float(getattr(self, "retr_noise", 0.0))
        if rn > 0:
            # 用「按步种子」的独立 Generator，而不是全局 RNG：
            # 本仓库开了 gradient_checkpointing，反向会重算前向。若用全局 RNG，
            # 重算时抽到的噪声与原始前向不同 → 重算输出 ≠ 原始输出 → 梯度是错的。
            # 按步种子保证同一步（含重算）拿到同一张掩码；且完全不消耗全局 RNG，
            # 数据采样流（_sample_nonempty_ix）不受影响。
            g = torch.Generator(device=idx.device)
            g.manual_seed(int(getattr(self, "noise_seed", 0)))
            m = torch.rand(idx.shape, device=idx.device, generator=g) < rn
            r = torch.randint(0, n, idx.shape, device=idx.device, generator=g)
            idx = torch.where(m, r, idx)
        return idx

    def _n_live(self):
        """活跃槽数（预构建库 = 整库；空库 = 0）。"""
        return max(0, min(int(self.live), int(self.store.shape[0])))

    @torch.no_grad()
    def write_online(self, h, targets, logits, *, quantile=0.02,
                     ignore_index=-100):
        """**规则式**种子写（no_grad）：把本 batch 里最答不上来的 chunk 均值键写进库里。

        返回 `(n_written, info)`。

        ## 判据是**人类给的规则**，不是模型的决定

        对每个 chunk 算它在有效 target 上的平均 CE（`ignore_index` 的位置不算），
        按 batch 内**人设的分位** `quantile` 取最惊讶的那批。信号（CE）是模型自己的，
        **阈值是我们的** —— 而"谁的阈值"才是决定"写什么"的那一步。
        所以本方法等价于**离线预热填表**的在线版本：它建起来的是**人灌的库**。
        要用它当"种子"（先灌一点让读路径能点火）可以，别把它当成模型在学写。

        ★ **本项目采用的、由模型自己决定读写的 NDB 是 `model/ngram_ndb.py`**：
        那里 `w_t = σ(W_w·h)` 带梯度并进 `read_gate` 的输入（`∂L/∂W_w ≠ 0`，
        `tests/test_ngram_ndb.py:292` 钉着）。

        ★ 键用的是**读取时同一套 chunk 均值**（`forward` 里 `q = hc.mean(dim=2)`），
        所以写进去的条目与检索空间同构，不需要额外的 key 投影。

        ★ 用 `ignore_index` 过滤掉 loss 掩码外的位置：不该产生梯度的 token
        也不该被写进记忆（否则人格层会把对方的话记成"自己答不上来的东西"）。
        """
        if h is None:
            return 0, {"skip": "no_hidden"}
        cap = int(self.store.shape[0])
        if self._n_live() >= cap:
            self.write_full_events += 1
            # 满了不清空、不淘汰：改成环形覆盖（下面 ptr 自动取模）
        B, T, D = h.shape
        C = T // self.chunk
        Tc = C * self.chunk
        if C == 0:
            return 0, {"skip": "too_short"}
        hc = h[:, :Tc].reshape(B, C, self.chunk, D).float()
        keys = hc.mean(dim=2).reshape(B * C, D)                     # (B*C, D)
        tgt = targets[:, :Tc].reshape(B, C, self.chunk)
        lg = logits[:, :Tc].reshape(B * C * self.chunk, -1).float()
        ce = F.cross_entropy(lg, tgt.reshape(-1), reduction="none",
                             ignore_index=ignore_index).reshape(B, C, self.chunk)
        valid = (tgt != ignore_index)
        n_valid = valid.sum(dim=-1)                                 # (B,C)
        loss = (ce * valid).sum(dim=-1) / n_valid.clamp(min=1)
        flat = loss.reshape(-1)
        cand = (n_valid.reshape(-1) > 0).nonzero(as_tuple=False).flatten()
        if cand.numel() == 0:
            return 0, {"skip": "no_valid_target"}
        lv = flat[cand]
        if quantile > 0:
            thr = float(torch.quantile(lv, 1.0 - float(quantile)))
            sel = cand[lv > thr]
            if sel.numel() == 0:                                    # 全相等时分位取不到
                sel = cand[lv >= thr]
        else:
            sel = cand
        if sel.numel() == 0:
            return 0, {"skip": "none_selected"}
        n = int(sel.numel())
        slots = (torch.arange(n, device=h.device) + self.ptr) % cap
        self.store[slots] = keys[sel].to(self.store.dtype)
        self.ptr = int((self.ptr + n) % cap)
        self.live = min(cap, self._n_live() + n)
        self.written_total += n
        return n, {"surprise_written": n, "mean_surprise": float(lv.mean()),
                   "thr": float(thr) if quantile > 0 else 0.0,
                   "live": self._n_live()}

    def forward(self, h, q_pos=None):
        """h: (B,T,D) → delta (B,T,D)，加到 h 上。"""
        B, T, D = h.shape
        C = T // self.chunk
        Tc = C * self.chunk
        if Tc == 0 or self._n_live() <= 0:
            # 在线写的库在第一次写之前是**空的** ⇒ 记忆必须是 no-op
            # （否则会拿全零槽去交叉注意力，产出没有意义的 delta）。
            return torch.zeros_like(h)
        hc = h[:, :Tc].reshape(B, C, self.chunk, D).float()
        q = hc.mean(dim=2)  # (B,C,D)
        idx = self._retrieve(q, q_pos)  # (B,C,k)
        # 检索相关性：对「实际取回的条目」重算余弦相似度（索引被污染时同样成立）。
        # 这一步是本模块最大的结构缺陷的补丁：检索用 sim 选完条目就把 sim 扔了，
        # 注意力只用学出来的 wq/wk 重新打分——于是随机条目和正确条目抢注意力的
        # 机会几乎相同。实测（2026-09-10 免训练探针，step 19000）：
        #   逐个污染 top-4 的任一槽，代价完全相同 +0.118~+0.121（模型完全不认排名）；
        #   把 4 个槽全换随机也只贵 1.83 倍；k 从 4 加到 64 只把单点污染代价降 46%。
        # 把 sim 加回 logit，等于免费告诉注意力「哪个条目其实是相关的」。
        mem_chunks = self.store[idx].float()  # (B,C,k,D)
        qn = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
        mn = mem_chunks / (mem_chunks.norm(dim=-1, keepdim=True) + 1e-6)
        sim_used = (qn.unsqueeze(-2) * mn).sum(-1)  # (B,C,k) ∈ [-1,1]
        # ★ 跨槽 z 分数：告诉注意力「这些条目里哪个真的更相关」。
        #   ⚠ 只在 k ≥ 2 时有定义：k = min(top_k, 库里的活跃条数)，库只有 1 条时 k=1，
        #   而 `std` 在单元素上是 NaN（correction=1），且 `0 * nan = nan`
        #   ⇒ 即使 `sim_gain=0` 也会把整个 forward 污染成 NaN。
        #   实测（2026-09-15）：live=1 → 输出 isnan=True；live≥2 → 干净。
        #   k=1 时槽之间**本来就没有名次可排** ⇒ 取 0（等价于「这条相关性信息不存在」），
        #   这是 k≥2 公式在退化情形下的正确极限，而不是补丁。
        if sim_used.shape[-1] >= 2:
            sim_z = ((sim_used - sim_used.mean(dim=-1, keepdim=True))
                     / (sim_used.std(dim=-1, keepdim=True) + 1e-6))
        else:
            sim_z = torch.zeros_like(sim_used)
        Q = self.wq(hc)  # (B,C,chunk,D)
        if self.per_token:
            mv = self.vals[idx].float()  # (B,C,k,chunk,rank)
            mv = mv.reshape(B, C, idx.shape[-1] * self.chunk, self.rank)
            K = self.wk(mv)  # (B,C,k*chunk,D)
            V = self.wv(mv)
        else:
            K = self.wk(mem_chunks)  # (B,C,k,D)
            V = self.wv(mem_chunks)
        nk = K.shape[2]
        # 相关性广播到「每个条目覆盖的那几个键」上
        if nk != sim_z.shape[-1]:
            sim_b = sim_z.repeat_interleave(nk // sim_z.shape[-1], dim=-1)
        else:
            sim_b = sim_z
        # 多头
        Q = Q.reshape(B, C, self.chunk, self.n_head, self.head_dim).permute(0, 1, 3, 2, 4)
        K = K.reshape(B, C, nk, self.n_head, self.head_dim).permute(0, 1, 3, 4, 2)
        V = V.reshape(B, C, nk, self.n_head, self.head_dim).permute(0, 1, 3, 2, 4)
        att = (Q @ K) / math.sqrt(self.head_dim)  # (B,C,H,chunk,nk)
        att = att + self.sim_gain * sim_b[:, :, None, None, :]
        att = att.softmax(dim=-1)
        out = (att @ V).permute(0, 1, 3, 2, 4).reshape(B, C, self.chunk, D)
        out = self.wo(out)  # (B,C,chunk,D)
        g = torch.sigmoid(self.gate_proj(hc) + self.gate)  # (B,C,chunk,1) 逐 token 门控
        out = (g * out).reshape(B, Tc, D)
        delta = torch.zeros(B, T, D, dtype=h.dtype, device=h.device)
        delta[:, :Tc] = out.to(h.dtype)
        return delta
