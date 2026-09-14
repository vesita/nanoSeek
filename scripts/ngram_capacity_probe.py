#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全量语料的 n-gram 记忆容量探针（流式，内存 O(槽位数)）。

## 为什么重写
`local/ngram_memory_probe.py` 一次性建 (N,) uint64 哈希数组 + `np.lexsort` 索引，
N=836M 时约 17GB —— 本机可用内存只有 ~9GB，**全量必 OOM**。
本脚本改成：
  ① 用**位图**（固定槽位表 present[M]）代替精确哈希表 —— 这正是真实 NDB 的形态
     （槽 = hash % M），内存从 O(N) 降到 O(M/8)；
  ② **流式**扫全量流，分块 + L-1 重叠，绝不一次载入；
  ③ 只在需要"多数续写"时再做一遍 top-1 累积（每槽 8 字节）。

## 为什么必须全量 + 全流随机采样
dev-notes/78 §13.1 证明 `train_char.bin` 是按文件名顺序拼接的（源有序）：
顺序前缀建库只覆盖前几个源，覆盖率被严重低估（同一配置 7.5% → 40.2%，5.4×）。
§12.3 只有 30M/100M 两个前缀点，外推区间 9%~43% —— 太宽，必须实测。

## 输出
1. **覆盖率曲线**：L ∈ {6,8,12} × 槽位 M ∈ {4.19M,16.8M,67M,268M}
   → val 位置中，其后缀在 train 全量流过出现过（位图置 1）的比例。
   ⚠ 读法（2026-09-10 实测后更正）：覆盖率随 M **下降**（M 越大空槽越多），
   而 M 越小越饱和 → M=4.2M 时恒为 100%。所以小 M 的 100% 是"饱和"而非"覆盖好"，
   **覆盖率不是瓶颈**；瓶颈是槽内 top-1 续写对不对（见第 2 段的命中率）。
2. **插值 Δ**：L=8 / M=67M 建 top-1，读时按 kNN-LM 凸组合
   `p = (1-λ)·p_model + λ·p_ng`，扫 λ，报配对 Δ（<0 有益）

## 2026-09-10 事故与加固（GPU Hang）
第一次运行在 GPU 段把显卡跑挂：内核日志
`ring gfx_0.0.0 timeout` → `gfx_0.1.0 reset failed` → `GPU reset begin! MODE1 reset`
→ `VRAM is lost due to GPU reset`。**连带把桌面一起打死**（VRAM 全丢）。
加固措施（本文件）：
  - 第 2 段默认 `--device cpu`：模型只有 81.58M，CPU 前向 ~12-15 分钟，**零 GPU 风险**；
  - 槽表查表在 **numpy 侧**做，GPU 上只保留 (N,) 的小张量，不再常驻 537MB；
  - 目标概率计算从 O(N·V) 降到 O(N)（不再 clone 整个 (N,8192) 概率矩阵）；
  - 每 `--log_every` 个 batch 打印进度 + 峰值显存，**下次挂能定位到具体 batch**；
  - top-1 表落盘缓存（`--cache_top1`），重跑不必再等 73s 建表。

用法：
  # 只跑覆盖率（CPU，约 3 分钟）
  .venv/bin/python -u scripts/ngram_capacity_probe.py --only_cover
  # 跑插值 Δ（CPU，建表 73s + 评测 ~15 分钟）
  .venv/bin/python -u scripts/ngram_capacity_probe.py --skip_cover
"""
import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("HSA_ENABLE_SDMA", "0")

import numpy as np
import torch
import torch.nn.functional as F

VOCAB = 8192
MULT = np.uint64(1000003)


def hashes(tok, L):
    """多项式滚动哈希：context = tok[i:i+L] → 要预测的下一个 token 是 tok[i+L]。

    返回值长度 len(tok)-L，第 i 个元素对应后缀 tok[i:i+L]。
    ★ 对齐关系（本文件历史上栽过两次）：
        hashes(tok,L)[i] 的上下文**结尾**在下标 i+L-1，目标是 tok[i+L]。
        因此把 val 的哈希放进"逐位置"数组时，必须放在下标 i+L-1：
            slot[L-1 : L-1+len(vh)] = vh % M
        写成 slot[L:] 就整体**错一个位置**（预测目标整体前移一位）。
        命名上极易记混，因为偏移量恰好是 L 与 L-1 之差 —— 别靠记忆，靠测试
        （tests/test_ngram_hash.py::test_lookup_alignment_toy_table）。
    """
    n = len(tok) - L
    h = np.zeros(n, dtype=np.uint64)
    for j in range(L):
        h = h * MULT + tok[j:j + n].astype(np.uint64)
    return h


def iter_chunks(path, chunk, L_max, seed=None, sample_runs=None):
    """流式产出 (tok, base_offset)。

    seed=None      ：顺序扫全流（仅用于对照，注意 §13.1 的源有序陷阱）
    sample_runs=k  ：全流**随机**取连续 k token 的 run（保留后缀结构）。
                     §13.1 要求建库必须全流随机采样，否则覆盖率被严重低估。
    """
    d = np.memmap(path, dtype=np.uint16, mode="r")
    n = len(d)
    if sample_runs is None:
        pos = 0
        while pos < n:
            end = min(pos + chunk, n)
            yield np.asarray(d[pos:end], dtype=np.int64), pos
            pos = end
    else:
        rng = np.random.default_rng(seed)
        total = 0
        target = sample_runs
        while total < target:
            take = min(chunk, target - total)
            i = int(rng.integers(0, max(n - take - L_max - 2, 1)))
            yield np.asarray(d[i:i + take], dtype=np.int64), i
            total += take
    del d


def build_bitmaps(path, Ls, Ms, chunk, sample_runs, L_max, seed=1234):
    """一次流式扫过语料，为每个 (L, M) 置位。返回 {(L,M): bitmap}。"""
    bits = {(L, M): np.zeros(M, dtype=bool) for L in Ls for M in Ms}
    n_tok_seen = 0
    t0 = time.time()
    for ci, (tok, _off) in enumerate(iter_chunks(path, chunk, L_max, seed, sample_runs)):
        if len(tok) <= L_max + 1:
            continue
        n_tok_seen += len(tok)
        for L in Ls:
            if len(tok) <= L + 1:
                continue
            h = hashes(tok, L)
            for M in Ms:
                bits[(L, M)][h % np.uint64(M)] = True
            del h
        if ci % 10 == 0:
            el = time.time() - t0
            print(f"    [bitmap] chunk {ci}  累计 {n_tok_seen/1e6:.0f}M token  "
                  f"{el:.0f}s", flush=True)
    print(f"    [bitmap] 扫完 {n_tok_seen/1e6:.0f}M token，用时 {time.time()-t0:.0f}s", flush=True)
    return bits, n_tok_seen


def build_top1(path, L, M, chunk, sample_runs, seed=1234):
    """流式建 (槽 → 多数续写 token, 计数)。**近似**：块内 top-1 与全局 top-1 合并。

    近似偏差：同一后缀在不同块里若各只出现 1 次且续写不同，则永不累积成多数。
    块越大偏差越小（默认 20M）；这个量级下同一后缀在块内通常已有多次观测。
    """
    g_tok = np.full(M, -1, dtype=np.int32)
    g_cnt = np.zeros(M, dtype=np.int32)
    t0 = time.time()
    for ci, (tok, _off) in enumerate(iter_chunks(path, chunk, L, seed, sample_runs)):
        if len(tok) <= L + 1:
            continue
        h = (hashes(tok, L) % np.uint64(M)).astype(np.int64)
        nxt = tok[L:L + len(h)].astype(np.int64)
        key = h * VOCAB + nxt                      # M*V ≤ 67M*8192 = 5.5e11 < 2^63 ✓
        uk, uc = np.unique(key, return_counts=True)
        us, ut = (uk // VOCAB).astype(np.int64), (uk % VOCAB).astype(np.int32)
        uc = uc.astype(np.int32)
        better = uc > g_cnt[us]
        g_tok[us[better]] = ut[better]
        g_cnt[us[better]] = uc[better]
        del h, nxt, key, uk, uc, us, ut, better
        if ci % 10 == 0:
            print(f"    [top1] chunk {ci}  {time.time()-t0:.0f}s", flush=True)
    return g_tok, g_cnt


def load_val(path, block, n_windows, seed=1234):
    d = np.memmap(path, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(d) - block - 2, n_windows)
    x = np.stack([np.asarray(d[i:i + block], dtype=np.int64) for i in ix])
    del d
    return torch.from_numpy(x)


def load_or_build_top1(cache_path, corpus, L, M, chunk, sample_runs):
    """top-1 表带磁盘缓存：命中就直接读，省掉重跑一遍语料（~73s / 67M 槽）。

    缓存键写在文件名里（L/M/采样量），换个配置不会读错表。
    """
    if cache_path and os.path.exists(cache_path):
        print(f"  复用缓存表 {cache_path}", flush=True)
        z = np.load(cache_path)
        return z["g_tok"], z["g_cnt"]
    print(f"  建 top-1：L={L}  M={M/1e6:.1f}M   （内存 {M*8/1e6:.0f}MB）", flush=True)
    t0 = time.time()
    g_tok, g_cnt = build_top1(corpus, L, M, chunk, sample_runs)
    used = int((g_tok >= 0).sum())
    print(f"  已填槽 {used/1e6:.2f}M / {M/1e6:.0f}M ({used/M*100:.1f}%)  "
          f"用时 {time.time()-t0:.0f}s", flush=True)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez(cache_path, g_tok=g_tok, g_cnt=g_cnt)
        print(f"  已写缓存 {cache_path}", flush=True)
    return g_tok, g_cnt


# ── λ 门控的 4 种口径（必须分开报，别混成一个数）─────────────────────────
# §12.3 原脚本（ngram_sample_capacity.py:130）**没有门控**：
#     pym = (1-λ)·pyf + λ·w，mask = 检索覆盖位，λ 是标量 → 即下面的 A_ungated。
# 其余三种是"λ 判据 dΔ/dλ|₀ = A − p"启发的门控，按**是否用到标签**分两类：
#   B_oracle   用正确 token 的概率 → **离线 oracle 上界**，推理时不可实现；
#   C_uncert   只用模型自身最大概率 → 可部署；
#   D_disagree 只用检索到的 token 的概率 → 可部署（tgt 来自库，推理时已知）。
#
# 为什么抽成纯函数：第一版把这段直接写在评测循环里，`lg` 还是 (B,T,V) 三维而
# `gather` 要二维，跑了几分钟才炸（GPU 版还曾把显卡跑挂）。现在 lp 的形状契约
# 由 tests/test_ngram_delta.py 里的小张量守住，几毫秒就能发现。
GATE_NAMES = ("A_ungated", "B_oracle", "C_uncert", "D_disagree")


def gate_factors(lp, p_y, p_max, tok_ng, ok):
    """返回 {门控名: (N,) 因子}（**不含 λ**，且已乘上"槽非空"掩码）。

    lp     : (N, V) 的 log 概率（**必须二维**，调用方自己 reshape）
    p_y    : (N,) 正确 token 的概率
    p_max  : (N,) 每行最大概率
    tok_ng : (N,) 检索到的续写（-1 表示无）
    ok     : (N,) bool，槽非空
    """
    if lp.dim() != 2:
        raise ValueError(f"lp 必须是 (N, V) 二维（调用方要先 reshape），实际 {tuple(lp.shape)}")
    okf = ok.float()
    # 检索到的 token 的概率：索引 clip 到 0 以避免 -1 越界；无效位置最后乘 okf 归零
    p_tgt = lp.gather(1, tok_ng.clamp(min=0).unsqueeze(1)).squeeze(1).exp()
    return {
        "A_ungated": torch.ones_like(p_y) * okf,
        "B_oracle": (1.0 - p_y).clamp_min(0) * okf,
        "C_uncert": (1.0 - p_max).clamp_min(0) * okf,
        "D_disagree": (1.0 - p_tgt).clamp_min(0) * okf,
    }


def nll_under_retrieval(p_y, hit_y, gate, lam):
    """检索结果按 kNN-LM 凸组合混进模型分布后，逐 token 的 NLL（形状 (N,)）。

        p_new = p_y·(1 − λ·gate) + (λ·gate)·[tgt == y]

    即 §12.3 公式的逐位置版：`λ·gate` 就是逐位置的 λ_eff。
    λ=0 或 gate=0 时退化为 `-log p_y`（= 无库基线）。
    """
    lam_eff = lam * gate
    p_new = p_y * (1 - lam_eff) + lam_eff * hit_y
    return -torch.log(p_new.clamp_min(1e-9))


def run_cover(a, Ls, Ms):
    """第 1 段：位图覆盖率曲线（纯 numpy，不碰 GPU）。返回 {(L,M): 覆盖率}。"""
    print("═══ 1) 覆盖率（位图，全量流式）═══")
    bits, n_seen = build_bitmaps(a.corpus, Ls, Ms, a.chunk, a.sample_runs or None, max(Ls))
    val_x = load_val(a.val, a.block, a.val_windows)
    print(f"  val: {tuple(val_x.shape)}  ")

    print(f"\n  {'L':>3s} {'槽位':>10s} {'位图占用':>9s} {'val 覆盖率':>10s}")
    cover = {}
    for L in Ls:
        vh = hashes(val_x.reshape(-1).numpy(), L)
        for M in Ms:
            b = bits[(L, M)]
            hit = float(b[vh % np.uint64(M)].mean())
            cover[f"L{L}_M{M}"] = hit
            print(f"  {L:>3d} {M/1e6:>9.1f}M {M/8/1e6:>8.1f}MB {hit*100:>9.2f}%")
        del vh
    print("\n  读法：覆盖率随 M **下降**（M 越大空槽越多）→ 小 M 的 100% 是「表已饱和」，")
    print("        不是「覆盖好」。覆盖率不是瓶颈，瓶颈是槽内 top-1 续写对不对（第 2 段命中率）。")
    return cover


def run_delta(a, L_d, M_d):
    """第 2 段：kNN-LM 凸组合插值 Δ。

    目标概率按 O(N) 算，不物化 (N, 8192)：
        p_y   = p_model[y]
        λ_eff = λ · clamp(1 − p_y, 0)         只在检索命中位置非零
        p_new = p_y·(1−λ_eff) + λ_eff·[tgt == y]
    """
    print(f"\n═══ 2) 插值 Δ（kNN-LM 凸组合，device={a.device}）═══")
    val_x = load_val(a.val, a.block, a.val_windows)
    g_tok, g_cnt = load_or_build_top1(a.cache_top1, a.corpus, L_d, M_d,
                                      a.chunk, a.sample_runs or None)

    from model.gpt import GPT, GPTConfig
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    ma = dict(ck["model_args"]); ma.pop("eos_token_id", None)
    model = GPT(GPTConfig.from_model_args(ma))
    sd = {(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in ck["model"].items()}
    model.load_state_dict(sd, strict=False)
    dev = torch.device(a.device)
    model = model.to(dev).eval()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"  基座 {a.ckpt}  参数 {n_par/1e6:.2f}M")
    print("  注意：它是在 v1 语料上训的，与 v2 不匹配 —— Δ 会是保守估计")

    x = val_x
    y = torch.zeros_like(x)
    y[:, :-1] = x[:, 1:]
    y[:, -1] = -100
    lam_list = [float(t) for t in a.lams.split(",")]

    acc = {(g, lam): [0.0, 0] for g in GATE_NAMES for lam in lam_list}
    acc["off"] = [0.0, 0]
    n_hit = n_tot = n_hit_correct = 0
    n_nonfinite = 0

    n_batch = (x.size(0) + a.bs - 1) // a.bs
    t0 = time.time()
    with torch.no_grad():
        for bi, i in enumerate(range(0, x.size(0), a.bs)):
            xb = x[i:i + a.bs].to(dev)
            yb = y[i:i + a.bs].to(dev)
            out = model(xb, yb)
            logits = out[0] if isinstance(out, tuple) else out
            lg = logits.float()
            if not torch.isfinite(lg).all():
                bad = int((~torch.isfinite(lg)).sum())
                n_nonfinite += bad
                print(f"    [!] batch {bi}: logits 有 {bad} 个非有限值（NaN/Inf）", flush=True)
            yflat = yb.reshape(-1)
            valid = yflat != -100
            # ★ 模型输出是 (B, T, V)，这里必须展平成 (N, V)：
            #   gather/max 都只接受二维，第一版漏了这步，跑几分钟后才炸。
            lp = F.log_softmax(lg.reshape(-1, lg.size(-1)), dim=-1)
            # 只在目标位置取 log 概率 → 不物化 (N, V) 的 nll
            p_y = lp.gather(1, yflat.clamp(min=0).unsqueeze(1)).squeeze(1).exp()
            nll_off = -torch.log(p_y.clamp_min(1e-9))
            acc["off"][0] += nll_off[valid].sum().item()
            acc["off"][1] += int(valid.sum())
            p_max = lp.max(dim=-1).values.exp()          # 模型自身最大概率（无标签）
            del lg, nll_off

            # 检索：用**同位置**的 L_d 后缀哈希查槽。
            # hashes(tok,L)[i] 的上下文结尾在下标 i+L-1 → 放到 slot[i+L-1]。
            flat = xb.reshape(-1).cpu().numpy()
            vh = hashes(flat, L_d)                       # (N - L,)
            slot = np.full(flat.size, -1, dtype=np.int64)
            slot[L_d - 1: L_d - 1 + len(vh)] = (vh % np.uint64(M_d)).astype(np.int64)
            # 查表留在 numpy 侧：GPU 上不再常驻 2×268MB 的槽表
            has = slot >= 0
            safe = np.where(has, slot, 0)
            tok_ng_np = np.where(has, g_tok[safe], -1)
            cnt_ng_np = np.where(has, g_cnt[safe], 0)
            tok_ng = torch.from_numpy(tok_ng_np.astype(np.int64)).to(dev)
            ok = torch.from_numpy(cnt_ng_np > 0).to(dev)
            hit_y = (tok_ng == yflat).float()
            n_hit += int(ok.sum().item()); n_tot += ok.numel()
            n_hit_correct += int((hit_y.bool() & ok).sum().item())
            del flat, vh, slot, has, safe, tok_ng_np, cnt_ng_np

            gates = gate_factors(lp, p_y, p_max, tok_ng, ok)
            for gname, gate in gates.items():
                for lam in lam_list:
                    nll = nll_under_retrieval(p_y, hit_y, gate, lam)
                    acc[(gname, lam)][0] += nll[valid].sum().item()
                    acc[(gname, lam)][1] += int(valid.sum())
                    del nll
            del tok_ng, ok, hit_y, p_y, p_max, gates, lp

            if bi % a.log_every == 0 or bi == n_batch - 1:
                msg = (f"    [eval] batch {bi+1}/{n_batch}  "
                       f"CE={acc['off'][0]/max(acc['off'][1],1):.4f}  {time.time()-t0:.0f}s")
                if dev.type == "cuda":
                    msg += f"  peak={torch.cuda.max_memory_allocated()/1e9:.2f}G"
                print(msg, flush=True)
            if dev.type == "cuda" and a.empty_cache_every and bi % a.empty_cache_every == 0:
                torch.cuda.empty_cache()

    base = acc["off"][0] / acc["off"][1]
    cov = n_hit / max(n_tot, 1)
    hit_correct = n_hit_correct / max(n_hit, 1)
    print(f"\n  槽命中率（槽非空）= {cov*100:.2f}%   "
          f"| 覆盖内 top-1 命中率 P(tgt==y) = {hit_correct*100:.2f}%")
    if n_nonfinite:
        print(f"  ⚠ logits 共有 {n_nonfinite} 个非有限值 —— 数值不稳，结果需谨慎")
    print(f"\n  {'口径':>12s} {'λ':>6s} {'CE':>10s} {'Δ vs off':>12s}   说明")
    NOTE = {"A_ungated": "= §12.3 原口径（无门控）",
            "B_oracle": "用标签，离线上界（推理不可实现）",
            "C_uncert": "可部署：模型不确定处才注入",
            "D_disagree": "可部署：模型与检索不一致才注入"}
    print(f"  {'off（无库）':>12s} {'—':>6s} {base:10.4f} {'—':>12s}   基线")
    best = {}
    for gname in GATE_NAMES:
        sub = []
        for lam in lam_list:
            ce = acc[(gname, lam)][0] / acc[(gname, lam)][1]
            sub.append((ce - base, lam, ce))
            print(f"  {gname:>12s} {lam:>6.2f} {ce:10.4f} {ce-base:+12.4f}   {NOTE[gname]}")
        best[gname] = min(sub)
    print("\n  各口径最优：")
    for gname, (dd, lam, ce) in best.items():
        print(f"    {gname:>12s}  λ*={lam:<5g} Δ={dd:+.4f}   {NOTE[gname]}")
    print(f"\n  本次 val：L={L_d} M={M_d/1e6:.0f}M  槽命中率 {cov*100:.2f}%  "
          f"覆盖内 top-1 命中 {hit_correct*100:.2f}%")
    print("  §12.3 参考（100M 顺序前缀，L=8）：覆盖 4.9%，命中 61.6%，最优 λ=0.2，Δ=−0.0161")
    return {"base": base, "slot_hit_rate": cov, "top1_hit_rate_in_covered": hit_correct,
            "nonfinite_logits": n_nonfinite,
            "by_scheme": {g: {"lam*": best[g][1], "delta": best[g][0]}
                          for g in GATE_NAMES},
            "curves": {f"{g}|{lam}": acc[(g, lam)][0] / acc[(g, lam)][1]
                       for g in GATE_NAMES for lam in lam_list}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/chinese/train_char_v2.bin")
    ap.add_argument("--val", default="data/chinese/val_char_v2.bin")
    ap.add_argument("--ckpt", default="out/ndb_run/last.pt")
    ap.add_argument("--orders", default="6,8,12")
    ap.add_argument("--slots", default="4194304,16777216,67108864,268435456")
    ap.add_argument("--chunk", type=int, default=20_000_000)
    ap.add_argument("--sample_runs", type=int, default=0,
                    help=">0 = 全流随机取这么多 token（§13.1 要求）；0 = 顺序全量")
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--val_windows", type=int, default=4096)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--device", default="cpu",
                    help="cpu（默认，零 GPU 风险）| cuda。2026-09-10 GPU Hang 事故后改默认 cpu")
    ap.add_argument("--lams", default="0.05,0.1,0.2,0.3,0.5")
    ap.add_argument("--delta_L", type=int, default=8, help="Δ 段用的后缀长度 L")
    ap.add_argument("--delta_M", type=int, default=67_108_864, help="Δ 段用的槽位数 M")
    ap.add_argument("--skip_delta", action="store_true", help="只跑覆盖率曲线")
    ap.add_argument("--only_cover", action="store_true", help="同 --skip_delta（语义更清楚）")
    ap.add_argument("--skip_cover", action="store_true", help="跳过覆盖率，直接跑插值 Δ")
    ap.add_argument("--cache_top1", default="",
                    help="top-1 表磁盘缓存路径；空 = 按 L/M 自动命名；'-' = 不缓存")
    ap.add_argument("--log_every", type=int, default=16, help="evaluate 进度打印间隔（batch）")
    ap.add_argument("--empty_cache_every", type=int, default=64, help="cuda 清缓存间隔（batch，0=关）")
    ap.add_argument("--out_json", default="out/ngram_full/result.json")
    a = ap.parse_args()

    # top-1 缓存按 (L, M) 自动命名：换配置不会读错表（读错表会给出看似合理的假结论）
    if a.cache_top1 == "-":
        a.cache_top1 = ""
    elif not a.cache_top1:
        a.cache_top1 = f"out/ngram_full/top1_L{a.delta_L}_M{a.delta_M // 1_000_000}M.npz"

    Ls = [int(x) for x in a.orders.split(",")]
    Ms = [int(x) for x in a.slots.split(",")]
    sruns = a.sample_runs or None
    corpus_n = os.path.getsize(a.corpus) // 2
    print(f"语料 {a.corpus}  {corpus_n/1e6:.1f}M token   "
          f"采样 {'顺序全量' if sruns is None else f'{sruns/1e6:.0f}M 随机 run'}")
    print(f"L={Ls}   槽位={[f'{m/1e6:.1f}M' for m in Ms]}   chunk={a.chunk/1e6:.0f}M\n", flush=True)

    result = {"corpus_tokens": corpus_n, "L": Ls, "M": Ms,
              "sample_runs": sruns, "val_windows": a.val_windows, "device": a.device}

    if not a.skip_cover:
        result["cover"] = run_cover(a, Ls, Ms)
    else:
        print("（--skip_cover：跳过第 1 段覆盖率）")

    if not (a.skip_delta or a.only_cover):
        result["delta"] = run_delta(a, a.delta_L, a.delta_M)

    if a.out_json:
        os.makedirs(os.path.dirname(a.out_json) or ".", exist_ok=True)
        with open(a.out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n结果已写入 {a.out_json}")


if __name__ == "__main__":
    main()
