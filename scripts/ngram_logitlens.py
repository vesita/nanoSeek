#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""NDB 旁路可行性审计（零训练）：中间层"知不知道" n-gram 表的答案？

用户构型（画布 2026-09-19）的核心假设：把 NDB 从输出端混合器改成**循环内部的工作记忆**
（层2 查表 → 门控启发层3）。这只有在"中间层表征与 n-gram 信息有可注入的接口"时才值得做。

方法（三步，全部零训练）：
  1. 从 train_char_v3_lang.bin 随机抽 N 个 4-gram 位置，建哈希表 key(前4字)→next（argmax 频次）。
  2. 取 val 窗口，前向 intent2，用 forward hook 抓每层输出；logit-lens（ln_f+lm_head）
     把每层表征投到词表，取 top-1。
  3. 逐层报告：模型 top-1 与表 top-1 的**一致率**（限表命中的位置），以及与最终层 top-1 的一致率。

读法：
  - 一致率随深度单调涨到很高 ⇒ 中间层已经"知道"表答案 ⇒ 内嵌检索冗余，优先级降。
  - 中层一致率明显低于最终层 ⇒ 中间存在"表知道但模型还没接上"的断层 ⇒ 你的旁路有想象空间。
★ 已知答案对照（§5.4）：最终层与表一致率必须显著高于随机基线（1/8192≈0.01%），且
  hit 位置上最终层 top-1 与"真实下一字"的一致率应明显高于表 top-1 与真实的一致率
  （否则模型不如表，审计对象先换掉）。

用法：
  .venv/bin/python scripts/ngram_logitlens.py --ckpt out/base_v3_intent2/last.pt \
      --out analysis/ngram_logitlens_intent2.txt
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from ckpt_paired_eval import load_model  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, 'data', 'chinese')
K = 3                    # n-gram 前缀长度（K=4 在 8192 词表下命中率 <0.1%，实测不可用）
TABLE_N = 40_000_000     # 建表采样位置数
N_WIN = 64               # val 窗口数
BLOCK = 256
QPOS = 32                # 每窗口抽多少位置做 logit-lens


def rolling_keys(toks: np.ndarray, K: int = K) -> np.ndarray:
    """长度 K 的滚动哈希（uint64）：key(i) = hash(toks[i:i+K])。"""
    h = np.zeros(len(toks) - K + 1, dtype=np.uint64)
    for k in range(K):
        h = h * np.uint64(1000003) + np.uint64(1_000_003) * np.uint64(toks[k:len(toks) - K + 1 + k] % 8192)
        h = h ^ (h >> 29)
    return h


def build_table(train_bin: str, rng: np.random.Generator, K: int = K):
    data = np.fromfile(train_bin, dtype=np.uint16)
    n = len(data)
    idx = rng.integers(0, n - K - 1, size=TABLE_N)
    # 逐位置算 key 太慢 ⇒ 向量化：取 idx 排序后仍逐 idx 取 K 个 token（K=4，可接受）
    toks = data  # noqa
    keys = np.empty(len(idx), dtype=np.uint64)
    nxt = data[idx + K].astype(np.uint64)
    for j, i in enumerate(idx):
        keys[j] = rolling_keys(data[i:i + K], K)[0]
    combo = keys * np.uint64(8192) + nxt
    uniq, cnt = np.unique(combo, return_counts=True)
    ukeys = uniq // np.uint64(8192)
    unext = (uniq % np.uint64(8192)).astype(np.uint16)
    order = np.lexsort((-cnt, ukeys))
    ukeys, unext = ukeys[order], unext[order]
    keep = np.ones(len(ukeys), dtype=bool)
    keep[1:] = ukeys[1:] != ukeys[:-1]          # 每个 key 留计数最大的 next
    return ukeys[keep], unext[keep], cnt.sum() / max(len(idx), 1)


def query(ukeys, unext, keys):
    pos = np.searchsorted(ukeys, keys)
    pos = np.clip(pos, 0, len(ukeys) - 1)
    hit = ukeys[pos] == keys
    return hit, unext[pos]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ckpt', default='out/base_v3_intent2/last.pt')
    ap.add_argument('--out', default='analysis/ngram_logitlens_intent2.txt')
    ap.add_argument('--table-n', type=int, default=TABLE_N)
    ap.add_argument('--k', type=int, default=K)
    ap.add_argument('--windows', type=int, default=N_WIN)
    args = ap.parse_args()

    rng = np.random.default_rng(20260919)
    train_bin = os.path.join(DATA, 'train_char_v3_lang.bin')
    val_bin = os.path.join(DATA, 'val_char_v3_lang.bin')
    print(f'== 建表（{args.table_n:,} 位置，K={K}）==', flush=True)
    ukeys, unext, _ = build_table(train_bin, rng, args.k)
    print(f'  表条目 {len(ukeys):,}')

    val = np.fromfile(val_bin, dtype=np.uint16)
    widx = rng.integers(0, len(val) - BLOCK - 1, size=args.windows)
    model, _ = load_model(os.path.join(REPO, args.ckpt), 'cuda')
    n_layer = model.config.n_layer

    acts = {}
    hooks = []
    for i, blk in enumerate(model.transformer.h):
        def mk(i):
            def hook(mod, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                acts[i] = o.detach()
            return hook
        hooks.append(blk.register_forward_hook(mk(i)))

    agree_layer = np.zeros(n_layer + 1)      # 各层(含最终) vs 表
    agree_final = np.zeros(n_layer + 1)      # 各层 vs 最终层
    n_hit = 0
    n_real = 0
    final_beats = np.zeros(2)                # [表 vs 真实命中数, 最终层 vs 真实命中数]
    ln_f = model.transformer.ln_f
    lmh = model.lm_head
    with torch.no_grad():
        for w in widx:
            win = val[w:w + BLOCK].astype(np.int64)
            x = torch.from_numpy(win[:-1]).unsqueeze(0).cuda()
            y = win[1:]
            _ = model(x)
            # 每窗口抽 QPOS 个位置，且要求 4-gram 上下文完整（前 K 个 token 跳过）
            pos = np.arange(K, BLOCK - 1)
            sel = rng.choice(pos, size=min(QPOS, len(pos)), replace=False)
            ctx_keys = np.array([rolling_keys(win[p - args.k:p], args.k)[0] for p in sel])
            # 位置 p 的前缀是 win[p-args.k:p]（预测 win[p]）；模型预测位是 token p-1 → p
            hit, table_next = query(ukeys, unext, ctx_keys)
            if not hit.any():
                continue
            n_hit += int(hit.sum())
            hidx = np.where(hit)[0]
            sel_h = x[0, torch.tensor(sel[hidx]) - 1]  # 预测位置 = p-1（token p-1 → p）
            real_next = torch.tensor(y[sel[hidx] - 1])
            tops = {}
            for li in range(n_layer):
                logits = lmh(ln_f(acts[li][0, torch.tensor(sel) - 1]))
                tops[li] = logits.argmax(-1).cpu()
            logits_f = lmh(ln_f(acts[n_layer - 1][0, torch.tensor(sel) - 1]))
            tops[n_layer] = logits_f.argmax(-1).cpu()
            tnext = torch.tensor(table_next[hidx].astype(np.int64))
            for li in range(n_layer + 1):
                t1 = tops[li][hidx]
                agree_layer[li] += (t1 == tnext).sum().item()
                agree_final[li] += (tops[li][hidx] == tops[n_layer][hidx]).sum().item()
            final_beats[0] += (tnext == real_next).sum().item()
            final_beats[1] += (tops[n_layer][hidx] == real_next).sum().item()
            n_real += len(hidx)

    L = [f'== NDB 旁路可行性审计（logit-lens × 4-gram 表）==',
         f'ckpt: {args.ckpt}  窗口 {args.windows}  表命中位置合计 {n_hit}',
         f'已知答案对照：表 top-1 命中真实下一字 {final_beats[0]/max(n_real,1):.3%}；'
         f'最终层 {final_beats[1]/max(n_real,1):.3%}（最终层应显著更高）',
         '',
         '层 | 与表一致率 | 与最终层一致率']
    for li in range(n_layer + 1):
        tag = '（最终）' if li == n_layer else f'{li:2d}'
        L.append(f'{tag} | {agree_layer[li]/max(n_hit,1):.3%} | {agree_final[li]/max(n_hit,1):.3%}')
    txt = '\n'.join(L)
    print(txt)
    with open(os.path.join(REPO, args.out), 'w') as f:
        f.write(txt + '\n')


if __name__ == '__main__':
    main()
