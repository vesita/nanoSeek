#!/usr/bin/env python
"""审计：探针用的那批 val 窗口，有多少能在训练 bin 里**原样**找到（污染率）。

## 为什么需要它

`scripts/per_source_ce_probe.py` 是 warm start 之后**唯一可信的验收尺子**。
但它给的是 CE —— CE 变好有两种来源：「学到了」和「见过」。分段训练尤其危险：
`v3_dlg` 是从同一批原始语料重新清洗/切分出来的，它的 val 完全可能和 train 近重复。
**不量污染率就把 val 改善记成能力提升，等于把"背过"当成"学会"。**

## 方法（★ 内存与 bin 大小无关）

反过来做：**查询集合很小（几百个），训练集很大**。所以不建训练集的索引，
而是把「val 窗口 + 对照样本」的哈希装进一个小数组，然后**流式扫**训练 bin：
每个 chunk 里 `np.isin(chunk_hashes, query_hashes)`。
（正着建索引需要 937M × 8B ≈ 7.5GB，v2 那个 train bin 上会把机器打爆 —— 实测被 OOM 杀过两次。）

哈希是**定长** 64 bit 的多项式滚动哈希（见 `chunk_hashes`）。64 token × 13 bit 装不进
128 bit，所以判据是「同一哈希函数下的等值」而非逐 token 相等；
碰撞上界 ≈ 查询数 × 训练 n-gram 数 / 2^64（写进报告，别声称"逐 token 核对"）。

## 判据与对照（§5.4：新测量必须先过"已知答案"）

- 对照 1：从 train 里随机截 5 段 → **必须命中**（假阴性检查）。
- 对照 2：把那 5 段在段内**打乱 token** → **必须不命中**（假阳性检查）。
- 对照 3：拿 v2 自己的 train 查 v2 自己的 val（`--control-train-bin`）——
  v2 的 val 是它自己的 holdout，命中率应 ≈0；若明显 >0，先修方法再报数。

## 用法

    .venv/bin/python scripts/val_train_contamination_probe.py \
        --windows out/val_windows_v2.json \
        --train-bin data/chinese/train_char_v3_dlg.bin \
        --control-train-bin data/chinese/train_char_v2.bin \
        --out analysis/val_train_contamination_v2val.md

`--windows` 必须来自 `per_source_ce_probe.py --dump-windows`，否则不是同一批窗口。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

M64 = 1 << 64
BASE = 1000003                     # 奇数 ⇒ 在模 2^64 下可逆
BASE_INV = pow(BASE, -1, M64)


def chunk_hashes(tokens: np.ndarray, n: int, i0: int) -> np.ndarray:
    """返回 tokens 的每 n-gram 一个 uint64 哈希；`i0` 是这段在整条流里的起点。

    定义（模 2^64，故意回绕）：
        h(i) = BASE^(n-1) * sum_{k=0..n-1} t[i+k] * BASE_INV^(i+k)
    ⇒ 同一窗口无论出现在哪里都得到同一个哈希；用 `BASE_INV^i` 前缀和把它变成
    两次向量化前缀运算（cumprod / cumsum），避免上亿次 Python 循环。
    （已对照朴素实现逐位验证过，见 tests/test_val_train_contamination_probe.py。）
    """
    m = len(tokens)
    if m < n:
        return np.empty(0, dtype=np.uint64)
    lp = np.empty(m, dtype=np.uint64)
    lp[0] = 1
    if m > 1:
        lp[1:] = np.cumprod(np.full(m - 1, np.uint64(BASE_INV), dtype=np.uint64))
    powers = lp * np.uint64(pow(BASE_INV, i0, M64))
    u = tokens.astype(np.uint64) * powers
    s = np.cumsum(u, dtype=np.uint64)
    head = np.empty(m - n + 1, dtype=np.uint64)
    head[0] = 0
    head[1:] = s[: m - n]
    return np.uint64(pow(BASE, n - 1, M64)) * (s[n - 1:] - head)


def window_hash(tokens: np.ndarray, n: int, start: int) -> np.uint64:
    """val 侧窗口的哈希 —— 必须传**全流内偏移** start，才与 train 侧定义一致。"""
    return np.uint64(chunk_hashes(np.asarray(tokens[:n], dtype=np.uint16), n, start)[0])


def scan_hits(bin_path: str, n: int, q_hashes: np.ndarray,
              chunk: int = 16_000_000) -> np.ndarray:
    """流式扫训练 bin，返回 `q_hashes` 里每个哈希"是否在训练流中出现过"的布尔数组。"""
    data = np.memmap(bin_path, dtype=np.uint16, mode='r')
    total = len(data)
    hits = np.zeros(len(q_hashes), dtype=bool)
    order = np.argsort(q_hashes, kind='stable')
    qs = q_hashes[order]
    i = 0
    while i < total - n + 1:
        j = min(i + chunk, total)
        seg = np.asarray(data[i:j], dtype=np.uint16)
        h = chunk_hashes(seg, n, i)
        mask = np.isin(h, qs)
        if mask.any():
            pos = np.searchsorted(qs, h[mask])
            for p in np.unique(pos):
                hits[order[int(p)]] = True
        if hits.all():
            break
        i = j - n + 1
    del data
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', required=True, help='per_source_ce_probe --dump-windows 产出的 JSON')
    ap.add_argument('--train-bin', required=True)
    ap.add_argument('--control-train-bin', default=None,
                    help='对照训练集（v2 自己的 train），用来验证方法无假阳性')
    ap.add_argument('--ngram', type=int, default=32)
    ap.add_argument('--special-from', type=int, default=2075)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    spec = np.zeros(65536, dtype=bool)
    spec[args.special_from:] = True

    meta = json.load(open(args.windows))
    starts = meta['starts']
    data_val = np.memmap(meta['data'], dtype=np.uint16, mode='r')

    # ---- 组织查询：val 窗口 + 5 段 train 原样 + 5 段打乱（两者用同一次扫描，同一条代码路径）
    queries = []          # (kind, source, start, hash)
    skipped = {}
    for name, st in starts.items():
        n_ok = 0
        for s in st:
            seg = np.asarray(data_val[int(s):int(s) + args.ngram], dtype=np.int64)
            if len(seg) < args.ngram or spec[seg].any():
                continue
            queries.append(('val', name, int(s), int(window_hash(seg.astype(np.uint16),
                                                                args.ngram, int(s)))))
            n_ok += 1
        skipped[name] = len(st) - n_ok
    n_val = len(queries)

    rng = np.random.default_rng(1234)
    for bin_path, tag in [(args.train_bin, 'train')] + (
            [(args.control_train_bin, 'ctrl')] if args.control_train_bin else []):
        d = np.memmap(bin_path, dtype=np.uint16, mode='r')
        got = 0
        tries = 0
        while got < 5 and tries < 50000:
            tries += 1
            s = int(rng.integers(0, len(d) - args.ngram - 1))
            seg = np.asarray(d[s:s + args.ngram], dtype=np.int64)
            if spec[seg].any():
                continue
            seg = seg.astype(np.uint16)
            queries.append((f'{tag}_raw', bin_path, s, int(window_hash(seg, args.ngram, s))))
            sh = seg.copy()
            rng.shuffle(sh)
            queries.append((f'{tag}_shuffled', bin_path, s,
                            int(window_hash(sh, args.ngram, s + 7))))
            got += 1
        del d

    q = np.array([x[3] for x in queries], dtype=np.uint64)
    print(f"val={meta['data']} 窗口总数={sum(len(v) for v in starts.values())}，"
          f"有效(前 {args.ngram} token 全普通)={n_val}；"
          f"查询总数={len(q)}（含对照）", flush=True)

    print(f"\n=== 扫描 {args.train_bin} ===", flush=True)
    hits_main = scan_hits(args.train_bin, args.ngram, q)
    print("  完成", flush=True)

    hits_ctrl = None
    if args.control_train_bin:
        print(f"=== 扫描 {args.control_train_bin} ===", flush=True)
        hits_ctrl = scan_hits(args.control_train_bin, args.ngram, q)
        print("  完成", flush=True)

    def report(hits, bin_path):
        per_src = {}
        for (kind, src, s, _), hit in zip(queries, hits):
            if kind != 'val':
                continue
            h, n = per_src.get(src, (0, 0))
            per_src[src] = (h + int(hit), n + 1)
        lines = [f"## 对 `{bin_path}`：命中 **{sum(h for h, _ in per_src.values())}/"
                 f"{sum(n for _, n in per_src.values())} = "
                 f"{100.0 * sum(h for h, _ in per_src.values()) / max(sum(n for _, n in per_src.values()), 1):.2f}%**", ""]
        ctrl = {}
        for (kind, src, s, _), hit in zip(queries, hits):
            if kind.endswith('_raw') or kind.endswith('_shuffled'):
                d = ctrl.setdefault(kind, [0, 0])
                d[0] += int(hit)
                d[1] += 1
        if ctrl:
            parts = [f"{k}: {v[0]}/{v[1]}" for k, v in sorted(ctrl.items())]
            lines.append(f"**已知答案对照**（同一次扫描、同一条代码路径）：{'; '.join(parts)}"
                         f" —— `_raw` 应全中、`_shuffled` 应全不中")
        lines.append("")
        lines.append("| 源 | 命中 | 有效窗口 | 命中率 | 窗口总采样数 | 跳过 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for src, (h, n) in sorted(per_src.items(), key=lambda kv: -(kv[1][0] / max(kv[1][1], 1))):
            lines.append(f"| {src} | {h} | {n} | {100.0 * h / max(n, 1):.1f}% | "
                         f"{n + skipped.get(src, 0)} | {skipped.get(src, 0)} |")
        lines.append("")
        return lines

    header = [f"# val/train 污染率审计（{args.ngram}-gram 定长哈希等值匹配）", ""]
    header.append(f"- val 窗口：`{args.windows}`（= `per_source_ce_probe.py` **实际用的那一批**，"
                  f"block={meta['block_size']}，seed={meta['seed']}）")
    header.append(f"- val bin：`{meta['data']}`")
    header.append(f"- 判据：窗口前 **{args.ngram}** 个 token 全是普通字符（id < {args.special_from}）才可查；"
                  f"含 `<eos>`/`<cont>`/`<unk>` 等的窗口跳过并单独计数")
    header.append(f"- 碰撞上界 ≈ 查询数 × 训练 n-gram 数 / 2^64 ≈ {len(q)} × 1.1e8 / 1.8e19 "
                  f"≈ {len(q) * 1.1e8 / 1.8e19:.1e} ⇒ 可视为精确等值匹配")
    header.append("")
    body = report(hits_main, args.train_bin)
    if hits_ctrl is not None:
        body += report(hits_ctrl, args.control_train_bin)
    txt = '\n'.join(header + body)
    print('\n' + txt)
    if args.out:
        Path(args.out).write_text(txt)
        print(f"已写入 {args.out}")


if __name__ == '__main__':
    main()
