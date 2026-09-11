#!/usr/bin/env python
"""量「有效 token 密度」：一个训练步读进去的 8192 个 token 里，有多少真的产生梯度？

背景：本仓库 char 级配置是 `stage=full` + `use_loss_masking=true` + `pack_nonempty=true`。
`build_assistant_mask` 是**行级**判据（一行里有 <eos>/<cont> 才整行算 loss），
所以 "窗口里的 token 数" 和 "产生梯度的 token 数" 是两个差很多的口径。
所有跟算力/数据量有关的结论（epoch 数、eval 噪声、有效样本量）都取决于这个比值，
所以它必须被实测，而不能从配置里推。

用法：
    .venv/bin/python scripts/mask_density_probe.py                 # train + val, 4000 窗口
    .venv/bin/python scripts/mask_density_probe.py --n 20000       # 更细
    .venv/bin/python scripts/mask_density_probe.py --split val --n 8000

自检（对照）：均匀抽窗口时「全空窗口占比」应当复现 data/chinese/DATASET_REPORT.md
§5 的 83.5%（train）/ 83.2%（val）。对不上说明本脚本的抽样或 masking 口径跑偏了，
先修脚本再看数字。
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.masking import build_assistant_mask  # noqa: E402


def load_vocab_ids(dataset='chinese'):
    from tokenizers import Tokenizer
    v = Tokenizer.from_file(os.path.join('data', dataset, 'char_tokenizer.json')).get_vocab()
    return [v['<eos>'], v['<cont>']]


def uniform_ix(data, n_windows, block_size, seed):
    g = np.random.default_rng(seed)
    hi = len(data) - block_size
    return g.integers(0, hi, n_windows)


def nonempty_ix(data, n_windows, block_size, terms, seed):
    """复刻 train.py 的 `_sample_nonempty_ix`（拒绝采样：y 窗口内必须出现终止符）。

    训练用的是 torch.randint；这里换成同分布的 numpy 采样。判据、窗口对齐
    （看 data[i+1 : i+1+block]）与批内配额逻辑保持一致。
    """
    g = np.random.default_rng(seed + 1)
    hi = len(data) - block_size
    out = []
    tries, limit = 0, 200 * n_windows
    while len(out) < n_windows and tries < limit:
        k = max((n_windows - len(out)) * 8, 32)
        for i in g.integers(0, hi, k).tolist():
            tries += 1
            w = data[i + 1: i + 1 + block_size]
            if any((w == t).any() for t in terms):
                out.append(i)
                if len(out) == n_windows:
                    break
    if len(out) < n_windows:
        print(f"  [warn] 拒绝采样只找到 {len(out)}/{n_windows} 个非空窗口，退回均匀采样补齐")
        out += g.integers(0, hi, n_windows - len(out)).tolist()
    return np.asarray(out, dtype=np.int64)


def effective_tokens(data, ix, block_size, terms):
    """返回每个窗口的有效 token 数（= build_assistant_mask 为 True 的个数）。"""
    out = []
    reply = [int(t) for t in terms]
    for i in ix:
        y = torch.from_numpy(data[i + 1: i + 1 + block_size].astype(np.int64)).unsqueeze(0)
        out.append(int(build_assistant_mask(y, reply, None).sum().item()))
    return np.asarray(out)


def summarize(tag, eff, block_size):
    n = len(eff)
    tot, mean = int(eff.sum()), float(eff.mean())
    print(f"  {tag}: 窗口 {n}  读入 {n*block_size:,} tok")
    print(f"    有效 token {tot:,}  占比 {100*tot/(n*block_size):.2f}%   "
          f"每窗口 均值 {mean:.1f} 中位 {np.median(eff):.0f} p10 {np.percentile(eff,10):.0f} "
          f"p90 {np.percentile(eff,90):.0f} max {eff.max()}")
    print(f"    全空窗口(0 有效 token) 占比 {100*(eff==0).mean():.2f}%")
    return tot, mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='chinese')
    ap.add_argument('--prefix', default='v2')
    ap.add_argument('--block-size', type=int, default=256)
    ap.add_argument('--n', type=int, default=4000, help='均匀抽样窗口数（非空窗口数同此）')
    ap.add_argument('--seed', type=int, default=20260911)
    ap.add_argument('--split', default='both', choices=['train', 'val', 'both'])
    args = ap.parse_args()

    terms = load_vocab_ids(args.dataset)
    print(f"终止符 id: <eos>={terms[0]} <cont>={terms[1]}   block_size={args.block_size}  "
          f"n={args.n}  seed={args.seed}")

    for split in (['train', 'val'] if args.split == 'both' else [args.split]):
        path = os.path.join('data', args.dataset, f'{split}_char_{args.prefix}.bin')
        data = np.memmap(path, dtype=np.uint16, mode='r')
        print(f"\n=== {split} ({path}) — {len(data):,} tok ===")

        ux = uniform_ix(data, args.n, args.block_size, args.seed)
        ueff = effective_tokens(data, ux, args.block_size, terms)
        _, u_mean = summarize('均匀采样（对照）', ueff, args.block_size)

        nx = nonempty_ix(data, args.n, args.block_size, terms, args.seed)
        neff = effective_tokens(data, nx, args.block_size, terms)
        ntot, n_mean = summarize('拒绝采样=实际训练所用', neff, args.block_size)

        print(f"    → 实际每步有效 token ≈ {n_mean:.0f}"
              f"（名义 {args.block_size*4}, 乘 grad_accum 8 后 {n_mean*8:.0f} / 名义 8192）")
        if u_mean > 0:
            print(f"    → 拒绝采样把每窗口有效 token 从 {u_mean:.1f} 提到 {n_mean:.1f}"
                  f"（{n_mean/max(u_mean,1e-9):.2f}×）")


if __name__ == '__main__':
    main()
