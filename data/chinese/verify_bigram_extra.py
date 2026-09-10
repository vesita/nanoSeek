"""bigram 交叉验证的稳健性补充（规则 3/4）：

  - k 敏感性：add-k 取 0.01 / 0.1 / 1.0
  - 来源顺序噪声底：CE(train前半段) vs CE(train后半段) —— 语料按来源拼接，
    "train后半段"本身不代表全库，这是该方法固有的偏差来源。
  - CE(train全域随机) vs CE(val)：不受顺序偏差影响的对照。

用法：
    .venv/bin/python data/chinese/verify_bigram_extra.py --train ... --val ... --label old
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_dataset import (VOCOB, build_bigram_counts, bigram_ce,  # noqa: E402
                            load_random_windows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--val', required=True)
    ap.add_argument('--label', default='')
    ap.add_argument('--n-windows', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=20260910)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    print(f'== bigram extra: {a.label} ==')
    counts, row = build_bigram_counts(a.train)
    rng = np.random.default_rng(a.seed + 11)
    first = load_random_windows(a.train, a.n_windows, 256, rng, 0.0, 0.5)
    second = load_random_windows(a.train, a.n_windows, 256, rng, 0.5, 1.0)
    whole = load_random_windows(a.train, a.n_windows, 256, rng, 0.0, 1.0)
    val = load_random_windows(a.val, a.n_windows, 256, rng, 0.0, 1.0)

    res = {}
    for k in (0.01, 0.1, 1.0):
        ce_f, _ = bigram_ce(counts, row, first, k=k)
        ce_s, _ = bigram_ce(counts, row, second, k=k)
        ce_w, _ = bigram_ce(counts, row, whole, k=k)
        ce_v, _ = bigram_ce(counts, row, val, k=k)
        res[f'k={k}'] = {'train_first': ce_f, 'train_second': ce_s,
                         'train_whole': ce_w, 'val': ce_v,
                         'gap_second_minus_first': ce_s - ce_f,
                         'gap_val_minus_second': ce_v - ce_s,
                         'gap_val_minus_whole': ce_v - ce_w}
        print(f'  k={k:<5} train前半={ce_f:.4f} train后半={ce_s:.4f} '
              f'train全域={ce_w:.4f} val={ce_v:.4f}')
        print(f'          [顺序噪声底] 后半−前半={ce_s-ce_f:+.4f} nats   '
              f'[主判据] val−后半={ce_v-ce_s:+.4f} nats   '
              f'[顺序无关] val−全域={ce_v-ce_w:+.4f} nats')

    # 规则 4 对照：同一份窗口自比
    r1 = np.random.default_rng(a.seed + 12)
    r2 = np.random.default_rng(a.seed + 12)
    w1 = load_random_windows(a.val, a.n_windows, 256, r1, 0.0, 1.0)
    w2 = load_random_windows(a.val, a.n_windows, 256, r2, 0.0, 1.0)
    c1, _ = bigram_ce(counts, row, w1)
    c2, _ = bigram_ce(counts, row, w2)
    res['selfcheck_abs_diff'] = abs(c1 - c2)
    print(f'  [对照] 同种子重取 val 两次 CE 差={abs(c1-c2):.2e} (应为 0)')
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f'  已写 {a.out}')


if __name__ == '__main__':
    main()
