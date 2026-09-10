"""验证 train/val 的分布一致性（任务书硬性方法论，规则 1-4）。

三项中间量（规则 2）：
  - 有效 token 占比：build_assistant_mask 命中的比例（True=算 loss）
  - 全空窗口占比：整窗 mask 全 False 的比例
  - 终止符密度：<eos>(128)+<cont>(130) 占比
一项独立判据（规则 3）：
  - bigram 交叉验证：在 train 上建 bigram 计数表（add-k 平滑），
    用同一张表给 train 后半段与 val 打分，比较平均负对数似然。

规则 1：抽样一律"全库随机"（rng.integers 全域取索引），绝不取文件前缀。
规则 4：本脚本内置已知答案自检（--selftest），对照不过就不采信实测。

用法：
    .venv/bin/python data/chinese/verify_dataset.py --selftest
    .venv/bin/python data/chinese/verify_dataset.py \
        --train data/chinese/train_char.bin --val data/chinese/val_char.bin \
        --label old --out data/chinese/verify_old.json
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, ROOT)

from training.masking import build_assistant_mask  # noqa: E402

EOS_ID, CONT_ID, NL_ID = 128, 130, 0
VOCOB = 8192
ADD_K = 0.1


def load_random_windows(path, n, block, rng, lo_frac=0.0, hi_frac=1.0):
    """全库随机（或区间内随机）取 n 个长度 block 的窗口，返回 (n, block) int64。"""
    d = np.memmap(path, dtype=np.uint16, mode='r')
    lo = int(len(d) * lo_frac)
    hi = int(len(d) * hi_frac) - block - 2
    if hi <= lo:
        raise ValueError(f'区间为空: lo={lo} hi={hi} len={len(d)}')
    ix = rng.integers(lo, hi, n)
    arr = np.stack([np.asarray(d[i:i + block], dtype=np.int64) for i in ix])
    return arr


def mask_stats(arr):
    """有效 token 占比 / 全空窗口占比 / 终止符密度。"""
    import torch
    y = torch.from_numpy(arr)
    m = build_assistant_mask(y, [EOS_ID, CONT_ID], None)
    n = y.numel()
    term = int(((y == EOS_ID) | (y == CONT_ID)).sum().item())
    eos = int((y == EOS_ID).sum().item())
    cont = int((y == CONT_ID).sum().item())
    return {
        'n_windows': int(arr.shape[0]),
        'n_tokens': int(n),
        'valid_token_ratio': float(m.sum().item()) / n,
        'empty_window_ratio': float((m.sum(dim=1) == 0).float().mean().item()),
        'terminator_density': term / n,
        'eos_density': eos / n,
        'cont_density': cont / n,
    }


def build_bigram_counts(path, vocab=VOCOB, chunk=10_000_000):
    """在整份 train 上建 bigram 计数表（float64 (V,V)）与 unigram 计数。"""
    d = np.memmap(path, dtype=np.uint16, mode='r')
    counts = np.zeros(vocab * vocab, dtype=np.float64)
    total = len(d)
    for s in range(0, total - 1, chunk):
        e = min(s + chunk, total - 1)
        a = np.asarray(d[s:e], dtype=np.int64)
        b = np.asarray(d[s + 1:e + 1], dtype=np.int64)
        counts += np.bincount(a * vocab + b, minlength=vocab * vocab)
    counts = counts.reshape(vocab, vocab)
    row = counts.sum(axis=1)
    return counts, row


def bigram_ce(counts, row, arr, vocab=VOCOB, k=ADD_K):
    """用同一张 add-k 平滑的 bigram 表给窗口打分，返回平均 NLL (nats)。"""
    a = arr[:, :-1].reshape(-1)
    b = arr[:, 1:].reshape(-1)
    c = counts[a, b]
    denom = row[a] + k * vocab
    nll = -np.log((c + k) / denom)
    return float(nll.mean()), int(nll.size)


def selftest():
    """规则 4：测量函数先在已知答案的输入上跑一遍。"""
    ok = True

    # --- mask_stats 已知答案 ---
    y = np.array([[1, 2, EOS_ID, NL_ID, 3, 4, 5, NL_ID],
                  [NL_ID] * 8], dtype=np.int64)
    st = mask_stats(y)
    exp_valid = 3 / 16
    exp_empty = 0.5
    exp_term = 1 / 16
    checks = [('valid', st['valid_token_ratio'], exp_valid),
              ('empty', st['empty_window_ratio'], exp_empty),
              ('term', st['terminator_density'], exp_term)]
    for name, got, exp in checks:
        good = abs(got - exp) < 1e-12
        ok &= good
        print(f'[selftest] mask.{name}: got={got:.6f} exp={exp:.6f} {"OK" if good else "FAIL"}')

    # --- 同一份数据自己跟自己比 → 差异必须为 0（规则 4 原话） ---
    rng = np.random.default_rng(7)
    toy = np.tile(np.array([5, 6, 7, EOS_ID, NL_ID, 8, 9, CONT_ID, NL_ID], dtype=np.int64), 200)
    a1 = np.stack([toy[i:i + 16] for i in rng.integers(0, 200, 50)])
    s1 = mask_stats(a1)
    s2 = mask_stats(a1.copy())
    d = max(abs(s1[k2] - s2[k2]) for k2 in s1 if isinstance(s1[k2], float))
    ok &= (d == 0.0)
    print(f'[selftest] 同数据自比 mask 统计最大差 = {d} (应为 0)')

    # --- bigram 已知答案：交替序列 0,1,0,1... 的 CE 应远低于乱序序列 ---
    seq = np.tile(np.array([0, 1], dtype=np.int64), 5000)
    arr_alt = seq.reshape(1, -1)
    counts, row = None, None
    c = np.zeros((VOCOB, VOCOB), dtype=np.float64)
    a = seq[:-1]
    b = seq[1:]
    c += np.bincount(a * VOCOB + b, minlength=VOCOB * VOCOB).reshape(VOCOB, VOCOB)
    r = c.sum(axis=1)
    ce_alt, _ = bigram_ce(c, r, arr_alt)
    shuffled = seq.copy()
    np.random.default_rng(1).shuffle(shuffled)
    ce_rand, _ = bigram_ce(c, r, shuffled.reshape(1, -1))
    good = ce_rand > ce_alt + 5.0
    ok &= good
    print(f'[selftest] bigram CE 规则交替={ce_alt:.4f} vs 乱序={ce_rand:.4f} '
          f'{"OK" if good else "FAIL"}')

    # --- add-k 与 log 数值稳定性 ---
    ce_zero, n = bigram_ce(np.zeros((VOCOB, VOCOB)), np.zeros(VOCOB), np.zeros((1, 10), dtype=np.int64))
    exp0 = np.log(VOCOB)
    good = abs(ce_zero - exp0) < 1e-9
    ok &= good
    print(f'[selftest] 空计数表 CE={ce_zero:.6f} 期望=log(8192)={exp0:.6f} {"OK" if good else "FAIL"}')

    print('[selftest]', 'ALL OK' if ok else 'FAILED')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default=None)
    ap.add_argument('--val', default=None)
    ap.add_argument('--label', default='')
    ap.add_argument('--out', default=None)
    ap.add_argument('--n-windows', type=int, default=3000)
    ap.add_argument('--block', type=int, default=256)
    ap.add_argument('--seed', type=int, default=20260910)
    ap.add_argument('--train-lo-frac', type=float, default=0.5,
                    help='bigram 打分用的 train 区间下界（默认后半段=0.5）')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        return 0 if selftest() else 1
    if not a.train or not a.val:
        raise SystemExit('需要 --train 与 --val')

    res = {'label': a.label, 'seed': a.seed, 'n_windows': a.n_windows, 'block': a.block,
           'train': a.train, 'val': a.val}
    rng = np.random.default_rng(a.seed)
    tr_w = load_random_windows(a.train, a.n_windows, a.block, rng, 0.0, 1.0)
    va_w = load_random_windows(a.val, a.n_windows, a.block, rng, 0.0, 1.0)
    res['train_stats'] = mask_stats(tr_w)
    res['val_stats'] = mask_stats(va_w)

    print(f'== {a.label} ==  (seed={a.seed}, n={a.n_windows}x{a.block})')
    for key in ('valid_token_ratio', 'empty_window_ratio', 'terminator_density',
                'eos_density', 'cont_density'):
        t = res['train_stats'][key]
        v = res['val_stats'][key]
        ratio = (v / t) if t else float('inf')
        print(f'  {key:22s} train={t:.4%}  val={v:.4%}  val/train={ratio:.3f}')

    print('  建 bigram 计数表（全 train）...')
    counts, row = build_bigram_counts(a.train)
    rng2 = np.random.default_rng(a.seed + 1)
    tr_half = load_random_windows(a.train, a.n_windows, a.block, rng2,
                                  a.train_lo_frac, 1.0)
    va_s = load_random_windows(a.val, a.n_windows, a.block, rng2, 0.0, 1.0)
    ce_tr, n_tr = bigram_ce(counts, row, tr_half)
    ce_va, n_va = bigram_ce(counts, row, va_s)
    # 同区间自比（已知答案=0）：同一 rng 状态重取 train 后半段
    rng3 = np.random.default_rng(a.seed + 2)
    h1 = load_random_windows(a.train, a.n_windows, a.block, rng3, a.train_lo_frac, 1.0)
    rng4 = np.random.default_rng(a.seed + 2)
    h2 = load_random_windows(a.train, a.n_windows, a.block, rng4, a.train_lo_frac, 1.0)
    ce_same1, _ = bigram_ce(counts, row, h1)
    ce_same2, _ = bigram_ce(counts, row, h2)
    res['bigram'] = {
        'train_ce': ce_tr, 'val_ce': ce_va, 'gap_val_minus_train': ce_va - ce_tr,
        'n_pairs_train': n_tr, 'n_pairs_val': n_va,
        'train_half_selfcheck_ce_1': ce_same1, 'train_half_selfcheck_ce_2': ce_same2,
        'selfcheck_abs_diff': abs(ce_same1 - ce_same2),
    }
    print(f'  bigram CE(train后半段)={ce_tr:.4f}  CE(val)={ce_va:.4f}  '
          f'gap={ce_va - ce_tr:+.4f} nats')
    print(f'  [对照] train后半段 两次同种子重取 CE 差={abs(ce_same1 - ce_same2):.2e} (应为 0)')
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f'  已写 {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
