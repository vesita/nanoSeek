#!/usr/bin/env python
"""在同一批 val 窗口上配对重评多个归档 checkpoint —— 把 eval 噪声和真实进度分开。

为什么需要它：`results.csv` 的 val 相邻差 sd ≈ 0.19，而真实进展只有 ~0.03/千步，
单点 eval 完全读不出趋势。配对（同一批窗口）能把「这批窗口好不好」这一项消掉。

顺带产出两个诊断量：
  * 200 批分块的块间 sd —— 直接量出训练用的 eval 口径的测量噪声
    （若它 ≈ results.csv 的相邻差 sd，就证明那 0.19 确实是测量噪声，不是模型抖动）
  * 每 token CE 的分位数 —— 判断这个指标是不是被少数灾难性 token 主导

用法：
    .venv/bin/python scripts/ckpt_paired_eval.py \
        --ckpts out/base_v2/ckpt_step_5000.pt out/base_v2/ckpt_step_10000.pt ... \
        --batches 1000

自检（对照）：本脚本对单个 checkpoint 重复跑（`--repeat 3`）拿到的重复间 sd，
应当与 `--batches 200` 分块 sd 同量级；两者都对不上训练日志，说明口径抄错了。
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GPT, GPTConfig  # noqa: E402
from training.masking import build_assistant_mask  # noqa: E402


def val_indices(data, n_windows, block_size, terms, seed):
    """复刻 train.py 的 `_sample_nonempty_ix`（y 窗口必须含 <eos>/<cont>）。

    与 train.py 的差别：用 numpy 固定种子采样，保证所有 checkpoint 看到**同一批**窗口。
    """
    g = np.random.default_rng(seed)
    hi = len(data) - block_size
    out = []
    tries, limit = 0, 200 * n_windows
    while len(out) < n_windows and tries < limit:
        k = max((n_windows - len(out)) * 8, 32)
        for i in g.integers(0, hi, k).tolist():
            tries += 1
            if any((data[i + 1: i + 1 + block_size] == t).any() for t in terms):
                out.append(i)
                if len(out) == n_windows:
                    break
    assert len(out) == n_windows, f"只找到 {len(out)}/{n_windows} 个非空窗口"
    return np.asarray(out, dtype=np.int64)


def build_batches(data, ix, block_size, batch_size):
    """把索引切成 (B, T) 张量；末尾不足一批的丢掉（配对时所有 ckpt 必须同批数）。"""
    n = (len(ix) // batch_size) * batch_size
    ix = ix[:n]
    xs, ys = [], []
    for b in range(0, n, batch_size):
        chunk = ix[b:b + batch_size]
        xs.append(np.stack([data[i:i + block_size] for i in chunk]).astype(np.int64))
        ys.append(np.stack([data[i + 1:i + 1 + block_size] for i in chunk]).astype(np.int64))
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    args = dict(ck['model_args'])
    fields = set(GPTConfig.__dataclass_fields__) if hasattr(GPTConfig, '__dataclass_fields__') else None
    if fields:
        dropped = sorted(set(args) - fields)
        if dropped:
            print(f"  [note] 丢弃 GPTConfig 不认的键: {dropped}")
        args = {k: v for k, v in args.items() if k in fields}
    model = GPT(GPTConfig(**args))
    sd = ck['model']
    if any(k.startswith('_orig_mod.') for k in sd):
        sd = {k.replace('_orig_mod.', '', 1): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(device).eval()
    return model, ck


@torch.no_grad()
def eval_paired(model, X, Y, terms, device, ctx, chunk=8):
    """返回 (每个窗口的 token 级平均 CE, 每 token CE 的采样)。

    ★ chunk 必须小：logits 是 (chunk*B, T, vocab)，fp32 upcast 那一份是
      chunk*4*256*8192*4 字节。chunk=50 时 = 1.7 GB，加上 bf16 原件会把 8G 卡吃满
      （实测 chunk=50 峰值 7.88 G，而训练峰值只有 1.70 G —— 探针自己制造了训练里
      不存在的内存画像，这是不允许的）。chunk=8 时峰值 ~270 MB。
    """
    per_window, per_token = [], []
    T = X.shape[-1]
    for s in range(0, X.shape[0], chunk):
        x = X[s:s + chunk].reshape(-1, T).to(device, non_blocking=True)
        y = Y[s:s + chunk].reshape(-1, T).to(device, non_blocking=True)
        m = build_assistant_mask(y, terms, None)
        with ctx:
            logits, _ = model(x, y)
        V = logits.size(-1)
        ce = F.cross_entropy(logits.view(-1, V), y.view(-1), reduction='none').view(y.shape)
        # ★ 向量化：逐窗口 .item()/.cpu() 会产生 2×窗口数 次设备同步，实测把单个
        # checkpoint 拖到 10 分钟以上。改成每 chunk 两次同步。
        n_i = m.sum(dim=1).clamp(min=1)
        per_window.append(((ce * m).sum(dim=1) / n_i).float().cpu())
        per_token.append(ce[m].float().cpu())
    return torch.cat(per_window).numpy(), torch.cat(per_token).numpy()


def fmt(x):
    return f"{x:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs='+', required=True)
    ap.add_argument('--data', default='data/chinese/val_char_v2.bin')
    ap.add_argument('--tokenizer', default='data/chinese/char_tokenizer.json')
    ap.add_argument('--block-size', type=int, default=256)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--batches', type=int, default=1000, help='前向批数（每批 batch_size 个窗口）')
    ap.add_argument('--seed', type=int, default=20260911)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--repeat', type=int, default=1,
                    help='对每个 ckpt 用不同 seed 重复评估几次（量重复间 sd）')
    args = ap.parse_args()

    from tokenizers import Tokenizer
    v = Tokenizer.from_file(args.tokenizer).get_vocab()
    terms = [v['<eos>'], v['<cont>']]
    data = np.memmap(args.data, dtype=np.uint16, mode='r')
    n_win = args.batches * args.batch_size
    print(f"val={args.data} ({len(data):,} tok)  窗口={n_win}  block={args.block_size}  "
          f"device={args.device}  终止符={terms}")

    dev = torch.device(args.device)
    ctx = (torch.autocast(device_type='cuda', dtype=torch.bfloat16)
           if dev.type == 'cuda' else torch.autocast('cpu', enabled=False))

    results = {}
    for ci, path in enumerate(args.ckpts):
        model, ck = load_model(path, dev)
        step = ck.get('iter_num', -1)
        rows = []
        for r in range(args.repeat):
            ix = val_indices(data, n_win, args.block_size, terms, args.seed + 7777 * r)
            X, Y = build_batches(data, ix, args.block_size, args.batch_size)
            pw, pt = eval_paired(model, X, Y, terms, dev, ctx)
            rows.append((pw, pt))
            del X, Y
        pw, pt = rows[0]
        # 训练侧 estimate_loss 是 **token 加权**（tot += loss.item()*n_i; n += n_i），
        # 而窗口数在 6~199 之间波动（p10=6, p90=199），所以「窗口均值」和「token 加权均值」
        # 是两个不同的口径。要跟训练日志比，必须用 token 加权。
        tok_mean = float(pt.mean())
        # 分块 sd = 训练那个 eval 口径的测量噪声。★ 块大小必须等于训练一次 eval 的
        # **窗口数**：eval_iters=200 × batch_size=4 = 800 个窗口（不是 200）。
        chunk_n = 800
        if len(pw) >= 2 * chunk_n:
            cmeans = [pw[i:i + chunk_n].mean() for i in range(0, len(pw) - chunk_n + 1, chunk_n)]
            block_sd = float(np.std(cmeans, ddof=1))
        else:
            block_sd = float('nan')
        rep_sd = float(np.std([r[0].mean() for r in rows], ddof=1)) if len(rows) > 1 else float('nan')
        q = np.percentile(pt, [50, 90, 99, 99.9, 99.99])
        # 尾部贡献：最高的 0.1% token 占均值多少
        srt = np.sort(pt)[::-1]
        top01 = srt[:max(1, len(srt) // 1000)].sum() / srt.sum()
        results[step] = dict(path=path, mean=tok_mean, win_mean=float(pw.mean()),
                             se=float(pw.std(ddof=1) / len(pw) ** .5),
                             ntok=len(pt), block_sd=block_sd, rep_sd=rep_sd, q=q, top01=top01)
        print(f"  [{os.path.basename(path)}] step {step}: "
              f"val(tok加权) {tok_mean:.4f}  val(窗口均) {pw.mean():.4f} "
              f"(SE {pw.std(ddof=1)/len(pw)**.5:.4f}, 有效 tok {len(pt):,})  "
              f"200批分块sd {block_sd:.4f}  尾部top0.1%占比 {top01:.1%}")
        del model
        if dev.type == 'cuda':
            torch.cuda.empty_cache()

    print("\n=== 全量 token 分布（第一个 ckpt）===")
    q = next(iter(results.values()))['q']
    print(f"  per-token CE 分位: p50 {q[0]:.3f}  p90 {q[1]:.3f}  p99 {q[2]:.3f}  "
          f"p99.9 {q[3]:.3f}  p99.99 {q[4]:.3f}")

    print("\n=== 配对重评结果（同一批窗口）===")
    print(f"{'step':>7} {'val(配对)':>10} {'SE':>7} {'有效tok':>10} {'200批块间sd':>12}")
    for step in sorted(results):
        r = results[step]
        print(f"{step:>7} {fmt(r['mean']):>10} {fmt(r['se']):>7} {r['ntok']:>10,} "
              f"{fmt(r['block_sd']):>12}")

    print("\n=== 相邻配对差（噪声抵消）===")
    steps = sorted(results)
    for a, b in zip(steps, steps[1:]):
        d = results[b]['mean'] - results[a]['mean']
        # 配对差的标准误：同一批窗口，用窗口级差算
        print(f"  {a:>6} → {b:<6}  Δ = {d:+.4f}")

    print("\n=== 与训练日志对比 ===")
    print("  日志 val: 19000 1.9203 · 20000 1.6432 · 21000 1.7924 · 22000 1.9107")
    print("  （单点口径；本表用同一批窗口 + 更多批数，可直接看趋势）")
    if not np.isnan(next(iter(results.values()))['rep_sd']):
        print(f"  重复评估间 sd = {next(iter(results.values()))['rep_sd']:.4f}")


if __name__ == '__main__':
    main()
