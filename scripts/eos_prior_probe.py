#!/usr/bin/env python
"""量一个**具体的先验**：模型在 `B：<think>\\n` 之后，有多想立刻吐 `<eos>`。

## 为什么要量这个

v3 之前的数据里，`prepare.py::annotate_replies` 把 `<eos>` 插在**回复首行之后**。
deepseek 蒸馏源占 27.3% 的字符、且 **100% 回复首行就是 `<think>`**，于是这些样本
在语料里长成：

    ... A：<问题> \\n B：<think> \\n <eos> ...真正的推理... 

⇒ 在教"**`<think>` 一开就收尾**"。v3 已经把新 bin 修好了
（终端符贴到整条回复末尾，验收 `rel<0.5 = 0.00%`），
但**`out/base_v2` 的权重是在旧 bin 上训了 61000 步的** —— 它带着这个先验。

A 段（`v3_know`，3k 步）的一半理由就是"冲刷这个先验"。这个脚本量的是
**冲刷之前它有多重**，所以：

* 跑 A 段**之前**跑一次 → `before`
* 跑 A 段**之后**再跑一次 → `after`，两者用同一 seed ⇒ 同一批位置，可直接配对比较

## 三组位置（自带已知答案对照）

| 组 | 位置 | 期望 |
|---|---|---|
| `after_think` | `<think>` `\\n` **紧后面** | ★ 被测的那个先验。**越高越病** |
| `reply_end` | 真实 `<eos>`/`<cont>` 所在位置 | **已知答案**：P(<eos>) 必须显著更高 |
| `random_mid` | 块内随机位置 | 应当低 |

`reply_end` 组是这把尺子的**已知答案对照**（AGENTS §5.4）：如果它不比 `random_mid`
显著高，说明探针本身是坏的，`after_think` 的数也不能信 —— 脚本会直接报出来。

## 用法

    HSA_OVERRIDE_GFX_VERSION=10.3.0 HSA_ENABLE_SDMA=0 \\
    .venv/bin/python scripts/eos_prior_probe.py \\
        --ckpt out/base_v2/last.pt --bin data/chinese/train_char_v3_know.bin \\
        --off data/chinese/train_char_v3_know.off --n 200 --tag before_A

输出：每个组的 P(<eos>) 均值/中位数、`<eos>` 的排名，以及一句判定。
`--dump-json <路径>` 可把逐位置明细落盘，便于事后配对比较。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

EOS, CONT, THINK, NL = 128, 130, 136, 0
BLOCK = 256          # 与训练的 block_size 一致（探针必须与真实负载同构，AGENTS §5.3）


def pick_positions(bin_path, off_path, n_per_group, seed=1337, block=BLOCK):
    """从 train bin + .off 里挑三组位置（全局下标 = 要预测的那个 token 的位置）。"""
    data = np.memmap(bin_path, dtype=np.uint16, mode='r')
    n = len(data)
    off = np.fromfile(off_path, dtype=np.int64)

    rng = np.random.default_rng(seed)
    after_think, reply_eos, reply_cont = [], [], []

    # 分块扫描，避免一次性对 4.9 亿 token 建布尔数组
    CH = 20_000_000
    for s in range(0, n, CH):
        e = min(s + CH + 1, n)
        a = np.asarray(data[s:e], dtype=np.int64)
        i = np.where((a[:-1] == THINK) & (a[1:] == NL))[0] + s + 2   # \n 之后那个位置
        after_think.extend(i.tolist())
        reply_eos.extend((np.where(a == EOS)[0] + s).tolist())
        reply_cont.extend((np.where(a == CONT)[0] + s).tolist())
        if len(after_think) > n_per_group * 40 and len(reply_eos) > n_per_group * 40:
            break

    def take(pool, label):
        pool = np.asarray([p for p in pool if block + 1 < p < n - 1], dtype=np.int64)
        if len(pool) == 0:
            raise SystemExit(f'错误：{label} 组一个位置都没找到（检查 bin/off 是否配错）')
        k = min(n_per_group, len(pool))
        return np.sort(rng.choice(pool, size=k, replace=False))

    # 随机组：块内随机位置，且避开特殊 token 附近，保证它真的是"回复中段"
    starts = off[:-1]
    lens = np.diff(off)
    big = starts[lens > block + 2]
    rand = np.asarray([int(s) + int(rng.integers(block + 1, max(block + 2, int(l) - 1)))
                       for s, l in zip(big[:n_per_group * 60], lens[lens > block + 2][:n_per_group * 60])],
                      dtype=np.int64)
    rand = rand[(rand > block) & (rand < n - 1)]
    spec = np.isin(np.asarray(data[rand], dtype=np.int64), [EOS, CONT, THINK, NL])
    rand = rand[~spec]
    rand = np.sort(rng.choice(rand, size=min(n_per_group, len(rand)), replace=False))

    return {'after_think': take(after_think, 'after_think'),
            'reply_eos': take(reply_eos, 'reply_eos'),
            'reply_cont': take(reply_cont, 'reply_cont'),
            'random_mid': rand}, data


SPECIALS = {EOS: '<eos>', CONT: '<cont>', THINK: '<think>', 129: '<unk>',
            131: '<pad>', 132: '<bos>', 133: '<sep>'}


def decode_ids(ids, tok_path):
    """把 id 序列解成可读文本（特殊 token 用名字表示），用于人工抽检位置是否选对。"""
    from tokenizers import Tokenizer
    inv = {i: s for s, i in Tokenizer.from_file(tok_path).get_vocab().items()}
    return ''.join(SPECIALS.get(i, inv.get(i, f'<{i}>')) for i in ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--bin', required=True)
    ap.add_argument('--off', required=True)
    ap.add_argument('--n', type=int, default=200, help='每组位置数')
    ap.add_argument('--batch', type=int, default=16, help='一次前向多少个窗口（探针同构性，勿乱调大）')
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--tag', default='')
    ap.add_argument('--dump-json', default='')
    ap.add_argument('--show', type=int, default=0,
                    help='每组抽检几个位置并打印其前文（AGENTS §5.9：判据改动前先读原文）')
    ap.add_argument('--tok', default='data/chinese/char_tokenizer.json')
    args = ap.parse_args()

    device = 'cuda'
    torch.manual_seed(args.seed)

    from model import GPT, GPTConfig
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = GPT(GPTConfig.from_model_args(ckpt['model_args']))
    sd = {k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k: v for k, v in ckpt['model'].items()}
    model.load_state_dict(sd)
    model.eval().to(device)
    print(f'模型：{args.ckpt}（step={ckpt.get("iter_num", "?")}）')

    groups, data = pick_positions(args.bin, args.off, args.n, args.seed, BLOCK)
    for k, v in groups.items():
        print(f'  位置组 {k:12s} n={len(v)}')

    out = {}
    with torch.no_grad():
        for name, pos in groups.items():
            pe, rk = [], []
            for i in range(0, len(pos), args.batch):
                chunk = pos[i:i + args.batch]
                x = np.stack([np.asarray(data[p - BLOCK:p], dtype=np.int64) for p in chunk])
                x = torch.from_numpy(x).to(device)
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits, _ = model(x)
                lp = torch.log_softmax(logits[:, -1, :].float(), dim=-1)
                pe.extend(lp[:, EOS].exp().cpu().tolist())
                rk.extend((lp > lp[:, EOS:EOS + 1]).sum(dim=-1).cpu().tolist())
            pe = np.asarray(pe); rk = np.asarray(rk)
            out[name] = {'p_eos': pe.tolist(), 'rank_eos': rk.tolist(),
                         'mean': float(pe.mean()), 'median': float(np.median(pe)),
                         'median_rank': float(np.median(rk))}

    print(f'\n=== P(<eos>) 结果（tag={args.tag or "-"}，每组 n={args.n}，seed={args.seed}）===')
    print(f'{"组":14s} {"mean P":>10s} {"median P":>10s} {"median rank":>12s} {"P>1%":>7s}')
    order = ('random_mid', 'after_think', 'reply_cont', 'reply_eos')
    for name in order:
        s = out[name]
        hi = float(np.mean(np.asarray(s['p_eos']) > 0.01))
        print(f'{name:14s} {s["mean"]:10.5f} {s["median"]:10.5f} {s["median_rank"]:12.0f} {hi:7.1%}')

    r, a, ec = out['random_mid'], out['after_think'], out['reply_eos']
    ctrl_ok = ec['median_rank'] < r['median_rank']
    print(f'\n已知答案对照：reply_eos.median_rank ({ec["median_rank"]:.0f}) '
          f'< random_mid.median_rank ({r["median_rank"]:.0f}) '
          f'→ {"✅ 探针有信号" if ctrl_ok else "❌ 探针坏了，下面的数不可信"}')
    if ctrl_ok:
        print('先验强度（rank 越小 = 模型越倾向在这里收尾；8192 个 token 里排第几）：')
        print(f'  random_mid  {r["median_rank"]:>6.0f}  ← 基线（回复中段，本不该收尾）')
        print(f'  after_think {a["median_rank"]:>6.0f}  ← 被测位置（`<think>` 换行之后）')
        print(f'  reply_eos   {ec["median_rank"]:>6.0f}  ← 已知答案（真的该收尾）')
        print(f'  比值 after_think / random_mid = '
              f'{a["median_rank"] / max(r["median_rank"], 1):.4f}'
              f'（≈1 ⇒ 先验不存在；≈reply_eos/random_mid ⇒ 先验很重）')
        print(f'  ★ 还要看绝对概率：after_think 的 mean P(<eos>) = {a["mean"]:.2e}，'
              f'reply_eos 的 = {ec["mean"]:.2e}')
        print('     rank 高但 P 仍极小 ⇒ 先验"可测，但采样时几乎不会真的兑现"')

    if args.show:
        print(f'\n=== 位置抽检（各取前 {args.show} 个，看位置是否选对 —— AGENTS §5.9）===')
        for name in order:
            print(f'--- {name} ---')
            for p in groups[name][:args.show]:
                ctx = np.asarray(data[p - 90:p], dtype=np.int64)
                nxt = int(data[p])
                print(f'  p={p} 下一 token={decode_ids([nxt], args.tok)!r}')
                print(f'    …{decode_ids(ctx.tolist(), args.tok)}⟨此处预测⟩')

    if args.dump_json:
        with open(args.dump_json, 'w', encoding='utf-8') as f:
            json.dump({'tag': args.tag, 'ckpt': args.ckpt, 'seed': args.seed,
                       'block': BLOCK, 'groups': out}, f, ensure_ascii=False, indent=1)
        print(f'明细已落盘 → {args.dump_json}')


if __name__ == '__main__':
    main()
