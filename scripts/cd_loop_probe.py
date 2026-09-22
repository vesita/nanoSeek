#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""收敛循环诊断：每圈到底改了多少？（"模块是否在干活"的空测试防线，§9）

读 out/_cd_rec/last.pt（use_cd），在真实 val 窗口上前向，报告：
  - 每圈适配器输出的范数 ‖delta_t‖（相对隐藏态范数）
  - 每圈之后隐藏态相对上一圈的变化率
  - 6 圈累计变化 与 主干 12 层累计变化 的比值
判读：若某圈 delta≈0 或各圈逐位相同 ⇒ 该圈没在干活（mHC 式鞍点）；若逐圈递减至 0 ⇒ 循环退化。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from ckpt_paired_eval import load_model  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ck = os.path.join(REPO, sys.argv[1] if len(sys.argv) > 1 else 'out/_cd_rec/last.pt')
    model, _ = load_model(ck, 'cuda')
    val = np.fromfile(os.path.join(REPO, 'data/chinese/val_char_v3_lang.bin'),
                      dtype=np.uint16)
    rng = np.random.default_rng(7)
    idx = rng.integers(0, len(val) - 257, size=8)
    x = torch.stack([torch.from_numpy(val[i:i + 256].astype(np.int64)) for i in idx]).cuda()

    # hook 每圈适配器
    deltas = []
    handles = []
    for t, ad in enumerate(model.cd_adapters):
        def mk(t):
            def hook(mod, inp, out):
                deltas.append((t, out.detach()))
            return hook
        handles.append(ad.register_forward_hook(mk(t)))

    with torch.no_grad():
        _ = model(x)
    for h in handles:
        h.remove()

    # 主干末层隐藏态范数（相对范数的分母，实测而非硬编码）
    acts = {}
    hs = []
    for i, blk in enumerate(model.transformer.h):
        def mk(i):
            def hook(mod, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                acts[i] = (o.detach().norm(dim=-1).mean().item(), o.detach())
            return hook
        hs.append(blk.register_forward_hook(mk(i)))
    with torch.no_grad():
        _ = model(x)
    for h in hs:
        h.remove()
    hnorm = acts[model.config.n_layer - 1][0]

    print(f'ckpt: {ck}')
    print(f'主干末层隐藏态范数 = {hnorm:.3f}（各圈相对范数的分母）')
    print('圈 | ‖delta‖均值 | 相对隐藏态范数 | 与上一圈 delta 的余弦')
    prev = None
    for t, d in sorted(deltas):
        n = d.norm(dim=-1).mean().item()
        cos = '—'
        if prev is not None:
            a, b = (prev - prev.mean(-1, keepdim=True)).flatten(), (d - d.mean(-1, keepdim=True)).flatten()
            cos = f'{(a @ b / (a.norm() * b.norm() + 1e-9)).item():+.3f}'
        print(f'{t:2d} | {n:11.4f} | {n / hnorm:14.4f} | {cos}')
        prev = d


if __name__ == '__main__':
    main()
