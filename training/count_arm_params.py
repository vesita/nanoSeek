#!/usr/bin/env python3
"""字级 A/B 各臂参数量核对脚本（以已训练 checkpoint 的 model_args 为准）。

训练过的臂直接读 out/<dir>/best.pt 的 model_args 重建模型数参数；
新组合臂以 Arm 4（全要素）的 model_args 为基底改 n_layer，保证与新臂训练配置
逐字节一致。数值口径 = get_num_params(non_embedding=True)（可训练参数，
不含 RoPE cos/sin 等非学习 buffer）。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from model.config import GPTConfig  # noqa: E402
from model.gpt import GPT           # noqa: E402


def count_from_args(name, model_args):
    model = GPT(GPTConfig(**{k: v for k, v in model_args.items()}))
    n = model.get_num_params(non_embedding=True)
    print(f"{name:<30} {n:>9,}   ({n/1e6:.3f}M)")
    return n


def load_ckpt_args(rel_dir):
    ck = torch.load(os.path.join(ROOT, rel_dir, "best.pt"), map_location="cpu", weights_only=False)
    return dict(ck["model_args"])


def main():
    print(f"{'实验臂':<30} {'可训练参数'}")
    print("-" * 52)
    base = count_from_args("Arm 0: 基准 Fact-6L", load_ckpt_args("out/char_fact24_300"))
    count_from_args("Arm 1: +输出门控 6L", load_ckpt_args("out/ab_char_gate_300"))
    count_from_args("Arm 2: +细粒度MoE 6L", load_ckpt_args("out/ab_char_moe8_300"))
    count_from_args("Arm 3: +2-Step MTP 6L", load_ckpt_args("out/ab_char_mtp2_300"))
    n_arm4 = count_from_args("Arm 4: 全要素 7L", load_ckpt_args("out/ab_char_all_300"))
    # 新臂：以 Arm 4 的完整 model_args 为基底，仅把 7 层改为 6 层
    new6 = load_ckpt_args("out/ab_char_all_300")
    new6["n_layer"] = 6
    n_new = count_from_args("NEW: 全要素组合 6L", new6)
    print("-" * 52)
    print(f"6层全组合 vs 7层全要素: {n_arm4 - n_new:+,} 参数（= 第 7 个 Transformer 块）")


if __name__ == "__main__":
    main()
