#!/usr/bin/env python3
"""
把训练好的 PyTorch checkpoint 转成 Rust 推理框架需要的格式：
    model.safetensors    权重（去掉 _orig_mod. 前缀、跳过 RoPE 的 cos/sin）
    model_config.json    模型配置（GPTConfig 字段）
    vocab.json           字符级词表（stoi / itos）

用法（从项目根目录）：
    uv run python runtime/scripts/convert.py --ckpt out/chinese/ckpt.pt --dataset chinese
"""
import argparse
import json
import os
import pickle

import torch
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='out/chinese/ckpt.pt', help='训练好的 checkpoint 路径')
    ap.add_argument('--dataset', default='chinese', help='数据集名，用来找 data/<dataset>/meta.pkl')
    ap.add_argument('--out', default='runtime', help='输出目录（Rust 项目根）')
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    sd = ck['model']
    model_args = ck['model_args']

    # 去掉 torch.compile 可能留下的 _orig_mod. 前缀
    sd = {k.removeprefix('_orig_mod.'): v for k, v in sd.items()}
    # 跳过 RoPE 的 cos/sin 预计算表（Rust 端按 rope_theta 自己算）
    sd = {k: v for k, v in sd.items() if not (k.endswith('.cos') or k.endswith('.sin'))}
    # 丢掉 lm_head.weight：它和 wte.weight 是 weight tying 共享的同一份数据，
    # safetensors 不允许共享内存张量重复保存；Rust 端直接用 wte 做输出投影
    sd = {k: v for k, v in sd.items() if k != 'lm_head.weight'}
    # 统一转 float32（Rust 端按 F32 处理）
    sd = {k: v.float().contiguous() for k, v in sd.items()}

    os.makedirs(args.out, exist_ok=True)
    save_file(sd, os.path.join(args.out, 'model.safetensors'))
    with open(os.path.join(args.out, 'model_config.json'), 'w', encoding='utf-8') as f:
        json.dump(model_args, f, indent=2, ensure_ascii=False)

    meta_path = os.path.join('data', args.dataset, 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    vocab = {
        'stoi': meta['stoi'],
        # JSON 的键必须是字符串，把 itos 的 int 键转成 str
        'itos': {str(i): ch for i, ch in meta['itos'].items()},
    }
    with open(os.path.join(args.out, 'vocab.json'), 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)

    n_params = sum(v.numel() for v in sd.values())
    print(f"转换完成 ✅ 权重 {n_params:,} 参数 → {os.path.join(args.out, 'model.safetensors')}")
    print(f"配置: {model_args}")
    print(f"词表: {len(meta['stoi'])} 字符")


if __name__ == '__main__':
    main()
