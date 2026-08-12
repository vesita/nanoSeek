#!/usr/bin/env python3
"""
把训练好的 PyTorch checkpoint 转成 Rust 推理框架需要的格式：
    model.safetensors    权重（去掉 _orig_mod. 前缀、跳过 RoPE 的 cos/sin）
    model_config.json    模型配置（GPTConfig 字段）
    tokenizer.json       BPE 分词器（从 data/<dataset>/ 复制，Python/Rust 共用）

用法（从项目根目录）：
    uv run python inference/runtime/scripts/convert.py --ckpt out/chinese-data2/best.pt --dataset chinese
"""
import argparse
import json
import os
import shutil

import torch
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='out/chinese-data2/best.pt', help='训练好的 checkpoint 路径（默认当前最佳模型）')
    ap.add_argument('--dataset', default='chinese', help='数据集名，用来找 data/<dataset>/tokenizer.json')
    ap.add_argument('--out', default='inference/runtime', help='输出目录（Rust 项目根）')
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    sd = ck['model']
    model_args = ck['model_args']

    # Rust 端（model_config.json）靠这些字段判断架构。它们已从 Python 的 GPTConfig
    # 硬编码移除（RMSNorm/SwiGLU 固定启用），但 Rust 的 Config 结构体默认 false，
    # 必须在写配置时显式注入 true，否则 Rust 端会退回 LayerNorm / GELU。
    model_args.setdefault('use_rmsnorm', True)
    model_args.setdefault('use_swiglu', True)

    # 去掉 torch.compile 可能留下的 _orig_mod. 前缀
    sd = {k.removeprefix('_orig_mod.'): v for k, v in sd.items()}
    # 跳过 RoPE 的 cos/sin 预计算表（Rust 端按 rope_theta 自己算）
    sd = {k: v for k, v in sd.items() if not (k.endswith('.cos') or k.endswith('.sin'))}
    # 跳过 MTP 模块权重：MTP 是训练时的辅助预测头，推理时主模型输出已含最终 logits，
    # Rust 端不实现 MTP，载入多余的 ~100 万参数纯属浪费（上次转换把 mtp_modules.* 全带进去了）
    sd = {k: v for k, v in sd.items() if 'mtp_modules' not in k}
    # 丢掉 lm_head.weight：它和 wte.weight 是 weight tying 共享的同一份数据，
    # safetensors 不允许共享内存张量重复保存；Rust 端直接用 wte 做输出投影
    sd = {k: v for k, v in sd.items() if k != 'lm_head.weight'}
    # 统一转 float32（Rust 端按 F32 处理）
    sd = {k: v.float().contiguous() for k, v in sd.items()}

    os.makedirs(args.out, exist_ok=True)
    save_file(sd, os.path.join(args.out, 'model.safetensors'))
    with open(os.path.join(args.out, 'model_config.json'), 'w', encoding='utf-8') as f:
        json.dump(model_args, f, indent=2, ensure_ascii=False)

    # BPE 分词器：直接从数据集目录复制，Python 端（tokenizers 库）和 Rust 端
    # （tokenizers crate）共用同一个 tokenizer.json，保证编解码完全一致。
    tokenizer_path = os.path.join('data', args.dataset, 'tokenizer.json')
    if not os.path.exists(tokenizer_path):
        raise SystemExit(f'错误：找不到 {tokenizer_path}，先跑 train_tokenizer.py')
    shutil.copy(tokenizer_path, os.path.join(args.out, 'tokenizer.json'))

    n_params = sum(v.numel() for v in sd.values())
    print(f"转换完成 ✅ 权重 {n_params:,} 参数 → {os.path.join(args.out, 'model.safetensors')}")
    print(f"配置: {model_args}")
    print(f"分词器: {tokenizer_path} → {os.path.join(args.out, 'tokenizer.json')}")


if __name__ == '__main__':
    main()
