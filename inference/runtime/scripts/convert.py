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
    ap.add_argument('--q8', action='store_true',
                    help='Q8 量化：权重按张量对称 int8 量化（scale=max|w|/127），'
                         '体积 ~1/4；Rust 端加载时按 scale 反量化。骨架版，未做精度校准')
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
    # 融合 QKV → 拆分回三个独立投影（Rust 端 attention.rs 按 q/k/v_proj_csa 加载；
    # c_qkv_csa.weight 形状 (3*n_embd, n_embd) = [Wq; Wk; Wv] 行拼接，split 顺序与
    # model/attention.py 的 c_qkv_csa(x).split(n_embd, dim=2) 一致）
    fused = [k for k in sd if '.c_qkv_csa.weight' in k]
    for k in fused:
        prefix = k.replace('.c_qkv_csa.weight', '')
        w = sd.pop(k)  # (3*n_embd, n_embd)
        ne = w.shape[0] // 3
        q, kv = w[:ne], w[ne:]
        kk, vv = kv[:ne], kv[ne:]
        sd[f'{prefix}.q_proj_csa.weight'] = q
        sd[f'{prefix}.k_proj_csa.weight'] = kk
        sd[f'{prefix}.v_proj_csa.weight'] = vv
        print(f"  融合 QKV 拆分：{prefix}.c_qkv_csa.weight ({w.shape}) → q/k/v_proj_csa")
    # 跳过 MTP 模块权重：MTP 是训练时的辅助预测头，推理时主模型输出已含最终 logits，
    # Rust 端不实现 MTP，载入多余的 ~100 万参数纯属浪费（上次转换把 mtp_modules.* 全带进去了）
    sd = {k: v for k, v in sd.items() if 'mtp_modules' not in k}
    # 丢掉 lm_head.weight：它和 wte.weight 是 weight tying 共享的同一份数据，
    # safetensors 不允许共享内存张量重复保存；Rust 端直接用 wte 做输出投影
    sd = {k: v for k, v in sd.items() if k != 'lm_head.weight'}
    # 统一转 float32（Rust 端按 F32 处理）
    sd = {k: v.float().contiguous() for k, v in sd.items()}

    # Q8 量化（骨架）：每张量对称 int8——scale = max|w|/127，q = round(w/scale)。
    # candle safetensors 不支持 I8，存 U8（int8+128 偏移）；权重名保持不变（存 uint8），
    # 另存 {name}_scale (f32 标量)；Rust 加载时反量化：(u8-128)*scale。
    if args.q8:
        qsd = {}
        n_q = 0
        for k, v in sd.items():
            if v.dim() >= 2 and v.numel() > 1:   # 矩阵权重量化，bias/标量不动
                scale = v.abs().max().item() / 127.0
                if scale > 0:
                    q = torch.clamp(torch.round(v / scale), -128, 127).to(torch.int8)
                    qsd[k] = (q.to(torch.uint8) + 128).contiguous()   # int8 → uint8 偏移
                    qsd[f'{k}_scale'] = torch.tensor([scale], dtype=torch.float32)
                    n_q += 1
                    continue
            qsd[k] = v
        sd = qsd
        model_args['quantized'] = 'q8'
        print(f"  Q8 量化：{n_q} 个权重张量 → uint8（含 scale）")

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
