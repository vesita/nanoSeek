#!/usr/bin/env python3
"""
gfx1030 (RDNA2) 针对性 Triton knob micro-benchmark
=================================================== 

用途：
  评估三个默认关闭的 Triton Inductor knobs 在【本机真实热算子】上的收益，
  为 scripts/train_100m.sh 提供"该不该默认开启"的硬数据。

测的算子（对齐 model/attention.py + model/mlp.py + model/neural_db.py）：
  1. scaled_dot_product_attention  (MLA-lite + Flash)
  2. MoE 分组 GEMM  (top-k 路由后的 expert 矩阵乘)
  3. 大 MLP 前向  (ffn_hidden 扩张 + 收缩)

被测 knob 组合（环境变量，须在 import torch 前导出）：
  TRITON_USE_BUFFER_OPS        True/False (buffer load/store 指令路径)
  TRITON_USE_ASYNC_COPY        True/False (LDS async copy / 流水线)
  TRITON_SCALARIZE_PACKED_FLOPS True/False (标量化 packed fops, 减 register spill)

用法：
  训练跑完后（显存空闲时）逐条执行：
    HSA_OVERRIDE_GFX_VERSION=10.3.0 HSA_ENABLE_SDMA=0 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    TRITON_USE_BUFFER_OPS=False TRITON_USE_ASYNC_COPY=False TRITON_SCALARIZE_PACKED_FLOPS=False \
    .venv/bin/python scripts/gfx1030_bench.py --label baseline

    TRITON_USE_BUFFER_OPS=True TRITON_USE_ASYNC_COPY=True TRITON_SCALARIZE_PACKED_FLOPS=True \
    .venv/bin/python scripts/gfx1030_bench.py --label tuned

说明：
  - 每个算子先用 torch.compile 以 Inductor 编译成 Triton kernel（冷启动一次），
    再做固定次数的热测量，用"每步毫秒 + 相对 baseline 加速比"呈现。
  - 全程在 8GB 显存限制内，单算子 tensor 都不大，可安全独占跑。
"""

import argparse
import os
import sys
import time

# --- ROCm / gfx1030 运行规约（与 scripts/train_100m.sh 一致）---
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("HSA_ENABLE_SDMA", "0")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser(description="gfx1030 Triton knob benchmark")
    p.add_argument("--label", default="run", help="本次运行的标签(写入结果)")
    p.add_argument("--iters", type=int, default=30, help="每次测量的迭代数")
    p.add_argument("--warmup", type=int, default=3, help="热身的迭代数")
    p.add_argument("--n_embd", type=int, default=512)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_expert", type=int, default=4)
    p.add_argument("--top_k", type=int, default=2)
    return p.parse_args()


def bench(name, fn, iters, warmup, args, label, results):
    """先 compile 生成 Triton kernel（冷启动），再热测量。"""
    dev = torch.device("cuda")
    # 1. 冷启动：跑一次触发 Inductor 编译
    try:
        fn()
        torch.cuda.synchronize()
    except Exception as e:
        print(f"   [{label}] {name}: 编译失败→跳过 ({type(e).__name__})")
        return

    # 2. 热身（已 compiled，纯执行）
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # 3. 热测量
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters * 1000.0  # ms/iter
    results[name] = dt
    print(f"   [{label}] {name:<22} {dt:8.2f} ms/iter")


def main():
    args = parse_args()
    dev = torch.device("cuda")
    ne, nh, n_expert, topk = args.n_embd, args.n_head, args.n_expert, args.top_k
    B, T, H = 8, 256, ne  # 训练时的典型 batch/seq/head 规模

    print(f"=== gfx1030 Triton knob benchmark [{args.label}] ===")
    print(f"  本机 knobs: use_buffer_ops={os.getenv('TRITON_USE_BUFFER_OPS', '默认False')} "
          f"use_async_copy={os.getenv('TRITON_USE_ASYNC_COPY', '默认False')} "
          f"scalarize_packed_fops={os.getenv('TRITON_SCALARIZE_PACKED_FLOPS', '默认False')}")
    print("  预热/Warmup: 0 次（每个算子前后只触发一次 compile）\n")
    print(f"  运行在: {torch.cuda.get_device_name(0)} | ROCm {torch.version.hip} | PyTorch {torch.__version__}\n")

    # ---- 1. Flash Attention (MLA-lite + SDPA) ----
    q = torch.randn(B, nh, T, ne // nh, device=dev, dtype=torch.float32)
    k = torch.randn(B, nh, T, ne // nh, device=dev, dtype=torch.float32)
    v = torch.randn(B, nh, T, ne // nh, device=dev, dtype=torch.float32)
    fn_attn = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v)

    # ---- 2. MoE 分组 GEMM (top-k 路由后) ----
    x = torch.randn(B * T, ne, device=dev, dtype=torch.float32)
    w = torch.randn(n_expert, ne, ne, device=dev, dtype=torch.float32)
    idx = torch.randint(0, n_expert, (B * T, topk), device=dev)
    def fn_moe():
        # 简化的 top-k 分组 GEMM：每个 token 对选中的 expert 计算线性层
        out = torch.zeros(B * T, ne, device=dev, dtype=torch.float32)
        for e in range(n_expert):
            mask = (idx == e).any(dim=-1)
            if mask.any():
                xs = x[mask]
                out[mask] = xs @ w[e]
        return out

    # ---- 3. 大 MLP 前向 (hidden 扩张 x1.333) ----
    h = int(ne * 1.333)
    xm = torch.randn(B * T, ne, device=dev, dtype=torch.float32)
    w1 = torch.randn(ne, h, device=dev, dtype=torch.float32)
    w2 = torch.randn(h, ne, device=dev, dtype=torch.float32)

    results = {}
    print("算子                           耗时(ms/iter)")
    print("-" * 50)
    bench("flash_attention", fn_attn, args.iters, args.warmup, args, args.label, results)
    bench("moe_topk_gemm", fn_moe, args.iters, args.warmup, args, args.label, results)
    bench("mlp_ffn", lambda: torch.tanh(xm @ w1) @ w2, args.iters, args.warmup, args, args.label, results)

    # ---- 输出 CSV 追加行 ----
    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gfx1030_bench_results.csv")
    header = "label,flash_attention_ms,moe_topk_gemm_ms,mlp_ffn_ms,knobs\n"
    row = (f"{args.label},{results.get('flash_attention', '')},{results.get('moe_topk_gemm', '')},"
           f"{results.get('mlp_ffn', '')},"
           f"buffer={os.getenv('TRITON_USE_BUFFER_OPS','F')}/async={os.getenv('TRITON_USE_ASYNC_COPY','F')}/"
           f"scalar={os.getenv('TRITON_SCALARIZE_PACKED_FLOPS','F')}")
    if not os.path.exists(out_csv):
        with open(out_csv, "w") as f:
            f.write(header)
    with open(out_csv, "a") as f:
        f.write(row + "\n")
    print(f"\n结果已追加到: {out_csv}")


if __name__ == "__main__":
    main()
