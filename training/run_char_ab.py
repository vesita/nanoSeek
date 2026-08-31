#!/usr/bin/env python3
"""字级架构优化 A/B 串行测试脚本。

评测目标（全部在 4623 词表 train_char.bin 上训练，步数见各臂 max_iters）：
  Arm 0 (基准): Fact-6L (out/char_fact24_300, 已有)
  Arm 1 (方案3): Fact-6L + KV 记忆输出门控 (out/ab_char_gate_300)
  Arm 2 (方案4): Fact-6L + 细粒度 MoE 8选2 (out/ab_char_moe8_300)
  Arm 3 (方案2): Fact-6L + 2-Step 级联 MTP (out/ab_char_mtp2_300)
  Arm 4 (全要素7L): Fact-7L + 门控 + 细粒度 MoE + 2-Step MTP (out/ab_char_all_300)
  Arm 5 (全要素6L): Fact-6L + 门控 + 细粒度 MoE + 2-Step MTP, 1500 步 (out/ab_char_all6_1500)
参数量为检查点实测可训练参数 (get_num_params, 不含 RoPE 缓冲)。
"""
import csv
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ALL_ARMS = [
    {
        "name": "Arm 0: 基准 (Fact-6L)",
        "out_dir": "out/char_fact24_300",
        "params": "2.51M",
        "max_iters": 300,
        "args": [],
    },
    {
        "name": "Arm 1: +输出门控",
        "out_dir": "out/ab_char_gate_300",
        "params": "2.55M",
        "max_iters": 300,
        "args": [
            "--char_level=true",
            "--factorized_emb_dim=24",
            "--n_layer=6",
            "--kv_memory_output_gate=true",
            "--out_dir=out/ab_char_gate_300",
            "--max_iters=300",
            "--eval_interval=50",
            "--eval_iters=25",
            "--lr_decay_iters=300",
            "--warmup_iters=20",
            "--compile=false",
        ],
    },
    {
        "name": "Arm 2: +细粒度MoE (8选2)",
        "out_dir": "out/ab_char_moe8_300",
        "params": "2.33M",
        "max_iters": 300,
        "args": [
            "--char_level=true",
            "--factorized_emb_dim=24",
            "--n_layer=6",
            "--n_experts=8",
            "--n_top_k=2",
            "--moe_hidden_scale=1.33333333",
            "--out_dir=out/ab_char_moe8_300",
            "--max_iters=300",
            "--eval_interval=50",
            "--eval_iters=25",
            "--lr_decay_iters=300",
            "--warmup_iters=20",
            "--compile=false",
        ],
    },
    {
        "name": "Arm 3: +2-Step MTP",
        "out_dir": "out/ab_char_mtp2_300",
        "params": "2.87M",
        "max_iters": 300,
        "args": [
            "--char_level=true",
            "--factorized_emb_dim=24",
            "--n_layer=6",
            "--n_mtp=2",
            "--out_dir=out/ab_char_mtp2_300",
            "--max_iters=300",
            "--eval_interval=50",
            "--eval_iters=25",
            "--lr_decay_iters=300",
            "--warmup_iters=20",
            "--compile=false",
        ],
    },
    {
        "name": "Arm 4: 全要素组合 (7层加深)",
        "out_dir": "out/ab_char_all_300",
        "params": "3.02M",
        "max_iters": 300,
        "args": [
            "--char_level=true",
            "--factorized_emb_dim=24",
            "--n_layer=7",
            "--kv_memory_output_gate=true",
            "--n_experts=8",
            "--n_top_k=2",
            "--moe_hidden_scale=1.33333333",
            "--n_mtp=2",
            "--kv_memory_checkpoint=true",
            "--out_dir=out/ab_char_all_300",
            "--max_iters=300",
            "--eval_interval=50",
            "--eval_iters=25",
            "--lr_decay_iters=300",
            "--warmup_iters=20",
            "--compile=false",
        ],
    },
    {
        "name": "Arm 5: 全要素组合 (6层, 1500步)",
        "out_dir": "out/ab_char_all6_1500",
        "params": "2.70M",
        "max_iters": 1500,
        "args": [
            "--char_level=true",
            "--factorized_emb_dim=24",
            "--n_layer=6",
            "--kv_memory_output_gate=true",
            "--n_experts=8",
            "--n_top_k=2",
            "--moe_hidden_scale=1.33333333",
            "--n_mtp=2",
            "--kv_memory_checkpoint=true",
            "--enable_early_stop=false",
            "--out_dir=out/ab_char_all6_1500",
            "--max_iters=1500",
            "--eval_interval=150",
            "--eval_iters=25",
            "--lr_decay_iters=1500",
            "--warmup_iters=100",
            "--compile=false",
        ],
    },
]


def check_completed(out_dir, max_iters):
    """按该臂的目标步数判断训练是否完成；返回 (done, (best_val, final_val, final_train))。"""
    csv_path = os.path.join(ROOT, out_dir, "results.csv")
    if not os.path.exists(csv_path):
        return False, None
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows and int(rows[-1]["step"]) >= max_iters:
                best_val = min(float(r["val/loss"]) for r in rows if "val/loss" in r)
                final_val = float(rows[-1]["val/loss"])
                final_train = float(rows[-1]["train/loss"])
                return True, (best_val, final_val, final_train)
    except Exception:
        pass
    return False, None


def run_one(arm):
    max_iters = arm["max_iters"]
    is_done, _ = check_completed(arm["out_dir"], max_iters)
    if is_done:
        print(f"⏩ {arm['name']} 已完成 {max_iters} 步训练，跳过。")
        return True

    print("\n" + "=" * 60)
    print(f"🚀 开始训练: {arm['name']} -> {arm['out_dir']}")
    print("=" * 60)

    args = list(arm["args"])
    ckpt_path = os.path.join(ROOT, arm["out_dir"], "best.pt")
    if os.path.exists(ckpt_path) and "--init_from=resume" not in args:
        args.append("--init_from=resume")

    cmd = [
        sys.executable,
        os.path.join(ROOT, "training", "train.py"),
        os.path.join(ROOT, "training", "config", "train_chinese.yaml"),
        *args,
    ]
    env = os.environ.copy()
    env["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
    t0 = time.time()
    ret = subprocess.call(cmd, cwd=ROOT, env=env)
    dt = time.time() - t0
    print(f"⏱ 完成 {arm['name']}，耗时 {dt:.1f}s，退出码 {ret}")
    return ret == 0


def summarize():
    print("\n" + "=" * 80)
    print("📊 字符直入架构优化 A/B 对比结果汇总")
    print("=" * 80)
    header = f"{'实验臂 (Arm)':<30} | {'参数量':<6} | {'Train Loss':<10} | {'Best Val':<10} | {'Final Val':<10} | {'Δ Val vs 基线':<12}"
    print(header)
    print("-" * len(header))

    base_val = None
    for arm in ALL_ARMS:
        done, stats = check_completed(arm["out_dir"], arm["max_iters"])
        if stats:
            best_val, final_val, final_train = stats
            if base_val is None:
                base_val = best_val
                delta_str = "0.0000 (基准)"
            else:
                diff = best_val - base_val
                delta_str = f"{diff:+.4f}"
            print(f"{arm['name']:<30} | {arm['params']:<6} | {final_train:<10.4f} | {best_val:<10.4f} | {final_val:<10.4f} | {delta_str:<12}")
        else:
            print(f"{arm['name']:<30} | {arm['params']:<6} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<12}")
    print("=" * 80)


def main():
    for arm in ALL_ARMS[1:]:
        ok = run_one(arm)
        if not ok:
            print(f"❌ {arm['name']} 训练失败！")
            return 1
    summarize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
