#!/usr/bin/env python3
"""字符直入全要素 7 层深层架构 (Arm 4: 3.02M) 1500 步正式长训脚本。

配置全要素装配：
  - 字级因式分解嵌入 (factorized_emb_dim=24, vocab=4623)
  - 7 层深层 Transformer 主干 (n_layer=7)
  - KV 记忆输出门控 (kv_memory_output_gate=True)
  - 细粒度 MoE (8 专家选 2, moe_hidden_scale=1.3333)
  - 2-Step 级联 MTP 预测头 (n_mtp=2)
  - 激活梯度检查点 (kv_memory_checkpoint=True, 显存安全)
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

cmd = [
    sys.executable,
    os.path.join(ROOT, "training", "train.py"),
    "training/config/train_chinese.yaml",
    "--char_level=true",
    "--factorized_emb_dim=24",
    "--n_layer=7",
    "--kv_memory_output_gate=true",
    "--n_experts=8",
    "--n_top_k=2",
    "--moe_hidden_scale=1.33333333",
    "--n_mtp=2",
    "--kv_memory_checkpoint=true",
    "--out_dir=out/char_fact24_all_1500",
    "--max_iters=1500",
    "--eval_interval=300",
    "--eval_iters=25",
    "--warmup_iters=20",
    "--lr_decay_iters=1500",
    "--compile=false",
]

print("=" * 80)
print("🚀 启动字符直入全要素 7 层 3.02M 架构 1500 步正式长训")
print("命令:", " ".join(cmd))
print("=" * 80)

env = os.environ.copy()
env["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

ret = subprocess.call(cmd, env=env, cwd=ROOT)
sys.exit(ret)
