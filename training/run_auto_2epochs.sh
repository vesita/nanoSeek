#!/usr/bin/env bash
# 自动执行 2-Epoch 续训 + 训练后自动化全流程评测
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

export HSA_OVERRIDE_GFX_VERSION=10.3.0
OUT_DIR="out/eos_fix_1epoch"
LOG="$OUT_DIR/resume_2epochs.log"

echo "==================================================" | tee -a "$LOG"
echo "🚀 启动 2-Epoch (17,200 步) 续训任务: $(date)" | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"

.venv/bin/python training/train.py \
  training/config/train_chinese.yaml \
  --char_level=true \
  --factorized_emb_dim=24 \
  --n_layer=6 \
  --kv_memory_output_gate=true \
  --n_experts=8 \
  --n_top_k=2 \
  --moe_hidden_scale=1.33333333 \
  --n_mtp=2 \
  --kv_memory_checkpoint=true \
  --enable_early_stop=false \
  --init_from=resume \
  --out_dir="$OUT_DIR" \
  --max_iters=17200 \
  --eval_interval=500 \
  --eval_iters=50 \
  --lr_decay_iters=17200 \
  --compile=true \
  >> "$LOG" 2>&1

echo "==================================================" | tee -a "$LOG"
echo "✅ 续训完成，自动触发全套评测报告: $(date)" | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"

bash training/run_post_train.sh >> "$LOG" 2>&1

echo "🎉 全部评测与分析完成: $(date)" | tee -a "$LOG"
