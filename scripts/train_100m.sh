#!/usr/bin/env bash
# nanoSeek-100M 训练启动脚本（封装 100M 架构 + 字符级 v3 词表 + 8GB 显卡适配参数）
#
# 用法：
#   bash scripts/train_100m.sh                          # 默认：32 batch × 30000 步（≈1 epoch）
#   bash scripts/train_100m.sh --max_iters 5000 --batch_size 16   # 小规模冒烟
#   bash scripts/train_100m.sh --out_dir out/nanoseek_100m_run1
#
# 架构（对齐 model/nanoseek_100m.py，总参数 ~82M）：
#   12 层 / 8 头 / 512 维 · MLA-Lite · MoE(1共享+4专家×top2, aux-free) · mHC(2流)
#   QK-Norm · RoPE · Attention Sink · Muon+Muon Split · MTP · 梯度检查点

set -euo pipefail
cd "$(dirname "$0")/.."

export HSA_ENABLE_SDMA=0
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TMPDIR="${TMPDIR:-/tmp}"
export PYTHONUNBUFFERED=1

# 默认训练超参（可被命令行覆盖）
MAX_ITERS="${MAX_ITERS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
OUT_DIR="${OUT_DIR:-out/nanoseek_100m}"
LR="${LR:-3e-4}"

exec .venv/bin/python training/train.py \
  --dataset=chinese --char-level=true \
  --n_layer=12 --n_head=8 --n_embd=512 \
  --use_rope=true --use_qk_norm=true \
  --use_moe=true --n_experts=4 --n_top_k=2 \
  --use_shared_expert=true --use_aux_free_balance=true \
  --use_sqrtsoftplus=true --moe_hidden_scale=1.333 \
  --num_hash_layers=2 \
  --use_mhc=true --hc_mult=2 \
  --use_mla=true --kv_lora_rank=96 --qk_rope_head_dim=32 \
  --use_attn_sink=true --use_kv_memory=false \
  --use_muon=true --muon_split=true --use_mtp=true \
  --gradient_checkpointing=true \
  --learning_rate="${LR}" --warmup_iters=100 \
  --batch_size="${BATCH_SIZE}" --gradient_accumulation_steps=2 \
  --max_iters="${MAX_ITERS}" \
  --out_dir="${OUT_DIR}" \
  --eval_interval=200 --log_interval=20 \
  --dtype=bfloat16 \
  "$@"
