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
# 抑制 AMD Triton / C++ 编译器的重复宏定义与头文件重定义警告
#   - CFLAGS/CXXFLAGS: 覆盖通用 C/C++ 编译
#   - TORCH_CXX_FLAGS: 专治 PyTorch C++ extension / Inductor 编译 __triton_launcher.c 时
#     pyconfig.h(_POSIX_C_SOURCE=200809L) 与 glibc features.h(202405L) 的宏重定义冲突
_COMMON_FLAGS="-Wno-macro-redefined -Wno-builtin-macro-redefined"
export CFLAGS="${CFLAGS:-} ${_COMMON_FLAGS}"
export CXXFLAGS="${CXXFLAGS:-} ${_COMMON_FLAGS}"
export TORCH_CXX_FLAGS="${TORCH_CXX_FLAGS:-} ${_COMMON_FLAGS}"
# Triton JIT / MLIR 路径下统一抑制编译期警告刷屏
export TRITON_EXTRA_COMPILE_ARGS="${TRITON_EXTRA_COMPILE_ARGS:-} -w"
export TRITON_HIP_COMPILER_ARGS="${TRITON_HIP_COMPILER_ARGS:-} -w"

# 消除 libdrm 探测 amdgpu.ids 的缺失噪音：提供一个空占位文件即可，无需影响 ROCm
_AMDGIDS="/opt/amdgpu/share/libdrm/amdgpu.ids"
if [ ! -e "${_AMDGIDS}" ] && [ -w "/opt/amdgpu/share/libdrm" ]; then
    : > "${_AMDGIDS}" 2>/dev/null || true
fi

# 默认训练超参（可被命令行覆盖）
MAX_ITERS="${MAX_ITERS:-30000}"
# 实测（gfx1030 8GB）：batch 越小单步越省显存、算力越能喂饱。
#   batch8：6.0G（eager）/ 5.6G（compile）；batch16 直接 OOM。
# 保持真实 batch = batch_size × grad_accum 不变（8×4 = 32，与原 16×2 = 32 等价），
# 但微批减半 → 显存大幅下降 + 吞吐提升。
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
OUT_DIR="${OUT_DIR:-out/nanoseek_100m}"
LR="${LR:-3e-4}"
# lr_decay_iters 默认对齐 max_iters（修复：防止余弦退火过早触底、LR 长期停在 min_lr）
LR_DECAY_ITERS="${LR_DECAY_ITERS:-${MAX_ITERS}}"

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
  --batch_size="${BATCH_SIZE}" --gradient_accumulation_steps="${GRAD_ACCUM}" \
  --max_iters="${MAX_ITERS}" \
  --lr_decay_iters="${LR_DECAY_ITERS}" \
  --out_dir="${OUT_DIR}" \
  --eval_interval=200 --log_interval=20 \
  --dtype=bfloat16 \
  "$@"
