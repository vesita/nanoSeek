#!/usr/bin/env bash
# 归档检查点的**兜底**稀疏化清理。
#
# ★ 2026-09-11 晚起，这个策略已经**搬进训练内部**（`train.py` 每次存档后自己调用
#   `training/checkpoints.prune_step_checkpoints_sparse`）。所以正常情况下你**不需要**跑本脚本。
#
# ★ 为什么还要留着它：它是对「代码改动之前就已经在跑的 run」的兜底 ——
#   那些进程加载的是旧代码（旧策略是"只留最新 5 个"，会在 step 25000 之后
#   把 ckpt_step_5000/10000/15000 这些阶段回溯点静默删掉）。重启训练后就用不上了。
#
# ★ 本脚本**不再自己实现策略**，而是直接调用 Python 侧的唯一实现，避免两份逻辑分叉：
#       python -m training.checkpoints <out_dir> [sparse_every] [newest_keep]
#   策略 = 每 sparse_every 步留一个 + 最新 newest_keep 个；best.pt/last.pt 永不删。
#
# 用法：
#   scripts/prune_ckpts.sh <out_dir> [sparse_every] [newest_keep]
#   scripts/prune_ckpts.sh out/base_v2 --dry-run      # 只看会删什么，不动手
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

OUT_DIR="${1:?用法: prune_ckpts.sh <out_dir> [sparse_every] [newest_keep] [--dry-run]}"
shift || true

exec .venv/bin/python -m training.checkpoints "$OUT_DIR" "$@"
