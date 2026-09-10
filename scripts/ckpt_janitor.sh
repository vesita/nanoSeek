#!/usr/bin/env bash
# 归档检查点守夜人：独立于 agent 的巡检定时器，兜住「会话/定时器链断了但训练还在跑」的场景。
#
# 为什么需要它：磁盘清理目前依赖 agent 每轮唤醒跑 scripts/watch.sh。但训练是
# `setsid nohup` 出去的，**会话死了训练还会继续**，而 agent 的定时器可能一起死。
# 那样归档会以 ~0.63GB/小时 无限增长（0.59GB/1000步，1000步≈56分钟）。
#
# 本守夜人：
#   * 每 INTERVAL 秒跑一次稀疏化（每 SPARSE_EVERY 步留一个 + 最新 NEWEST_KEEP 个）
#   * **训练进程消失后自动退出**，不会变成需要人类记得杀的孤儿进程
#   * 幂等、只删归档，绝不碰 best.pt / last.pt
#
# 启动（必须只负责启动并立刻返回，铁律 6）：
#   setsid nohup bash scripts/ckpt_janitor.sh > out/janitor.log 2>&1 < /dev/null &
# 停止：
#   pkill -f ckpt_janitor.sh

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

OUT_DIR="${1:-out/base_v2}"
INTERVAL="${2:-1800}"
SPARSE_EVERY="${3:-5000}"
NEWEST_KEEP="${4:-2}"
TRAIN_PAT="training/train.py"

echo "[janitor] 启动 $(date '+%F %T')  out_dir=$OUT_DIR interval=${INTERVAL}s"
while true; do
  sleep "$INTERVAL"
  if ! pgrep -f "$TRAIN_PAT" >/dev/null; then
    echo "[janitor] $(date '+%F %T') 训练进程已消失 → 退出"
    exit 0
  fi
  echo "[janitor] $(date '+%F %T')"
  bash scripts/prune_ckpts.sh "$OUT_DIR" "$SPARSE_EVERY" "$NEWEST_KEEP" 2>&1 | sed 's/^/  /'
done
