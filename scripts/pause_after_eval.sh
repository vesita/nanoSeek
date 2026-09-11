#!/usr/bin/env bash
# 等到下一个 eval 存档点落盘后，优雅暂停训练。
#
# 为什么要等：train.py **没有任何信号处理器**，checkpoint 只在
# `iter_num % eval_interval == 0` 时写。直接 SIGINT 会丢掉从上次 eval 到现在的所有步
# （实测 ~630 步 / 36 分钟训练量）。
#
# 为什么可以放心杀：先等到 `ckpt_step_<N>.pt` **出现在磁盘上**（save_checkpoint_async
# 是先写 .tmp 再 rename，所以文件一出现就是完整的），再发信号——顺序不能反。
#
# 用法：scripts/pause_after_eval.sh <target_step> [train_pat] [out_dir] [timeout_s]

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

TARGET="${1:?用法: pause_after_eval.sh <target_step> [train_pat] [out_dir] [timeout_s]}"
TRAIN_PAT="${2:-training/train.py}"
OUT_DIR="${3:-out/base_v2}"
TIMEOUT="${4:-5400}"

CKPT="$OUT_DIR/ckpt_step_${TARGET}.pt"
echo "[pause] $(date '+%F %T') 等待 $CKPT 落盘（最多 ${TIMEOUT}s）…"

waited=0
while [[ ! -f "$CKPT" ]]; do
  if ! pgrep -f "$TRAIN_PAT" >/dev/null; then
    echo "[pause] $(date '+%F %T') 🔴 训练进程已自行退出，放弃等待"
    exit 2
  fi
  if (( waited >= TIMEOUT )); then
    echo "[pause] $(date '+%F %T') ⏰ 等待超时（${TIMEOUT}s），$CKPT 仍未出现"
    exit 3
  fi
  sleep 20
  waited=$((waited + 20))
done

echo "[pause] $(date '+%F %T') ✅ $CKPT 已落盘（等待 ${waited}s）"
# 再等一下，让同一步的 last.pt 也写完（同一批异步线程，重名 .tmp→rename）
sleep 25

PID=$(pgrep -f "$TRAIN_PAT" | head -1)
if [[ -z "$PID" ]]; then
  echo "[pause] 训练进程已不在"
  exit 0
fi

echo "[pause] 向 pid $PID 发 SIGINT（优雅退出）…"
kill -INT "$PID" 2>/dev/null

for _ in $(seq 1 15); do
  sleep 2
  pgrep -f "$TRAIN_PAT" >/dev/null || break
done

if pgrep -f "$TRAIN_PAT" >/dev/null; then
  echo "[pause] SIGINT 15 秒内未退出 → 升级为 SIGTERM"
  pkill -TERM -f "$TRAIN_PAT" 2>/dev/null
  sleep 5
fi

if pgrep -f "$TRAIN_PAT" >/dev/null; then
  echo "[pause] 仍未退出 → SIGKILL"
  pkill -KILL -f "$TRAIN_PAT" 2>/dev/null
  sleep 3
fi

if pgrep -f "$TRAIN_PAT" >/dev/null; then
  echo "[pause] 🔴 进程仍在，需要人工介入"
  exit 4
fi

echo "[pause] $(date '+%F %T') 🟢 训练已停止"
echo "[pause] --- 存档清单 ---"
ls -la --time-style=+%F_%H:%M:%S "$OUT_DIR"/*.pt 2>/dev/null | awk '{print "  ", $6, $7, $5}'
