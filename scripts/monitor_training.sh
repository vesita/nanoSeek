#!/usr/bin/env bash
# nanoSeek 训练监控辅助：给定 out_dir，输出「是否存活 + 最近进度 + NaN 扫描」的统一口径。
# 用法：
#   bash scripts/monitor_training.sh out/nanoseek_100m        # 默认扫 train/loss val/loss 两列
#   bash scripts/monitor_training.sh out/nanoseek_100m_smoke
# 判定标准（阀值便于人工按节奏复核）：
#   - alive：是否有 train.py 进程在跑（训练中 vs 已结束/崩溃）
#   - step：results.csv 最新的评估点步数
#   - 最近 3 行 train/loss、val/loss、lr、mfu、time
#   - NaN 扫描：train/loss 或 val/loss 出现 nan / inf 即报警

set -uo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="${1:?用法: $0 <out_dir>}"

echo "================ 训练监控：$OUT_DIR ================"

# 1) 是否存活
if pgrep -af "training/train.py" | grep -q .; then
  echo "状态: ✅ 训练进程存活"
  pgrep -af "training/train.py" | head -3
else
  echo "状态: ⛔ 无 train.py 进程（已结束/崩溃）"
fi

CSV="$OUT_DIR/results.csv"
if [[ ! -f "$CSV" ]]; then
  echo "results.csv 尚未生成。检查 stdout 日志：$OUT_DIR/train.log 或最近后台日志。"
  echo "out_dir 内容:"; ls -la "$OUT_DIR" 2>/dev/null
else
  echo "--- 最近 3 个评估点 (step, train/loss, val/loss, lr, mfu, time) ---"
  tail -n 3 "$CSV"
  echo "--- 已评估点数: $(($(wc -l < "$CSV") - 1)) ---"

  # 2) NaN / inf 扫描（跳过表头）
  NAN_HITS=$(tail -n +2 "$CSV" | awk -F, '$2 ~ /nan|inf/ || $3 ~ /nan|inf/')
  if [[ -n "$NAN_HITS" ]]; then
    echo "🚨 发现 NaN/Inf！以下行异常："
    echo "$NAN_HITS"
  else
    echo "NaN 扫描: ✅ 无 NaN/Inf"
  fi

  # 3) 趋势摘要（如有 >=2 个评估点）
  N=$(($(wc -l < "$CSV") - 1))
  if [[ "$N" -ge 2 ]]; then
    LAST=$(sed -n '${p}' "$CSV")
    PREV=$(tail -n 2 "$CSV" | head -n 1)
    L_S=$(echo "$LAST" | cut -d, -f1); L_T=$(echo "$LAST" | cut -d, -f2); L_V=$(echo "$LAST" | cut -d, -f3)
    P_V=$(echo "$PREV" | cut -d, -f3)
    echo "最新: step=$L_S train/loss=$L_T val/loss=$L_V"
    # 若已有前一点 val，给归一化趋势
    if [[ "$P_V" =~ ^[0-9eE.+-]+$ && "$L_V" =~ ^[0-9eE.+-]+$ ]]; then
      D=$(awk -v a="$P_V" -v b="$L_V" 'BEGIN{print b-a}')
      echo "val/loss 变化: $P_V -> $L_V (Δ=$D)"
    fi
  fi
fi

echo "====================================================="
