#!/usr/bin/env bash
# 串行跑完整 2×2 A/B：base → qk → z → qkz，各 1500 步，关早停。
# 输出：每个 arm 的 out/chinese-data2-ab-<arm>/{results.csv,best.pt,last.pt,loss_curve.png}
#       汇总日志 out/ab_serial.log，逐步状态 out/ab_serial.status
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=out/ab_serial.log
STATUS=out/ab_serial.status
: > "$LOG"
: > "$STATUS"

declare -A ARMS=( [base]="train_chinese_ab_base.yaml" [qk]="train_chinese_ab_qk.yaml" \
                  [z]="train_chinese_ab_z.yaml" [qkz]="train_chinese_ab_qkz.yaml" )
RC=0
for arm in base qk z qkz; do
  cfg="training/config/${ARMS[$arm]}"
  echo "RUNNING $arm" >> "$STATUS"
  echo "==============================" >> "$LOG"
  echo ">>> ARM=$arm CFG=$cfg $(date)" >> "$LOG"
  start=$(date +%s)
  $PY training/train.py "$cfg" >> "$LOG" 2>&1
  rc=$?
  end=$(date +%s)
  echo ">>> ARM=$arm rc=$rc wall=$((end-start))s $(date)" >> "$LOG"
  if [ $rc -ne 0 ]; then
    echo "FAILED $arm rc=$rc" >> "$STATUS"
    RC=1
  else
    echo "DONE $arm wall=$((end-start))s" >> "$STATUS"
    # 关键指标立即落盘，方便中途查看
    tail -n +2 "out/chinese-data2-ab-$arm/results.csv" >> out/ab_serial.log 2>/dev/null
  fi
done
echo "ALL_DONE rc=$RC" >> "$STATUS"
exit $RC