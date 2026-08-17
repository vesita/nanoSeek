#!/usr/bin/env bash
# 对 4 个 A/B 臂做「续训 500 步」（resume，保留优化器/LR 调度连续性），串行。
# 续训前模型保留在 out/chinese-data2-ab-<arm>（不动）；
# 续训后模型存 out/chinese-data2-ab-<arm>-ft500（new 目录内的 best.pt 为续训产物）。
# 判定成功 = 新目录 results.csv 出现 step=2000 行（段错误 rc=139 发生于收尾阶段，产物已完整）。
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=out/ab_ft500.log
STATUS=out/ab_ft500.status
: > "$LOG"
: > "$STATUS"
for arm in base qk z qkz; do
  src="out/chinese-data2-ab-$arm"
  dst="out/chinese-data2-ab-$arm-ft500"
  mkdir -p "$dst"
  cp -f "$src/best.pt" "$dst/best.pt"   # resume 要求目标目录内已有 checkpoint
  echo "RUNNING $arm" >> "$STATUS"
  echo ">>> ARM=$arm $(date)" >> "$LOG"
  start=$(date +%s)
  $PY training/train.py "training/config/train_chinese_ab_$arm.yaml" \
      --init_from=resume --out_dir="$dst" --max_iters=2000 --enable_early_stop=false \
      >> "$LOG" 2>&1
  rc=$?
  end=$(date +%s)
  echo ">>> ARM=$arm rc=$rc wall=$((end-start))s $(date)" >> "$LOG"
  if [ -f "$dst/results.csv" ] && grep -q ',2000,' "$dst/results.csv"; then
    echo "DONE $arm wall=$((end-start))s" >> "$STATUS"
  else
    echo "FAILED $arm rc=$rc" >> "$STATUS"
  fi
done
echo "ALL_DONE" >> "$STATUS"