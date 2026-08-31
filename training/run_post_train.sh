#!/usr/bin/env bash
# 训练后自动验证 + 分析套件
# 等训练完成 → 依次执行：你好体检、多轮对话、激活分析、loss 分析
set -u
cd "$(dirname "$0")/.." || exit 1

DIR="out/eos_fix_1epoch"
OLD="out/ab_char_all6_1500"  # 旧 Arm5 对照
LOG="$DIR/post_train_report.txt"

export HSA_OVERRIDE_GFX_VERSION=10.3.0

echo "═══════════════════════════════════════════════════════" | tee "$LOG"
echo "  EOS 修复后 1 epoch 训练 — 完整验证报告" | tee -a "$LOG"
echo "  $(date)" | tee -a "$LOG"
echo "═══════════════════════════════════════════════════════" | tee -a "$LOG"

# 0. 训练结果摘要
echo "" | tee -a "$LOG"
echo "▶ 训练结果摘要" | tee -a "$LOG"
if [ -f "$DIR/results.csv" ]; then
  echo "  最后几行 results.csv:" | tee -a "$LOG"
  tail -5 "$DIR/results.csv" | tee -a "$LOG"
fi
echo "" | tee -a "$LOG"

# 1. 你好体检（新模型 + 旧模型对照）
echo "▶ [1/4] 你好体检" | tee -a "$LOG"
DIRS="--dirs $DIR"
[ -f "$OLD/best.pt" ] && DIRS="$DIRS $OLD"
.venv/bin/python training/health_check_hello.py $DIRS --samples 3 2>&1 | tee -a "$LOG"
echo "" | tee -a "$LOG"

# 2. 多轮对话自然度实测
echo "▶ [2/4] 多轮对话自然度实测" | tee -a "$LOG"
CHAT_OUT="$DIR/chat_nat"
mkdir -p "$CHAT_OUT"

# 天气场景 4 轮
printf '你好，今天天气怎么样？\n那适合出门跑步吗？\n如果下雨该带什么？\n谢谢你的建议！\nexit\n' | \
  .venv/bin/python inference/scripts/chat.py --out_dir "$DIR" --seed 42 --max-new-tokens 200 \
  > "$CHAT_OUT/s1-weather.txt" 2>&1
echo "  天气场景 → $CHAT_OUT/s1-weather.txt" | tee -a "$LOG"
cat "$CHAT_OUT/s1-weather.txt" | tee -a "$LOG"

# 心情场景 4 轮
printf '我今天心情有点低落。\n工作上遇到了点挫折。\n感觉大家都不理解我。\n听你这么说我好多了。\nexit\n' | \
  .venv/bin/python inference/scripts/chat.py --out_dir "$DIR" --seed 42 --max-new-tokens 200 \
  > "$CHAT_OUT/s2-mood.txt" 2>&1
echo "  心情场景 → $CHAT_OUT/s2-mood.txt" | tee -a "$LOG"
cat "$CHAT_OUT/s2-mood.txt" | tee -a "$LOG"

# 通用开场 6 题
printf '你好！\n请介绍一下你自己。\n如何学好人工智能？\n什么是量子计算？\n推荐两本好看的历史小说。\n为什么天空是蓝色的？\nexit\n' | \
  .venv/bin/python inference/scripts/chat.py --out_dir "$DIR" --seed 42 --max-new-tokens 200 \
  > "$CHAT_OUT/gen.txt" 2>&1
echo "  通用开场 → $CHAT_OUT/gen.txt" | tee -a "$LOG"
cat "$CHAT_OUT/gen.txt" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# 3. 网络激活分析
echo "▶ [3/4] 网络激活分析" | tee -a "$LOG"
.venv/bin/python training/analyze_activations.py --dir "$DIR" 2>&1 | tee -a "$LOG"
echo "" | tee -a "$LOG"

# 4. Loss 曲线分析
echo "▶ [4/4] Loss 曲线分析" | tee -a "$LOG"
if [ -f "$DIR/results.csv" ]; then
  .venv/bin/python -c "
import csv
rows = list(csv.DictReader(open('$DIR/results.csv')))
if rows:
    first, last = rows[0], rows[-1]
    best_val = min(float(r['val/loss']) for r in rows)
    best_step = [r['step'] for r in rows if float(r['val/loss']) == best_val][0]
    print(f'  起始 val: {first[\"val/loss\"]} → 最终 val: {last[\"val/loss\"]}')
    print(f'  最佳 val: {best_val:.4f} @ step {best_step}')
    print(f'  起始 train: {first[\"train/loss\"]} → 最终 train: {last[\"train/loss\"]}')
    gap = float(last['train/loss']) - float(last['val/loss'])
    print(f'  最终 train-val gap: {gap:.4f} ({\"过拟合\" if gap < -0.3 else \"正常\"})')
    print(f'  总耗时: {last[\"time\"]}s')
" 2>&1 | tee -a "$LOG"
fi
echo "" | tee -a "$LOG"

echo "═══════════════════════════════════════════════════════" | tee -a "$LOG"
echo "  完整报告已保存: $LOG" | tee -a "$LOG"
echo "═══════════════════════════════════════════════════════" | tee -a "$LOG"
