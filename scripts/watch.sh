#!/usr/bin/env bash
# 巡检脚本：一轮唤醒的全部检查动作都放这里。
#
# ★ 为什么要有这个脚本（而不是每次现敲一条长命令）：
#   2026-09-11 我**口头**把 prune_ckpts.sh 写进了文档里的巡检命令，
#   但实际挂出去的定时器里只有 `ls | wc -l`（数个数），**没有真的清理**。
#   人（和压缩后的自己）会照文档以为在清理，实际没清 —— 典型的"以为在监控"。
#   → 检查动作本身必须是**一个不可分割、可被调用、可被审计的脚本**，
#     定时器只负责 `sleep N && bash scripts/watch.sh`，没机会漏掉任何一步。
#
# 用法：scripts/watch.sh [out_dir] [log] [sparse_every] [newest_keep]

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

# ★ 默认值必须跟**当前正在跑的 run** 走。2026-09-17 起是**答案段掩码实验的对照臂**
#   （`configs/base_v3_mask_off.yaml` → `out/base_v3_mask_off`，`data_prefix: v3_dlg`，
#   `use_loss_masking: false`，3k 步 —— 与 `out/base_v3_mask` 同起点/同语料/同步数，
#   唯一差别是掩码开关）；
#   掩码臂 `out/base_v3_mask` 与通识段 `out/base_v3_know2` 都已跑完。
#   默认值写死旧 run 的后果不是"少看几行日志"：
#   新目录的归档 ckpt 不会被 prune（见第 49 行），**磁盘会被写满**。
#   换 run 时**必须**同步改这两行；要巡检别的 run 就显式传参。
OUT_DIR="${1:-out/base_v3_intent5}"
LOG="${2:-out/base_v3_intent5_train.log}"
SPARSE_EVERY="${3:-5000}"
NEWEST_KEEP="${4:-2}"
CONFIG_PAT="training/train.py"

echo "===== 巡检 $(date '+%F %T') ====="

echo "--- 事件行（非 tqdm）---"
tr '\r' '\n' < "$LOG" 2>/dev/null | grep -v '^训练中' | tail -4

echo "--- 最新进度 ---"
tr '\r' '\n' < "$LOG" 2>/dev/null | grep '^训练中' | tail -1 | cut -c1-150

echo "--- 指标 CSV ---"
tail -3 "$OUT_DIR/results.csv" 2>/dev/null || echo "(无 CSV)"

_anom=$(grep -cE 'Traceback|OutOfMemory|非有限值' "$LOG" 2>/dev/null)
# ★ 坑：写成 `$(grep -c ... || echo 0)` 会得到两行 "0"（grep -c 命中 0 次时
#   仍然打印 0，但退出码是 1，于是 `|| echo 0` 又补一个 0），
#   字符串比较 `!= "0"` 恒真 → **每轮都误报"出现异常"**。
#   诊断代码自己不能撒谎。grep -c 在文件存在时必定打印数字，不需要兜底。
[[ -z "$_anom" ]] && _anom=0
echo "--- 异常关键字：$_anom ---"
if [[ "$_anom" != "0" ]]; then
  echo "⚠ 出现异常，最近现场："
  grep -E 'Traceback|OutOfMemory|非有限值' -A 6 "$LOG" 2>/dev/null | tail -25
fi

echo "--- 磁盘 ---"
df -h "$OUT_DIR" 2>/dev/null | tail -1

echo "--- 归档检查点清理 ---"
bash scripts/prune_ckpts.sh "$OUT_DIR" "$SPARSE_EVERY" "$NEWEST_KEEP"

if pgrep -f "$CONFIG_PAT" >/dev/null; then
  echo "🟢 仍在跑（pid $(pgrep -f "$CONFIG_PAT" | tr '\n' ' '))"
else
  echo "🔴 已停止——先看上面 traceback，别急着重启"
fi
