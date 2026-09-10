#!/usr/bin/env bash
# 归档检查点的外部稀疏化清理（适用于**已经在跑**的训练，改不了它的代码时用）。
#
# 背景：train.py 逢 1000 步写 ckpt_step_<N>.pt（约 0.6GB）。70000 步 ⇒ 70 个 ⇒ 42GB。
# 2026-09-11 巡检发现 out/ 已占 74GB、分区只剩 87GB，而当时的运行进程没有清理逻辑
# （保留策略已加进 train.py，但只对**下次重启之后**生效）。
#
# 本脚本策略（保守，保证可回滚性）：
#   * 永远保留 best.pt / last.pt（固定文件名，续训只用 last.pt）
#   * 保留所有 step % SPARSE_EVERY == 0 的归档（阶段回溯用，默认每 5000 步留一个）
#   * 额外保留最新的 NEWEST_KEEP 个（无论步号，防"刚写的就被删"）
#   * 其它删除
#
# 用法：scripts/prune_ckpts.sh <out_dir> [sparse_every] [newest_keep]
set -uo pipefail

OUT_DIR="${1:?用法: prune_ckpts.sh <out_dir> [sparse_every] [newest_keep]}"
SPARSE_EVERY="${2:-5000}"
NEWEST_KEEP="${3:-2}"

if [[ ! -d "$OUT_DIR" ]]; then
  echo "✗ 目录不存在：$OUT_DIR"
  exit 1
fi

# 保护：进程还在写盘时不要动手（异步保存线程会写 .tmp，但保险起见看主进程）
if pgrep -f "training/train.py" >/dev/null; then
  : # 训练在跑是正常情况，清理历史归档是安全的（不会碰正在写的那一个）
fi

removed=0
freed_mb=0
# 按步号数值升序排列所有归档
mapfile -t all < <(
  find "$OUT_DIR" -maxdepth 1 -name 'ckpt_step_*.pt' -printf '%f\n' 2>/dev/null |
    sed -n 's/^ckpt_step_\([0-9]\+\)\.pt$/\1/p' | sort -n
)

n=${#all[@]}
if (( n == 0 )); then
  echo "（无归档检查点）"
  exit 0
fi

# 最新的 NEWEST_KEEP 个步号进保护名单
declare -A keep=()
for (( i = n - NEWEST_KEEP; i < n; i++ )); do
  (( i >= 0 )) && keep["${all[$i]}"]=1
done

for step in "${all[@]}"; do
  [[ -n "${keep[$step]:-}" ]] && continue
  if (( SPARSE_EVERY > 0 && step % SPARSE_EVERY == 0 )); then
    continue
  fi
  f="$OUT_DIR/ckpt_step_${step}.pt"
  sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
  if rm -f "$f"; then
    removed=$((removed + 1))
    freed_mb=$((freed_mb + sz / 1048576))
  fi
done

echo "🧹 归档稀疏化：保留 $((${#keep[@]})) 个最新 + 每 ${SPARSE_EVERY} 步一个；删除 ${removed} 个，释放 ${freed_mb}MB"
echo "   现存归档：$(find "$OUT_DIR" -maxdepth 1 -name 'ckpt_step_*.pt' | wc -l) 个"
df -h "$OUT_DIR" | tail -1
