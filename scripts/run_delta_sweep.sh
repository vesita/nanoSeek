#!/usr/bin/env bash
# Δ 扫描：把「槽位够不够、λ 最优点在哪」一次跑清楚。
#
# 背景（2026-09-10 全量探针结论）：L=8/M=67M 的覆盖率是 99.99%（不是 §12.3 说的 4.9%），
# 但覆盖内 top-1 命中率只有 18.52%（§12.3 是 61.6%）—— 瓶颈是**槽碰撞**而非覆盖率。
# 两个待验证假设：
#   H1  λ*=0.05 卡在网格下界，真正最优更小（A/p−1 判据：A 只有 0.185，只能轻轻混）
#   H2  把槽位从 67M 加到 268M（负载因子 14→3.5，覆盖率仍 93.5%）能明显提升命中率与 Δ
#
# 用已有的 top-1 缓存时 H1 只要 1 分钟；H2 要重建 268M 槽的表（约 5 分钟）。
set -u
cd "$(dirname "$0")/.." || exit 1
export HSA_OVERRIDE_GFX_VERSION=10.3.0 HSA_ENABLE_SDMA=0
PY=.venv/bin/python
COMMON="--corpus data/chinese/train_char_v2.bin --val data/chinese/val_char_v2.bin \
  --ckpt out/ndb_run/last.pt --skip_cover --device cuda --bs 16 \
  --val_windows 4096 --log_every 64 --empty_cache_every 16 \
  --lams 0.01,0.02,0.03,0.05,0.08,0.12"

run () {  # run <标签> <额外参数...>
  local tag="$1"; shift
  echo "════════ $(date '+%H:%M:%S')  $tag ════════" | tee -a out/ngram_full/sweep.log
  $PY -u scripts/ngram_capacity_probe.py $COMMON "$@" \
      >> "out/ngram_full/sweep_${tag}.log" 2>&1
  echo "[exit=$?] $tag 完成" | tee -a out/ngram_full/sweep.log
  grep -E "各口径最优|槽命中率|命中率 =|λ\*=" "out/ngram_full/sweep_${tag}.log" | tail -8 | tee -a out/ngram_full/sweep.log
}

# C：L=8 / 67M —— 复用缓存，只在更低的 λ 上找最优点（验证 H1）
run "C_L8_M67M"   --delta_L 8  --delta_M 67108864 \
    --out_json out/ngram_full/delta_L8_M67M_finelam.json

# A：L=8 / 268M —— 验证 H2（槽位翻 4 倍）
run "A_L8_M268M"  --delta_L 8  --delta_M 268435456 \
    --out_json out/ngram_full/delta_L8_M268M.json

# B：L=12 / 268M —— 更长后缀 + 大表（精度最高的组合）
run "B_L12_M268M" --delta_L 12 --delta_M 268435456 \
    --out_json out/ngram_full/delta_L12_M268M.json

echo "════════ 全部完成 $(date '+%H:%M:%S') ════════" | tee -a out/ngram_full/sweep.log
