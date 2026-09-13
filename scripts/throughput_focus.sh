#!/usr/bin/env bash
# 定向吞吐对照：bs=4/ga=8（现配置） vs bs=2/ga=16（更小微批）
#
# 背景：throughput_sweep.sh 的 A/B/C/E 四臂给出**单调**结论 —— 微批越小越快：
#   bs=4  → 3.86 s/步  (8192 tok/步)  → 2124 tok/s   <- 最快
#   bs=8  → 4.51 s/步                 → 1815 tok/s
#   bs=16 → 4.71 s/步                 → 1738 tok/s
# 按微批归一化：bs=4 每微批 8480 tok/s、bs=8 7500、bs=16 6940，单调下降。
# 但线性拟合截距为负（非物理），所以**不能**外推到 bs<4，必须实测。
#
# 注意（测量纪律）：
#   * 机上有用户的游戏在占 GPU（busy 27~51%），绝对值不可信，只能看**同批次内的比值**。
#   * 因此用 ABAB 交替而不是 AABB —— 游戏负载漂移时交替能把漂移抵消掉。
#   * token/步固定 8192（4*8*256 = 2*16*256），所以直接比 s/步即可。
#
# 用法：cd /home/vesita/coding/my/nanoSeek && bash scripts/throughput_focus.sh
set -u
cd /home/vesita/coding/my/nanoSeek

export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HSA_ENABLE_SDMA=0
PY=.venv/bin/python
LOG=out/throughput_focus.log
: > "$LOG"

# 臂定义：标签:batch_size:grad_accum  —— ABAB 交替
ARMS=(
  "A1:4:8"
  "X1:2:16"
  "A2:4:8"
  "X2:2:16"
)

vram() { awk '{printf "%.0f", $1/1048576}' /sys/class/drm/card1/device/mem_info_vram_used; }

echo "=== 定向吞吐对照开始 $(date '+%F %T')  STEPS=40  ABAB ===" | tee -a "$LOG"

for arm in "${ARMS[@]}"; do
  tag="${arm%%:*}"; rest="${arm#*:}"; bs="${rest%%:*}"; ga="${rest##*:}"
  out="out/_foc_${tag}"
  log="out/_foc_${tag}.log"
  echo "--- [${tag}] batch_size=${bs} grad_accum=${ga} block_size=256  跑前显存 $(vram) MiB" | tee -a "$LOG"
  env timeout 1800 "$PY" -u training/train.py configs/base_v2.yaml \
    --init_from=scratch --out_dir="$out" \
    --batch_size="$bs" --gradient_accumulation_steps="$ga" --block_size=256 \
    --max_iters=40 --eval_interval=1000000 --eval_iters=1 --eval_train_split=false \
    --log_interval=5 --always_save_checkpoint=false \
    > "$log" 2>&1
  rc=$?
  echo "    rc=$rc  跑后显存 $(vram) MiB" | tee -a "$LOG"
done

echo "=== 定向对照结束 $(date '+%F %T') ===" | tee -a "$LOG"

# ---- 解析：用 tqdm 时间戳差分算稳态 s/步（不用累计均值，它含编译预热）----
"$PY" - <<'PY' | tee -a "$LOG"
import re, glob, os
def tosec(t):
    p = [int(x) for x in t.split(':')]
    return p[0]*3600 + p[1]*60 + p[2] if len(p) == 3 else p[0]*60 + p[1]
rows = []
for f in sorted(glob.glob('out/_foc_*.log')):
    txt = open(f, errors='ignore').read()
    d = {int(m.group(1)): tosec(m.group(2))
         for m in re.finditer(r'(\d+)/40 \[((?:\d+:)?\d+:\d+)<', txt)}
    ks = sorted(d)
    if len(ks) < 5:
        print(f"{os.path.basename(f):22s} 数据不足 ({len(ks)} 行)"); continue
    lo = next((k for k in ks if k >= 5), ks[0]); hi = ks[-1]
    sp = (d[hi] - d[lo]) / (hi - lo)
    tag = os.path.basename(f)[5:-4]
    rows.append((tag, lo, hi, d[hi]-d[lo], sp, 8192/sp))

print()
print(f"{'臂':>4s} {'区间':>10s} {'Δt':>6s} {'s/步':>7s} {'tok/s':>7s}")
for tag, lo, hi, dt, sp, tps in rows:
    print(f"{tag:>4s} {lo:>4d}->{hi:<4d} {dt:>5d}s {sp:>7.2f} {tps:>7.0f}")

A = [r[4] for r in rows if r[0].startswith('A')]
X = [r[4] for r in rows if r[0].startswith('X')]
if A and X:
    a, x = sum(A)/len(A), sum(X)/len(X)
    print(f"\nA(bs4/ga8) 平均 {a:.2f} s/步   X(bs2/ga16) 平均 {x:.2f} s/步")
    print(f"X 相对 A: {x/a*100:.1f}%  (步时)  → tok/s 提升 {(a/x-1)*100:+.1f}%")
    if len(A) == 2:
        drift = abs(A[0]-A[1]) / a * 100
        print(f"A 臂自身漂移（游戏负载漂移的估计）: {drift:.1f}%")
        print(f"=> 若 |X/A - 1| 小于漂移，则判为『无差异』")
PY
