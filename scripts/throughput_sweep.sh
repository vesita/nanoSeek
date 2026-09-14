#!/usr/bin/env bash
# 吞吐扫描：找出「每步多少 token / 每次前向多大」对 tok/s 的影响。
#
# ★ 结论（2026-09-11 实跑，见 dev-notes/83 §0.6）：
#   原假设「瓶颈是 kernel 粒度，所以微批越大越快」**被推翻**。实测单调相反：
#     A bs4/ga8  → 3.86 s/步 = 2124 tok/s   ← 最快（就是现配置）
#     B bs8/ga4  → 4.51 s/步 = 1815 tok/s
#     C bs16/ga2 → 4.71 s/步 = 1738 tok/s
#   按微批归一化（每微批 tok/s）：bs=4 → 8480、bs=8 → 7500、bs=16 → 6940，单调下降。
#   且 gpu_busy_percent = 99% —— GPU **不是空闲**，是被大量低效 kernel 占满。
#   所以方向不是「填满空闲」，而是「让 kernel 变少/变高效」。
#   ⚠ E/F 臂是 **2× token/步**（ga 没随 bs 改），不能和 A/B/C/D 直接比 s/步，只能比 tok/s。
#
# 设计（关键）：A→B→C→D 是**同样 8192 token/步**、只变「每次前向的 token 数」的阶梯。
# 这样总工作量完全相同，tok/s 的差别只能来自 kernel 粒度 —— 这是干净的对照。
#
# 安全性：每个臂**独立进程、串行执行**（不同时占显存）；用 --init_from=scratch
# 建同尺寸随机模型（不碰 out/base_v2/ 的任何存档）；eval_interval 拉到极大
# → 不评估、不写 checkpoint。
# ⚠ D_bs32 是 8× 基线激活，接近本机显存上限，且若桌面/游戏在占显存会拖死桌面。
#   默认已从臂表移除；要用请显式传参。
#
# 用法：
#   bash scripts/throughput_sweep.sh                    # 跑默认臂
#   bash scripts/throughput_sweep.sh "A 4 8 256" "B 8 4 256"
#
# 读结果：scripts/throughput_sweep.sh 跑完会打印汇总表。
# 要**可信的比值**请优先用 scripts/throughput_focus.sh（ABAB 交替，抗负载漂移）。
set -u
cd "$(dirname "$0")/.."
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HSA_ENABLE_SDMA=0

STEPS=${STEPS:-40}          # 总步数（含编译预热；分析时自动跳过前 10 步）
WARM=${WARM:-10}            # 分析时跳过的预热步数
PY=.venv/bin/python

# ---- 测量纪律工具（2026-09-11 补：上一轮扫描踩过这三个坑）------------------
# 坑 1：显存读数在上一个进程释放前就采样 → 打印出「阶梯」（2231→4059→4721→6121），
#       看起来像显存泄漏，其实是读数时刻的假象。必须先等进程消失再等显存稳定。
# 坑 2：机上有别的进程（游戏/浏览器）占 GPU 时，绝对值完全不可信。每臂记录
#       gpu_busy_percent 作为混杂指示。**比值仍可用**，绝对值不可引用。
# 坑 3：臂串行不交替 → 外国负载漂移无法被检测。想得到可信比值请用
#       scripts/throughput_focus.sh / throughput_knobs.sh（ABAB 交替设计）。
vram_mib() { awk '{printf "%.0f", $1/1048576}' /sys/class/drm/card1/device/mem_info_vram_used 2>/dev/null || echo '?'; }
gpu_busy() { cat /sys/class/drm/card1/device/gpu_busy_percent 2>/dev/null || echo '?'; }
settle() {   # 等显存回到稳定值（连续两次读数相同），最多 20 秒
  local prev=-1 cur t=0
  while [ $t -lt 20 ]; do
    cur=$(vram_mib); [ "$cur" = "$prev" ] && { echo "$cur"; return; }
    prev=$cur; sleep 2; t=$((t+2))
  done
  echo "$cur"
}

if [ "$#" -gt 0 ]; then
  ARMS=("$@")
else
  # tag  batch_size  grad_accum  block_size   tok/前向   tok/步
  ARMS=(
    "A_base      4  8  256   #  1024   8192   ← 现状（实测最快）"
    "B_bs8       8  4  256   #  2048   8192"
    "C_bs16     16  2  256   #  4096   8192"
    "E_ga8       8  8  256   #  2048  16384   ← 2× token/步，只比 tok/s"
    "F_blk512    4  8  512   #  2048  16384   ← 2× token/步，只比 tok/s"
    # "D_bs32    32  1  256   #  8192   8192  ← 8× 激活，接近显存上限，默认禁用
  )
fi

echo "=== 吞吐扫描开始 $(date '+%F %T')  STEPS=$STEPS ==="
echo "（混杂提示：每臂记录 gpu_busy%；>10% 说明有别的进程在抢 GPU，绝对值不可引用）"
for spec in "${ARMS[@]}"; do
  spec="${spec%%#*}"                       # 去掉注释
  read -r tag bs ga blk <<< "$spec"
  [ -z "${tag:-}" ] && continue
  log="out/_bench_${tag}.log"
  echo "--- [$tag] batch_size=$bs grad_accum=$ga block_size=$blk -> $log"
  echo "    跑前：显存 $(settle) MiB   gpu_busy $(gpu_busy)%"
  timeout 1800 $PY -u training/train.py configs/base_v2.yaml \
    --init_from=scratch --out_dir="out/_bench_${tag}" \
    --batch_size="$bs" --gradient_accumulation_steps="$ga" --block_size="$blk" \
    --max_iters="$STEPS" --eval_interval=1000000 --eval_iters=1 --eval_train_split=false \
    --log_interval=5 --always_save_checkpoint=false > "$log" 2>&1
  rc=$?
  echo "    rc=$rc  跑后：显存 $(settle) MiB   gpu_busy $(gpu_busy)%"
  # OOM / 崩溃就记录下来继续，不要让一条腿拖垮整轮
  grep -qE "显存不足|OutOfMemory|Traceback" "$log" && echo "    ⚠ 出现 OOM/Traceback"
done

echo
echo "=== 汇总 $(date '+%F %T') ==="
$PY - "$WARM" <<'PYEOF'
import re, sys, glob, os, datetime
warm = int(sys.argv[1])
# tqdm 的 elapsed 字段：<1h 时是 MM:SS，否则 H:MM:SS
LINE = re.compile(r"\|\s*(\d+)/\d+ \[(\d+(?::\d+){1,2})<.*?,\s*([\d.]+)s/it")
def secs(s):
    p = [int(x) for x in s.split(':')]
    while len(p) < 3: p.insert(0, 0)
    return p[0]*3600 + p[1]*60 + p[2]

rows = []
for log in sorted(glob.glob('out/_bench_*.log')):
    tag = os.path.basename(log)[len('_bench_'):-len('.log')]
    txt = open(log, errors='ignore').read()
    # 配置行里的 tokens/步
    m = re.search(r'batch_size=(\d+).*?grad', txt)
    pts = re.findall(r"tokens/步", txt)
    steps = [(int(a), secs(b), float(c)) for a, b, c in LINE.findall(txt)]
    steps = [(i, t) for i, t, _ in steps]
    peak = re.findall(r"peak=([\d.]+)", txt)
    if len(steps) < warm + 3:
        rows.append((tag, None, None, None, peak[-1] if peak else '?', '步数不足/OOM'))
        continue
    (i0, t0), (i1, t1) = steps[warm], steps[-1]
    # 从日志里拿 tokens/step
    mt = re.search(r"(\d[\d,]*)\s*tokens/步", txt)
    tok_step = int(mt.group(1).replace(',', '')) if mt else None
    if not tok_step or i1 == i0:
        rows.append((tag, None, None, None, peak[-1] if peak else '?', '无法解析'))
        continue
    dt = t1 - t0
    tps = (i1 - i0) * tok_step / dt if dt > 0 else 0
    sit = dt / (i1 - i0)
    rows.append((tag, tok_step, sit, tps, peak[-1] if peak else '?', ''))
print(f"{'臂':<12}{'tok/步':>9}{'稳态s/it':>10}{'tok/s':>9}{'峰值显存G':>11}  备注")
print("-"*66)
base = None
for tag, ts, sit, tps, pk, note in rows:
    if tps is None:
        print(f"{tag:<12}{'-':>9}{'-':>10}{'-':>9}{pk:>11}  {note}")
        continue
    if base is None: base = tps
    print(f"{tag:<12}{ts:>9,}{sit:>10.2f}{tps:>9.0f}{pk:>11}  {tps/base:.2f}x vs 第一臂 {note}")
PYEOF
echo "（显存基线：空载约 0.9G）"
