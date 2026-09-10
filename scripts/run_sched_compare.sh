#!/usr/bin/env bash
# cosine vs WSD 的**快速对照**（不是"哪个更好"的实验，那要几十小时；
# 这里只验证三件事：① 两者都能跑 ② LR 轨迹符合预期 ③ WSD 中断后能接上）
#
# 预算压到 150 步，但把三段结构压缩到可见：
#   warmup 10 步 → 稳定段到 0.6×150 = 90 步 → 退火 50 步到 min_lr
#
# 判据（跑完自动打印）：
#   - 记录在 train_loss_window.csv 里的 lr 列必须与 training/schedules.py::lr_at 逐点相符
#   - 第 3 组（中断续训）的 LR 轨迹必须与第 2 组（一气跑完）**完全一致**
#
# ⚠ 坑（第一版踩过）：续训段必须用**同一个 out_dir**（checkpoint 在那里），
#   但启动函数若每次都 `rm -rf out_dir`，就会把刚存的 best.pt 删掉再 resume
#   → FileNotFoundError。所以清理与执行要分开（run vs run_keep）。
set -u
cd "$(dirname "$0")/.." || exit 1
export HSA_OVERRIDE_GFX_VERSION=10.3.0 HSA_ENABLE_SDMA=0
PY=.venv/bin/python

COMMON=(configs/base_v2.yaml
  --device=cuda --compile=false
  --batch_size=4 --gradient_accumulation_steps=1
  --warmup_iters=10 --max_iters=150 --lr_decay_iters=150 --stable_frac=0.6
  --log_interval=10 --tensorboard_log=false --health_enabled=false
  --ndb_debug_mem=false --data_prefix=v2 --init_from=scratch)

_exec () {   # _exec <标签> <out_dir> <附加参数...>
  local tag="$1" out="$2"; shift 2
  echo "════════ $tag ════════"
  "$PY" -u training/train.py "${COMMON[@]}" --out_dir="$out" "$@" \
      > "out/_sched_${tag}.log" 2>&1
  echo "[exit=$?] $tag"
}
run () {     # 先清空 out_dir
  local tag="$1" out="$2"; shift 2
  rm -rf "$out"; _exec "$tag" "$out" "$@"
}
run_keep () {  # 保留 out_dir（续训用）
  local tag="$1" out="$2"; shift 2
  _exec "$tag" "$out" "$@"
}

RESUME_DIR=out/_sched_wsd_resume
rm -rf out/_sched_cosine out/_sched_wsd "$RESUME_DIR"

run      cosine    out/_sched_cosine --schedule=cosine
run      wsd_whole out/_sched_wsd    --schedule=wsd
# 第 3 组：WSD 跑 60 步存档 → 从存档续训到 150 步
run      wsd_seg1  "$RESUME_DIR" --schedule=wsd --max_iters=60 \
    --eval_interval=60 --always_save_checkpoint=true
run_keep wsd_seg2  "$RESUME_DIR" --schedule=wsd --init_from=resume \
    --max_iters=150 --eval_interval=100000

echo
echo "════════ 对照结果 ════════"
"$PY" - <<'PYEOF'
import csv, os, sys
sys.path.insert(0, '.')
from training.schedules import lr_at

def lr_curve(path):
    f = os.path.join(path, 'train_loss_window.csv')
    if not os.path.exists(f):
        return None
    rows = list(csv.DictReader(open(f, encoding='utf-8')))
    lr = {int(r['step']): float(r['lr']) for r in rows if float(r['lr']) > 0}
    ls = {int(r['step']): float(r['window_mean']) for r in rows}
    return lr, ls

KW = dict(learning_rate=3e-4, min_lr=1e-4, warmup_iters=10,
          lr_decay_iters=150, stable_frac=0.6)
cos, wsd, res = (lr_curve('out/_sched_cosine'),
                 lr_curve('out/_sched_wsd'),
                 lr_curve('out/_sched_wsd_resume'))
if not all((cos, wsd, res)):
    print("⚠ 有组缺产物：", [n for n, v in (('cosine', cos), ('wsd', wsd), ('resume', res)) if not v])
    raise SystemExit(1)

steps = sorted(set(cos[0]) | set(wsd[0]) | set(res[0]))
print(f"{'step':>6} {'cosine':>11} {'wsd(整跑)':>11} {'wsd(续训)':>11} {'lr_at(wsd)':>11} {'一致?':>7}")
bad = 0
for s in steps:
    exp = lr_at(s, schedule='wsd', **KW)
    w, r = wsd[0].get(s), res[0].get(s)
    ok = (w is None or abs(w - exp) < 1e-12) and (r is None or abs(r - exp) < 1e-12)
    bad += 0 if ok else 1
    fmt = lambda v: f"{v:.3e}" if v is not None else "-"
    print(f"{s:>6} {fmt(cos[0].get(s)):>11} {fmt(w):>11} {fmt(r):>11} {exp:>11.3e} {'OK' if ok else '✗':>7}")

common = sorted(set(wsd[0]) & set(res[0]))
print()
if common:
    dlr = max(abs(wsd[0][s] - res[0][s]) for s in common)
    dls = max(abs(wsd[1][s] - res[1][s]) for s in common)
    print(f"续训 vs 整跑：{len(common)} 个公共点，最大 LR 差 {dlr:.2e}，最大窗口 loss 差 {dls:.4f}")
    print("→ LR 完全一致 ✓" if dlr == 0 else "→ ⚠ LR 不一致")
    dls_note = "→ loss 连续（差 <0.3 属单步噪声）✓" if dls < 0.3 else "→ ⚠ loss 不连续"
    print(dls_note)
else:
    print("⚠ 无公共步点，无法比较续训")
print(f"\n与 lr_at 不符的点数：{bad}（应为 0）")
for name, (lrs, _) in (('cosine', cos), ('wsd', wsd), ('wsd续训', res)):
    f, l = min(lrs), max(lrs)
    print(f"{name:>8}: step {f}..{l}  lr {lrs[f]:.3e} → {lrs[l]:.3e}")
PYEOF
