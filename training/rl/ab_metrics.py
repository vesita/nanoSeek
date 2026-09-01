"""A/B 对比并统计训练过程退化次数 + 最终体检。用 stage_12_1200 各训100步。
用法: python ab_metrics.py <ckpt_path> <outdir> <shape> [--winsorize K] [--dyn_tau] [--steps N]
输出: 退化计数, 平均组均分, 最终体检
"""
import os, sys, subprocess, re, math
root = "/home/vesita/coding/my/nanoSeek"
ckpt = sys.argv[1]; outdir = sys.argv[2]; shape = sys.argv[3]
wins = sys.argv[sys.argv.index("--winsorize")+1] if "--winsorize" in sys.argv else "3.0"
dyn = "--dyn_tau" in sys.argv
steps = sys.argv[sys.argv.index("--steps")+1] if "--steps" in sys.argv else "100"
cmd = [".venv/bin/python", "training/rl/grpo_char.py", "--ckpt", ckpt, "--out", outdir,
       "--steps", steps, "--group_size", "4", "--lr", "8e-6", "--tau", "1.5",
       "--beta_kl", "0.4", "--temperature", "1.0", "--repeat_penalty", "1.4", "--div_weight", "1.5",
       "--shape", shape, "--winsorize", wins]
if dyn: cmd += ["--dyn_tau"]
env = dict(os.environ, **{"HSA_OVERRIDE_GFX_VERSION": "10.3.0", "PYTHONUNBUFFERED": "1"})
print("CMD:", " ".join(cmd))
r = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True)
log = r.stdout
deg = len(re.findall(r"组退化", log))
means = [float(x) for x in re.findall(r"组均分:\s+(-?[\d.]+)", log)]
mean_avg = sum(means)/len(means) if means else float('nan')
# 极端负优势异常
n_extreme = sum(1 for m in means if m < -50)
print(f"\n=== {shape}{' dyn' if dyn else ''} ===")
print(f"  退化次数: {deg}/{steps}")
print(f"  平均组均分: {mean_avg:.2f}")
print(f"  极端负优势(< -50)次数: {n_extreme}")
print(f"  EXIT: {r.returncode}")
# 体检
hc = subprocess.run([".venv/bin/python", "/tmp/run_health_base.py", os.path.join(outdir,"best.pt")],
                    cwd=root, env=env, capture_output=True, text=True)
print("  " + (hc.stdout.strip().splitlines()[-1] if hc.stdout else "体检失败"))
