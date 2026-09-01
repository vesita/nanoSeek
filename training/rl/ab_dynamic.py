"""A/B 对比: 不同塑形形态+鲁棒化。在 stage_12_1200 上各训100步, 体检对比。
用法: python ab_dynamic.py <ckpt_path> <outdir> --shape {exp|log|tanh} [--winsorize K] [--dyn_tau] [--steps N]
"""
import os, sys, subprocess
root = "/home/vesita/coding/my/nanoSeek"
ckpt = sys.argv[1]; outdir = sys.argv[2]
shape = sys.argv[sys.argv.index("--shape")+1] if "--shape" in sys.argv else "exp"
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
r = subprocess.run(cmd, cwd=root, env=env)
print("EXIT:", r.returncode)
