#!/usr/bin/env python3
"""nanoSeek 1500 轮课程总控. 每阶段后做「你好体检」退化门控, 退化即中断."""
import argparse, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from training.rl.curriculum_lib import CURRICULUM_STAGES, checkpoint_healthy

def run_stage(cfg, initial_ckpt, base_out="out/curriculum"):
    stage_id = cfg["stage"]
    out_dir = os.path.join(base_out, f"stage_{stage_id:02d}_{stage_id*100}")
    os.makedirs(out_dir, exist_ok=True)
    if stage_id == 1:
        ckpt_in = initial_ckpt
    else:
        prev_dir = os.path.join(base_out, f"stage_{stage_id-1:02d}_{(stage_id-1)*100}")
        ckpt_in = os.path.join(prev_dir, "best.pt")
        if not os.path.exists(ckpt_in):
            ckpt_in = initial_ckpt
    print("=" * 70)
    print(f"🔧 Stage {stage_id:02d}/{len(CURRICULUM_STAGES)} (第 {(stage_id-1)*100+1}-{stage_id*100} 轮) | {cfg['task'].upper()} | lr={cfg['lr']}")
    print(f"  ckpt_in: {ckpt_in} -> out: {out_dir}")
    print("=" * 70)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = os.environ.copy(); env["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"; env["PYTHONUNBUFFERED"] = "1"
    t0 = time.time(); sys.stdout.flush()
    if cfg["task"] == "grpo":
        cmd = [sys.executable, "training/rl/grpo_char.py", "--ckpt", ckpt_in, "--out", out_dir,
               "--steps", str(cfg["rounds"]), "--lr", cfg["lr"], "--group_size", "4",
               "--tau", "1.5", "--beta_kl", "0.4",
               "--temperature", "1.0", "--repeat_penalty", "1.4", "--div_weight", "1.5",
               "--shape", "exp", "--winsorize", "3.0", "--dyn_tau"]
    elif cfg["task"] == "memory":
        cmd = [sys.executable, "training/rl/memory_rl.py", "--ckpt", ckpt_in, "--out", out_dir,
               "--steps", str(cfg["rounds"]), "--lr", cfg["lr"], "--group_size", "4",
               "--tau", "1.5", "--beta_kl", "0.4"]
    elif cfg["task"] == "selfplay":
        cmd = [sys.executable, "training/rl/dual_chat_selfplay.py", "--ckpt", ckpt_in, "--out", out_dir,
               "--rounds", str(cfg["rounds"]), "--lr", cfg["lr"], "--tau", "1.5",
               "--beta_kl", "0.4", "--min_reward", "0.5"]
    else:
        raise ValueError(f"Unknown task: {cfg['task']}")
    res = subprocess.run(cmd, env=env, cwd=root)
    sys.stdout.flush()
    dt = time.time() - t0
    ckpt_out = os.path.join(out_dir, "best.pt")
    if res.returncode != 0 or not os.path.exists(ckpt_out):
        print(f"❌ Stage {stage_id} 未成功 (rc={res.returncode}, ckpt={os.path.exists(ckpt_out)})")
        return False
    # 阶段间退化门控
    healthy, report = checkpoint_healthy(ckpt_out)
    dt_s = f"{dt:.1f}s"
    if not healthy:
        print(f"⛔ Stage {stage_id} 检查点退化! {report} ({dt_s}) -> 中断流水线")
        return False
    print(f"✅ Stage {stage_id} 完成! {dt_s} | {report}")
    return True

def main():
    ap = argparse.ArgumentParser(description="nanoSeek 1500 轮课程总控")
    ap.add_argument("--stage", type=int, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--start", type=int, default=1, help="--all 模式下从第几个阶段开始续跑 (保留前面已训阶段)")
    ap.add_argument("--ckpt", default="out/eos_fix_1epoch/best.pt")
    ap.add_argument("--out", default="out/curriculum")
    args = ap.parse_args()
    if args.stage is not None:
        cfg = next((c for c in CURRICULUM_STAGES if c["stage"] == args.stage), None)
        if not cfg:
            print(f"错误: 未找到 Stage {args.stage}")
            return
        run_stage(cfg, args.ckpt, base_out=args.out)
    elif args.all:
        print(f"🌟 启动对话串联课程训练 ({len(CURRICULUM_STAGES)} 阶段, 从 Stage {args.start} 起)\n")
        for cfg in CURRICULUM_STAGES:
            if cfg["stage"] < args.start:
                print(f"  (跳过已完成的 Stage {cfg['stage']:02d})")
                continue
            if not run_stage(cfg, args.ckpt, base_out=args.out):
                print("流水线中断!")
                break
        print("\n🎉 对话课程训练收官!")
    else:
        print("请指定 --stage <1-12> 或 --all")
        for c in CURRICULUM_STAGES:
            print(f"  Stage {c['stage']:02d}: Phase {c['phase']} [{c['task']:8s}] {c['desc']} ({c['rounds']} 轮)")

if __name__ == "__main__":
    main()