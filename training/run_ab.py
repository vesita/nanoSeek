#!/usr/bin/env python3
"""通用 A/B 架构对比：同种子 / 同数据 / 同步数跑多个配置，输出对比表 + 叠加曲线。

探索新网络结构的核心工作流：一个基线配置 + 若干变体配置，所有 arm 用相同
seed（train.py 固定 1337）、相同数据集、相同步数训练，然后直接对比 val loss。

用法（从项目根目录）：

    uv run python cli.py ab \
        --base=training/config/test.yaml \
        --variant=training/config/test.yaml \
        --iters=300 \
        --dataset=synth --compile=false

    # --base 和 --variant 可以重复传（多臂对比）；--iters 默认 1500
    # 其余 --key=value 参数原样透传给每次训练（如 --dataset / --compile / --n_layer）

输出：out/ab_<时间戳>/ 下每个 arm 一个实验目录（含 results.csv / best.pt），
另生成 compare.csv（对比表）和 compare.png（train/val 曲线叠加）。
"""
import csv
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def parse_args(argv):
    """返回 (base_cfgs, variant_cfgs, iters, out_root, passthrough)。"""
    base, variants, iters, out_root = [], [], 1500, None
    passthrough = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--base="):
            base.append(a.split("=", 1)[1])
        elif a.startswith("--variant="):
            variants.append(a.split("=", 1)[1])
        elif a.startswith("--iters="):
            iters = int(a.split("=", 1)[1])
        elif a.startswith("--out="):
            out_root = a.split("=", 1)[1]
        else:
            passthrough.append(a)  # 其余原样透传给 train.py
        i += 1
    return base, variants, iters, out_root, passthrough


def arm_dir_name(cfg, idx, is_base):
    """从配置文件名推导 arm 目录名（保留，兼容外部调用）。"""
    name = os.path.splitext(os.path.basename(cfg))[0]
    return "base" if is_base else f"{name}{idx}"


def run_arm(cfg, arm_dir, iters, passthrough):
    """跑一次训练；返回 (ok, results_csv_path)。"""
    cfg = os.path.join(ROOT, cfg) if not os.path.isabs(cfg) else cfg
    if not os.path.exists(cfg):
        print(f"✗ 配置不存在：{cfg}")
        return False, None
    overrides = [
        f"--out_dir={arm_dir}",
        f"--max_iters={iters}",
        f"--lr_decay_iters={iters}",
        f"--warmup_iters={min(100, max(1, iters // 10))}",
        f"--eval_interval={max(1, iters // 20)}",
        "--enable_early_stop=False",   # 保证所有 arm 严格跑满 iters 步（公平对比）
        "--always_save_checkpoint=False",
        *passthrough,
    ]
    print(f"\n=== arm: {os.path.basename(arm_dir)} ← {cfg}")
    code = subprocess.call([sys.executable, os.path.join("training", "train.py"), cfg, *overrides],
                           cwd=ROOT)
    if code != 0:
        print(f"✗ arm {os.path.basename(arm_dir)} 训练失败（exit {code}）")
        return False, None
    return True, os.path.join(arm_dir, "results.csv")


def load_results(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                rows.append((int(float(r["step"])), float(r["train/loss"]), float(r["val/loss"])))
            except (KeyError, ValueError):
                continue
    return rows


def summarize(rows):
    if not rows:
        return None
    best = min((v for _, _, v in rows), default=float("nan"))
    final = rows[-1][2]
    return {"best_val": best, "final_val": final, "steps": rows[-1][0], "rows": rows}


def main(argv):
    base_cfgs, variant_cfgs, iters, out_root, passthrough = parse_args(argv)
    if not base_cfgs or not variant_cfgs:
        print("用法: cli.py ab --base=<config.yaml> --variant=<config.yaml> [--variant=...] "
              "[--iters=N] [--out=dir] [透传参数...]")
        return 2
    if any("=" not in a for a in passthrough):
        print("透传参数必须是 --key=value 形式（透传参数: %r）" % passthrough)
        return 2

    # arm 命名：base 用 "base"；variant 用配置文件名，重名时加序号
    arms = [("base", c) for c in base_cfgs]
    seen = {}
    for c in variant_cfgs:
        n = os.path.splitext(os.path.basename(c))[0]
        seen[n] = seen.get(n, 0) + 1
        arms.append((n if seen[n] == 1 else f"{n}{seen[n]}", c))
    ts = time.strftime("%m%d-%H%M%S")
    out_root = out_root or os.path.join("out", f"ab_{ts}")
    os.makedirs(out_root, exist_ok=True)
    print(f"A/B 对比输出目录：{out_root}（iters={iters}）")
    summaries, failures = {}, []
    for name, cfg in arms:
        arm_dir = os.path.join(out_root, name)
        ok, csv_path = run_arm(cfg, arm_dir, iters, passthrough)
        if ok:
            summaries[name] = summarize(load_results(csv_path))
        else:
            failures.append(name)

    # 对比表 + 曲线
    comp_path = os.path.join(out_root, "compare.csv")
    with open(comp_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "best_val", "final_val", "steps"])
        for name, s in summaries.items():
            if s:
                w.writerow([name, f"{s['best_val']:.4f}", f"{s['final_val']:.4f}", s["steps"]])
    print("\n================ A/B 对比结果 ================")
    print(f"{'arm':<16}{'best_val':>12}{'final_val':>12}{'Δbest':>10}  steps")
    base_s = summaries.get("base")
    for name, s in summaries.items():
        if not s:
            print(f"{name:<16}{'失败':>12}")
            continue
        delta = "" if base_s is None or name == "base" else f"{s['best_val'] - base_s['best_val']:+.4f}"
        print(f"{name:<16}{s['best_val']:>12.4f}{s['final_val']:>12.4f}{delta:>10}  {s['steps']}")
    if failures:
        print(f"失败 arm：{failures}")
    print(f"对比表：{comp_path}")

    # 叠加曲线（ASCII 图例，避免中文字体问题）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for name, s in summaries.items():
            if s:
                steps = [r[0] for r in s["rows"]]
                plt.plot(steps, [r[2] for r in s["rows"]], label=f"{name} (val)")
        plt.xlabel("step"); plt.ylabel("val loss"); plt.legend(); plt.grid(alpha=0.3)
        png = os.path.join(out_root, "compare.png")
        plt.savefig(png, dpi=120); plt.close()
        print(f"曲线图：{png}")
    except Exception as e:
        print(f"（画图失败，不影响结果：{e}）")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
