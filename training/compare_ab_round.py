"""通用 A/B 对比：比较两个实验臂的 results.csv，输出 val/train 轨迹表 + 汇总。

用法：
    .venv/bin/python training/compare_ab_round.py out/chinese-data2-ab1-sdpa out/chinese-data2-lm
参数：两个 out/ 子目录（第一个 = 新臂，第二个 = 基线）。
输出：终端逐评估点 val/train 对比 + 终值/best 汇总 + 总训练时长（速度信号）。
"""
import csv
import os
import sys


def load_results(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def main():
    if len(sys.argv) != 3:
        print("用法: python training/compare_ab_round.py <新臂 out_dir> <基线 out_dir>")
        sys.exit(1)
    new_dir, base_dir = sys.argv[1:3]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    new_rows = load_results(os.path.join(root, new_dir, "results.csv"))
    base_rows = load_results(os.path.join(root, base_dir, "results.csv"))
    key = lambda r: int(float(r["step"]))
    new_rows.sort(key=key)
    base_rows.sort(key=key)

    print(f"{'step':>7} | {'train 新臂':>10} {'train 基线':>10} {'Δtrain':>8} | "
          f"{'val 新臂':>9} {'val 基线':>9} {'Δval':>8}")
    print("-" * 80)
    for nr, br in zip(new_rows, base_rows):
        ns, bs = key(nr), key(br)
        dtr = float(nr["train/loss"]) - float(br["train/loss"])
        dva = float(nr["val/loss"]) - float(br["val/loss"])
        print(f"{ns:>7} | {float(nr['train/loss']):>10.4f} {float(br['train/loss']):>10.4f} "
              f"{dtr:>+8.4f} | {float(nr['val/loss']):>9.4f} {float(br['val/loss']):>9.4f} {dva:>+8.4f}")

    n_last, b_last = new_rows[-1], base_rows[-1]
    n_best = min(float(r["val/loss"]) for r in new_rows)
    b_best = min(float(r["val/loss"]) for r in base_rows)
    print("\n========== 汇总 ==========")
    print(f"{'':<22}{'新臂':>12}{'基线':>12}{'Δ(新-基)':>12}")
    print(f"{'train@终':<22}{float(n_last['train/loss']):>12.4f}{float(b_last['train/loss']):>12.4f}"
          f"{float(n_last['train/loss'])-float(b_last['train/loss']):>+12.4f}")
    print(f"{'val@终':<22}{float(n_last['val/loss']):>12.4f}{float(b_last['val/loss']):>12.4f}"
          f"{float(n_last['val/loss'])-float(b_last['val/loss']):>+12.4f}")
    print(f"{'val@best':<22}{n_best:>12.4f}{b_best:>12.4f}{n_best-b_best:>+12.4f}")
    if "time" in n_last and "time" in b_last:
        print(f"{'总时长(秒)':<22}{float(n_last['time']):>12.1f}{float(b_last['time']):>12.1f}"
              f"{float(n_last['time'])-float(b_last['time']):>+12.1f}（负=新臂更快）")
    if "mfu" in n_last and "mfu" in b_last:
        print(f"{'MFU@终(%)':<22}{float(n_last['mfu'])*100:>12.2f}{float(b_last['mfu'])*100:>12.2f}")
    print(f"\n结论：val@终 Δ={float(n_last['val/loss'])-float(b_last['val/loss']):+.4f}；"
          f"val@best Δ={n_best-b_best:+.4f}（负=新臂更优）")


if __name__ == "__main__":
    main()
