"""A/B 结果对比：读取 4 个 arm 的 results.csv，输出对比表 + 叠加 loss 曲线。

用法：.venv/bin/python training/compare_ab.py
输出：out/ab_compare.csv（对比表）、out/ab_compare.png（train/val 曲线叠加）、终端摘要。
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARMS = [("base", "基线（默认架构）"),
        ("qk",   "QK-Norm"),
        ("z",    "Router Z-Loss"),
        ("qkz",  "QK-Norm + Z-Loss")]
OUT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")

def load_results(arm):
    path = os.path.join(OUT_ROOT, f"chinese-data2-ab-{arm}", "results.csv")
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items() if k != "step"})
            rows[-1]["step"] = int(float(r["step"]))
    return rows

def main():
    series = {}
    for arm, label in ARMS:
        rows = load_results(arm)
        series[arm] = (label, rows)
        print(f"{arm:<5} {label:<20} eval点数={len(rows)}")

    # ---- 对比表 ----
    table_rows = []
    for arm, label in ARMS:
        rows = series[arm][1]
        last = rows[-1]
        best_val = min(r["val/loss"] for r in rows)
        first = rows[0]
        table_rows.append({
            "arm": arm, "label": label,
            "train@0": first["train/loss"], "val@0": first["val/loss"],
            "train@last": last["train/loss"], "val@last": last["val/loss"],
            "fin_step": last["step"],
            "best_val": best_val,
            "delta_train": last["train/loss"] - first["train/loss"],
            "delta_val": best_val - first["val/loss"],
        })
    # 相对 base 的收获（val 终值 / best）
    base_last = next(t for t in table_rows if t["arm"] == "base")["val@last"]
    base_best = next(t for t in table_rows if t["arm"] == "base")["best_val"]
    for t in table_rows:
        t["val@last - base"] = t["val@last"] - base_last
        t["best_val - base"] = t["best_val"] - base_best

    out_csv = os.path.join(OUT_ROOT, "ab_compare.csv")
    fields = ["arm", "label", "train@0", "train@last", "val@0", "val@last",
              "best_val", "fin_step", "val@last - base", "best_val - base",
              "delta_train", "delta_val"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(table_rows)

    # ---- 终端摘要 ----
    print("\n========== A/B 对比 (1500 步) ==========")
    print(f"{'arm':<6}{'label':<20}{'train@0':>9}{'train@1500':>11}{'val@1500':>10}"
          f"{'best_val':>10}{'Δval终(best)-base':>18}")
    for t in table_rows:
        print(f"{t['arm']:<6}{t['label']:<20}{t['train@0']:>9.3f}{t['train@last']:>11.3f}"
              f"{t['val@last']:>10.3f}{t['best_val']:>10.3f}"
              f"{t['val@last']-base_last:>+10.3f} ({t['best_val']-base_best:+.3f})")

    # ---- 叠加曲线 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        for _font in ("Noto Sans CJK SC", "Source Han Sans CN", "WenQuanYi Zen Hei",
                      "Microsoft YaHei", "SimHei"):
            try:
                font_manager.findfont(_font, fallback_to_default=False)
                plt.rcParams["font.family"] = _font
                break
            except Exception:
                continue
        plt.rcParams["axes.unicode_minus"] = False
        colors = {"base": "#333333", "qk": "#c0392b", "z": "#2471a3", "qkz": "#1e8449"}
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for arm, (label, rows) in series.items():
            steps = [r["step"] for r in rows]
            axes[0].plot(steps, [r["train/loss"] for r in rows], color=colors[arm],
                         marker="o", ms=3, label=label)
            axes[1].plot(steps, [r["val/loss"] for r in rows], color=colors[arm],
                         marker="o", ms=3, label=label)
        for ax, title in zip(axes, ("train loss", "val loss")):
            ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_title(title)
            ax.legend(); ax.grid(alpha=0.3)
        fig.suptitle("QK-Norm / Router Z-Loss A/B · 1500 步（val 越低越好）")
        fig.tight_layout()
        out_png = os.path.join(OUT_ROOT, "ab_compare.png")
        fig.savefig(out_png, dpi=130)
        print(f"\n已生成：{out_csv}\n       {out_png}")
    except Exception as e:
        print(f"画图失败（不影响表格）：{e}")

if __name__ == "__main__":
    main()