"""采样对比：4 个 A/B 臂（+ 生产参考 chinese-data2）用同一组 prompt/seed 生成对话，
输出客观指标表 + 完整样本（落盘 markdown），供逐条对比「不同效果」。

用法：.venv/bin/python training/compare_sampling.py
输出：终端指标表 + dev-notes/sample_compare.md（全部样本）
"""
import sys
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from tokenizers import Tokenizer

from inference.scripts.sample_py import build_model_from_checkpoint, generate
from inference.scripts.eval_dialogue import (DIALOGUE_PROMPTS, MAX_NEW_TOKENS, TEMPERATURE,
                                             TOP_K, REPEAT_PENALTY, SEED, evaluate_one)

MODELS = [
    ("out/chinese-data2-ab-base", "base 基线"),
    ("out/chinese-data2-ab-qk",   "qk   QK-Norm"),
    ("out/chinese-data2-ab-z",    "z    Router Z-Loss"),
    ("out/chinese-data2-ab-qkz",  "qkz  QK+Z 都开"),
    ("out/chinese-data2",         "prod 生产参考 (chinese-data2)"),
]

def main():
    tok = Tokenizer.from_file("data/chinese/tokenizer.json")
    results = []
    for d, label in MODELS:
        if not (Path(d) / "best.pt").exists():
            print(f"⚠ 跳过（无 best.pt）: {d}")
            continue
        try:
            model, ckpt = build_model_from_checkpoint(d)  # sample_py.generate 用 CPU 张量，模型保持 CPU
        except Exception as e:
            print(f"⚠ 跳过（加载失败）: {d} → {type(e).__name__}: {str(e)[:100]}")
            continue
        r = evaluate_one(model, tok, d)
        r["label"] = label
        results.append(r)
        print(f"=== {label} ({d}) val={ckpt['best_val_loss'].item() if hasattr(ckpt['best_val_loss'],'item') else ckpt['best_val_loss']:.4f} "
              f"iter={ckpt['iter_num']} ===")
        print(f"  avg_len={r['avg_len_tokens']} rep2={r['rep2']} rep3={r['rep3']} rep4={r['rep4']} "
              f"ws={r['ws_ratio']} d1={r['distinct1']} d2={r['distinct2']} turns={r['turn_structure']}")
        del model
        torch.cuda.empty_cache()

    # ---- 汇总表 ----
    print("\n\n======== 采样智能度对比（同 6 prompt × seed 1337，temp 0.8 / top-k 200 / rep-pen 1.2） ========")
    print(f"{'模型':<28} | {'val@1500':>8} | {'len':>5} | {'rep2':>6} | {'rep3':>6} | {'rep4':>6} | "
          f"{'ws%':>5} | {'d1':>6} | {'d2':>6} | {'turns':>5}")
    print("-" * 108)
    for r in results:
        val = "—"
        print(f"{r['label']:<28} | {val:>8} | {r['avg_len_tokens']:>5} | {r['rep2']:>6} | {r['rep3']:>6} | "
              f"{r['rep4']:>6} | {int(r['ws_ratio']*100):>4}% | {r['distinct1']:>6} | "
              f"{r['distinct2']:>6} | {r['turn_structure']:>5}")
    print("\nrepN=字符 n-gram 重复率(越低越好) | ws%=空白占比(越低越好) | d1/d2=多样性(越高越好) | turns=轮次结构(越高越好)")

    # ---- 完整样本落盘（可读 markdown） ----
    lines = ["# 采样对比：QK-Norm / Router Z-Loss A/B（同 prompt、同 seed，逐条对照）\n",
             f"采样参数：temperature={TEMPERATURE} top_k={TOP_K} repeat_penalty={REPEAT_PENALTY} "
             f"max_new_tokens={MAX_NEW_TOKENS} seed={SEED}\n"]
    for i, p in enumerate(DIALOGUE_PROMPTS):
        lines.append(f"\n---\n\n### Prompt {i+1}: `{p}`\n")
        for r in results:
            lines.append(f"\n**{r['label']}**（rep3={r['rep3']} turns={r['turn_structure']}）\n")
            lines.append(f"\n> {r['samples'][i]}\n")
    out = ROOT / "dev-notes" / "sample_compare.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n完整样本已写入 {out}")

if __name__ == "__main__":
    main()