"""解码端"喋喋不休"治理对比：同一模型、同一 prompt/seed，现状策略 vs 治理策略逐条对照。

治理策略（不动模型）：
  1. stop_on_turn  : 检测到 \n用户：/\n模型： 标签即截断（轮次边界=说话结束）
  2. stop_on_eos   : 采样到 <eos> 即停止
  3. clip_sentence : 预算用尽回退到最后一个句号/问号/叹号
  4. 收紧预算 max_new_tokens 200→120 + 加强 repeat-penalty 1.2→1.6

用法：.venv/bin/python training/compare_verbosity.py [--dirs out/... out/...]
输出：终端指标表 + dev-notes/sample_verbosity.md（完整逐条样本）
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from tokenizers import Tokenizer

from inference.scripts.sample_py import build_model_from_checkpoint, generate
from inference.scripts.eval_dialogue import (DIALOGUE_PROMPTS, SEED,
                                             ngram_repetition, whitespace_ratio,
                                             distinct_n, dialogue_turn_structure, _encode_len)

POLICIES = [
    ("现状（顶格200·惩罚1.2·不停）",
     dict(max_new_tokens=200, temperature=0.8, top_k=200, repeat_penalty=1.2,
          stop_on_turn=False, stop_on_eos=False, clip_at_sentence=False)),
    ("治理（≤120·惩罚1.6·轮次截断·EOS·句末收尾）",
     dict(max_new_tokens=120, temperature=0.8, top_k=200, repeat_penalty=1.6,
          stop_on_turn=True, stop_on_eos=True, clip_at_sentence=True)),
]


def run_policy(model, tok, kwargs):
    """在 6 个标准 prompt 上按该策略采样，返回 (metrics_dict, samples)。"""
    metrics = {"len": 0.0, "rep2": 0.0, "rep3": 0.0, "rep4": 0.0,
               "ws": 0.0, "d1": 0.0, "d2": 0.0, "turns": 0.0, "stopped_early": 0}
    samples = []
    n = len(DIALOGUE_PROMPTS)
    for p in DIALOGUE_PROMPTS:
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        gen = generate(model, tok, p, **kwargs)
        prompt_dec = tok.decode(tok.encode(p).ids)
        text = gen[len(prompt_dec):] if gen.startswith(prompt_dec) else gen
        samples.append(text)
        metrics["len"] += _encode_len(tok, text)
        metrics["rep2"] += ngram_repetition(text, 2)
        metrics["rep3"] += ngram_repetition(text, 3)
        metrics["rep4"] += ngram_repetition(text, 4)
        metrics["ws"] += whitespace_ratio(text)
        metrics["d1"] += distinct_n(text, 1)
        metrics["d2"] += distinct_n(text, 2)
        metrics["turns"] += dialogue_turn_structure(text)["turns"]
        if _encode_len(tok, text) < kwargs["max_new_tokens"]:   # 提前停了（轮次/EOS 截断）
            metrics["stopped_early"] += 1
    for k in ("len", "rep2", "rep3", "rep4", "ws", "d1", "d2", "turns"):
        metrics[k] = round(metrics[k] / n, 4)
    return metrics, samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=["out/chinese-data2-ab-qkz-ft500"],
                    help="要对比的模型目录（默认 qkz 续训后，val 0.4905 最优）")
    a = ap.parse_args()

    tok = Tokenizer.from_file("data/chinese/tokenizer.json")
    print("=" * 78)
    print("解码端喋喋不休治理对比（同一模型、同 6 prompt、同 seed 1337）")
    for d in a.dirs:
        if not (Path(d) / "best.pt").exists():
            print(f"⚠ 跳过（无 best.pt）: {d}")
            continue
        model, ckpt = build_model_from_checkpoint(d)
        print(f"\n模型: {d}（val={ckpt['best_val_loss'].item() if hasattr(ckpt['best_val_loss'],'item') else ckpt['best_val_loss']:.4f} @ {ckpt['iter_num']} 步）")
        results = []
        for name, kw in POLICIES:
            m, samples = run_policy(model, tok, kw)
            results.append((name, m, samples))
            print(f"\n  [{name}]")
            print(f"   avg_len={m['len']} rep2={m['rep2']} rep3={m['rep3']} rep4={m['rep4']} "
                  f"ws={m['ws']} d1={m['d1']} d2={m['d2']} turns={m['turns']} "
                  f"| 提前停止 {m['stopped_early']}/6 prompt")
        # 摘要行
        n0, m0, _ = results[0]
        n1, m1, _ = results[1]
        print(f"\n  摘要：avg_len {m0['len']}→{m1['len']} token | rep3 {m0['rep3']}→{m1['rep3']} | "
              f"d2 {m0['d2']}→{m1['d2']} | 提前停止 {m0['stopped_early']}→{m1['stopped_early']}/6")

        # 完整样本落盘
        lines = [f"# 解码端喋喋不休治理对比：{d}\n",
                 f"同一模型、同 6 prompt、同 seed {SEED}，仅采样策略不同\n"]
        for i, p in enumerate(DIALOGUE_PROMPTS):
            lines.append(f"\n---\n\n### Prompt {i+1}: `{p}`\n")
            for name, m, samples in results:
                lines.append(f"\n**{name}**（len={m['len']} rep3={m['rep3']} turns={m['turns']}）\n")
                lines.append(f"\n> {samples[i]}\n")
        out = ROOT / "dev-notes" / "sample_verbosity.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"完整样本已写入 {out}")


if __name__ == "__main__":
    main()