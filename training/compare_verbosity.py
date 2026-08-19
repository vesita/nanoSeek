"""解码端"喋喋不休"治理对比：同一模型、同一 prompt/seed，不同采样策略逐条对照。

策略：
  1. 现状     ：顶格 200 token、惩罚 1.2、不截断（治理前的行为）
  2. 治理     ：≤120 + 惩罚 1.6 + 轮次截断 + EOS 停止 + 句末收尾
  3. EOS 探测 ：原始生成（不截断），token 级检测「模型自己吐不吐 <eos>」——
              验证 turn-level EOS 训练是否让模型学会"话说完→吐终止符"
                （eos_hits = 6 个 prompt 里吐了 <eos> 的个数，avg_eos_pos = 吐的位置，
                 位置越靠前 = 模型越早知道收尾）

用法：.venv/bin/python training/compare_verbosity.py --dirs out/... out/...
输出：终端指标表 + dev-notes/sample_verbosity.md（完整逐条样本）
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from tokenizers import Tokenizer

from inference.scripts.sample_py import build_model_from_checkpoint, generate, generate_ids
from inference.scripts.eval_dialogue import (DIALOGUE_PROMPTS, SEED,
                                             ngram_repetition, whitespace_ratio,
                                             distinct_n, dialogue_turn_structure, _encode_len)

POLICIES = [
    ("现状（顶格200·惩罚1.2·不停）",
     dict(max_new_tokens=200, temperature=0.8, top_k=200, repeat_penalty=1.2,
          stop_on_turn=False, stop_on_eos=False, clip_at_sentence=False, probe=False)),
    ("治理（≤120·惩罚1.6·轮次截断·EOS·句末收尾）",
     dict(max_new_tokens=120, temperature=0.8, top_k=200, repeat_penalty=1.6,
          stop_on_turn=True, stop_on_eos=True, clip_at_sentence=True, probe=False)),
    ("EOS 学习探测（原始生成·看模型自己吐不吐 <eos>）",
     dict(max_new_tokens=200, temperature=0.8, top_k=200, repeat_penalty=1.2,
          stop_on_turn=False, stop_on_eos=False, clip_at_sentence=False, probe=True)),
]


def run_policy(model, tok, kwargs):
    """在 6 个标准 prompt 上按该策略采样，返回 (metrics, samples)。"""
    # 必须复制 kwargs 再弹 probe：POLICIES 是全局共享的 dict，直接 pop 会污染
    # 下一个模型（第二个模型的探测策略会拿到 probe=False → EOS 永远 0/6）。
    kw = dict(kwargs)
    probe = kw.pop("probe", False)
    metrics = {"len": 0.0, "rep2": 0.0, "rep3": 0.0, "rep4": 0.0,
               "ws": 0.0, "d1": 0.0, "d2": 0.0, "turns": 0.0,
               "stopped_early": 0, "eos_hits": 0, "avg_eos_pos": 0.0}
    samples = []
    n = len(DIALOGUE_PROMPTS)
    prompt_lens = [len(tok.encode(p).ids) for p in DIALOGUE_PROMPTS]
    for p, plen in zip(DIALOGUE_PROMPTS, prompt_lens):
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        if probe:
            ids, eos_pos = generate_ids(model, tok, p, **kw)
            text = tok.decode(ids[plen:])
            if eos_pos >= 0:
                metrics["eos_hits"] += 1
                metrics["avg_eos_pos"] += eos_pos
        else:
            gen = generate(model, tok, p, **kw)
            text = gen[len(tok.decode(tok.encode(p).ids)):] if gen.startswith(
                tok.decode(tok.encode(p).ids)) else gen
        samples.append(text)
        metrics["len"] += _encode_len(tok, text)
        metrics["rep2"] += ngram_repetition(text, 2)
        metrics["rep3"] += ngram_repetition(text, 3)
        metrics["rep4"] += ngram_repetition(text, 4)
        metrics["ws"] += whitespace_ratio(text)
        metrics["d1"] += distinct_n(text, 1)
        metrics["d2"] += distinct_n(text, 2)
        metrics["turns"] += dialogue_turn_structure(text)["turns"]
        if not probe and _encode_len(tok, text) < kwargs["max_new_tokens"]:
            metrics["stopped_early"] += 1
    for k in ("len", "rep2", "rep3", "rep4", "ws", "d1", "d2", "turns"):
        metrics[k] = round(metrics[k] / n, 4)
    if probe and metrics["eos_hits"]:
        metrics["avg_eos_pos"] = round(metrics["avg_eos_pos"] / metrics["eos_hits"], 1)
    return metrics, samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=["out/chinese-data2-ab-qkz"],
                    help="要对比的模型目录（默认 qkz 非 EOS 版 1500 步）")
    a = ap.parse_args()

    tok = Tokenizer.from_file("data/chinese/tokenizer.json")
    print("=" * 80)
    print("喋喋不休治理对比（同 6 prompt、同 seed 1337；EOS 探测 = token 级）")
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
            extra = f"| EOS 自吐 {m['eos_hits']}/6" + (f"（平均位置 {m['avg_eos_pos']} token）" if m['eos_hits'] else "")
            print(f"\n  [{name}]")
            print(f"   avg_len={m['len']} rep2={m['rep2']} rep3={m['rep3']} rep4={m['rep4']} "
                  f"ws={m['ws']} d1={m['d1']} d2={m['d2']} turns={m['turns']} "
                  f"| 提前停止 {m['stopped_early']}/6 {extra}")
        # 摘要
        m0 = results[0][1]; m1 = results[1][1]; m2 = results[2][1]
        print(f"\n  摘要：avg_len {m0['len']}→{m1['len']} | rep3 {m0['rep3']}→{m1['rep3']} | "
              f"d2 {m0['d2']}→{m1['d2']} | EOS 自吐 {m2['eos_hits']}/6（位置 {m2['avg_eos_pos']}）")

        # 完整样本落盘
        lines = [f"# 喋喋不休治理对比：{d}\n",
                 f"同一模型、同 6 prompt、同 seed {SEED}，仅采样策略不同\n"]
        for i, p in enumerate(DIALOGUE_PROMPTS):
            lines.append(f"\n---\n\n### Prompt {i+1}: `{p}`\n")
            for name, m, samples in results:
                lines.append(f"\n**{name}**（len={m['len']} rep3={m['rep3']} "
                             f"turns={m['turns']} EOS自吐={m['eos_hits']}/6）\n")
                lines.append(f"\n> {samples[i]}\n")
        out = ROOT / "dev-notes" / "sample_verbosity.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"完整样本已写入 {out}")


if __name__ == "__main__":
    main()