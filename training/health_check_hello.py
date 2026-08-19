#!/usr/bin/env python3
"""你好体检：以「用户：你好\n模型：」为固定提示词，检验对话自然度/重复度/EOS 自吐。

用法：
    .venv/bin/python training/health_check_hello.py --dirs out/chinese-data2-reb
    .venv/bin/python training/health_check_hello.py --dirs out/a out/b out/c

输出（每模型 10 个 seed，temp 0.8 / topk 200 / rep 1.2，原始生成不截断）：
    EOS 自吐 n/10 + 平均位置；平均 len；rep2/3/4；d1/d2；ws；turns
    + 前 3 个 seed 的原始文本（自然度人工判读）

兼容性：旧 checkpoint（无 use_csa_fused_qkv 键，独立 QKV 布局）自动按 False 加载。
"""
import sys, re, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from collections import Counter
from tokenizers import Tokenizer
from model import GPTConfig, GPT
from inference.scripts.sample_py import generate_ids

PROMPT = "用户：你好\n模型："
SEEDS = range(10)


def load(dirpath):
    ck = torch.load(f"{dirpath}/best.pt", map_location="cpu")
    args = dict(ck["model_args"])
    if "use_csa_fused_qkv" not in args:   # 旧 checkpoint → 独立 QKV 布局
        args["use_csa_fused_qkv"] = False
    m = GPT(GPTConfig(**args))
    state = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
             for k, v in ck["model"].items()}
    m.load_state_dict(state)
    m.eval()
    return m, ck


def ngram_rep(s, n):
    seq = re.sub(r"\s+", "", s)
    if len(seq) < 2 * n:
        return 0.0
    g = [seq[i:i + n] for i in range(len(seq) - n + 1)]
    c = Counter(g)
    return sum(1 for x in g if c[x] > 1) / len(g)


def distinct_n(s, n):
    seq = re.sub(r"\s+", "", s)
    if len(seq) < n:
        return 0.0
    g = [seq[i:i + n] for i in range(len(seq) - n + 1)]
    return len(set(g)) / len(g)


def ws_ratio(s):
    return sum(1 for ch in s if ch.isspace()) / max(len(s), 1)


def turns(s):
    return max(len(re.findall(r"用户[:：]", s)), len(re.findall(r"模型[:：]", s)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", required=True)
    ap.add_argument("--samples", type=int, default=3,
                    help="打印前几个 seed 的原始文本（默认 3）")
    a = ap.parse_args()

    tok = Tokenizer.from_file("data/chinese/tokenizer.json")
    plen = len(tok.encode(PROMPT).ids)

    for d in a.dirs:
        model, ck = load(d)
        rows = []
        for s in SEEDS:
            torch.manual_seed(s); torch.cuda.manual_seed(s)
            ids, eos_pos = generate_ids(model, tok, PROMPT, 200, 0.8, 200, 1.2,
                                        stop_on_turn=False, stop_on_eos=False,
                                        clip_at_sentence=False)
            text = tok.decode(ids[plen:])
            rows.append(dict(seed=s, eos=eos_pos, len=len(ids) - plen, text=text,
                             rep2=ngram_rep(text, 2), rep3=ngram_rep(text, 3),
                             rep4=ngram_rep(text, 4), d1=distinct_n(text, 1),
                             d2=distinct_n(text, 2), ws=ws_ratio(text),
                             turns=turns(text)))
        hits = sum(1 for r in rows if r["eos"] >= 0)
        avg_pos = sum(r["eos"] for r in rows if r["eos"] >= 0) / max(hits, 1)
        avg = lambda k: sum(r[k] for r in rows) / len(rows)
        print("=" * 74)
        print(f"{d}（val {float(ck['best_val_loss']):.4f} @1500）你好体检 · {len(rows)} seed")
        print("=" * 74)
        print(f"EOS 自吐 {hits}/{len(rows)}（平均位置 {avg_pos:.1f} token） | "
              f"平均 len {avg('len'):.1f} | 续轮 turns {sum(1 for r in rows if r['turns'] >= 2)}/{len(rows)}")
        print(f"rep2 {avg('rep2'):.3f} | rep3 {avg('rep3'):.4f} | rep4 {avg('rep4'):.4f} | "
              f"d1 {avg('d1'):.3f} | d2 {avg('d2'):.3f} | ws {avg('ws'):.3f}")
        for r in rows[:a.samples]:
            print(f"\n--- seed {r['seed']} (len={r['len']}, EOS@{r['eos']}, turns={r['turns']}) ---")
            print(r["text"][:280])
        print()


if __name__ == "__main__":
    main()
