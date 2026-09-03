#!/usr/bin/env python3
"""探查 v3 坍缩: 用 out/rl_cont_v3/best.pt 自采样, 量化空回复率与数学首位数准确率。
用法: HSA_ENABLE_SDMA=0 HSA_OVERRIDE_GFX_VERSION=10.3.0 uv run python dev_scripts/probe_v3.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import torch
from inference.scripts.sample_py import build_model_from_checkpoint, load_tokenizer
from training.rl.sampler import sample_candidates_batch
from training.rl.reward import RewardEngine

CKPT = "out/rl_cont_v5"
DEV = "cuda"
ENGINE = RewardEngine()

# (中文, kind, 数学正确结果)
PROMPTS = [
    ("3加4等于几？", "arith", 7),
    ("15减7等于多少？", "arith", 8),
    ("8乘3等于多少？", "arith", 24),
    ("24除以6等于几？", "arith", 4),
    ("12加9等于？", "arith", 21),
    ("20减13等于多少？", "arith", 7),
    ("今天天气怎么样？", "general", None),
    ("你觉得如何？", "sentiment", None),
]


def decode(rid, tok):
    raw = tok.decode(rid)
    has_cont = "<cont>" in raw
    has_eos = "<eos>" in raw
    body = raw.replace("<eos>", "").replace("<cont>", "").strip()
    return raw, body, has_cont, has_eos


def main():
    model, ckpt = build_model_from_checkpoint(CKPT, device=DEV)
    tok = load_tokenizer(ckpt)
    eos_id, cont_id = 117, 119
    model.eval()

    print(f"=== 探查 {CKPT} ===")
    totals = {"n": 0, "empty": 0, "len1": 0, "arith_n": 0, "arith_correct": 0}
    for ptext, kind, ans in PROMPTS:
        pids = tok.encode(ptext).ids
        replies = sample_candidates_batch(
            model, tok, pids, eos_id, group_size=8, max_new_tokens=55,
            temperature=1.0, top_k=200, repeat_penalty=1.4, device=DEV, cont_id=cont_id,
        )
        print(f"\n--- {ptext} ---")
        for i, rid in enumerate(replies):
            raw, body, has_cont, has_eos = decode(rid, tok)
            cl = len(body)
            totals["n"] += 1
            tag = ("[cont]" if has_cont else "") + ("[eos]" if has_eos else "")
            if cl == 0:
                totals["empty"] += 1
            elif cl <= 1:
                totals["len1"] += 1
            _, s, _ = ENGINE.evaluate_reply(
                ptext, raw, rid, eos_id, kind=kind, cont_id=cont_id)
            extra = ""
            if kind == "arith":
                totals["arith_n"] += 1
                m = re.search(r"(\d{1,3})", body)
                first = int(m.group(1)) if m else None
                hit = (first == ans)
                if hit:
                    totals["arith_correct"] += 1
                extra = f" first={first} hit={hit}"
            print(f"  {i}: len={cl} {tag} score={s:6.2f}{extra}  {body[:36]!r}")

    n = totals["n"]
    print(f"\n=== 统计 (n={n}) ===")
    print(f"  空回复: {totals['empty']} ({100*totals['empty']/n:.1f}%)")
    print(f"  长度<=1: {totals['len1']} ({100*totals['len1']/n:.1f}%)")
    if totals["arith_n"]:
        print(f"  数学首位命中: {totals['arith_correct']}/{totals['arith_n']} "
              f"({100*totals['arith_correct']/totals['arith_n']:.1f}%)")


if __name__ == "__main__":
    main()