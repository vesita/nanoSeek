#!/usr/bin/env python3
"""训练后 RL 数学/对话健康度验证 probe。

用法:
  HSA_ENABLE_SDMA=0 HSA_OVERRIDE_GFX_VERSION=10.3.0 PYTHONUNBUFFERED=1 \\
    .venv/bin/python dev_scripts/rl_probe_math.py --ckpt_dir out/rl_math_v1 --label rl_math_v1

对指定 checkpoint:
  1. 数学: 用 arith_rules.toml 配置生成 K 条各运算符/噪音写法的题, 对每条采样 G 个候选,
     用与奖励引擎同逻辑的 extract_answer_number 判"组内最优命中率"(任一候选答对即命中)。
  2. 对话: 对若干自然 prompt 采样, 统计 均长/EOS-cont 自控/空回复率/汉字占比/多样性,
     判断是否坍缩。

基座对照: 也传 --base_dir out/cont_v1_1epoch, 可并排对比提升。
"""
import argparse
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from inference.scripts.sample_py import build_model_from_checkpoint, load_tokenizer
from training.rl.sampler import sample_candidates_batch
from training.rl.reward import arith_gen

# 各运算符构造一组配置驱动数学题, 保证四则全覆盖 + 噪音写法多样
OPS_CNT = 4          # 每个运算符生成几条
DIALOGUE = [
    "用户：你好！\n模型：",
    "用户：你今天心情怎么样？\n模型：",
    "用户：如果有机会出去旅游，你最想去哪？\n模型：",
    "用户：给你自己起个名字吧！\n模型：",
    "用户：你平时性格是什么样？用一个词形容自己\n模型：",
    "用户：今天攻克了一个难题，心情太棒了！\n模型：",
    "用户：如果有超能力你最想要什么？\n模型：",
    "用户：你觉得保持积极乐观的秘诀是什么？\n模型：",
]


def decode(rid, tok):
    raw = tok.decode(rid)
    has_cont = "<cont>" in raw
    has_eos = "<eos>" in raw
    body = raw.replace("<eos>", "").replace("<cont>", "").strip()
    return raw, body, has_cont, has_eos


def probe_ckpt(ckpt_dir, label, device, seed=0):
    model, ckpt = build_model_from_checkpoint(ckpt_dir, device=device)
    tok = load_tokenizer(ckpt)
    eos_id, cont_id = 117, 119
    model.eval()

    print(f"\n{'='*70}\n=== {label} ({ckpt_dir}) ===\n{'='*70}")

    # ---- 数学题池 (配置驱动) ----
    # 从配置读各 op 生成提示词需要底层函数; 这里直接 gen 一批并按其分布统计即可,
    # 但为保各运算符均衡覆盖, 用固定 seed 大批生成再按比例校验。
    arith_pool = arith_gen.gen_prompts(count=80, seed=seed)
    # 运算符标签不可直接从 prompt 提, 但可从 已知配置的正则取; 简化: 全池混测。
    op_hits = {"add": [0, 0], "sub": [0, 0], "mul": [0, 0], "div": [0, 0]}
    total = [0, 0]   # [命中, 总数]
    for ptext, kw in arith_pool:
        ar = arith_gen.extract_arith(ptext)   # (a, opname, b, result)
        opname = ar[1] if ar else "?"
        pids = tok.encode(ptext).ids
        replies = sample_candidates_batch(
            model, tok, pids, eos_id, group_size=4, max_new_tokens=55,
            temperature=1.0, top_k=200, repeat_penalty=1.4, device=device, cont_id=cont_id)
        hit = False
        for rid in replies:
            raw, body, _, _ = decode(rid, tok)
            if ar and arith_gen.extract_answer_number(body) == ar[3]:
                hit = True
                break
        total[1] += 1
        if hit:
            total[0] += 1
            if opname in op_hits:
                op_hits[opname][0] += 1
        if opname in op_hits:
            op_hits[opname][1] += 1
        if ar:
            ans_ar = f"(答案{ar[3]})"
        else:
            ans_ar = ""
        print(f"  arith: {'✓' if hit else '✗'} {ptext}{ans_ar}")

    n_math = total[1]
    print(f"\n  [数学] 组级命中 {total[0]}/{n_math} = "
          f"{100*total[0]/n_math:.1f}%" if n_math else "  无数学题")
    for op, (h, c) in op_hits.items():
        if c:
            print(f"      {op:5s}: {h}/{c}")

    # ---- 对话健康度 ----
    n = 0
    empty, len1, lens, eos_c, cont_c, texts = 0, 0, [], 0, 0, []
    for ptext in DIALOGUE:
        pids = tok.encode(ptext).ids
        replies = sample_candidates_batch(
            model, tok, pids, eos_id, group_size=2, max_new_tokens=55,
            temperature=1.0, top_k=200, repeat_penalty=1.4, device=device, cont_id=cont_id)
        for rid in replies:
            raw, body, has_cont, has_eos = decode(rid, tok)
            cl = len(body)
            n += 1
            lens.append(cl)
            texts.append(body)
            if has_eos: eos_c += 1
            if has_cont: cont_c += 1
            if cl == 0: empty += 1
            elif cl <= 1: len1 += 1
            print(f"  dial: {'e' if has_eos else 'c' if has_cont else '.'} len={cl:2d}  {body[:38]!r}")

    avg_len = sum(lens) / max(n, 1)
    empty_r = empty / max(n, 1)
    len1_r = len1 / max(n, 1)
    seq = "".join(t for t in texts if t)
    han = sum(1 for ch in seq if '\u4e00' <= ch <= '\u9fff')
    han_r = han / max(len(seq), 1)
    chars = [c for c in seq if not c.isspace()]
    d1 = len(set(chars)) / max(len(chars), 1) if chars else 0.0
    counter = Counter([t for t in texts if t])
    top1_r = counter.most_common(1)[0][1] / max(n, 1) if counter else 0.0
    print(f"\n  [对话] n={n} 均长={avg_len:.1f} 空={empty_r:.0%} len<=1={len1_r:.0%} "
          f"EOS={eos_c} cont={cont_c} 汉字比={han_r:.0%} d1={d1:.2f} 短语重复={top1_r:.0%}")
    return {"label": label, "math": total, "avg_len": avg_len, "empty": empty_r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--base_dir", default=None, help="基座对照, 提供则先跑一次")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda"
    if args.base_dir:
        probe_ckpt(args.base_dir, f"BASE({os.path.basename(args.base_dir)})", device, seed=args.seed)
    probe_ckpt(args.ckpt_dir, args.label, device, seed=args.seed)


if __name__ == "__main__":
    main()
