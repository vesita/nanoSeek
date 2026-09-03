#!/usr/bin/env python3
"""对话质量综合评估器: 多样话题 × 针对性 × 乱码 × 语感。

单指标(均长/EOS/d1)不足以判断对话好坏 —— 基座 d1 也高但全是无意义碎片。
本工具用 8 个不同话题 prompt 采样, 综合看:
  1. 是否"会说人话"(语义通顺度, 人工看样例)
  2. 是否"有针对性"(不同话题不同回复, 不车轱辘)
  3. 是否混入乱码字符(/字母/数字碎片)
  4. 温和确认框架是否过度(counseling 腔)

用法:
  HSA_ENABLE_SDMA=0 HSA_OVERRIDE_GFX_VERSION=10.3.0 TMPDIR=... PYTHONUNBUFFERED=1 \
    .venv/bin/python dev_scripts/dialog_quality.py --ckpt_dir out/rl_coh_dialog --label COH300
"""
import argparse
import os
import re
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from inference.scripts.sample_py import build_model_from_checkpoint, load_tokenizer, generate_ids

TOPICS = [
    ("greeting", "用户：你好！\n模型："),
    ("mood",     "用户：你今天心情怎么样？\n模型："),
    ("travel",   "用户：如果有机会出去旅游，最想去哪？\n模型："),
    ("naming",   "用户：给你自己起个名字吧！\n模型："),
    ("comfort",  "用户：我最近有点累了，压力很大。\n模型："),
    ("hobby",    "用户：你平时喜欢做些什么？\n模型："),
    ("weather",  "用户：今天天气真好。\n模型："),
    ("food",     "用户：今天吃了什么好吃的？\n模型："),
]

# 乱码信号: 非汉字/非常见标点的字符(字母/数字/外语/符号)密度
_GARBAGE = re.compile(r"[A-Za-z0-9_%&@#!$*()=+\[\]{}|\\/;:~`'\"<>0-9]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--max_tokens", type=int, default=35)
    args = ap.parse_args()

    model, ck = build_model_from_checkpoint(args.ckpt_dir)
    dev = "cuda"; model.eval()
    tok = load_tokenizer(ck)

    print(f"=== 对话质量 [{args.label or args.ckpt_dir}] ===")
    bodies = []
    for topic, p in TOPICS:
        ids, ep = generate_ids(model, tok, p, args.max_tokens, 0.8, 200, 1.2,
                               stop_on_turn=True, stop_on_eos=True)
        plen = len(tok.encode(p).ids)
        body = tok.decode(ids[plen:]).strip()
        bodies.append(body)
        print(f"  {topic:9s} -> {body[:36]!r}")

    # 指标
    n = len([b for b in bodies if b])
    uniq = len(set(b for b in bodies if b))
    # 乱码密度: 每个回复里非中文标点+字母数字字符占比(去标点后)
    tot_c = sum(len([c for c in b if not c.isspace()]) for b in bodies)
    gar_c = sum(len(_GARBAGE.findall(b)) for b in bodies)
    # 温和确认腔计数 (过度同质框架信号)
    confirm = sum(1 for b in bodies if re.search(r"我明白|嗯，|嗯我|哪一块|磨人|撑着", b))
    avg_len = sum(len(b) for b in bodies) / max(n, 1)
    print(f"\n  n={n} 不同回复={uniq}/{n} ({100*uniq/max(n,1):.0f}%多样) "
          f"均长={avg_len:.1f} 乱码字占比={100*gar_c/max(tot_c,1):.0f}% "
          f"温和确认腔={confirm}/{n}")
    print(f"  解读: 好的对话 = 多样接近100% + 乱码≈0 + 均长适中 + 各话题语义相关。")


if __name__ == "__main__":
    main()
