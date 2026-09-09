#!/usr/bin/env python3
"""多轮对话体检：固定脚本式多轮（默认 你好→你是谁→你在哪），token 级跨轮累积。

验证"连续对话框架"（dev-notes/archive/31 修复 + chat.py 重写）是否真的让模型能撑住
简单多轮：每轮 用户：{输入}\n模型： 追加进上下文，模型自吐的 <eos> 保留
（训练格式 回复+\n<eos>+\n），下一轮在训练分布内继续。

用法：
    .venv/bin/python training/multiturn_check.py --dirs out/chinese-reb/old out/chinese-reb
    # --script "你好|你是谁|你在哪" 自定义轮次；--seeds 3 每个 seed 独立跑一遍完整多轮

输出（每模型）：
    每轮 EOS 自吐 n/seeds + 平均回复 len；完整对话原文（自然度判读）
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from tokenizers import Tokenizer

from inference.scripts.sample_py import build_model_from_checkpoint, generate_ids

EOS_ID = 3
DEFAULT_SCRIPT = ["你好", "你是谁", "你在哪"]


def run_multiturn(model, tok, script, max_new=200, temp=0.6, top_k=200, rep=1.2):
    """跑一遍完整多轮脚本，返回 (turns, context, notes)。

    turns: [{user, reply, eos, len}]；context: 最终 token 上下文（调试用）。
    逻辑镜像 chat.py：EOS 命中才把 回复+<eos>+\n 续进上下文，否则不补。
    """
    context: list[int] = []
    nl_id = tok.encode("\n").ids[0]
    turns = []
    for u in script:
        context.extend(tok.encode(f"用户：{u}\n模型：").ids)
        plen = len(context)
        gen_ids, eos_pos = generate_ids(model, tok, None, max_new, temp, top_k, rep,
                                        stop_on_turn=False, stop_on_eos=False,
                                        clip_at_sentence=False, context_ids=context)
        reply_ids = gen_ids[plen:]
        reply = tok.decode(reply_ids)
        turns.append(dict(user=u, reply=reply, eos=eos_pos >= 0, len=len(reply_ids)))
        if eos_pos >= 0:
            context.extend(reply_ids + [EOS_ID, nl_id])
    return turns, context


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", required=True)
    ap.add_argument("--script", default="|".join(DEFAULT_SCRIPT),
                    help="用 | 分隔的用户轮次（默认 你好|你是谁|你在哪）")
    ap.add_argument("--seeds", type=int, default=3, help="每个模型跑几个 seed")
    a = ap.parse_args()

    script = [s.strip() for s in a.script.split("|") if s.strip()]
    tok = Tokenizer.from_file("data/chinese/tokenizer.json")

    for d in a.dirs:
        model, ck = build_model_from_checkpoint(d)
        print("=" * 74)
        print(f"{d}（val {float(ck['best_val_loss']):.4f}）多轮体检 · {a.seeds} seed · 脚本 {script}")
        print("=" * 74)
        for s in range(a.seeds):
            torch.manual_seed(s); torch.cuda.manual_seed(s)
            turns, _ = run_multiturn(model, tok, script)
            eos_cnt = sum(1 for t in turns if t["eos"])
            print(f"\n--- seed {s}：EOS 自吐 {eos_cnt}/{len(turns)} ---")
            for i, t in enumerate(turns):
                mark = "✓" if t["eos"] else "✗"
                print(f"  用户{i + 1}：{t['user']}")
                print(f"  模型{i + 1} [{mark} EOS, len={t['len']}]：{t['reply'].strip()[:150]}")
        # 汇总
        eos_total = 0; len_total = 0; n = 0
        for s in range(a.seeds):
            torch.manual_seed(s); torch.cuda.manual_seed(s)
            turns, _ = run_multiturn(model, tok, script)
            for t in turns:
                eos_total += 1 if t["eos"] else 0
                len_total += t["len"]
                n += 1
        print(f"\n汇总：EOS 自吐 {eos_total}/{n} 轮（{eos_total / n * 100:.0f}%） | 平均回复 {len_total / n:.1f} token")
        print()


if __name__ == "__main__":
    main()
