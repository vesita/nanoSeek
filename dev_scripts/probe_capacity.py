#!/usr/bin/env python3
"""判别实验: 2.7M 模型到底能不能输出"个位数加法答案"?

目的: 判断 RL 数学学不会是"奖励地形"还是"模型容量天花板"。
  - 若连最简个位数加法 (a+b, a,b in 1..9, 结果<20) 在监督式(非RL)都给不出
    一致正确数字 → 容量/表示天花板, RL 调奖励无济于事。
  - 提供基座自由生成 + argmax 解码两种探测, 看模型是否"有数字作答的倾向"。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from inference.scripts.sample_py import build_model_from_checkpoint, load_tokenizer
from training.rl.reward import arith_gen

BASE = "out/cont_v1_1epoch"
DEV = "cuda"

SIMPLE = [(f"{a}加{b}等于几？", a+b) for a in range(1,9) for b in range(1, min(9,10-a))]  # 结果<10最简


def main():
    model, ckpt = build_model_from_checkpoint(BASE, device=DEV)
    tok = load_tokenizer(ckpt)
    model.eval()
    print(f"=== {BASE} 最简个位数加法 (结果<10, {len(SIMPLE)} 题) ===")
    top1_ok = 0
    gen_ok = 0
    n = len(SIMPLE)
    for p, ans in SIMPLE:
        pids = tok.encode(p).ids
        # argmax 贪婪解码 (确定性, 看模型"最想说啥")
        idx = torch.tensor([pids], dtype=torch.long, device=DEV)
        out = []
        for _ in range(12):
            logits, _ = model(idx)
            nxt = logits[0,-1,:].argmax()
            if int(nxt) in (117,119): break  # eos/cont
            out.append(int(nxt))
            idx = torch.cat([idx, nxt.unsqueeze(0).unsqueeze(0)], dim=1)
        greedy = tok.decode(out)
        # 贪心解码是否含正确答案数字
        got = arith_gen.extract_answer_number(greedy)
        if got == ans:
            top1_ok += 1
        if len(out) <= 4:
            pass
        print(f"  {p:16s} 答案={ans:2d} | greedy={greedy[:22]!r} got={got}")
    print(f"\n  argmax 首数字命中: {top1_ok}/{n}")
    print(f"\n结论: 若 argmax 在个位数加法上都给不出答案数字, 则 2.7M 模型做"
          f"\n      100以内四则 = 容量天花板, 需改任务设定或加 SFT 教'输出数字'行为。")


if __name__ == "__main__":
    main()
