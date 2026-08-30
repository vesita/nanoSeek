#!/usr/bin/env python3
"""多轮对话压力测试：终结符（EOS）+ 多轮崩溃检测。

针对项目反复记录的失败模式：
  1. 终结符：模型有没有学会在回复结束时吐 <eos>（turn-level EOS 训练的目标，
     dev-notes/26：自吐终止符才算学会"话说完"）；不吐 = 回复收不住、上下文越滚越长。
  2. 多轮崩溃：连续追问 N 轮后是否退化——回复变极短/变长失控、重复率飙升、
     上下文越长越崩（重复坍缩、碎片的累积效应）。
  3. 轮次结构：回复里是否自己重开 用户：/模型： 轮次（碎片拼贴的反面）。

用 generate_ids 的 token 级结果（eos_pos / 轮次截断），不依赖字符串检测。

用法（从项目根目录）：
    uv run python inference/scripts/eval_multiturn.py --dirs out/obs_zh_base out/obs_zh_mem
"""
import argparse
import sys
import re
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from inference.scripts.sample_py import build_model_from_checkpoint, generate_ids

MAX_NEW_TOKENS = 120
TEMPERATURE = 0.8
TOP_K = 200
REPEAT_PENALTY = 1.2
SEED = 1337
N_TURNS = 4

# 三个对话场景 + 固定追问（模拟真实多轮对话，追问与首轮风格一致）
SCENARIOS = [
    ("压力", "用户：最近工作压力好大，怎么办啊？\n模型：",
     ["用户：具体说说怎么放松吧。\n模型：", "用户：可是我没时间运动啊。\n模型：",
      "用户：那熬夜工作是不是更不行？\n模型："]),
    ("闲聊", "用户：好想出去玩\n模型：",
     ["用户：可是没钱去远地方。\n模型：", "用户：那周边游呢？\n模型：",
      "用户：周末两天够不够？\n模型："]),
    ("荐书", "用户：帮我推荐一本小说吧。\n模型：",
     ["用户：有没有轻松一点的？\n模型：", "用户：悬疑的也行。\n模型：",
      "用户：最近有什么新书吗？\n模型："]),
]


def ngram_rep(s: str, n: int = 3) -> float:
    seq = re.sub(r"\s+", "", s)
    if len(seq) < 2 * n:
        return 0.0
    grams = [seq[i:i + n] for i in range(len(seq) - n + 1)]
    c = Counter(grams)
    return sum(1 for g in grams if c[g] > 1) / len(grams)


def run_conversation(model, tok, scenario, window=None, no_resume=False):
    """跑 4 轮对话，返回每轮指标。

    window（dev-notes/46 推理状态选择性续传）：非 None 时输入只保留最近 window
    token，记忆状态跨轮续传（长对话不随轮次增长）；no_resume 禁用续传（对照）。
    """
    opening, followups = scenario
    ctx = opening
    mem_state = None
    turns = []
    for t in range(N_TURNS):
        gen, eos_pos = generate_ids(model, tok, ctx, MAX_NEW_TOKENS, TEMPERATURE,
                                    TOP_K, REPEAT_PENALTY,
                                    stop_on_turn=True, stop_on_eos=False,
                                    window=window, no_resume=no_resume,
                                    resume_state=None if (no_resume or window is None) else mem_state)
        if window is not None and not no_resume:
            mem_state = model.get_memory_state()      # 跨轮续传：存本轮末态
        # generate_ids 返回 (完整 token 列表, eos_pos)；文本 = prompt 之后的部分
        prompt_ids = tok.encode(ctx).ids
        new_tok = gen[len(prompt_ids):]
        text = tok.decode(new_tok)
        rep3 = ngram_rep(text)
        # 轮次截断：如果模型自己开了 用户：/模型： 轮次，generate_ids 会截断
        turn_cut = len(new_tok) < MAX_NEW_TOKENS and eos_pos == -1 and any(
            s in text for s in ("用户：", "用户:", "模型：", "模型:"))
        turns.append(dict(
            turn=t, len_tokens=len(new_tok), rep3=round(rep3, 3),
            eos_hit=eos_pos != -1, turn_cut=turn_cut,
            text=text.strip()[:100],
        ))
        # 把模型回复接回上下文（模型没吐 EOS 也截断到轮次/上限，避免上下文失控）
        reply = tok.decode(new_tok)
        # 去掉回复里可能自己开的 用户： 之后的尾巴（保留到轮次截断点即可）
        ctx = ctx + reply + "\n"
        if t < len(followups):
            ctx += followups[t]
    return turns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=["out/obs_zh_base", "out/obs_zh_mem"])
    ap.add_argument("--window", type=int, default=None,
                    help="窗口+状态续传（dev-notes/46）：输入只保留最近 N token，记忆状态跨轮续传")
    ap.add_argument("--no-resume", action="store_true", help="窗口模式下禁用状态续传（对照）")
    a = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file("data/chinese/tokenizer.json")

    results = {}
    for d in a.dirs:
        if not (Path(d) / "best.pt").exists():
            print(f"⚠ 跳过（无 best.pt）: {d}")
            continue
        rope_len = 8192 if a.window is not None else None
        model, _ = build_model_from_checkpoint(d, rope_len=rope_len)
        model.eval()
        print(f"\n===== {d} (window={a.window}, resume={not a.no_resume}) =====")
        per_model = []
        for sname, opening, followups in SCENARIOS:
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            turns = run_conversation(model, tok, (opening, followups),
                                     window=a.window, no_resume=a.no_resume)
            per_model.extend(turns)
            print(f"  [{sname}] 开场: {opening.strip()[:20]}…")
            for tt in turns:
                flag = " ⚠" if (tt["rep3"] > 0.3 or tt["len_tokens"] <= 2
                                or (tt["len_tokens"] >= MAX_NEW_TOKENS - 1 and tt["turn"] == 0)) else ""
                print(f"    轮{tt['turn']+1}: {tt['len_tokens']:>3} tok | rep3 {tt['rep3']:.3f} | "
                      f"EOS {tt['eos_hit']} | 自开轮次 {tt['turn_cut']}{flag} | {tt['text'][:60]}")
        results[d] = per_model

    print("\n\n======== 多轮汇总 ========")
    print(f"{'模型':<22} | {'EOS率':>6} | {'自开轮次率':>7} | {'平均len':>7} | {'rep3':>6} | "
          f"{'末轮len':>7} | {'末轮rep3':>8} | {'崩溃数':>5}")
    print("-" * 100)
    for d, turns in results.items():
        n = len(turns)
        eos_rate = sum(t["eos_hit"] for t in turns) / n
        cut_rate = sum(t["turn_cut"] for t in turns) / n
        avg_len = sum(t["len_tokens"] for t in turns) / n
        avg_rep3 = sum(t["rep3"] for t in turns) / n
        # 末轮（第 4 轮）平均
        last = [t for t in turns if t["turn"] == N_TURNS - 1]
        last_len = sum(t["len_tokens"] for t in last) / len(last)
        last_rep3 = sum(t["rep3"] for t in last) / len(last)
        # 崩溃 = rep3>0.3 或 len<=2 或 首轮就打满上限（收不住）
        crash = sum(1 for t in turns if t["rep3"] > 0.3 or t["len_tokens"] <= 2
                    or (t["len_tokens"] >= MAX_NEW_TOKENS - 1 and t["turn"] == 0))
        print(f"{d:<22} | {eos_rate:>5.0%} | {cut_rate:>6.0%} | {avg_len:>7.1f} | "
              f"{avg_rep3:>6.3f} | {last_len:>7.1f} | {last_rep3:>8.3f} | {crash:>5}")
    print("\n指标：EOS率=自然吐<eos>的比例 | 自开轮次=回复里自己重开用户/模型轮次(碎片信号) | "
          "崩溃=rep3>0.3 或长度≤2 或首轮顶满上限")


if __name__ == "__main__":
    main()
