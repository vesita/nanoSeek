#!/usr/bin/env python3
"""交互式流式对话（打字机输出，GPU 默认，对话历史跨轮积累）。

用法（从项目根目录）：
    uv run python inference/scripts/chat.py                                  # 默认基线模型
    uv run python inference/scripts/chat.py --out_dir out/obs_zh_mem1500glm
    uv run python inference/scripts/chat.py --window 64                      # 窗口+状态续传
    uv run python inference/scripts/chat.py --system "你是小寻，一个温柔耐心的倾听者"

交互：输入问题回车 → 模型逐 token 流式输出 → 继续下一轮。Ctrl+C / 空行 / exit 退出。
上下文管理：
    默认 = 完整上下文（对话历史全部保留，随轮次增长）；
    --window N = 只保留最近 N token（RoPE 绝对位置），更早历史由记忆状态
    跨轮续传承接（dev-notes/46 推理状态选择性续传）——输入不随对话增长。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from tokenizers import Tokenizer

from inference.scripts.sample_py import build_model_from_checkpoint, generate_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="out/obs_zh_mem1500glm",
                    help="训练输出目录（读 best.pt）")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--repeat-penalty", type=float, default=1.2)
    ap.add_argument("--seed", type=int, default=None,
                    help="固定随机种子（默认 None = 每次对话随机）")
    ap.add_argument("--window", type=int, default=None,
                    help="推理状态选择性续传：只保留最近 N token，历史由记忆状态承接")
    ap.add_argument("--system", default=None, help="系统提示（身份/风格设定）")
    a = ap.parse_args()

    if a.seed is not None:
        torch.manual_seed(a.seed)
        torch.cuda.manual_seed(a.seed)
    rope_len = 8192 if a.window is not None else None
    model, ckpt = build_model_from_checkpoint(a.out_dir, rope_len=rope_len)
    tok = Tokenizer.from_file("data/chinese/tokenizer.json")
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"[{a.out_dir}] {n:,} 参数 | window={a.window} | 对话开始（Ctrl+C / 空行 / exit 退出）")
    print("─" * 60)

    ctx = (a.system + "\n") if a.system else ""
    mem_state = None
    while True:
        try:
            user = input("🙂 用户：")
        except (EOFError, KeyboardInterrupt):
            print("\n再见 👋")
            break
        if not user.strip():
            break
        if user.strip() in ("exit", "quit", "退出"):
            print("再见 👋")
            break
        ctx += f"用户：{user}\n模型："

        # 流式打字机：回调里全量 decode 求增量（BPE 分片自愈）
        printed = ""
        gen_tokens = []

        def cb(tid):
            nonlocal printed
            gen_tokens.append(tid)
            text = tok.decode(gen_tokens)
            sys.stdout.write(text[len(printed):])
            sys.stdout.flush()
            printed = text

        print("🤖 模型：", end="", flush=True)
        gen, _ = generate_ids(
            model, tok, ctx, a.max_new_tokens, a.temperature, a.top_k,
            a.repeat_penalty,
            stop_on_turn=True, stop_on_eos=True, clip_at_sentence=True,
            window=a.window, resume_state=None if a.window is None else mem_state,
            token_callback=cb)
        if a.window is not None:
            mem_state = model.get_memory_state()      # 跨轮续传：存本轮末态

        # 更新上下文：回复 = 生成里 prompt 之后的部分
        reply = tok.decode(gen[len(tok.encode(ctx).ids):])
        ctx += reply + "\n"
        print()


if __name__ == "__main__":
    main()
