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

from inference.scripts.sample_py import build_model_from_checkpoint, generate_ids, load_tokenizer


def _trunc_need(b):
    """若字节串 b 是截断的多字节 UTF-8 字符前缀，返回还缺的续字节数；否则 0。"""
    if not b:
        return 0
    lead = b[0]
    if 0xC2 <= lead <= 0xDF:
        need = 1
    elif 0xE0 <= lead <= 0xEF:
        need = 2
    elif 0xF0 <= lead <= 0xF4:
        need = 3
    else:
        return 0
    cont = b[1:]
    if len(cont) >= need:
        return 0                       # 完整字符（或字符+更多），不是截断
    if all(0x80 <= c <= 0xBF for c in cont):
        return need - len(cont)        # 截断：还差这么多字节
    return 0                           # 续字节非法


def _tail_pending(raw):
    """返回字节流尾部滞留的跨组分片字节数（0 = 无）。截断字符等下一组补全。"""
    for k in (1, 2, 3):
        if _trunc_need(raw[-k:]) > 0:
            return k
    return 0


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
                    help="推理状态选择性续传：只保留最近 N token（字节模型 = N 字节，"
                         "如 192 ≈ 64 字），历史由记忆状态承接")
    ap.add_argument("--system", default=None, help="系统提示（身份/风格设定）")
    a = ap.parse_args()

    if a.seed is not None:
        torch.manual_seed(a.seed)
        torch.cuda.manual_seed(a.seed)
    rope_len = 8192 if a.window is not None else None
    model, ckpt = build_model_from_checkpoint(a.out_dir, rope_len=rope_len)
    tok = load_tokenizer(ckpt)      # 字节直入模型自动切 ByteTokenizer
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    byte_flag = bool(ckpt["model_args"].get("byte_level"))
    print(f"[{a.out_dir}] {n:,} 参数 | byte_level={byte_flag} | window={a.window} | 对话开始（Ctrl+C / 空行 / exit 退出）")
    print("─" * 60)

    ctx = (a.system + "\n") if a.system else ""
    mem_state = None
    # dev-notes/61：字级模型训练数据用 A：/B：（70% 说话人标注），其余模式保持旧 用户：/模型：
    char_style = bool(ckpt["model_args"].get("char_level"))
    user_prefix = "A：" if char_style else "用户："
    model_prefix = "B：" if char_style else "模型："
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
        ctx += f"{user_prefix}{user}\n{model_prefix}"

        # 流式打字机：回调里全量 decode 求增量。回调载荷：BPE = 单个 token id；
        # 字节直入 = 一组 3 字节（list），统一按列表处理。
        # 字节模式：3 字节组可能切在 UTF-8 字符中间（如 [0xE4] 独字节）——若直接把
        # 截断的 lead 字节输出成 �，下组补全时增量会丢字符。用 _tail_pending 识别
        # 尾部"待补全"分片：截断字符滞留等下一组，真非法字节才 replace 消化。
        printed = ""
        gen_tokens = []

        def cb(tids):
            nonlocal printed
            if isinstance(tids, int):
                tids = [tids]
            gen_tokens.extend(tids)
            if byte_flag:
                raw = bytes(t for t in gen_tokens if t < 256)
                n = len(raw) - _tail_pending(raw)
                text = raw[:n].decode("utf-8", errors="replace")
            else:
                text = tok.decode(gen_tokens)
            sys.stdout.write(text[len(printed):])
            sys.stdout.flush()
            printed = text

        print("🤖 模型：", end="", flush=True)
        stop_kind_ref = [None]
        gen, _ = generate_ids(
            model, tok, ctx, a.max_new_tokens, a.temperature, a.top_k,
            a.repeat_penalty,
            stop_on_turn=True, stop_on_eos=True, clip_at_sentence=True,
            stop_on_cont=True, window=a.window,
            resume_state=None if a.window is None else mem_state,
            token_callback=cb, stop_kind_ref=stop_kind_ref)
        if a.window is not None:
            mem_state = model.get_memory_state()      # 跨轮续传：存本轮末态

        # 更新上下文：回复 = 生成里 prompt 之后的部分
        reply = tok.decode(gen[len(tok.encode(ctx).ids):])
        ctx += reply + "\n"
        # dev-notes/61 待续符自控：显示模型本轮以何种方式收尾
        sk = stop_kind_ref[0] if stop_kind_ref else None
        tag = {"eos": "🔴 说完收尾", "cont": "🔵 说完递回（期待你回应）",
               "turn": "（自然收尾）", "maxlen": "（到生成上限）"}.get(sk, "")
        print(f"  ↳ {tag}" if tag else "")
        print()


if __name__ == "__main__":
    main()
