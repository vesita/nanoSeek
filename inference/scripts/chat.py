#!/usr/bin/env python3
"""交互式对话：加载模型，输入 prompt，看回复。Ctrl+C 退出。

连续对话框架（2026-08-19）：上下文按训练格式跨轮累积——
    「用户：{输入}\n模型：{回复}\n<eos>\n用户：{输入}...」
模型自吐的 <eos> 保留在上下文里（训练里每条回复后都有 <eos>），下一轮才能
在训练分布内继续；没有 EOS 的轮次不补（跑满 max_new_tokens = 模型没学会终止）。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from inference.scripts.sample_py import build_model_from_checkpoint, generate_ids
from tokenizers import Tokenizer

OUT_DIR = "out/chinese-reb"  # ← 换模型改这里（默认指针：v0.2 定版，reb）
EOS_ID = 3

model, ckpt = build_model_from_checkpoint(OUT_DIR)
tok = Tokenizer.from_file("data/chinese/tokenizer.json")
n = sum(p.numel() for p in model.parameters())
block_size = model.config.block_size
nl_id = tok.encode("\n").ids[0]
print(f"已加载 [{OUT_DIR}] {n:,} 参数（上下文 {block_size}）")
print("输入 prompt 直接回车，空行退出\n")

context: list[int] = []   # token 级上下文，跨轮累积
while True:
    try:
        user = input("用户：").strip()
        if not user:
            break
        context.extend(tok.encode(f"用户：{user}\n模型：").ids)
        # 上下文超长时裁剪到 block_size 内（尾部始终以 模型： 结尾，仍是合法 prompt）
        if len(context) > block_size:
            context = context[-block_size:]

        plen = len(context)
        gen_ids, eos_pos = generate_ids(model, tok, None, 200, 0.6, 200, 1.2,
                                        stop_on_turn=False, stop_on_eos=False,
                                        clip_at_sentence=False, context_ids=context)
        reply_ids = gen_ids[plen:]
        reply = tok.decode(reply_ids)
        print(f"模型：{reply}\n")

        # 训练格式续上下文：回复 + <eos> + \n（仅当模型确实吐了 EOS）
        if eos_pos >= 0:
            context.extend(reply_ids + [EOS_ID, nl_id])
        else:
            print("  ⚠ 模型未输出终止符（跑满长度）——本轮不补 <eos> 到上下文\n")
    except KeyboardInterrupt:
        print("\n退出")
        break
