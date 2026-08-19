#!/usr/bin/env python3
"""交互式对话：加载模型，输入 prompt，看回复。Ctrl+C 退出。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from inference.scripts.sample_py import build_model_from_checkpoint, generate
from tokenizers import Tokenizer

OUT_DIR = "out/chinese-data2-ab2-qkv"  # ← 换模型改这里（默认指针：A/B 三连胜者）

model, ckpt = build_model_from_checkpoint(OUT_DIR)
tok = Tokenizer.from_file("data/chinese/tokenizer.json")
n = sum(p.numel() for p in model.parameters())
print(f"已加载 [{OUT_DIR}] {n:,} 参数")
print("输入 prompt 直接回车，空行退出\n")

while True:
    try:
        user = input("用户：").strip()
        if not user:
            break
        prompt = f"用户：{user}\n模型："
        out = generate(model, tok, prompt,
                       max_new_tokens=200,
                       temperature=0.8,
                       top_k=200,
                       repeat_penalty=1.2,
                       stop_on_eos=True,
                       stop_on_turn=True,
                       clip_at_sentence=True)
        reply = out[len(prompt):] if out.startswith(prompt) else out
        print(f"模型：{reply}\n")
    except KeyboardInterrupt:
        print("\n退出")
        break