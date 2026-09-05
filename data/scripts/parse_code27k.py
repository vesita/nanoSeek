#!/usr/bin/env python3
"""解析 CodeExercise-Python-27k 代码语料为标准对话格式。"""
import os
import json
import unicodedata

DATA_CHINESE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chinese")

def clean(s):
    if not s:
        return ""
    return unicodedata.normalize("NFKC", str(s).strip())

def parse_code27k():
    src = os.path.join(DATA_CHINESE_DIR, "_code27k.jsonl")
    if not os.path.exists(src):
        print("_code27k.jsonl 不存在")
        return
    out = os.path.join(DATA_CHINESE_DIR, "code_alpaca_dialogue.txt")
    count = 0
    with open(src, "r", encoding="utf-8") as fin, open(out, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            rounds = item.get("chat_rounds", [])
            if len(rounds) < 2:
                continue
            human = clean(rounds[0].get("content", ""))
            bot = clean(rounds[1].get("content", ""))
            if not human or not bot:
                continue
            fout.write(f"用户：{human}\n模型：{bot}\n\n")
            count += 1
    print(f"✓ 代码语料解析完成: {count} 条 -> {out} ({os.path.getsize(out)/1e6:.1f} MB)")
    os.remove(src)

if __name__ == "__main__":
    parse_code27k()
