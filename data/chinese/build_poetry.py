#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 data/external/chinese-poetry 抽取高质量诗词蒙学经典，
清洗输出为适合语言模型的自然文本 data/chinese/classical_poetry.txt。
"""
import os, json, glob

SRC = "data/external/chinese-poetry"
OUT = "data/chinese/classical_poetry.txt"

texts = []

# 1. 蒙学经典 (三字经、百家姓、千字文、弟子规、幼学琼林等)
print("正在抽取蒙学经典...")
for fp in glob.glob(f"{SRC}/蒙学/*.json"):
    try:
        data = json.load(open(fp, encoding="utf-8"))
        if isinstance(data, dict) and "paragraphs" in data:
            title = data.get("title", "")
            paras = data.get("paragraphs", [])
            texts.append(f"【{title}】\n" + "\n".join(paras))
        elif isinstance(data, list):
            for item in data:
                t = item.get("title", "")
                p = item.get("paragraphs", []) or item.get("content", [])
                if p:
                    texts.append(f"【{t}】\n" + "\n".join(p))
    except Exception:
        continue

# 2. 论语、诗经
print("正在抽取论语与诗经...")
for name in ["论语", "诗经"]:
    for fp in glob.glob(f"{SRC}/{name}/*.json"):
        try:
            items = json.load(open(fp, encoding="utf-8"))
            if isinstance(items, list):
                for it in items:
                    title = it.get("chapter", "") or it.get("title", "")
                    paras = it.get("paragraphs", []) or it.get("content", [])
                    if paras:
                        texts.append(f"《{title}》\n" + "\n".join(paras))
        except Exception:
            continue

# 3. 精选唐诗宋词（按名家/卷抽样，控制体量，避免生僻字污染词表）
print("正在抽取精选唐诗宋词...")
for fp in sorted(glob.glob(f"{SRC}/全唐诗/poet.tang.*.json"))[:15]: # 前15卷精选
    try:
        items = json.load(open(fp, encoding="utf-8"))
        for it in items:
            author = it.get("author", "无名氏")
            title = it.get("title", "")
            paras = it.get("paragraphs", [])
            if paras and len(paras) <= 12: # 过滤过长歌行，保留精练律诗绝句
                texts.append(f"《{title}》 {author}\n" + "\n".join(paras))
    except Exception:
        continue

for fp in sorted(glob.glob(f"{SRC}/宋词/ci.song.*.json"))[:10]: # 宋词精选
    try:
        items = json.load(open(fp, encoding="utf-8"))
        for it in items:
            author = it.get("author", "无名氏")
            title = it.get("rhythmic", "")
            paras = it.get("paragraphs", [])
            if paras:
                texts.append(f"《{title}》 {author}\n" + "\n".join(paras))
    except Exception:
        continue

final_text = "\n\n".join(texts)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(final_text)

print(f"✅ 古典文采语料生成完成: {OUT}")
print(f"   共抽取 {len(texts):,} 篇目，总字符数: {len(final_text):,} 字符 ({os.path.getsize(OUT)/1024:.0f} KB)")
