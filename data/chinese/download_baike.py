#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流式下载 pleisto/wikipedia-cn-20230720-filtered 高质量中文维基百科。
过滤低信噪比条目，清洗抽取自然文本输出为 data/chinese/wikipedia_cn.txt。
"""
import os, sys, json, requests, re

OUT = "data/chinese/wikipedia_cn.txt"
TMP = OUT + ".tmp"
TARGET_CHARS = 30_000_000  # 抽取约 3000 万字百科正文 (约合 40MB 纯文本，适合 100M 模型体量)

URL = "https://hf-mirror.com/datasets/pleisto/wikipedia-cn-20230720-filtered/resolve/main/wikipedia-cn-20230720-filtered.json"

print(f"开始流式下载清洗中文维基百科，目标规模: {TARGET_CHARS/1e6:.1f}M 字符...")

total_chars = 0
total_articles = 0

# 过滤无实质内容的词条（如纯消歧义、无中文正文等）
def is_valid_article(text: str) -> bool:
    if len(text) < 80:  # 太短丢弃
        return False
    if "消歧义" in text[:30]:
        return False
    # 中文字符比例检查
    cjk_count = len(re.findall(r'[\u4e00-\u9fff]', text))
    if cjk_count / max(len(text), 1) < 0.5:
        return False
    return True

with requests.get(URL, stream=True, timeout=30) as resp:
    resp.raise_for_status()
    buffer = ""
    with open(TMP, "w", encoding="utf-8") as out_f:
        for chunk in resp.iter_content(chunk_size=1024*1024):
            if not chunk:
                break
            buffer += chunk.decode("utf-8", errors="ignore")
            # 按 json 对象简单匹配切割
            while True:
                pos = buffer.find('{"completion":')
                if pos == -1:
                    pos = buffer.find('{\n    "completion":')
                if pos == -1:
                    if len(buffer) > 200_000:
                        buffer = buffer[-50_000:]
                    break
                
                # 寻找当前 json 结尾
                end_pos = buffer.find('\n  }', pos)
                if end_pos == -1:
                    end_pos = buffer.find('}', pos)
                    if end_pos == -1:
                        break
                
                raw_json = buffer[pos:end_pos+1]
                buffer = buffer[end_pos+1:]
                
                try:
                    obj = json.loads(raw_json)
                    text = obj.get("completion", "").strip()
                    if is_valid_article(text):
                        # 清理多余空行与参考资料垃圾
                        clean_text = re.sub(r'\n{3,}', '\n\n', text)
                        out_f.write(clean_text + "\n\n")
                        total_chars += len(clean_text)
                        total_articles += 1
                        if total_articles % 5000 == 0:
                            print(f"已处理 {total_articles:,} 篇条目，累计 {total_chars/1e6:.2f}M 字符...")
                        if total_chars >= TARGET_CHARS:
                            break
                except Exception:
                    continue
            
            if total_chars >= TARGET_CHARS:
                break

if os.path.exists(TMP):
    os.replace(TMP, OUT)
    print(f"✅ 维基百科抽取完成: {OUT}")
    print(f"   有效条目: {total_articles:,} 篇 | 字符数: {total_chars:,} ({os.path.getsize(OUT)/1024:.0f} KB)")
