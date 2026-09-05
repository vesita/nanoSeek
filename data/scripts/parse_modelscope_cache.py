#!/usr/bin/env python3
"""
从 ModelScope 缓存及直接接口解析开源高质量数据集 (COIG-CQIA / GSM8K)
并写入 data/chinese/ 标准对话格式。
"""

import os
import glob
import json
import unicodedata

DATA_CHINESE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chinese")
CACHE_DOWNLOADS = os.path.expanduser("~/.cache/modelscope/hub/datasets/downloads")

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", str(s).strip())
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s

def extract_coig_cqia_from_cache():
    """解析已下载的 14MB COIG-CQIA 高质知乎子集 jsonl 文件。"""
    out_path = os.path.join(DATA_CHINESE_DIR, "coig_cqia_dialogue.txt")
    json_files = glob.glob(os.path.join(CACHE_DOWNLOADS, "*"))
    
    count = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for fpath in json_files:
            if fpath.endswith(".json") or fpath.endswith(".lock"):
                continue
            try:
                with open(fpath, "r", encoding="utf-8") as in_f:
                    for line in in_f:
                        line = line.strip()
                        if not line or not line.startswith("{"):
                            continue
                        try:
                            item = json.loads(line)
                        except Exception:
                            continue
                        instruction = clean_text(item.get("instruction", ""))
                        inp = clean_text(item.get("input", ""))
                        out = clean_text(item.get("output", ""))
                        
                        user_msg = instruction if not inp else f"{instruction}\n{inp}"
                        if len(user_msg) < 5 or len(out) < 10:
                            continue
                        out_f.write(f"用户：{user_msg}\n模型：{out}\n\n")
                        count += 1
            except Exception as e:
                continue

    if os.path.exists(out_path):
        print(f"✓ 成功从魔搭缓存提纯 COIG-CQIA 精选数据: {out_path} ({count} 条问答, {os.path.getsize(out_path)/(1024*1024):.2f} MB)")

def parse_gsm8k_from_cache():
    """解析已下载的 GSM8K 缓存文件并加入思维链语料。"""
    out_path = os.path.join(DATA_CHINESE_DIR, "gsm8k_cot_dialogue.txt")
    cache_file = os.path.join(CACHE_DOWNLOADS, "1065526710517f26b3e0721d7efb1efcfc5941c5ce4895ac619822ca5df5babb")
    if not os.path.exists(cache_file):
        return
    count = 0
    with open(out_path, "w", encoding="utf-8") as out_f, open(cache_file, "r", encoding="utf-8") as in_f:
        for line in in_f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            q = clean_text(item.get("question", ""))
            a = clean_text(item.get("answer", ""))
            if not q or not a:
                continue
            out_f.write(f"用户：请解答这道数学应用题并给出详细的思考步骤：{q}\n模型：【详细解题思路与推理】：\n{a}\n\n")
            count += 1
    print(f"✓ 成功从魔搭缓存提纯 GSM8K 逻辑思维链数据: {out_path} ({count} 条题目, {os.path.getsize(out_path)/(1024*1024):.2f} MB)")

if __name__ == "__main__":
    extract_coig_cqia_from_cache()
    parse_gsm8k_from_cache()
