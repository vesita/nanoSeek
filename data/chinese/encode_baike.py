# -*- coding: utf-8 -*-
"""把维基百科正文编码为 char token ids (baike_char.bin)。

用与基座训练完全相同的 char tokenizer (WordLevel，8192 词表)，逐字编码。
百科正文以段落自然分隔；每段落末尾插入 <eos> 作为 turn 级终止符（与现有对话数据一致）。
输出：data/chinese/baike_char.bin（uint16）
"""
import os, sys, numpy as np
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("HSA_ENABLE_SDMA", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizers import Tokenizer

SRC = "data/chinese/wikipedia_cn.txt"
OUT = "data/chinese/baike_char.bin"
TOKENIZER = "data/chinese/char_tokenizer.json"
CHUNK = 100_000

assert os.path.exists(SRC), f"缺 {SRC}，先跑 download_baike_v2.py"
tok = Tokenizer.from_file(TOKENIZER)
eos_id = tok.token_to_id("<eos>")
if eos_id is None:
    eos_id = 128  # 兜底
print(f"char tokenizer vocab={tok.get_vocab_size()} | <eos>={eos_id}")

with open(SRC, "r", encoding="utf-8", errors="replace") as f:
    text = f.read()

# 按空行分段（每段一条百科条目）
blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
print(f"百科段落数: {len(blocks):,}")

# 逐段编码，段间插 <eos>
total_ids = []
for b in blocks:
    chunk_ids = tok.encode(b).ids
    total_ids.extend(chunk_ids)
    total_ids.append(eos_id)

arr = np.array(total_ids, dtype=np.uint16)
arr.tofile(OUT)
print(f"已写出 {OUT}")
print(f"  token 数: {len(arr):,} | 段落: {len(blocks):,} | 字符数: {len(text):,}")
print(f"  文件大小: {os.path.getsize(OUT)/1e6:.1f} MB")

# 额外写训练用 meta（供加载脚本读取 token 数）
meta = {"tokens": int(len(arr)), "blocks": len(blocks)}
import json
with open("data/chinese/baike_meta.json", "w") as f:
    json.dump(meta, f)
print("已写 baike_meta.json:", meta)
