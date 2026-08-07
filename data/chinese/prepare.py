"""
为语言建模准备中文对话数据集（BPE 分词）。

流程：download_dialogue.py 下载对话语料 → train_tokenizer.py 训出 tokenizer.json
→ 本脚本把所有 txt 编码成 token ids，写出 train.bin / val.bin / meta.pkl。

用法（从项目根目录）：
    uv run python data/chinese/prepare.py
"""
import os
import pickle
import requests
import numpy as np
from tokenizers import Tokenizer

# 待补齐的书目：(本地文件名, URL 里的中文书名)，下载不到就跳过
BOOKS = [
    ('西游记.txt',  '西游记'),
    ('红楼梦.txt',  '红楼梦'),
    ('三国演义.txt','三国演义'),
    ('水浒传.txt',  '水浒传'),
]
RAW_URL    = 'https://raw.githubusercontent.com/tennessine/corpus/master/{enc}.txt'
MIRROR_URL = 'https://cdn.jsdelivr.net/gh/tennessine/corpus@master/{enc}.txt'

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH = os.path.join(DATA_DIR, 'tokenizer.json')
CHUNK = 1_000_000  # 编码分块大小（字符），控制内存


def download_if_missing(local_name, book_name):
    """本地已有就不再下载。"""
    path = os.path.join(DATA_DIR, local_name)
    if os.path.exists(path):
        return
    import urllib.parse
    enc = urllib.parse.quote(book_name)
    for tmpl in (RAW_URL, MIRROR_URL):
        url = tmpl.format(enc=enc)
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(r.text)
                print(f'已下载《{book_name}》 -> {local_name}')
                return
        except Exception:
            continue
    print(f'警告：未能下载《{book_name}》，跳过。')


def encode_to_bin(text, tokenizer, out_path):
    """分块编码文本为 uint16 token ids，增量写入 bin 文件。"""
    with open(out_path, 'wb') as f:
        for i in range(0, len(text), CHUNK):
            ids = tokenizer.encode(text[i:i + CHUNK]).ids
            np.array(ids, dtype=np.uint16).tofile(f)


def main():
    import argparse
    ap = argparse.ArgumentParser(description='编码语料为 token ids')
    ap.add_argument('--with-books', action='store_true',
                    help='顺带下载四大名著补充语料（默认只用手头已有的 txt）')
    args = ap.parse_args()

    # 1) 可选：补齐四大名著（次要语料，网络不稳时默认跳过）
    if args.with_books:
        for local_name, book_name in BOOKS:
            download_if_missing(local_name, book_name)

    # 2) 加载 BPE 分词器（必须先跑 train_tokenizer.py）
    if not os.path.exists(TOKENIZER_PATH):
        raise SystemExit(f'错误：找不到 {TOKENIZER_PATH}，先跑 uv run python data/chinese/train_tokenizer.py')
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()
    print(f'BPE 分词器：{vocab_size} token')

    # 3) 读取所有 .txt 拼成一个语料
    parts = []
    for fn in sorted(os.listdir(DATA_DIR)):
        if fn.endswith('.txt'):
            with open(os.path.join(DATA_DIR, fn), 'r', encoding='utf-8') as f:
                parts.append(f.read())
    data = '\n'.join(parts)
    print(f'合并 {len(parts)} 个文件，{len(data):,} 字符')

    # 4) 90/10 分切
    n = len(data)
    train_data = data[:int(n * 0.9)]
    val_data = data[int(n * 0.9):]

    # 5) 编码 + 写 bin
    encode_to_bin(train_data, tokenizer, os.path.join(DATA_DIR, 'train.bin'))
    encode_to_bin(val_data, tokenizer, os.path.join(DATA_DIR, 'val.bin'))
    print(f'train token 数：{os.path.getsize(os.path.join(DATA_DIR, "train.bin")) // 2:,}')
    print(f'val token 数：{os.path.getsize(os.path.join(DATA_DIR, "val.bin")) // 2:,}')

    # 6) meta 信息（train.py 只读 vocab_size；推理端用 tokenizer.json 编解码）
    meta = {
        'vocab_size': vocab_size,
        'tokenizer_path': os.path.basename(TOKENIZER_PATH),
    }
    with open(os.path.join(DATA_DIR, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    print('完成 ✅ train.bin / val.bin / meta.pkl 已生成')


if __name__ == '__main__':
    main()
