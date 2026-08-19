"""
为语言建模准备中文对话数据集（BPE 分词）。

流程：download_dialogue.py 下载对话语料 → train_tokenizer.py 训出 tokenizer.json
→ 本脚本把所有 txt 编码成 token ids，写出 train.bin / val.bin / meta.pkl。

用法（从项目根目录）——默认即「全部数据 + turn-level EOS」（与默认
train_chinese.yaml 对应）：
    uv run python data/chinese/prepare.py
    # --task-ratio：非对话(任务/指令)样本保留比例。1.0=全保留（默认，所有 txt 都进训练）；
    #   0=剔除（train 只剩对话）；0.1=留 10%。
    # --insert-eos：每条「模型：」回复后插 <eos>（turn-level 终止符，默认开启）。
    # agent_dialogue.txt 归入 DIALOGUE_FILES，享受 90/10 验证切分。
"""
import datetime
import hashlib
import json
import os
import pickle
import random
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


def insert_eos_after_replies(block: str) -> str:
    """在每条「模型：」回复结束后插入结束符 <eos>（字面量，编码时映射为 special token id 3）。

    动机（2026-08-17，dev-notes/26）：训练分布里从来没有「结束」概念，小模型自回归只会
    一直续写下一轮 → 喋喋不休。给每条回复补 <eos>（chat 微调的 turn-level 终止符惯例，
    Llama-3 <|eot_id|> / Qwen <|im_end|> 同思路），模型才能学会「话说完 → 吐终止符」，
    解码端遇 <eos> 即停（sample_py.generate_ids 的 stop_on_eos；Rust 端按 dev-notes/02）。
    行级状态机处理多行回复：只在回复的最后一行后才插；用户轮次/空行不插。
    """
    lines = block.split("\n")
    out = []
    in_reply = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("用户：") or stripped.startswith("模型："):
            if in_reply:
                out.append("<eos>")            # 上一个模型回复在此结束
            in_reply = stripped.startswith("模型：")
            out.append(line)
        elif not stripped:
            if in_reply:
                out.append("<eos>")            # 回复被空行截断（兜底）
                in_reply = False
            if line:
                out.append(line)
        else:
            out.append(line)                   # 回复续行（多行回复）
    if in_reply:                               # 块尾仍是模型回复
        out.append("<eos>")
    return "\n".join(out)


def encode_to_bin(text, tokenizer, out_path):
    """分块编码文本为 uint16 token ids，增量写入 bin 文件。"""
    with open(out_path, 'wb') as f:
        for i in range(0, len(text), CHUNK):
            ids = tokenizer.encode(text[i:i + CHUNK]).ids
            np.array(ids, dtype=np.uint16).tofile(f)


def sha256_file(path):
    """计算文件 SHA-256，用于数据溯源/一致性校验。"""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(args, tokenizer, train_samples, val_samples,
                   train_data, val_data, meta):
    """把数据集的来源、切分、哈希等信息写进 manifest.json。

    以后训练 checkpoint 可以记录这个文件的哈希，就能回答“这个模型用的哪版数据”。
    """
    manifest = {
        "dataset": "chinese",
        "created_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "prepare_args": {
            "with_books": args.with_books,
            "task_ratio": args.task_ratio,
            "insert_eos": args.insert_eos,
        },
        "tokenizer": meta,
        "counts": {
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "train_chars": len(train_data),
            "val_chars": len(val_data),
            "train_tokens": os.path.getsize(os.path.join(DATA_DIR, 'train.bin')) // 2,
            "val_tokens": os.path.getsize(os.path.join(DATA_DIR, 'val.bin')) // 2,
        },
        "source_files": [],
        "artifacts": {},
    }
    for fn in sorted(os.listdir(DATA_DIR)):
        if not fn.endswith('.txt'):
            continue
        path = os.path.join(DATA_DIR, fn)
        manifest["source_files"].append({
            "file": fn,
            "size_bytes": os.path.getsize(path),
            "sha256": sha256_file(path),
            "mtime": datetime.datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds"),
        })
    for name in ('train.bin', 'val.bin', 'tokenizer.json', 'meta.pkl'):
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            manifest["artifacts"][name] = {
                "size_bytes": os.path.getsize(path),
                "sha256": sha256_file(path),
            }
    manifest_path = os.path.join(DATA_DIR, 'manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f'数据清单已写入：{manifest_path}')


def main():
    import argparse
    ap = argparse.ArgumentParser(description='编码语料为 token ids')
    ap.add_argument('--with-books', action='store_true',
                    help='顺带下载四大名著补充语料（默认只用手头已有的 txt）')
    ap.add_argument('--task-ratio', type=float, default=1.0,
                    help='非对话(任务/指令)样本保留比例：1.0=全保留(默认,所有数据)，0=剔除，0.1=留10%')
    ap.add_argument('--source-ratio', action='append', default=[], metavar='NAME=RATIO',
                    help='按文件名前缀降采样某源（仅 train 侧，val 不变保持可比）。'
                         '可重复：--source-ratio multi_turn=0.15 --source-ratio zhuangxialie=0.2')
    ap.add_argument('--insert-eos', action='store_true', default=True,
                    help='每条 模型： 回复后插入 <eos>（turn-level 终止符，治喋喋不休，默认开启）')
    ap.add_argument('--no-insert-eos', action='store_true',
                    help='关闭 --insert-eos（不插 <eos>，旧数据行为）')
    args = ap.parse_args()
    if args.no_insert_eos:
        args.insert_eos = False

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

    # 3) 读取所有 .txt，按"空行分隔的样本"（每条对话）拆开。
    #    旧实现是按文件拼接后整段硬切 90/10，会让 val 恰好落在最后一个文件
    #    （zhuangxialie 单轮指令）的后半段，而 train 主要是对话 → 分布错位，
    #    train-val gap 巨大（val 7.17 vs train 4.44）。
    #    策略（用户 2026-08-11）：val 只验证"对话"（核心目标），train 学全面。
    #    对话类文件（闲聊/多轮）单独 90/10 切：对话 90% 进 train、10% 进 val；
    #    非对话类文件（单轮指令/逻辑/古风）全部进 train，不参与 val。
    DIALOGUE_FILES = {'dailychat_dialogue.txt', 'muice_dialogue.txt', 'multi_turn_dialogue.txt',
                       'agent_dialogue.txt'}
    train_samples, val_samples = [], []
    for fn in sorted(os.listdir(DATA_DIR)):
        if not fn.endswith('.txt'):
            continue
        with open(os.path.join(DATA_DIR, fn), 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        blocks = [b.strip() for b in text.split('\n\n') if b.strip()]
        # --source-ratio NAME=RATIO：按文件名降采样该源（只作用于 train 侧，
        # val 保持 90/10 全量 → val.bin 不变，跨实验 val 可比）。NAME 是文件名前缀，
        # 如 multi_turn=0.15 / zhuangxialie=0.2。优先级高于 task_ratio。
        src_ratio = None
        for spec in args.source_ratio:
            name, ratio = spec.split('=', 1)
            if fn.startswith(name):
                src_ratio = float(ratio)
        if fn in DIALOGUE_FILES:
            random.seed(1337)  # 固定 seed：重复运行切分一致，实验结果可复现
            random.shuffle(blocks)
            n = int(len(blocks) * 0.9)
            train_blocks, val_blocks = blocks[:n], blocks[n:]
            if src_ratio is not None and src_ratio < 1.0:
                random.seed(1337 + sum(ord(c) for c in fn) + 1)
                random.shuffle(train_blocks)
                train_blocks = train_blocks[:max(1, int(len(train_blocks) * src_ratio))]
            train_samples += train_blocks
            val_samples += val_blocks
        elif src_ratio is not None:
            random.seed(1337 + sum(ord(c) for c in fn) + 1)
            random.shuffle(blocks)
            n = max(1, int(len(blocks) * src_ratio))
            train_samples += blocks[:n]
        elif args.task_ratio >= 1.0:
            train_samples += blocks  # 默认：非对话(任务/指令)全部进训练，让模型学全面
        elif args.task_ratio > 0:
            # 数据治理（2026-08-12）：任务/指令样本按比例抽样进 train。
            # 根因：zhuangxialie(149MB 单轮指令)占 train 55%，模型自由生成学成
            # "碎片拼贴"（对对联/实体识别/热评等任务模板拼贴，dev-notes 见 21）。
            # 降比例让对话主导；每个文件独立 seed 保证可复现。
            random.seed(1337 + sum(ord(c) for c in fn))
            random.shuffle(blocks)
            n = max(1, int(len(blocks) * args.task_ratio))
            train_samples += blocks[:n]
        # task_ratio == 0：任务/指令样本剔除，train 只剩对话
    if args.insert_eos:
        train_samples = [insert_eos_after_replies(b) for b in train_samples]
        val_samples = [insert_eos_after_replies(b) for b in val_samples]
    print(f'训练 {len(train_samples)} 条 / 验证 {len(val_samples)} 条（仅对话）')
    train_data = '\n\n'.join(train_samples)
    val_data = '\n\n'.join(val_samples)
    print(f'{len(train_data):,} 训练字符 / {len(val_data):,} 验证字符')

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
    write_manifest(args, tokenizer, train_samples, val_samples,
                   train_data, val_data, meta)
    print('完成 ✅ train.bin / val.bin / meta.pkl / manifest.json 已生成')


if __name__ == '__main__':
    main()
