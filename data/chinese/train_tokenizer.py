#!/usr/bin/env python3
"""训练 BPE 分词器：把所有语料 txt 训成一个 tokenizer.json。

为什么换掉字符级分词器：
    字符级 = 每个汉字 1 个 token，序列长、语义被拆散、信息密度低。
    BPE = 高频字/词组合并成子词 token，序列更短、语义更完整（对话领域尤其明显）。

用法（先跑 download_dialogue.py 下载语料，再跑本脚本）：
    uv run python data/chinese/train_tokenizer.py                 # 默认 8000 词表（日常中文）
    uv run python data/chinese/train_tokenizer.py --vocab-size 12000
"""
import os

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def collect_corpus_files():
    """收集 data/chinese/ 下所有 txt 语料文件。"""
    files = [f for f in sorted(os.listdir(DATA_DIR)) if f.endswith('.txt')]
    if not files:
        raise SystemExit(f'错误：{DATA_DIR} 下没有 txt 语料，先跑 download_dialogue.py 下载')
    paths = [os.path.join(DATA_DIR, f) for f in files]
    print(f'训练语料：{len(paths)} 个文件')
    for p in paths:
        print(f'  {os.path.basename(p)}  {os.path.getsize(p) / 1e6:.1f} MB')
    return paths


def main():
    import argparse
    ap = argparse.ArgumentParser(description='训练 BPE 分词器（全量语料）')
    ap.add_argument('--vocab-size', type=int, default=8000,
                    help='词表大小。默认 8000（日常中文）：嵌入表 8000×n_embd 很小，'
                         'transformer 参数占比高；词表越大压缩越好，但嵌入表越占参数。')
    args = ap.parse_args()

    tokenizer = Tokenizer(models.BPE(unk_token='<unk>'))
    # ByteLevel 对所有 Unicode（含中文）都友好，和 GPT-2 系模型一致
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,           # 只出现 1 次的字/片段不建 token，过滤噪声
        special_tokens=['<pad>', '<unk>', '<bos>', '<eos>'],
        show_progress=True,
    )

    tokenizer.train(files=collect_corpus_files(), trainer=trainer)

    out = os.path.join(DATA_DIR, 'tokenizer.json')
    tokenizer.save(out)
    print(f'\n分词器已保存 ✅ {out}')
    print(f'词表大小：{tokenizer.get_vocab_size()} token')

    # 演示：编码/解码往返
    demo = '悟空问：师傅，我们去西天取经要走多远？'
    ids = tokenizer.encode(demo).ids
    back = tokenizer.decode(ids)
    print(f'往返测试：{demo}')
    print(f'  → {len(demo)} 字符编码成 {len(ids)} token')
    print(f'  → 解码还原: {back}')


if __name__ == '__main__':
    main()
