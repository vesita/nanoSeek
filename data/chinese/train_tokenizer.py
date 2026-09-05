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

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers, Regex

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


def build_char_wordlevel(paths, out_json, hanzi_top=4400, mech_slots=16,
                         base_slots=128, mech_zone_slots=256, hanzi_zone_slots=7808):
    """字级词表 v3（dev-notes/50 + 61 + 76 稀疏分区布局）。

    四区稀疏布局（改良自 v2，为未来扩充预留稀疏空间，新增机制符/汉字不再错位）：
      1. 基础文本字符区 (base_slots=128)   : 换行 + 空格 + ASCII 32-126 + 全角标点
          实际占用 ~117 位，其余填 <reserved_base_i> 稀疏占位（对齐 128 边界）。
      2. 机制符区间 (mech_zone_slots=256)  : <eos>/<unk>/<cont> + 命名机制符
          (<pad>/<bos>/<sep>/<call>/<result>/<think>/<answer>...) + <res_i> 稀疏占位。
          预留充足，未来 Agent/RL 新增机制符直接占空位，汉字区 id 零错位。
      3. 汉字区 (hanzi_zone_slots=7808)   : 按治理后语料词频降序，top hanzi_top 个常用字，
          其余 <res_han_i> 稀疏占位，未来语料出现新低频字可无缝追加。

    总词表 = 128 + 256 + 7808 = 8192 (2^13，对齐硬件 GEMM 高效边界)。
    稀疏空间总预留 ≈ (128-117) + (256-命名符) + (7808-hanzi_top) 位。
    """
    import collections

    # ── 区 1: 基础文本字符（固定顺序）──
    fullwidth = "。！？，、；：\"\"''（）《》…—·～「」『』【】"
    base = ["\n", " "] + [chr(c) for c in range(32, 127)] + list(fullwidth)
    base = list(dict.fromkeys(base))   # 去重保序
    # 稀疏占位：不足 base_slots 的用 <reserved_base_i> 填充
    base += [f"<reserved_base{i}>" for i in range(max(0, base_slots - len(base)))]

    # ── 区 2: 机制符区间（统一管理 + 稀疏预留）──
    # 命名机制符（前几个固定语义，顺序即 id，绝不更改）：
    mech_names = ["<eos>", "<unk>", "<cont>", "<pad>", "<bos>", "<sep>",
                  "<call>", "<result>", "<think>", "<answer>", "<tool>", "<search>"]
    # 剩余稀疏占位：未来机制符扩展区
    mech_reserved = [f"<res{i}>" for i in range(max(0, mech_zone_slots - len(mech_names)))]
    mec = mech_names + mech_reserved

    # ── 区 3: 汉字（治理后语料词频降序，稀疏尾预留）──
    counter = collections.Counter()
    for p in paths:
        with open(p, encoding="utf-8", errors="replace") as f:
            counter.update(f.read())
    hans = [(c, n) for c, n in counter.items()
            if "\u4e00" <= c <= "\u9fff" and c not in base and c not in mec]
    hans.sort(key=lambda x: -x[1])
    han_toks = [c for c, _ in hans[:hanzi_top]]
    # 稀疏占位：不足 hanzi_zone_slots 的用 <res_han_i> 填充（未来追加低频字）
    han_toks += [f"<res_han{i}>" for i in range(max(0, hanzi_zone_slots - len(han_toks)))]

    # ── 拼装词表 ──
    vocab: dict = {}
    def _add(items):
        for it in items:
            if it not in vocab:
                vocab[it] = len(vocab)
    _add(base)          # 0..127       基础文本字符区（128 位，稀疏对齐）
    _add(mec)           # 128..383     机制符区间（256 位，含命名符 + 稀疏预留）
    _add(han_toks)      # 384..8191    汉字区（7808 位，词频降序 + 稀疏尾）

    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    tok.decoder = decoders.ByteLevel(add_prefix_space=False)
    # WordLevel 需要特殊 token 声明(added_tokens)使 <eos>/<cont> 可被 encode/decode 完整保留
    from tokenizers import AddedToken
    for name in mech_names:
        tok.add_special_tokens([AddedToken(name, special=True)])
    tok.save(out_json)
    print(f"\n字级词表 v3 {len(vocab)} 项 → {out_json}（WordLevel, 四区稀疏布局）")
    print(f"  基础字符区 {len(base)} | 机制符区间 {len(mec)} (命名{len(mech_names)}+预留{len(mech_reserved)}) | 汉字区 {len(han_toks)} (实际{hanzi_top}+稀疏{max(0, hanzi_zone_slots-hanzi_top)})")
    for i, name in enumerate(mech_names):
        print(f"    机制符 {name:>10} → id {base_slots + i}")
    return tok

def main():
    import argparse
    ap = argparse.ArgumentParser(description='训练 BPE 分词器（全量语料）')
    ap.add_argument('--vocab-size', type=int, default=8000,
                    help='词表大小。默认 8000（日常中文）：嵌入表 8000×n_embd 很小，'
                         'transformer 参数占比高；词表越大压缩越好，但嵌入表越占参数。')
    ap.add_argument('--char', action='store_true',
                    help='构建字级 WordLevel 词表（dev-notes/50, 三区布局见 61）替代 BPE：'
                         '汉字=1 token，对齐字符 + 无话术固化（产出 char_tokenizer.json）')
    ap.add_argument('--hanzi-top', type=int, default=4400,
                    help='字级词表汉字数（词频降序截断），基础字符121 + 机制符16 + 汉字n')
    args = ap.parse_args()

    if args.char:
        paths = collect_corpus_files()
        build_char_wordlevel(paths, os.path.join(DATA_DIR, 'char_tokenizer.json'),
                             hanzi_top=args.hanzi_top)
        return

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
