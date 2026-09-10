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

============================================================================
变更记录 2026-09-10（任务：重建中文语料数据集）
============================================================================
【问题】修前的 train/val 切分是"双标准"：
    - DIALOGUE_FILES 硬编码 5 个文件名（其中 agent_dialogue.txt 实际不存在，
      真正命中的只有 dailychat / muice / multi_turn / zhihu_kol 4 个）按 10% 抽 val；
    - 其它所有来源（含 deepseek_r1_distill、qwen3_235b_distill、coig_wiki、
      lccc、c4_zh、wikipedia_cn 等）只在 `--val-all` 下按 `--val-ratio`（实跑 0.01）抽 val。
    后果：对话类在 val 里被放大约 6.5 倍。实测修前 train 中 4 个对话文件只占 7.08%
    字符，却占了 val 的 45.70% 字符；val 的 <eos>/<cont> 密度和"有效 token 占比"
    远高于 train，两个 split 实际不是同一个任务（见 DATASET_REPORT.md）。

【修复】`--val-all` 现在真正统一：**所有来源**都用同一个 `--val-ratio` 抽 val，
    DIALOGUE_FILES 不再有第二套 90/10 标准。DIALOGUE_FILES 常量保留，只在
    *不传* `--val-all` 时（旧默认行为）生效，因此默认调用方的语义不变。
    同时去掉了 val 侧 `max(1, ...)`：块数 <100 的小源不再被整段抽进 val。
    统一后 val 的来源构成与 train 一致（目标：bigram CE 差距 < 0.15 nats）。

【新增】`--out-prefix PREFIX`：所有产物加后缀，例如 `--out-prefix v2` 写出
    train_char_v2.bin / val_char_v2.bin / meta_char_v2.pkl / manifest_v2.json。
    默认空 = 旧文件名，保证不覆盖既有数据集。
    `--seed`：可选，固定 annotate_replies 的随机种子，让重建结果可复现
    （不传时保持旧行为 = 不设种子）。

【注意】`--source-ratio` 仍然只作用于 train 侧（val 保持不变以跨实验可比）；
    它与统一 val 比例叠加时，会让被降采样的源在 val 中占比偏高。本次重建
    未使用任何 `--source-ratio`。

【另修一个真 bug】encode_to_bin 原先只在"两个特殊符之间的整段"处理完后才
    flush；train 拼接的第一个来源 c4_zh.txt（~1.89 亿字符、无 <eos>/<cont>）
    会被整段塞进 buffer → 实测 RSS 7.1GB、系统可用内存剩 469MB、swap 打满，
    险些 OOM。现改为在 1M 字符分块内 flush；已用已知输入做逐字节等价自检
    （输出与旧实现完全一致）。

【留痕】manifest 的 prepare_args 现在记录完整 `argv`。教训：旧构建的
    source_ratio 未落盘，产物里发现 ~9.5 万 block 的 train 缺口却无法反查命令。
"""
import datetime
import hashlib
import json
import os
import pickle
import random
import sys
import requests
import numpy as np
from split_sentences import split_text  # 数据分句器（dev-notes/49）
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
CHAR_TOKENIZER_PATH = os.path.join(DATA_DIR, 'char_tokenizer.json')
CHUNK = 1_000_000  # 编码分块大小（字符），控制内存

# 旧默认行为里享受 90/10 切分的"对话类"文件（agent_dialogue.txt 实际不存在，
# 留着以兼容历史语义）。仅在 *不传* --val-all 时生效；传 --val-all 时所有来源统一比例。
DIALOGUE_FILES = {'dailychat_dialogue.txt', 'muice_dialogue.txt', 'multi_turn_dialogue.txt',
                  'agent_dialogue.txt', 'zhihu_kol_dialogue.txt'}


def named_output(stem, prefix, ext):
    """按 --out-prefix 生成产物文件名：空 prefix 保持旧名，否则 stem_prefix.ext。"""
    return f'{stem}{("_" + prefix) if prefix else ""}{ext}'


def split_one_source(fn, blocks, args, src_ratio=None):
    """切分单个来源的 blocks 为 (train_blocks, val_blocks)。

    修复点（2026-09-10）：`args.val_all` 为真时，**所有来源**（包括
    DIALOGUE_FILES）统一用 `args.val_ratio` 抽 val —— val 的来源构成与 train 一致。
    不传 `--val-all` 时完全保持旧行为（对话 90/10，非对话全部进 train）。
    """
    n = len(blocks)
    if args.val_all:
        # 统一比例切分（修复双标准）。seed 按文件名派生 → 各源独立、可复现。
        random.seed(1337 + sum(ord(c) for c in fn) + 7)
        random.shuffle(blocks)
        n_val = int(n * args.val_ratio)          # 注意：不再 max(1,·)，小源可贡献 0 条 val
        val_blocks, train_blocks = blocks[:n_val], blocks[n_val:]
        if src_ratio is not None and src_ratio < 1.0:
            random.seed(1337 + sum(ord(c) for c in fn) + 1)
            random.shuffle(train_blocks)
            train_blocks = train_blocks[:max(1, int(len(train_blocks) * src_ratio))]
        return train_blocks, val_blocks
    if fn in DIALOGUE_FILES:
        random.seed(1337)  # 固定 seed：重复运行切分一致，实验结果可复现
        random.shuffle(blocks)
        k = int(n * 0.9)
        train_blocks, val_blocks = blocks[:k], blocks[k:]
        if src_ratio is not None and src_ratio < 1.0:
            random.seed(1337 + sum(ord(c) for c in fn) + 1)
            random.shuffle(train_blocks)
            train_blocks = train_blocks[:max(1, int(len(train_blocks) * src_ratio))]
        return train_blocks, val_blocks
    if src_ratio is not None:
        random.seed(1337 + sum(ord(c) for c in fn) + 1)
        random.shuffle(blocks)
        k = max(1, int(n * src_ratio))
        return blocks[:k], []
    if args.task_ratio >= 1.0:
        return blocks, []  # 默认：非对话(任务/指令)全部进训练，让模型学全面
    if args.task_ratio > 0:
        # 数据治理（2026-08-12）：任务/指令样本按比例抽样进 train。
        # 根因：zhuangxialie(149MB 单轮指令)占 train 55%，模型自由生成学成
        # "碎片拼贴"（对对联/实体识别/热评等任务模板拼贴，dev-notes 见 21）。
        random.seed(1337 + sum(ord(c) for c in fn))
        random.shuffle(blocks)
        k = max(1, int(n * args.task_ratio))
        return blocks[:k], []
    return [], []  # task_ratio == 0：任务/指令样本剔除，train 只剩对话


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


# 追问/待续判定：回复"期待用户继续回应"→ 标注 <cont>；否则收尾 → <eos>。
# 语义(dev-notes/61)：<eos> = 本轮话说完可以停；<cont> = 本轮说完但请对话继续(递回/追问)。
CONTINUE_QUESTION = ["？", "?", "吧", "呢", "对不对", "是不是", "你觉得", "你说呢",
                     "怎么样", "想不想", "要不要", "怎么样？", "如何", "们看"]
CONTINUE_PHRASE = ["你觉", "你呢", "怎么样", "是不是", "想不想", "要不要",
                   "你说呢", "对不对", "如何", "可以吗", "好吗", "吗?", "吗？"]


def _should_continue(reply: str) -> bool:
    """回复是否属"待续/递回(期待用户继续)"类型。启发式: 含疑问标点或追问/递回短语。"""
    r = reply.strip()
    if not r:
        return False                      # 空回复不标注(调用方兜底)
    for q in ("？", "?"):
        if q in r:
            return True                   # 带问号 → 倾向于待续
    for p in CONTINUE_PHRASE:
        if p in r:
            return True
    return False


def annotate_replies(block: str, ab_rate: float = 0.7, quote_rate: float = 0.8) -> str:
    """样本级去开口标签 + 待续符标注（dev-notes/61）。

    把「用户：/模型：」强标签对话转换成更自然的混合样式，并在每条模型回复后
    插终止符 <eos>(收尾停) 或 <cont>(说完请继续/递回)：
      - 说话人维度（样本级随机, 概率 ab_rate=0.7）:
          70% → A：/B： 前缀 (A=对方/用户, B=模型)
          30% → 无说话人前缀（纯文本, 换行分句, 上下文内一致）
      - 引号维度（样本级随机, 概率 quote_rate=0.8, 与说话人正交）:
          80% → 每句用 "..." 引号包裹
          20% → 无引号
      两种维度独立 → 四种组合: "A：..." / A：... / "..." / 裸文本
    <eos>/<cont> 统一插在模型(B/偶数段)回复后：待续/追问 → <cont>, 收尾 → <eos>。
    """
    import random
    use_ab = random.random() < ab_rate
    use_quote = random.random() < quote_rate

    lines = block.split("\n")
    out = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("用户：") or stripped.startswith("模型："):
            is_model = stripped.startswith("模型：")
            body = stripped.split("：", 1)[1] if "：" in stripped else stripped
            if use_ab:
                speaker = "B" if is_model else "A"
                text = f"{speaker}：{body}"
            else:
                text = body
            # 引号维度（正交）；终止符紧随回复正文同行（引号外），不独立成行
            if use_quote:
                text = f'"{text}"'
            if is_model:
                ann = "<cont>" if _should_continue(body) else "<eos>"
                text = text + ann
            out.append(text)
        elif not stripped:
            if line:
                out.append(line)
        else:
            out.append(line)               # 非对话行（备注/续行）原样保留
    return "\n".join(out)


def insert_eos_after_replies(block: str) -> str:
    """兼容旧名：一律插 <eos>（标注 <cont> 由 --no-annotate 关闭时使用）。"""
    return annotate_replies(block)


def encode_to_bin(text, tokenizer, out_path):
    """分块编码文本为 uint16 token ids，增量写入 bin 文件。

    字面量 <eos>/<cont> 映射为 tokenizer 的对应 special id（而非逐字符编码），
    与 encode_bytes_to_bin 逻辑一致——先按特殊符分割，各段独立编码，
    段间插入对应 id。分块 flush 控制内存。
    """
    # 特殊符字面量 → 目标 id 映射（只映射词表中存在且为特殊符的）
    markers = {}
    for sym in ("<eos>", "<cont>"):
        sid = tokenizer.token_to_id(sym)
        if sid is not None:
            markers[sym] = sid
    import re
    # 通用分割：按任一特殊符字面量切分，并保留分隔物
    pattern = re.compile("|".join(re.escape(m) for m in markers) or r"$^")
    with open(out_path, 'wb') as f:
        buf = []
        prev_end = 0

        def _flush():
            """把 buf 写成 uint16；在分块内调用，避免长段（如无终止符的 c4_zh
            189M 字符）把 buf 撑到数 GB 导致 OOM。输出字节与一次性写完全一致。"""
            nonlocal buf
            if buf:
                np.array(buf, dtype=np.uint16).tofile(f)
                buf = []

        for m in pattern.finditer(text):
            seg = text[prev_end:m.end() - len(m.group())]
            # 段前普通文本
            if seg:
                for j in range(0, max(len(seg), 1), CHUNK):
                    chunk = seg[j:j + CHUNK]
                    if chunk:
                        buf.extend(tokenizer.encode(chunk).ids)
                        if len(buf) >= 1 << 20:
                            _flush()
            # 特殊符 id
            if markers.get(m.group()) is not None:
                buf.append(markers[m.group()])
            prev_end = m.end()
            if len(buf) >= 1 << 20:
                _flush()
        # 结尾余段
        seg = text[prev_end:]
        if seg:
            for j in range(0, max(len(seg), 1), CHUNK):
                chunk = seg[j:j + CHUNK]
                if chunk:
                    buf.extend(tokenizer.encode(chunk).ids)
                    if len(buf) >= 1 << 20:
                        _flush()
        _flush()

EOS_ID = 256  # 字节直入模式（dev-notes/48）：0-255 = UTF-8 字节，256 = <eos>


def encode_bytes_to_bin(text, out_path):
    """字节直入版：文本 → UTF-8 字节 id（0-255），字面量 <eos> 映射 256。

    不经过 BPE 分词（Mamba-Byte 思想：无分词器、无 OOV、话术片段不固化在词表）。
    整段编码（<eos> 分割避免跨块切半），uint16 增量写 bin。
    """
    with open(out_path, 'wb') as f:
        buf = []
        parts = text.split('<eos>')
        for i, part in enumerate(parts):
            buf.extend(part.encode('utf-8'))       # 0-255 字节
            if i < len(parts) - 1:
                buf.append(EOS_ID)
            if len(buf) >= 1 << 20:                # 每 ~1M id flush，控峰值内存
                np.array(buf, dtype=np.uint16).tofile(f)
                buf = []
        if buf:
            np.array(buf, dtype=np.uint16).tofile(f)


def sha256_file(path):
    """计算文件 SHA-256，用于数据溯源/一致性校验。"""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(args, tokenizer, train_samples, val_samples,
                   train_data, val_data, meta,
                   train_bin=None, val_bin=None, meta_path=None,
                   manifest_path=None, source_stats=None,
                   n_train_samples=None, n_val_samples=None):
    """把数据集的来源、切分、哈希等信息写进 manifest（默认 manifest.json）。

    train_bin/val_bin/meta_path 传入真实产物名（支持 --out-prefix），
    source_stats 传入逐来源 (blocks/train_chars/val_chars) 统计用于审计。
    """
    stem = getattr(args, 'out_prefix', '') or ''
    if manifest_path is None:
        manifest_path = os.path.join(DATA_DIR, named_output('manifest', stem, '.json'))
    if train_bin is None:
        train_bin = os.path.join(DATA_DIR, 'train.bin')
    if val_bin is None:
        val_bin = os.path.join(DATA_DIR, 'val.bin')
    manifest = {
        "dataset": "chinese",
        "created_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "prepare_args": {
            "with_books": args.with_books,
            "task_ratio": args.task_ratio,
            "insert_eos": args.insert_eos,
            "pretrain": getattr(args, 'pretrain', False),
            "char_level": getattr(args, 'char_level', False),
            "byte_level": getattr(args, 'byte_level', False),
            "val_all": getattr(args, 'val_all', False),
            "val_ratio": getattr(args, 'val_ratio', None),
            "source_ratio": list(getattr(args, 'source_ratio', []) or []),
            "seed": getattr(args, 'seed', None),
            "out_prefix": getattr(args, 'out_prefix', ''),
            # 完整命令行：2026-09 事故教训——旧构建的 source_ratio 没落盘，
            # 导致 train 缺口 9.5 万 block 无法从产物反查。以后一律留痕。
            "argv": list(sys.argv),
        },
        "tokenizer": meta,
        "counts": {
            "train_samples": (n_train_samples if n_train_samples is not None
                              else len(train_samples)),
            "val_samples": (n_val_samples if n_val_samples is not None
                            else len(val_samples)),
            "train_chars": len(train_data),
            "val_chars": len(val_data),
            "train_tokens": os.path.getsize(train_bin) // 2,
            "val_tokens": os.path.getsize(val_bin) // 2,
        },
        "source_files": [],
        "source_breakdown": source_stats or [],
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
    for name in (os.path.basename(train_bin), os.path.basename(val_bin),
                 'tokenizer.json', os.path.basename(meta_path) if meta_path else 'meta.pkl'):
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            manifest["artifacts"][name] = {
                "size_bytes": os.path.getsize(path),
                "sha256": sha256_file(path),
            }
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f'数据清单已写入：{manifest_path}')


def main():
    import argparse
    ap = argparse.ArgumentParser(description='编码语料为 token ids')
    ap.add_argument('--with-books', action='store_true',
                    help='顺带下载四大名著补充语料（默认只用手头已有的 txt）')
    ap.add_argument('--task-ratio', type=float, default=1.0,
                    help='非对话(任务/指令)样本保留比例：1.0=全保留(默认,所有数据)，0=剔除，0.1=留10%%')
    ap.add_argument('--val-ratio', type=float, default=0.1,
                    help='当 --val-all 开启时，**所有来源统一**抽到验证集的样本比例'
                         '（默认0.1=10%%）。推荐 0.01=1%%：val 仍 ≈9.4M token，'
                         '且 train 只损失 1%%。不传 --val-all 时此参数不生效'
                         '（旧行为：DIALOGUE_FILES 固定 90/10）。')
    ap.add_argument('--val-all', action='store_true',
                    help='让所有来源(含百科/网页/指令)都参与验证集抽样，'
                         '并统一使用 --val-ratio（修复"对话10%%/其它1%%"的双标准 bug），'
                         'val 来源构成与 train 一致。')
    ap.add_argument('--out-prefix', default='', metavar='PREFIX',
                    help='产物文件名加后缀，避免覆盖现有数据集：'
                         '--out-prefix v2 → train_char_v2.bin / val_char_v2.bin / '
                         'meta_char_v2.pkl / manifest_v2.json。默认空=旧文件名。')
    ap.add_argument('--seed', type=int, default=None,
                    help='可选：固定 annotate_replies 的随机种子，使重建结果可复现。'
                         '不传则保持旧行为（不设种子，每次标注随机）。')
    ap.add_argument('--source-ratio', action='append', default=[], metavar='NAME=RATIO',
                    help='按文件名前缀降采样某源（仅 train 侧，val 不变保持可比）。'
                         '可重复：--source-ratio multi_turn=0.15 --source-ratio zhuangxialie=0.2')
    ap.add_argument('--insert-eos', action='store_true', default=True,
                    help='每条 模型： 回复后插入 <eos>（turn-level 终止符，治喋喋不休，默认开启）')
    ap.add_argument('--no-insert-eos', action='store_true',
                    help='关闭 --insert-eos（不插 <eos>，旧数据行为）')
    ap.add_argument('--byte-level', action='store_true',
                    help='字节直入模式（dev-notes/48）：输出 train_byte.bin/val_byte.bin，'
                         '0-255=UTF-8 字节 + 256=<eos>，不经过 BPE 分词。不动 BPE 的 train.bin')
    ap.add_argument('--split-sentences', action='store_true',
                    help='数据分句（dev-notes/49）：按句末标点切分（保留标点、超长句二次切），'
                         '模型按完整句子阅读/生成，不硬截断句子')
    ap.add_argument('--char-level', action='store_true',
                    help='字级模式（dev-notes/50）：汉字=1 token，输出 train_char.bin/val_char.bin，'
                         '词表见 char_tokenizer.json（对齐字符 + 无话术固化）')
    ap.add_argument('--pretrain', action='store_true',
                    help='预训练模式：所有 txt 按原始文本编码（不解析 用户：/模型： 结构、'
                         '不插 <eos>、无 loss mask 概念），输出 pretrain.bin/val.bin——'
                         '对应 train.py --stage=pretrain 的无掩码全 token 训练')
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

    # 0) 预训练模式：全部 txt 按原始文本 90/10 字符切分，不解析对话结构、不插 <eos>。
    #    输出 pretrain.bin（train.py --stage=pretrain 读它做无掩码全 token 训练）。
    if args.pretrain:
        train_parts, val_parts = [], []
        for fn in sorted(os.listdir(DATA_DIR)):
            if not fn.endswith('.txt'):
                continue
            with open(os.path.join(DATA_DIR, fn), 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
            n = int(len(text) * 0.9)
            train_parts.append(text[:n])
            val_parts.append(text[n:])
        train_data = ''.join(train_parts)
        val_data = ''.join(val_parts)
        pfx = args.out_prefix or ''
        pretrain_bin = os.path.join(DATA_DIR, named_output('pretrain', pfx, '.bin'))
        val_bin = os.path.join(DATA_DIR, named_output('val', pfx, '.bin'))
        meta_path = os.path.join(DATA_DIR, named_output('meta', pfx, '.pkl'))
        encode_to_bin(train_data, tokenizer, pretrain_bin)
        encode_to_bin(val_data, tokenizer, val_bin)
        meta = {'vocab_size': vocab_size, 'tokenizer_path': os.path.basename(TOKENIZER_PATH)}
        print(f'预训练数据：{len(train_data):,} 训练字符 / {len(val_data):,} 验证字符')
        print(f'pretrain token 数：{os.path.getsize(pretrain_bin) // 2:,}')
        print(f'val token 数：{os.path.getsize(val_bin) // 2:,}')
        with open(meta_path, 'wb') as f:
            pickle.dump(meta, f)
        write_manifest(args, tokenizer, train_parts, val_parts,
                       train_data, val_data, meta,
                       train_bin=pretrain_bin, val_bin=val_bin, meta_path=meta_path)
        print('完成 ✅ pretrain.bin / val.bin / meta.pkl / manifest.json 已生成（预训练模式）')
        return

    # 3) 读取所有 .txt，按"空行分隔的样本"（每条对话）拆开。
    #    旧实现是按文件拼接后整段硬切 90/10，会让 val 恰好落在最后一个文件
    #    （zhuangxialie 单轮指令）的后半段，而 train 主要是对话 → 分布错位，
    #    train-val gap 巨大（val 7.17 vs train 4.44）。
    #    2026-09-10 修复：--val-all 时所有来源统一 --val-ratio，val 来源构成对齐 train。
    #    切分逻辑见 split_one_source()（模块级函数，审计脚本可复用）。
    train_samples, val_samples = [], []
    source_stats = []
    for fn in sorted(os.listdir(DATA_DIR)):
        if not fn.endswith('.txt'):
            continue
        with open(os.path.join(DATA_DIR, fn), 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        blocks = [b.strip() for b in text.split('\n\n') if b.strip()]
        # --source-ratio NAME=RATIO：按文件名降采样该源（只作用于 train 侧，
        # val 保持全量 → val.bin 不变，跨实验 val 可比）。NAME 是文件名前缀，
        # 如 multi_turn=0.15 / zhuangxialie=0.2。优先级高于 task_ratio。
        src_ratio = None
        for spec in args.source_ratio:
            name, ratio = spec.split('=', 1)
            if fn.startswith(name):
                src_ratio = float(ratio)
        train_blocks, val_blocks = split_one_source(fn, blocks, args, src_ratio)
        train_samples += train_blocks
        val_samples += val_blocks
        source_stats.append({
            'file': fn,
            'size_bytes': os.path.getsize(os.path.join(DATA_DIR, fn)),
            'blocks': len(blocks),
            'train_blocks': len(train_blocks),
            'val_blocks': len(val_blocks),
            'train_chars_raw': sum(len(b) for b in train_blocks),
            'val_chars_raw': sum(len(b) for b in val_blocks),
        })
    if args.seed is not None:
        random.seed(args.seed)   # 可选：固定标注随机性，使重建可复现
    if args.insert_eos:
        train_samples = [insert_eos_after_replies(b) for b in train_samples]
        val_samples = [insert_eos_after_replies(b) for b in val_samples]
    print(f'训练 {len(train_samples)} 条 / 验证 {len(val_samples)} 条')
    train_data = '\n\n'.join(train_samples)
    val_data = '\n\n'.join(val_samples)
    # 释放 block 列表（~2GB），降低编码阶段峰值内存
    n_train_samples, n_val_samples = len(train_samples), len(val_samples)
    del train_samples, val_samples
    print(f'{len(train_data):,} 训练字符 / {len(val_data):,} 验证字符')

    # 5) 编码 + 写 bin（--char-level：字级；--byte-level：字节直入；默认：BPE）
    #    --out-prefix 给所有产物加后缀（如 v2），绝不覆盖既有数据集。
    pfx = args.out_prefix or ''
    if args.char_level:
        train_bin = os.path.join(DATA_DIR, named_output('train_char', pfx, '.bin'))
        val_bin = os.path.join(DATA_DIR, named_output('val_char', pfx, '.bin'))
        char_tok = Tokenizer.from_file(CHAR_TOKENIZER_PATH)   # WordLevel 字级（dev-notes/50）
        encode_to_bin(train_data, char_tok, train_bin)
        encode_to_bin(val_data, char_tok, val_bin)
        vocab_size = char_tok.get_vocab_size()
        meta = {'vocab_size': vocab_size, 'char_level': True,
                'tokenizer_path': os.path.basename(CHAR_TOKENIZER_PATH)}
        meta_path = os.path.join(DATA_DIR, named_output('meta_char', pfx, '.pkl'))  # 独立 meta
    elif args.byte_level:
        train_bin = os.path.join(DATA_DIR, named_output('train_byte', pfx, '.bin'))
        val_bin = os.path.join(DATA_DIR, named_output('val_byte', pfx, '.bin'))
        encode_bytes_to_bin(train_data, train_bin)
        encode_bytes_to_bin(val_data, val_bin)
        vocab_size = 257  # 0-255 字节 + <eos>=256
        meta = {'vocab_size': vocab_size, 'byte_level': True}
        meta_path = os.path.join(DATA_DIR, named_output('meta_byte', pfx, '.pkl'))
    else:
        train_bin = os.path.join(DATA_DIR, named_output('train', pfx, '.bin'))
        val_bin = os.path.join(DATA_DIR, named_output('val', pfx, '.bin'))
        encode_to_bin(train_data, tokenizer, train_bin)
        encode_to_bin(val_data, tokenizer, val_bin)
        vocab_size = tokenizer.get_vocab_size()
        meta = {
            'vocab_size': vocab_size,
            'tokenizer_path': os.path.basename(TOKENIZER_PATH),
        }
        meta_path = os.path.join(DATA_DIR, named_output('meta', pfx, '.pkl'))
    print(f'train token 数：{os.path.getsize(train_bin) // 2:,}')
    print(f'val token 数：{os.path.getsize(val_bin) // 2:,}')

    # 6) meta 信息（train.py 只读 vocab_size；推理端用 tokenizer.json 编解码）
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
    # 逐来源占比：把 raw 字符统计换算成占比写进 manifest，便于审计
    tot_t = sum(s['train_chars_raw'] for s in source_stats) or 1
    tot_v = sum(s['val_chars_raw'] for s in source_stats) or 1
    for s in source_stats:
        s['train_share'] = round(s['train_chars_raw'] / tot_t, 6)
        s['val_share'] = round(s['val_chars_raw'] / tot_v, 6)
    write_manifest(args, tokenizer, None, None,
                   train_data, val_data, meta,
                   train_bin=train_bin, val_bin=val_bin, meta_path=meta_path,
                   source_stats=source_stats,
                   n_train_samples=n_train_samples, n_val_samples=n_val_samples)
    print(f'完成 ✅ {os.path.basename(train_bin)} / {os.path.basename(val_bin)} / '
          f'{os.path.basename(meta_path)} / '
          f'{named_output("manifest", pfx, ".json")} 已生成')


if __name__ == '__main__':
    main()
