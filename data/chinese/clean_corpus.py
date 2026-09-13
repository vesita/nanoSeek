#!/usr/bin/env python
"""语料清洗工具：在 `prepare.py` **之前**跑，把清洗后的 `*.txt` 写到一个新目录。

## 为什么要它

v2 语料实测病症（`manifest_v2.json` / 父 agent 的扫描）：

    1,582,859 block / 946,882,745 字符
    精确重复 block（整块）             5.00%
    含 rep3>0.3 回复的 block           2.18%
    逐条精确去重只覆盖                 0.11% 字符  ← 主症不是整块重复

主症是**短语级套话复用**：同一批句子（"你说的很有道理" / "我在，慢慢说就行"）
嵌在大量**互不相同**的整块回复里。整块哈希看不见它们，只有按字符 K-gram 频次
才能看见 —— 所以有了规则 3。

## 用法（默认 dry-run，只有 `--apply` 才写数据文件）

    .venv/bin/python data/chinese/clean_corpus.py \
        --src data/chinese --dst data/chinese/clean_v3 [规则参数] [--apply]

规则（**全部默认关闭**；阈值由调用方按实测拍板，工具不自带默认值）：

    --dedup-blocks                      block 级精确去重（去空白归一化后哈希，全局跨文件）
    --max-reply-rep3 F                  block 内任一 `模型：` 回复 rep3 > F → 丢
    --max-ngram-freq N --ngram-size K --max-ngram-cover C
                                        短语级：全语料回复里频次 > N 的字符 K-gram
                                        覆盖某 block 回复字符的比例 > C → 丢
    --min-reply-chars M                 block 内**所有** `模型：` 回复都 < M 字 → 丢
    --filter-nondialogue-rep3           对非对话 block 也套 rep3（默认**不套**）
    --nondialogue-max-rep3 G            非对话 block 的 rep3 阈值（不给则 = F）。
                                        为什么不能复用 F：rep3 随文本长度单调上升
                                        （同一段干净文本 50 字 rep3=0.024 → 1600 字 0.131），
                                        而对话回复均长 65 字、c4_zh 块均长 998 字 —— 一个阈值
                                        不可能同时对两者正确。仅在 --filter-nondialogue-rep3
                                        时生效。

## 纪律

* 默认 dry-run（只写报告），与 `scripts/cleanup_out.py` 的约定一致。
* **绝不修改 `--src`**：`--apply` 只往 `--dst` 写同名文件；`dst == src` 直接报错。
* 确定性：文件按文件名排序、块按原顺序、去重保留首次出现 —— 同样的输入永远同样的输出。
* 非对话 block（不含 `模型：`）默认**原样保留**（c4_zh / wikipedia / 四大名著占大头）。

## 内存策略（2.2GB 语料）

回复字符总量实测 **443,105,769**（946M 字符里）。K-gram 出现次数同量级，
用 Python dict 精确统计所有不同 K-gram 需要 >10GB —— 会 OOM（训练还占着内存）。
所以规则 3 走 **Count-Min 草图 + 精确复核** 的两阶段，共扫 3 遍：

1. **第 1 遍**（只扫对话文件）：把每个 K-gram 的 64 位滚动哈希塞进 d=4 张
   `2**--ngram-sketch-bits` 的 uint32 计数表（默认 24 位 → 4×16.7M×4B ≈ 268MB）。
2. **第 2 遍**：对每个 K-gram 取 4 张表的**最小值**作为估计。Count-Min 的性质是
   `估计 >= 真值`（哈希碰撞只会**高估**）⇒ 真高频项**不可能被漏掉**；被误标成
   候选的只是碰撞受害者。对候选做**精确计数**（dict），得到**精确**的 heavy 集合。
   ⇒ 最终判定不是近似值，草图只影响候选集大小。
3. **第 3 遍**（扫全部文件）：用精确 heavy 集合判覆盖率，同时做去重/rep3/min-chars，
   写报告与（`--apply` 时）输出。

取舍：`--ngram-sketch-bits` 越小内存越省、候选越多（第 2 遍 dict 越大、越容易撞
`--ngram-max-candidates` 上限）。N 越小越危险（真 heavy 项本身就多）。撞上限会**明确报错**
并告诉你调大哪个旋钮，不会静默降级。第 1/2 遍只读含 `模型：` 的文件（先按字节探测），
跳过 c4_zh/wikipedia 那 ~380M 字符。
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import os
import sys

import numpy as np

MODEL_TAG = '模型：'
USER_TAG = '用户：'
MODEL_TAG_BYTES = MODEL_TAG.encode('utf-8')

# 64 位多项式滚动哈希：H[i] = sum_j code[i+j] * BASE^(K-1-j) mod 2^64
BASE = np.uint64(0x100000001B3)
# Count-Min 草图的 d 个独立种子（每张表一个 splitmix64 种子）
SKETCH_SEEDS = (0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9,
                0x94D049BB133111EB, 0x2545F4914F6CDD1D)
SKETCH_CHUNK = 4_000_000      # 第 1 遍：每积攒这么多 K-gram 哈希就更新一次草图
CAND_CHUNK = 4_000_000        # 第 2 遍：候选哈希缓冲，避免把整文件候选留在内存

RULE_DEDUP = 'dedup_blocks'
RULE_REP3 = 'max_reply_rep3'
RULE_NGRAM = 'max_ngram_cover'
RULE_MIN = 'min_reply_chars'
RULE_REP3_NONDIALOGUE = 'nondialogue_rep3'


# ==========================================================================
# 纯文本工具
# ==========================================================================
def strip_ws(text: str) -> str:
    """去掉**所有**空白字符（去空白归一化）。"""
    return ''.join(text.split())


def normalized_hash(block: str) -> bytes:
    """block 级去重用的哈希：去空白归一化后的 blake2b-128。"""
    return hashlib.blake2b(strip_ws(block).encode('utf-8'), digest_size=16).digest()


def has_model_reply(block: str) -> bool:
    """block 是否含 `模型：` 行 ⇒ 是否算"对话 block"。"""
    return any(ln.strip().startswith(MODEL_TAG) for ln in block.split('\n'))


def split_replies(block: str) -> list[str]:
    """抽出 block 内所有 `模型：` 回复。

    一条回复 = `模型：` 那一行的正文 + 其后所有**不**以 `用户：`/`模型：` 开头的
    续行（deepseek 的 `<think>...</think>` 就是多行回复）。续行按 strip 后拼 `\\n`。
    """
    replies: list[str] = []
    cur: list[str] | None = None
    for line in block.split('\n'):
        s = line.strip()
        if s.startswith(MODEL_TAG):
            if cur is not None:
                replies.append('\n'.join(cur))
            cur = [s[len(MODEL_TAG):]]
        elif s.startswith(USER_TAG):
            if cur is not None:
                replies.append('\n'.join(cur))
                cur = None
        elif cur is not None:
            cur.append(s)
    if cur is not None:
        replies.append('\n'.join(cur))
    return replies


def rep_n(text: str, n: int = 3) -> float:
    """字符 n-gram 重复率 = **落在「重复类型」里的 n-gram 数 / 总 n-gram 数**。

    ★ 与评估侧**逐字同口径**（2026-09-13 对齐）。`inference/scripts/eval_dialogue.py
    ::ngram_repetition` 与 `eval_multiturn.py::ngram_rep` 都是「先去空白 → 按本公式」，
    而阈值 0.3（项目判定"重复坍缩"的判据）也是按那个口径、用对照标定的
    （干净源误伤 0.1%）。两处必须同一个函数，否则同一个 0.3 含义不同。

    ⚠ 不要改成 CTRL 口径的 `1 - 不同/总数`：**两者不等价**，前者恒 ≥ 后者
    （例：grams=[a,a,b] → 0.667 vs 0.333；重数越高差越大）。2026-09-13 实测
    multi_turn 回复：本公式均值 0.202、CTRL 公式 0.104，差近一倍 —— 阈值搬不过去。
    """
    seq = strip_ws(text)
    if n <= 0 or len(seq) < 2 * n:      # 与 eval 侧同一条守卫：太短不判（'aaaa' → 0.0）
        return 0.0
    grams = [seq[i:i + n] for i in range(len(seq) - n + 1)]
    c = collections.Counter(grams)
    return sum(1 for g in grams if c[g] > 1) / len(grams)


# ==========================================================================
# 滚动哈希 / Count-Min 草图（全部 numpy，mod 2^64 自动回绕）
# ==========================================================================
def char_codes(text: str) -> np.ndarray:
    """文本 → uint64 码点数组（按 UTF-32 解，一个"字符"= 一个码点）。"""
    return np.frombuffer(text.encode('utf-32-le'), dtype='<u4').astype(np.uint64)


def window_hashes(codes: np.ndarray, k: int) -> np.ndarray:
    """长度 k 的滑动窗口多项式哈希（位置无关）。

    H[i] = (((c[i]*BASE + c[i+1])*BASE + ...)*BASE + c[i+k-1]) mod 2^64
    """
    length = codes.shape[0]
    if k <= 0 or length < k:
        return np.empty(0, dtype=np.uint64)
    m = length - k + 1
    h = codes[:m].copy()
    for j in range(1, k):
        h *= BASE
        h += codes[j:j + m]
    return h


def _splitmix64(x: np.ndarray, seed: int) -> np.ndarray:
    """splitmix64 终结器（向量化）：把 64 位哈希打散成互不相关的 64 位。"""
    z = x + np.uint64(seed)
    z ^= z >> np.uint64(30)
    z *= np.uint64(0xBF58476D1CE4E5B9)
    z ^= z >> np.uint64(27)
    z *= np.uint64(0x94D049BB133111EB)
    z ^= z >> np.uint64(31)
    return z


class CountMinSketch:
    """只增不减的 Count-Min 草图。估计值**恒 >= 真值**（碰撞只高估）⇒ 不漏真高频项。"""

    def __init__(self, bits: int):
        if bits < 4:
            raise ValueError('--ngram-sketch-bits 至少 4')
        self.bits = bits
        self.size = 1 << bits
        self.mask = np.uint64(self.size - 1)
        self.tables = np.zeros((len(SKETCH_SEEDS), self.size), dtype=np.uint32)

    @property
    def nbytes(self) -> int:
        return self.tables.nbytes

    def add(self, hashes: np.ndarray) -> None:
        for t, seed in enumerate(SKETCH_SEEDS):
            idx = (_splitmix64(hashes, seed) & self.mask).astype(np.intp)
            self.tables[t] += np.bincount(idx, minlength=self.size).astype(np.uint32)

    def estimate(self, hashes: np.ndarray) -> np.ndarray:
        """d 张表取最小值（Count-Min 估计）。"""
        out = np.full(hashes.shape, np.iinfo(np.uint32).max, dtype=np.uint32)
        for t, seed in enumerate(SKETCH_SEEDS):
            idx = (_splitmix64(hashes, seed) & self.mask).astype(np.intp)
            np.minimum(out, self.tables[t][idx], out=out)
        return out


# ==========================================================================
# 文件 / 块扫描（确定性顺序：文件名排序 + 块原顺序）
# ==========================================================================
def list_sources(src_dir: str) -> list[str]:
    """`src_dir` 下所有 `*.txt`，按文件名排序。"""
    return sorted(f for f in os.listdir(src_dir)
                  if f.endswith('.txt') and os.path.isfile(os.path.join(src_dir, f)))


def file_has_model_tag(path: str, chunk: int = 1 << 23) -> bool:
    """按字节探测 `模型：`，避免为了看一眼标签就解码几百 MB 的非对话文件。"""
    keep = len(MODEL_TAG_BYTES) - 1
    tail = b''
    with open(path, 'rb') as f:
        while True:
            data = f.read(chunk)
            if not data:
                return False
            if MODEL_TAG_BYTES in tail + data:
                return True
            tail = data[-keep:] if keep else b''


def read_text(path: str) -> str:
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()


def iter_raw_blocks(path: str):
    """按 `\\n\\n` 切 block，丢掉纯空白块（与 `prepare.py` 的口径一致）。"""
    for b in read_text(path).split('\n\n'):
        if b.strip():
            yield b


# ==========================================================================
# 规则
# ==========================================================================
def reply_coverage(replies: list[str], k: int, heavy_sorted: np.ndarray):
    """回复里「high-freq K-gram 覆盖的字符数 / 回复总字符数」。

    返回 (coverage, covered_chars, reply_chars)。覆盖率 = 被至少一个高频 K-gram
    覆盖的字符占全部回复字符的比例（用差分数组求区间并集）。
    """
    total_chars = sum(len(r) for r in replies)
    if total_chars == 0 or k <= 0 or heavy_sorted.size == 0:
        return 0.0, 0, total_chars
    covered = 0
    for reply in replies:
        codes = char_codes(reply)
        length = codes.shape[0]
        if length < k:
            continue
        h = window_hashes(codes, k)
        pos = np.searchsorted(heavy_sorted, h)
        pos = np.minimum(pos, heavy_sorted.size - 1)
        hit = heavy_sorted[pos] == h                     # heavy 里有这个哈希
        idx = np.nonzero(hit)[0]
        if idx.size == 0:
            continue
        diff = np.bincount(idx, minlength=length + 1).astype(np.int64)
        diff -= np.bincount(idx + k, minlength=length + 1)
        covered += int((np.cumsum(diff) > 0).sum())
    if total_chars == 0:
        return 0.0, 0, 0
    return covered / total_chars, covered, total_chars


def evaluate_block(block, replies, is_dialogue, heavy_sorted, seen_hashes, args):
    """返回命中的规则名列表（空 = 保留）。一条 block 可同时命中多条规则。"""
    hits = []
    if args.dedup_blocks:
        h = normalized_hash(block)
        if h in seen_hashes:
            hits.append(RULE_DEDUP)
        else:
            seen_hashes.add(h)

    if is_dialogue:
        if args.max_reply_rep3 > 0:
            if any(rep_n(r, 3) > args.max_reply_rep3 for r in replies):
                hits.append(RULE_REP3)
        if args.min_reply_chars > 0 and replies:
            if all(len(r) < args.min_reply_chars for r in replies):
                hits.append(RULE_MIN)
        if heavy_sorted.size and args.max_ngram_cover > 0:
            cov, _, _ = reply_coverage(replies, args.ngram_size, heavy_sorted)
            if cov > args.max_ngram_cover:
                hits.append(RULE_NGRAM)
    else:
        # 非对话块用**独立**阈值 G：rep3 随长度单调上升，对话回复与长 block 不能共用一个阈值。
        if args.filter_nondialogue_rep3 and args.nondialogue_max_rep3 > 0:
            if rep_n(block, 3) > args.nondialogue_max_rep3:
                hits.append(RULE_REP3_NONDIALOGUE)
    return hits


# ==========================================================================
# 第 1 遍：Count-Min 草图
# ==========================================================================
def build_sketch(src_dir: str, dialogue_files, k: int, bits: int, log=print) -> CountMinSketch:
    sketch = CountMinSketch(bits)
    log(f'[1/3] 统计 {k}-gram 频次（草图 {bits} 位 × {len(SKETCH_SEEDS)} 表，'
        f'{sketch.nbytes / 2**20:.0f} MB）…')
    parts: list[np.ndarray] = []
    n_parts = 0
    n_grams = 0
    for fn in dialogue_files:
        for block in iter_raw_blocks(os.path.join(src_dir, fn)):
            for reply in split_replies(block):
                codes = char_codes(reply)
                if codes.shape[0] < k:
                    continue
                h = window_hashes(codes, k)
                parts.append(h)
                n_parts += h.size
                if n_parts >= SKETCH_CHUNK:          # 攒够一批再合并，避免反复 concatenate
                    buf = parts[0] if len(parts) == 1 else np.concatenate(parts)
                    sketch.add(buf)
                    n_grams += buf.size
                    parts, n_parts = [], 0
        log(f'      已扫 {fn}（累计 {n_grams + n_parts:,} 个 {k}-gram）')
    if n_parts:
        buf = parts[0] if len(parts) == 1 else np.concatenate(parts)
        sketch.add(buf)
        n_grams += buf.size
    log(f'      第 1 遍完成：{n_grams:,} 个 {k}-gram')
    return sketch


# ==========================================================================
# 第 2 遍：候选集 + 精确复核
# ==========================================================================
def build_heavy_set(src_dir: str, dialogue_files, sketch: CountMinSketch, k: int,
                    min_freq: int, max_candidates: int, log=print) -> np.ndarray:
    """返回**精确**频次 > min_freq 的 K-gram 哈希（已排序，供 searchsorted）。"""
    log(f'[2/3] 复核候选（频次 > {min_freq}）…')
    counts: collections.Counter = collections.Counter()
    parts: list[np.ndarray] = []
    n_parts = 0
    n_occ = 0
    # 上限比缓冲还小时，早点 flush，别等攒到 4M 个候选才发现爆了
    flush_at = max(1, min(CAND_CHUNK, max_candidates))

    def _flush():
        nonlocal parts, n_parts, n_occ
        if not n_parts:
            return
        buf = parts[0] if len(parts) == 1 else np.concatenate(parts)
        _merge_counts(counts, buf)
        n_occ += buf.size
        parts, n_parts = [], 0
        if len(counts) > max_candidates:
            raise SystemExit(
                f'候选 K-gram 超过上限 {max_candidates:,}（当前 {len(counts):,}）。\n'
                f'  说明 --max-ngram-freq={min_freq} 偏小或草图太密。请调大 '
                f'--ngram-sketch-bits（当前 {sketch.bits}，更省候选、更费内存），'
                f'或调大 --ngram-max-candidates / --max-ngram-freq。')

    for fn in dialogue_files:
        for block in iter_raw_blocks(os.path.join(src_dir, fn)):
            for reply in split_replies(block):
                codes = char_codes(reply)
                if codes.shape[0] < k:
                    continue
                h = window_hashes(codes, k)
                cand = h[sketch.estimate(h) > min_freq]
                if cand.size:
                    parts.append(cand)
                    n_parts += cand.size
                if n_parts >= flush_at:
                    _flush()
        log(f'      已复核 {fn}（候选占用 {len(counts):,}）')
    _flush()
    heavy = np.fromiter((h for h, c in counts.items() if c > min_freq),
                        dtype=np.uint64, count=-1)
    heavy.sort()
    log(f'      第 2 遍完成：{n_occ:,} 个候选出现，{len(counts):,} 个不同候选，'
        f'{heavy.size:,} 个确认为高频（> {min_freq}）')
    return heavy


def _merge_counts(counts: collections.Counter, hashes: np.ndarray) -> None:
    uniq, cnt = np.unique(hashes, return_counts=True)
    for h, c in zip(uniq.tolist(), cnt.tolist()):
        counts[h] += c


# ==========================================================================
# 第 3 遍：过滤 + 报告 + 输出
# ==========================================================================
def _new_file_stat(fn):
    return {'file': fn, 'blocks_in': 0, 'blocks_out': 0, 'chars_in': 0, 'chars_out': 0,
            'replies_in': 0, 'rules': collections.Counter()}


def run(args, log=print):
    src = args.src
    dst = args.dst
    if os.path.realpath(src) == os.path.realpath(dst):
        raise SystemExit(f'拒绝执行：--dst 与 --src 相同（{src}）——本工具绝不修改 --src。')
    if not os.path.isdir(src):
        raise SystemExit(f'--src 不是目录：{src}')

    ngram_on = args.max_ngram_freq > 0 and args.max_ngram_cover > 0
    if (args.max_ngram_freq > 0) != (args.max_ngram_cover > 0):
        raise SystemExit('--max-ngram-freq 与 --max-ngram-cover 必须同时 > 0 才启用 K-gram 规则'
                         '（0 = 关闭）。')
    if args.ngram_size < 1:
        raise SystemExit('--ngram-size 至少为 1')

    files = list_sources(src)
    if not files:
        raise SystemExit(f'{src} 下没有 *.txt')

    # 第 1/2 遍只关心含「模型：」的文件
    dialogue_files = [fn for fn in files if file_has_model_tag(os.path.join(src, fn))]
    log(f'源目录 {src}：{len(files)} 个 txt，其中 {len(dialogue_files)} 个含对话标签')

    heavy = np.empty(0, dtype=np.uint64)
    if ngram_on:
        sketch = build_sketch(src, dialogue_files, args.ngram_size,
                              args.ngram_sketch_bits, log=log)
        heavy = build_heavy_set(src, dialogue_files, sketch, args.ngram_size,
                                args.max_ngram_freq, args.ngram_max_candidates, log=log)

    log(f'[3/3] 过滤{"（--apply：写数据文件）" if args.apply else "（dry-run：只写报告）"} …')
    if args.apply:
        os.makedirs(dst, exist_ok=True)

    seen_hashes: set = set()
    per_file = []
    total_rules = collections.Counter()
    cov_before = cov_after = rep_before = rep_after = 0
    for fn in files:
        path = os.path.join(src, fn)
        stat = _new_file_stat(fn)
        kept = []
        for block in iter_raw_blocks(path):
            replies = split_replies(block) if MODEL_TAG in block else []
            is_dialogue = bool(replies) or has_model_reply(block)
            stat['blocks_in'] += 1
            stat['chars_in'] += len(block)
            stat['replies_in'] += len(replies)
            if ngram_on and is_dialogue:
                cov, covered, rchars = reply_coverage(replies, args.ngram_size, heavy)
                cov_before += covered
                rep_before += rchars
            hits = evaluate_block(block, replies, is_dialogue, heavy,
                                  seen_hashes, args)
            for r in hits:
                stat['rules'][r] += 1
                total_rules[r] += 1
            if hits:
                continue
            kept.append(block)
            stat['blocks_out'] += 1
            stat['chars_out'] += len(block)
            if ngram_on and is_dialogue:
                cov, covered, rchars = reply_coverage(replies, args.ngram_size, heavy)
                cov_after += covered
                rep_after += rchars
        per_file.append(stat)
        if args.apply:
            with open(os.path.join(dst, fn), 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(kept))
        log(f'      {fn}: {stat["blocks_in"]:,} → {stat["blocks_out"]:,} block，'
            f'命中 {sum(stat["rules"].values()):,}')

    coverage = {
        'enabled': ngram_on,
        'k': args.ngram_size,
        'min_freq': args.max_ngram_freq,
        'before': (cov_before / rep_before) if rep_before else 0.0,
        'after': (cov_after / rep_after) if rep_after else 0.0,
        'reply_chars_before': rep_before,
        'reply_chars_after': rep_after,
        'heavy': int(heavy.size),
    }
    report_path = args.report or os.path.join(dst, 'CLEANING_REPORT.md')
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(render_report(args, files, per_file, total_rules, coverage))
    log(f'报告已写入：{report_path}')
    if not args.apply:
        log('（dry-run：未写任何数据文件；加 --apply 才输出清洗后的 txt）')

    return {
        'files': len(files),
        'blocks_in': sum(s['blocks_in'] for s in per_file),
        'blocks_out': sum(s['blocks_out'] for s in per_file),
        'chars_in': sum(s['chars_in'] for s in per_file),
        'chars_out': sum(s['chars_out'] for s in per_file),
        'rule_hits': dict(total_rules),
        'coverage': coverage,
        'per_file': per_file,
        'report': report_path,
    }


def render_report(args, files, per_file, total_rules, coverage) -> str:
    def pct(a, b):
        return f'{100.0 * a / b:.2f}%' if b else 'n/a'

    rules_on = []
    if args.dedup_blocks:
        rules_on.append('`--dedup-blocks`（block 级精确去重，全局）')
    if args.max_reply_rep3 > 0:
        rules_on.append(f'`--max-reply-rep3 {args.max_reply_rep3}`（角色 3-gram 重复率）')
    if coverage['enabled']:
        rules_on.append(f'`--max-ngram-freq {args.max_ngram_freq} --ngram-size '
                        f'{args.ngram_size} --max-ngram-cover {args.max_ngram_cover}`（短语级 K-gram）')
    if args.min_reply_chars > 0:
        rules_on.append(f'`--min-reply-chars {args.min_reply_chars}`')
    if args.filter_nondialogue_rep3:
        rules_on.append(f'`--filter-nondialogue-rep3`（非对话块 rep3 阈值 '
                        f'{args.nondialogue_max_rep3}，'
                        f'{"显式 G" if args.nondialogue_max_rep3 != args.max_reply_rep3 else "= F"}）')
    if not rules_on:
        rules_on.append('（无：所有规则都关闭，本报告只是一次基线扫描）')

    b_in = sum(s['blocks_in'] for s in per_file)
    b_out = sum(s['blocks_out'] for s in per_file)
    c_in = sum(s['chars_in'] for s in per_file)
    c_out = sum(s['chars_out'] for s in per_file)

    lines = [
        '# 语料清洗报告（clean_corpus.py）',
        '',
        f'- 时间：{datetime.datetime.now().astimezone().isoformat(timespec="seconds")}',
        f'- 模式：**{"--apply（已写数据文件）" if args.apply else "dry-run（未写数据文件）"}**',
        f'- 源目录（只读）：`{args.src}`',
        f'- 输出目录：`{args.dst}`',
        f'- 命令行：`{" ".join(sys.argv)}`',
        f'- 启用规则：{"; ".join(rules_on)}',
        '',
        '## 全局',
        '',
        '| 指标 | 清洗前 | 清洗后 | 变化 |',
        '|---|---|---|---|',
        f'| block 数 | {b_in:,} | {b_out:,} | -{b_in - b_out:,}（{pct(b_in - b_out, b_in)}） |',
        f'| 字符数 | {c_in:,} | {c_out:,} | -{c_in - c_out:,}（{pct(c_in - c_out, c_in)}） |',
    ]
    if coverage['enabled']:
        cb, ca = coverage['before'], coverage['after']
        lines.append(f'| 高频 {coverage["k"]}-gram 覆盖率（回复字符） | {cb:.4%} | {ca:.4%} | '
                     f'{(ca - cb) * 100:+.4f} pp |')
        lines += ['', f'（heavy 集合：频次 > {coverage["min_freq"]} 的 {coverage["heavy"]:,} 个 '
                      f'{coverage["k"]}-gram；覆盖口径 = 回复里被至少一个 heavy K-gram 覆盖的字符占比。'
                      f'回复字符 {coverage["reply_chars_before"]:,} → {coverage["reply_chars_after"]:,}）']
    else:
        lines += ['', '（未启用 K-gram 规则，跳过覆盖率统计。）']

    lines += [
        '',
        '## 各规则命中 block 数',
        '',
        '| 规则 | 命中 |',
        '|---|---|',
    ]
    label = {
        RULE_DEDUP: 'dedup_blocks',
        RULE_REP3: 'max_reply_rep3',
        RULE_NGRAM: 'max_ngram_cover',
        RULE_MIN: 'min_reply_chars',
        RULE_REP3_NONDIALOGUE: 'nondialogue_rep3',
    }
    for rule in label.values():
        lines.append(f'| {label[rule]} | {total_rules.get(rule, 0):,} |')

    lines += [
        '',
        '> 一条 block 可能同时命中多条规则，所以"命中数之和"≥ 被丢的 block 数。',
        '',
        '## 逐来源',
        '',
        '| 文件 | blocks 进 | blocks 出 | 字符 进 | 字符 出 | dedup | rep3 | ngram | min | 非对话rep3 |',
        '|---|---|---|---|---|---|---|---|---|---|',
    ]
    for s in per_file:
        r = s['rules']
        lines.append(
            f'| {s["file"]} | {s["blocks_in"]:,} | {s["blocks_out"]:,} | {s["chars_in"]:,} | '
            f'{s["chars_out"]:,} | {r.get(RULE_DEDUP, 0):,} | {r.get(RULE_REP3, 0):,} | '
            f'{r.get(RULE_NGRAM, 0):,} | {r.get(RULE_MIN, 0):,} | '
            f'{r.get(RULE_REP3_NONDIALOGUE, 0):,} |')
    lines += [
        f'| **合计** | **{b_in:,}** | **{b_out:,}** | **{c_in:,}** | **{c_out:,}** | '
        f'**{total_rules.get(RULE_DEDUP, 0):,}** | **{total_rules.get(RULE_REP3, 0):,}** | '
        f'**{total_rules.get(RULE_NGRAM, 0):,}** | **{total_rules.get(RULE_MIN, 0):,}** | '
        f'**{total_rules.get(RULE_REP3_NONDIALOGUE, 0):,}** |',
        '',
        f'（默认：不含 `模型：` 的非对话 block 只受 `--dedup-blocks` 约束、'
        f'**不受** rep3 / min-reply-chars / K-gram 覆盖率约束；'
        f'要连非对话块一起按 rep3 过滤，显式加 `--filter-nondialogue-rep3`。）',
        '',
        f'- 源文件数：{len(files)}',
        '',
    ]
    return '\n'.join(lines)


# ==========================================================================
# CLI
# ==========================================================================
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description='语料清洗（默认 dry-run，--apply 才写数据文件）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--src', required=True, help='只读源目录（扫它下面的 *.txt）')
    ap.add_argument('--dst', required=True, help='输出目录（--apply 时写清洗后的同名 txt）')
    ap.add_argument('--report', default=None, help='报告路径（默认 <dst>/CLEANING_REPORT.md）')
    ap.add_argument('--apply', action='store_true',
                    help='真正写数据文件；不传则只报告（dry-run）')
    # --- 规则（全部默认关闭；阈值由调用方拍板）---
    ap.add_argument('--dedup-blocks', action='store_true',
                    help='block 级精确去重（去空白归一化哈希，全局跨文件，保留首次出现）')
    ap.add_argument('--max-reply-rep3', type=float, default=0.0, metavar='F',
                    help='block 内任一 模型： 回复的 rep3 > F 就丢（0=关闭）')
    ap.add_argument('--max-ngram-freq', type=int, default=0, metavar='N',
                    help='K-gram 高频阈值：全语料回复里频次 > N 才算高频（0=关闭）')
    ap.add_argument('--ngram-size', type=int, default=5, metavar='K',
                    help='K-gram 的 K（字符级）')
    ap.add_argument('--max-ngram-cover', type=float, default=0.0, metavar='C',
                    help='回复里高频 K-gram 覆盖字符占比 > C 就丢该 block（0=关闭）')
    ap.add_argument('--min-reply-chars', type=int, default=0, metavar='M',
                    help='block 内所有 模型： 回复都 < M 字就丢（0=关闭）')
    ap.add_argument('--filter-nondialogue-rep3', action='store_true',
                    help='对非对话 block（不含 模型：）也套 rep3（默认不套）')
    ap.add_argument('--nondialogue-max-rep3', type=float, default=None, metavar='G',
                    help='非对话 block 的 rep3 阈值 G（不给则回退到 --max-reply-rep3，'
                         '保持向后兼容）。rep3 随长度单调上升，对话回复与 c4_zh 长块不能'
                         '共用同一个阈值。只在 --filter-nondialogue-rep3 时生效。')
    # --- 内存旋钮 ---
    ap.add_argument('--ngram-sketch-bits', type=int, default=24,
                    help='Count-Min 草图每张表的位数（内存 4*2^bits*4B；越大候选越少）')
    ap.add_argument('--ngram-max-candidates', type=int, default=12_000_000,
                    help='第 2 遍精确复核的候选上限，超过就报错（防 OOM）')
    args = ap.parse_args(argv)
    # 向后兼容：没给 G 时，非对话阈值 = 对话阈值 F（旧行为一字不变）
    if args.nondialogue_max_rep3 is None:
        args.nondialogue_max_rep3 = args.max_reply_rep3
    return args


def main(argv=None):
    args = parse_args(argv)
    # flush=True：长跑时用 tail 看进度（否则重定向到文件会块缓冲，半天不吐字）
    run(args, log=lambda *a: print(*a, flush=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
