"""分句器 —— **全项目统一的句子切分**（canonical）。

谁该用它：`training/dialogue_stream.py`（对话流滑窗）、`data/chinese/prepare.py`
（`--split-sentences`）、以及所有语料构建器。
`data/chinese/split_sentences.py` 现在只是本模块的**薄壳转发**（保留兼容与 CLI）。

## 为什么要有它（而不是继续散在 `data/chinese/`）
1. **它是纯函数模块** ⇒ 属于 `training/`（本项目的规矩：已抽出的纯函数放这里），
   而不是 `data/`（那是语料与 tokenizer 的地盘）；
2. `training/dialogue_stream.py` 原先只能靠 `importlib` **按文件路径**去 load 它
   （因为 `data/chinese/` 不是包），现在改成正常 import；
3. 它承载一条**实测过的教训**（见下），这条教训不该只活在某一个调用方的补丁里。

## ★ 机制符必须"原子且归属上一句"（2026-09-14 实测事故）
朴素分句器会把 `<eos>` 切出来**单独成句**：

    split_line('我是小寻。<eos>')  →  ['我是小寻。', '<eos>']      ← 错

后果不是"格式难看"，而是**真坏数据**：对话流按句滑窗，`<eos>` 会被**单独弹掉**，
于是模型看到一条没有终止符的回复。本模块用两条规则修掉：

- **规则一（不切进去）**：`<...>` 形式的机制符内部绝不作为句边界
  （` thinking第一步。 response` 里的 `。` 不会把 token 切开）；
- **规则二（不单独成句）**：只由机制符与空白构成的片段，**并入前一句**
  （`<eos>` / `<resp>` 这类是"后缀标记"，天然属于它终止的那一句）。

## 核心：一个"句队列"（2026-09-15 重构）
本模块的核心只有一个类 `SentenceQueue` —— **字符流 → 有界句子队列**。
它把原 `StreamSegmenter`（字符→句子的切分）与 `ContextWindow`（有界窗口）合并成
**一套状态、两个用法**。`StreamSegmenter` / `ContextWindow` 现在只是它的**薄 facade**
（公开 API 逐字不变，`DialogueStream` / `iter_sentences` / `prepare.py` 都不用改）。
合并的意义：窗口的"弹出/溢出/测量"与切分的"压尾巴/机制符归属"规则集中在**一处**，
以后要加"增量 token 计数 / KV 感知弹出 / 超长句索引"都只改这一个类。

## 与旧实现的行为等价性
对**不含机制符**的输入，本模块与旧 `split_sentences.split_line` **逐字等价**
（`tests/test_segmentation.py::test_matches_legacy_behaviour_on_plain_text` 用
同一批样例双向对拍）。所以 `prepare.py --split-sentences` 的行为只在
"文本里真的出现机制符"时才变 —— 而那正是修 bug。
"""
from __future__ import annotations

import re
from typing import List

__all__ = ['SENT_PAT', 'SUB_SEP', 'SPECIAL_RE', 'ContextWindow', 'SentenceQueue',
           'StreamSegmenter', 'is_special_only', 'iter_sentences', 'normalize_text',
           'prepare_natural_text', 'process_file', 'run_cli', 'split_line', 'split_text']

# 句末标点：全角句号/问号/叹号/省略号 + 半角 ?!，以及**不在数字之间**的半角 `.`
# （保护 3.14 / U.S. 这类）。★ 与旧实现逐字相同，别改。
SENT_PAT = re.compile(r'[。！？…?!]|(?<!\d)\.(?!\d)')

# 超长句的二次切分点
SUB_SEP = '，,；;、'

# 机制符 / 特殊 token：`<` + 字母数字下划线**或半角点**或汉字 + `>`。
# - 覆盖 `<eos>` `<unk>` `<cont>` `<pad>` `<bos>` `<sep>` ` thinking` `<answer>` `<resp>` 等；
# - 也覆盖中文机制符（如早期的 `<你该说话了>`）；
# - ★ 允许 `.` 是**规则一存在的理由**：`<stdio.h>` 这种名字里的 `.` 不该当句边界
#   （不含 `.` 时规则一是空规则 —— token 里不可能出现 `。`，那字符不在字符类里）；
# - 会顺带把代码/HTML 里的 `<div>` 当成"不切进去"的片段 —— 无害（本来也不该在那里切）。
SPECIAL_RE = re.compile(r'<[A-Za-z0-9_.\u4e00-\u9fff]+>')


def _protected(line: str) -> List[bool]:
    """每个位置是否落在机制符内部。"""
    flags = [False] * len(line)
    for m in SPECIAL_RE.finditer(line):
        for i in range(m.start(), m.end()):
            flags[i] = True
    return flags


def _is_boundary(line: str, i: int, flags: List[bool]) -> bool:
    """位置 `i` 是不是句边界。**机制符内部一律不是。**"""
    if flags[i]:
        return False
    ch = line[i]
    if ch in '。！？…?!':
        return True
    if ch == '.':
        prev_d = i > 0 and line[i - 1].isdigit()
        next_d = i + 1 < len(line) and line[i + 1].isdigit()
        return not (prev_d or next_d)
    return False


def is_special_only(piece: str) -> bool:
    """片段是否**只由机制符与空白构成**（没有任何实义字符）。"""
    return not SPECIAL_RE.sub('', piece).strip()


def _subsplit(sentence: str, max_len: int) -> List[str]:
    """超长句按逗号/分号/顿号二次切；**同样不切进机制符**。"""
    if len(sentence) <= max_len:
        return [sentence]
    flags = _protected(sentence)
    segs, cur = [], ''
    for i, ch in enumerate(sentence):
        cur += ch
        if ch in SUB_SEP and not flags[i]:
            segs.append(cur)
            cur = ''
    if cur:
        segs.append(cur)
    return segs or [sentence]


def split_line(line: str, max_len: int = 80) -> List[str]:
    """单行内按句末标点切分（标点保留在前一句），超长句按逗号/分号二次切。

    机制符的处理见模块 docstring 的两条规则。
    """
    if not line.strip():
        return []
    flags = _protected(line)
    sentences: List[str] = []
    buf = ''
    for i, ch in enumerate(line):
        buf += ch
        if _is_boundary(line, i, flags):
            sentences.append(buf)
            buf = ''
    if buf.strip():
        sentences.append(buf)
    if not sentences:
        return []

    # ★ 规则二：只由机制符/空白构成的片段并入前一句（`<eos>` 属于它终止的那一句）
    merged: List[str] = []
    for s in sentences:
        if merged and is_special_only(s):
            merged[-1] += s
        else:
            merged.append(s)

    out: List[str] = []
    for s in merged:
        out.extend(_subsplit(s, max_len))
    return out


def _tail_is_open_token(s: str) -> bool:
    """`s` 的结尾是不是一个**尚未闭合**的潜在机制符（如 `<eo`）。

    ★ 流式场景必需：`<eos>` 可能**跨块**到达（`'<eo'` + `'s>'`）。
      若把 `<` 当普通字符，`'我是小寻。<'` 会被切成 `['我是小寻。', '<']`，
      于是 `'我是小寻。'` **先被发出去**，`<eos>` 只能单独成句 ——
      那个"`<eos>` 被单独弹掉"的老 bug 就在流式下复发了。
    """
    i = s.rfind('<')
    if i < 0 or '>' in s[i:]:
        return False
    return re.fullmatch(r'<[A-Za-z0-9_.\u4e00-\u9fff]*', s[i:]) is not None


# ============================================================================
# 句队列（唯一核心）：字符流 → 有界句子队列
# ============================================================================


class SentenceQueue:
    """句队列 —— **全项目统一的"字符流 → 有界句子队列"单核**。

    它把原 `StreamSegmenter`（字符→句子的切分）与 `ContextWindow`（有界窗口）
    合并成**一套状态、两个用法**：

    - **窗口用法**（原 `ContextWindow`）：调用方喂**已切好的单元**（`push`/`extend`），
      它维护有界窗口 —— 超预算从头部**弹出整单元**、至少保留 `keep_min` 个、
      单个超长单元置 `overflow`（软上限）、`budget >= NO_LIMIT` 时**完全不测量**
      （否则解析整份语料是 **O(n²)**，2026-09-14 实测：10 万字符把验证脚本跑到
      60s 超时被杀；修好后 0.01s）。
    - **流式用法**（原 `StreamSegmenter`）：调用方喂**原始字符流**（`feed`/`flush`），
      它切分并逐句吐出；`pending` 是压着没吐的尾巴（机制符要粘在它终止的那一句上，
      所以最后一段总是留到下一次 `feed()` 或 `flush()` 才吐）。

    两个公开入口类 `StreamSegmenter` / `ContextWindow` 只是本类的**薄 facade**
    （公开 API 与旧版逐字相同，`DialogueStream` / `iter_sentences` 不用改）。

    **合并的意义**：窗口的"弹出/溢出/测量"与切分的"压尾巴/机制符归属"规则集中在
    **一处** —— 以后要加"增量 token 计数 / KV 感知弹出 / 超长句索引"都只改这一个类。
    """

    #: 预算 ≥ 此值 ⇒ 视为"不裁剪"，完全跳过测量（窗口用法）
    NO_LIMIT = 10 ** 8

    def __init__(self, *, measure=None, unit_size=None, sep_size: int = 0,
                 budget: int = NO_LIMIT, keep_min: int = 1,
                 max_len: int = 80, newline_is_boundary: bool = True,
                 hold_last: bool = True, keep_empty_lines: bool = True) -> None:
        if keep_min < 1:
            raise ValueError('keep_min 至少为 1，否则会把窗口弹空')
        # --- 窗口状态（原 ContextWindow）---
        self._measure = measure          # 整窗重算的测量回调（`DialogueStream` 的合并轮次用它）
        self._unit_size = unit_size      # 增量模式：每个单元的单独大小（**可分解**测量）
        self._sep_size = int(sep_size)   # 增量模式：单元间统一分隔符的大小
        self._incremental = unit_size is not None  # 可分解 ⇒ 用 O(1) 运行计数，不整窗重算
        self._size = 0                   # 增量模式的运行长度
        self.budget = int(budget)
        self.keep_min = int(keep_min)
        self._units: List = []
        self.dropped = 0        # 累计弹出的**单元**数
        self.overflow = False   # 单个单元本身就超预算 ⇒ 软上限放行
        # --- 切分状态（原 StreamSegmenter）---
        self.max_len = int(max_len)
        self.newline_is_boundary = newline_is_boundary
        self.hold_last = hold_last
        self.keep_empty_lines = keep_empty_lines
        self._buf = ''
        self._after_newline = False

    # ---------- 窗口：访问 ----------

    def _add_size(self, units) -> None:
        """增量模式：把 `units` 追加进窗口时同步累计长度（O(len)，不重算整窗）。

        规则：总长 = Σ unit_size(u) + sep_size × 单元间分隔数。
        空窗 加 n 个 ⇒ 分隔数 n−1；非空窗 加 n 个 ⇒ 分隔数 n。"""
        if not self._incremental:
            return
        n_new = len(units)
        seps = (n_new - 1) if not self._units else n_new
        self._size += sum(self._unit_size(u) for u in units) + self._sep_size * seps

    def _sub_size(self) -> None:
        """增量模式：从**首部**弹出一个单元时同步扣长度（单元大小 + 一个分隔符）。"""
        if not self._incremental:
            return
        self._size -= self._unit_size(self._units[0]) + self._sep_size

    @property
    def units(self) -> List:
        """★ 返回**活的**列表：调用方会就地改最后一个单元（例如给最后一句贴 `<eos>`）。"""
        return self._units

    def size(self) -> int:
        """当前长度（`budget >= NO_LIMIT` 时返回 0；增量模式返回 O(1) 运行计数）。"""
        if self.budget >= self.NO_LIMIT:
            return 0
        return self._size if self._incremental else self._measure(self._units)

    # ---------- 窗口：维护 ----------

    def enforce(self) -> int:
        """弹出首部整单元直到放得下；返回本次弹出的单元数。"""
        if self.budget >= self.NO_LIMIT:
            self.overflow = False
            return 0
        n = 0
        while len(self._units) > self.keep_min and self._over_budget():
            self._sub_size()
            self._units.pop(0)
            self.dropped += 1
            n += 1
        self.overflow = self._over_budget()
        return n

    def _over_budget(self) -> bool:
        return self.size() > self.budget

    def push(self, unit) -> int:
        self._add_size([unit])
        self._units.append(unit)
        return self.enforce()

    def extend(self, units) -> int:
        self._add_size(units)
        self._units.extend(units)
        return self.enforce()

    # ---------- 切分：内部 ----------

    def _drain_line(self, final: bool) -> List[str]:
        """吐出当前行**能确定**的部分。`final=True` 表示这一行已经结束（换行或流结束）。

        判定规则（2026-09-14 修）：
          - **末尾不是句边界** ⇒ 最后一段是"还没写完的文本"，**必须**留在缓冲区（与
            `hold_last` 无关）。早先的实现漏了这条，导致 `hold_last=False` 时**每喂一个字
            就吐一个字**（"我" 都算一句）。
          - **末尾是句边界** ⇒ 全部片段都完整；但若 `hold_last=True`，最后一句仍可能被
            `<eos>` 之类的后缀粘上，所以**再压一段**。
          - **尾巴是未闭合的 `<`** ⇒ 额外压一段（它可能要粘到上一句上）。
        """
        if not self._buf:
            return []
        pieces = split_line(self._buf, self.max_len)
        if not pieces:
            return []
        if final:
            self._buf = ''
            return pieces

        flags = _protected(self._buf)
        terminated = _is_boundary(self._buf, len(self._buf) - 1, flags)
        if terminated:
            complete, tail = list(pieces), ''
        else:
            complete, tail = list(pieces[:-1]), pieces[-1]

        if _tail_is_open_token(self._buf) and complete:
            tail, complete = complete[-1] + tail, complete[:-1]
        elif self.hold_last and terminated and complete:
            tail, complete = complete[-1] + tail, complete[:-1]

        self._buf = tail
        return complete

    # ---------- 切分：公开 ----------

    def feed(self, chunk: str) -> List[str]:
        """喂一块文本，返回本次**已经确定**的句子。"""
        out: List[str] = []
        for ch in chunk:
            if ch == '\n' and self.newline_is_boundary:
                if self._buf:
                    out.extend(self._drain_line(final=True))
                elif self.keep_empty_lines:
                    out.append('')
                self._buf = ''
                self._after_newline = True
                continue
            self._buf += ch
            self._after_newline = False
        if self._buf:
            out.extend(self._drain_line(final=False))
        return out

    def flush(self) -> List[str]:
        """输入结束：把残余全部吐出来。"""
        out = self._drain_line(final=True)
        if not self._buf and self.keep_empty_lines and self._after_newline:
            out.append('')
        self._after_newline = False
        return out

    @property
    def pending(self) -> str:
        """还压在缓冲区里、没吐出来的尾巴。"""
        return self._buf

    # ---------- 共享 ----------

    def reset(self) -> None:
        """清空全部状态（窗口 + 切分尾巴）。"""
        self._buf = ''
        self._after_newline = False
        self._units.clear()
        self.dropped = 0
        self.overflow = False
        self._size = 0

    def __len__(self) -> int:
        return len(self._units)

    def __repr__(self) -> str:
        return (f'SentenceQueue(单元={len(self._units)}, 预算={self.budget}, '
                f'已弹出={self.dropped}, 溢出={self.overflow}, 待定={len(self._buf)})')


class StreamSegmenter(SentenceQueue):
    """增量（流式）分句器：喂任意大小的块，吐出**已经确定完整**的句子。

    ★ 这是 `SentenceQueue` 的**薄 facade**（流式用法），行为与旧版逐字相同。

    三种用法都覆盖：

    - **用户输入自动分句**：`feed(用户输入)` + `flush()`；
    - **长文本（书籍等）**：`feed(整本书)` + `flush()`，或直接用 `split_text()`；
    - **真流式**：循环 `feed(chunk)`，边界一确定就吐出来（`pending` 是压着没吐的尾巴）。

    边界 ＝ 句末标点 **或** 换行（`newline_is_boundary=True`）。
    换行同时充当**轮次/段落分隔**：空行产出一个空串（保持结构，与 `split_text` 一致）。

    ★ **为什么默认会"压一段"（`hold_last=True`）**：`<eos>` / `<resp>` 这类机制符要
      **粘在它终止/开启的那一句**上，而流式里你无法预知某句后面还会不会来一个 `<eos>`。
      所以最后一段总是留到**下一次 `feed()` 或 `flush()`** 才吐。
      要低延迟、且确定输入里不会有机制符时，设 `hold_last=False`（`split_text` 用的就是它）。
    """

    def __init__(self, max_len: int = 80, *, newline_is_boundary: bool = True,
                 hold_last: bool = True, keep_empty_lines: bool = True) -> None:
        super().__init__(max_len=max_len, newline_is_boundary=newline_is_boundary,
                         hold_last=hold_last, keep_empty_lines=keep_empty_lines)


class ContextWindow(SentenceQueue):
    """按单元（通常就是"句"）滑动的上下文窗口 —— 本项目**统一**的上下文管理。

    ★ 这是 `SentenceQueue` 的**薄 facade**（窗口用法），行为与旧版逐字相同。

    规则（用户原话）：**总长超过上下文窗口后，弹出首部的句子，直到能够放进新句子。**

    - **与"怎么算长度"解耦**：`measure(units) -> int` 由调用方给。对
      `DialogueStream` 来说，长度是"把当前记录渲染成文本 + 末尾 `<resp>` 再编码"的
      token 数（所以标签、换行都算进去）。
    - **只在单元边界弹出** ⇒ 永远不会留下半句。
    - **至少保留 `keep_min` 个单元** ⇒ 单个超长单元时不死循环，改为置
      `overflow=True`（软上限；调用方据此决定要不要硬截）。
    - **`budget >= NO_LIMIT` 时完全不做测量**（见 `SentenceQueue.NO_LIMIT`）。

    用法：
        win = ContextWindow(measure=len, budget=256)
        win.extend(['a。', 'b。', 'c。'])
        win.units        # 只保留放得下的那几段
    """

    def __init__(self, measure, budget: int, *, keep_min: int = 1) -> None:
        super().__init__(measure=measure, budget=budget, keep_min=keep_min)


def split_text(text: str, max_len: int = 80) -> str:
    """整段文本分句：按行保持轮次/段落结构，行内分句后用换行连接。

    等价于 `StreamSegmenter(max_len, hold_last=False).feed(text)` + `flush()` 再用换行连接 ——
    批量场景没有"下一块"，所以不需要压段（`hold_last=False`）。
    """
    seg = StreamSegmenter(max_len, hold_last=False)
    return '\n'.join(seg.feed(text) + seg.flush())


def iter_sentences(chunks, max_len: int = 80, **kwargs):
    """把任意可迭代的文本块**流式**转成句子迭代器。

    例：`for s in iter_sentences(['你好。', '世界。']): ...` → `'你好。'`, `'世界。'`
    """
    seg = StreamSegmenter(max_len, **kwargs)
    for c in chunks:
        yield from seg.feed(c)
    yield from seg.flush()


# ============================================================================
# 自然文本 → 训练可用文本（**数据集进入训练框架的统一入口**）
# ============================================================================

# 控制符（保留 \t 与 \n，去掉其余 C0/C1 与 DEL）
_CTRL_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def normalize_text(text: str) -> str:
    """把**原始自然文本**规整到可分句的形态（不做任何有损改写）。

    - `\\r\\n` / `\\r` → `\\n`
    - 去掉控制符（保留 `\\t` `\\n`）
    - 去掉行尾空白
    - 3 个以上连续换行压成 2 个（**空行是块分隔符，必须保留一个**）
    - 去掉首尾空行

    ⚠ 故意**不**做的：不合并句内空格（代码/中英混排靠它）、不过滤标点、
      不做繁简/全半角转换 —— 这些都会改内容，该由数据清洗负责，不该藏在分句器里。
    """
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = _CTRL_RE.sub('', text)
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip('\n')


def prepare_natural_text(
    text: str,
    *,
    max_len: int = 80,
    newline_is_boundary: bool = True,
    min_chars: int = 0,
    normalize: bool = True,
    keep_empty_lines: bool = True,
) -> str:
    """**自然文本 → 训练可用文本**：规整 → 分句 → 输出。

    产出形态：**一句一行**，段落之间保留一个空行。

    ★ 为什么保留空行：`prepare.py` 用**空行**切训练块（block），
      所以空行 = 样本边界。丢了它，整本书会变成一个巨大样本。

    `min_chars > 0` 时丢掉过短的非空行（噪声碎片）；空行（块边界）**永远保留**。
    """
    if normalize:
        text = normalize_text(text)
    seg = StreamSegmenter(max_len, newline_is_boundary=newline_is_boundary,
                          hold_last=False, keep_empty_lines=keep_empty_lines)
    lines = seg.feed(text) + seg.flush()
    if min_chars > 0:
        lines = [s for s in lines if not s.strip() or len(s.strip()) >= min_chars]
    return '\n'.join(lines)


def process_file(src: str, dst: str | None = None, **kwargs) -> dict:
    """把一份**原始自然文本数据集**处理成训练可用文本，落盘并返回统计。

    默认输出 `<stem>_seg.txt`（同目录）。这是"以后有数据集就直接过一遍"的入口。
    """
    import os

    text = open(src, encoding='utf-8', errors='replace').read()
    out = prepare_natural_text(text, **kwargs)
    if dst is None:
        stem, _ = os.path.splitext(src)
        dst = f'{stem}_seg.txt'
    with open(dst, 'w', encoding='utf-8') as f:
        f.write(out + '\n')

    src_chars, out_chars = len(text), len(out)
    blocks = sum(1 for b in out.split('\n\n') if b.strip())
    lines = [ln for ln in out.split('\n') if ln.strip()]
    return {
        'src': src, 'dst': dst,
        'src_chars': src_chars, 'out_chars': out_chars,
        'blocks': blocks, 'lines': lines,
        'avg_line': (out_chars / len(lines)) if lines else 0.0,
    }


def run_cli(argv=None, default_suffix: str = '_seg') -> int:
    """`python -m training.segmentation --file X`：自然文本 → 训练可用文本。"""
    import argparse
    import os

    ap = argparse.ArgumentParser(
        prog='python -m training.segmentation',
        description='统一分句器：把自然文本处理成训练可用文本（一句一行、空行分块）')
    ap.add_argument('--file', required=True, help='原始文本（txt）')
    ap.add_argument('--out', default=None, help=f'输出路径（默认 <stem>{default_suffix}.txt）')
    ap.add_argument('--max-len', type=int, default=80, help='超长句二次切阈值（字）')
    ap.add_argument('--min-chars', type=int, default=0, help='丢弃短于该长度的非空行（0=不丢）')
    ap.add_argument('--no-normalize', action='store_true', help='跳过规整（不折叠换行/去控制符）')
    a = ap.parse_args(argv)

    if not os.path.exists(a.file):
        print(f'❌ 文件不存在：{a.file}')
        return 1
    st = process_file(a.file, a.out, max_len=a.max_len, min_chars=a.min_chars,
                      normalize=not a.no_normalize)
    print(f"{os.path.basename(st['src'])} → {os.path.basename(st['dst'])}")
    print(f"  字符 {st['src_chars']:,} → {st['out_chars']:,}"
          f" | 块 {st['blocks']:,} | 句 {len(st['lines']):,}"
          f" | 平均句长 {st['avg_line']:.1f} 字")
    return 0


def main(argv=None) -> int:
    return run_cli(argv, default_suffix='_seg')


if __name__ == '__main__':
    raise SystemExit(main())