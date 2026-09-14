"""对话流与上下文窗口管理 —— 「按句滑窗 + `<resp>` 内联 + 单流」（v2）。

设计来源：用户 2026-09-14 提出并拍板。四条规则：

1. **按句分布** —— 对话流是一串「句子」，每句带说话人标签；
2. **滑窗按句弹出** —— 总长超过上下文窗口时，从**头部弹出整句**，直到新句子放得下；
3. **模型自己的轮次内联标记** —— 用**单 token** `<resp>`（id 140）标在轮首；
4. **保持单条连续流**（用户 2026-09-14 选「乙」）—— 不再有"临时的尾部 cue"。

★ 由此得到的**日志形态**（`render()`）：

    对象A：你好，我是李华。
    <resp>你好呀。我是小寻。<eos>
    对象A：你多大了？
    <resp>还在长个儿呢。<eos>

  而**模型输入**（`prompt()`）＝日志 + 末尾一行 `<resp>`。

★★ **零替换的严格连续性**（这是选「乙」的全部意义，有测试钉住）：

    prompt_k + target_k  ==  日志_{k+1}          ← 逐字符相等，**没有任何替换**

  对比选「甲」时的形态：那里 `prompt_{k+1}` 里 cue 要被 `自己：` **替换**掉，
  所以数据只能是 (prompt, target) **对**、喂不进「一条长流 + 按终止符掩码」的现有管线。
  「乙」把它变成连续流，现有管线能用。

★ **`自己：` 哪去了**（2026-09-14 实现时的一个判断，见 §决策）：
  `自己：` 的角色被 `<resp>` **吸收**了 —— 两者都是"我的轮次从这里开始"。
  合并的好处：**1 token 取代 3 token**（`自己：` = `自`+`己`+`：`），
  而且消除"两个标记说同一件事"的冗余。**语义意图完全保留**（上下文里自动出现
  一个自我标记），只是它现在叫 `<resp>`。
  ⇒ 因此 `SELF_LABEL`（`'自己'`）只是**内部哨兵**，**永远不会被渲染成 `自己：`**。

★ **训练目标**（`iter_training_samples` 产出）＝「内容 + `<eos>`」，**不含** `<resp>`
  （`<resp>` 由 harness 喂，不由模型生成）。哪些 token 该算 loss 见 `loss_token_spans()`。

★ 与已有 `<eos>` / 退休的 `<cont>` 的关系（用户 2026-09-14 决定：**合并 `<cont>`、保留 `<eos>`**）：

   - `<eos>`(128)＝「本轮说完，可以停」——**保留**，贴在每轮模型回复末尾；
   - `<cont>`(130)＝「说完但请对方继续」——**新数据不再生成**（语义已被轮次机制取代）。
     ⚠ **token 本身绝不能删**：既有 bin 里它有 **1,003,220** 个
     （v2 344,673 / know 174,405 / dlg 163,989 / 旧 char 320,153），删了旧 bin 就解不出来。

★ ★ **`prepare.py` 一个字都不用改**：`annotate_replies` 只认 `用户：`/`模型：`（`:222`），
  非匹配行**原样保留**（`:251-255`）⇒ 我们的 `对象A：`/`<resp>` 格式**完全绕过**那套标注
  （不插 `<eos>`/`<cont>`、不做 70% `A：/B：` 改写、不加引号、不走 `_should_continue`）。
  **终止符由本模块自己插。**

★ **`<resp>` 是单 token（`id=140`）**，2026-09-14 落地：
  机制符区间的第一个预留位 `<res0>` 被改名成 `<resp>` 并登记为 special，
  用法与 `<eos>`(128) / `<cont>`(130) 同族 ⇒ `encode('<resp>') == [140]`（改前是 7 个 token）。

  ★ **为什么用新位（140）而不回收退休的 `<cont>`（130）**：
    回收是**零收益**（词表大小与 id 数都不变，130 和 140 都照样占位），
    代价却是污染既有数据 —— id 130 在既有 bin 里共 1,003,220 个 `<cont>`，改名会让这
    100 万 token 的含义变化。而且它违反 `train_tokenizer.py:57`
    「命名机制符顺序即 id，**绝不更改**」；B 段 ckpt 还在这 16 万个 `<cont>` 上训过
    1 epoch（强先验），换名是主动和已学关联对撞。**全新 id 零先验，严格更优。**

  ★ **安全性已实测**（这是能这么改的前提）：
    - 词表仍 **8192**；id `0..139` 逐位未变；**汉字区 `384..8191` 逐位未变**；
    - 29 个语料源各取 5KB，编码**逐位相同**（改前 vs 改后）；
    - 所有 char-level bin 在 id `140..383` 区间**零命中**（只有废弃的旧 BPE `train.bin` 有，
      它属另一套 tokenizer）。
    ⇒ **既有 bin 与 checkpoint 全部继续有效，不需要重建任何数据。**

  注意：`Tokenizer.decode` 默认 `skip_special_tokens=True`，所以 `decode([140])` 返回 `''`；
  要拿回文本得 `decode([140], skip_special_tokens=False)`（`<eos>` 一直也是这行为）。

★ 本模块**不含任何人格内容**，是纯机制 —— 人格定义在仓库外（`~/datasets/persona/`）。

★ **与 `training/segmentation.py` 的分工**（2026-09-14 抽象出来）：
  分句（含"机制符原子且归属上一句"那条规则）归 **`segmentation.py`**；
  本模块只管**对话流、滑窗、渲染、loss 区间**。
  语料文件的读写在 `read_corpus()` / `write_corpus()` —— 全项目统一走这两个，
  别再各自 `split('\\n\\n')`。
"""
from __future__ import annotations

from typing import Callable, Iterator, List, Sequence, Tuple

# 分句器：**canonical 在 `training/segmentation.py`**，本模块正常 import 它。
# 2026-09-14 之前这里用 `importlib` 按文件路径去 load `data/chinese/split_sentences.py`
# （因为 `data/chinese/` 不是包）；那件事现在搬成了正式模块，`split_sentences.py`
# 退化成薄壳转发。`split_line` 仍在本模块 re-export，供既有调用方使用。
from training.segmentation import ContextWindow, split_line  # noqa: F401  (split_line: re-export)

SELF_LABEL = '自己'      # ★ 内部哨兵：表示"这是模型的轮次"。**不渲染成 `自己：`**，渲染成 `<resp>`
TURN_CUE = '<resp>'      # 模型轮次的内联标记；tokenizer id = 140，**单 token**
TURN_CUE_ID = 140
EOS = '<eos>'            # 轮末终止符（保留；`<cont>` 已退休）
EOS_ID = 128

# 「不裁剪」阈值与其 O(n²) 的来龙去脉，现在住在 `segmentation.ContextWindow.NO_LIMIT`
# （上下文管理已归分句器模块）。这里留个别名，方便沿用旧名字的调用方/文档。
_NO_WINDOW_LIMIT = ContextWindow.NO_LIMIT

Entry = Tuple[str, str]  # (speaker, sentence)


def _load_split_line_removed():
    """占位：分句器已搬到 `training/segmentation.py`（见模块头部说明）。

    保留这个名字只为让"旧写法被删掉了"这件事可被 grep 到；不要使用。
    """
    raise RuntimeError('分句器已搬到 training/segmentation.py，请直接 import split_line')


class DialogueStream:
    """按句滑窗的对话流 + 单流渲染（模型输入 / 日志是**同一条流**）。

    参数
    ----
    encode : Callable[[str], Sequence[int]]
        **必须传真实 tokenizer 的 encode** —— 窗口按 token 算，不按字符算。
        测试里传 `lambda s: list(s)` 就能拿到「1 字 = 1 token」的已知答案。
    window : int
        上下文窗口（token）。**约束模型输入**（即日志 + 末尾 `<resp>`），
        不含模型将要生成的回复。
    self_label : str
        内部哨兵，默认 `自己`；**不参与渲染**（渲染成 `<resp>`）。
    cue : str
        模型轮次的内联标记，默认 `<resp>`（单 token）。
    colon : str
        其他人标签后的冒号，默认全角 `：`（与仓库既有语料一致）。
    max_sentence_len : int
        交给 `split_line` 的超长句二次切阈值。
    group_turns : bool
        True（默认）＝把连续同一说话人的句子**合并成一行**。
        内部仍**按句存储**，所以窗口照样在句子边界弹出。
        这是保证「`prompt + target` 等于下一段日志」的必要条件。
    emit_eos : bool
        True（默认）＝每轮模型回复末尾追加 `<eos>`。
    """

    def __init__(
        self,
        encode: Callable[[str], Sequence[int]],
        window: int = 256,
        *,
        self_label: str = SELF_LABEL,
        cue: str = TURN_CUE,
        colon: str = '：',
        max_sentence_len: int = 80,
        group_turns: bool = True,
        emit_eos: bool = True,
    ) -> None:
        self.encode = encode
        self.window = int(window)
        self.self_label = self_label
        self.cue = cue
        self.colon = colon
        self.max_sentence_len = max_sentence_len
        self.group_turns = group_turns
        self.emit_eos = emit_eos
        # ★ 上下文管理（按句滑窗）**不在本模块实现** —— 用 `training/segmentation.py`
        #   的 `ContextWindow`（全项目统一）。这里只负责"怎么把记录渲染成文本"，
        #   长度测量通过 `self._context_len_of` 回调交给它。
        self._win = ContextWindow(self._context_len_of, self.window)

    # ---------- 输入记录（转发给 ContextWindow） ----------

    @property
    def _entries(self) -> List[Entry]:
        """★ **活的**记录列表（`ContextWindow.units`）—— 因为要就地改最后一个单元。"""
        return self._win.units

    @property
    def dropped(self) -> int:
        """累计弹出的**整句**数（不是 token 数）。"""
        return self._win.dropped

    @property
    def overflow(self) -> bool:
        """单句本身就超窗 ⇒ 软上限放行（否则会死循环）。"""
        return self._win.overflow

    # ---------- 内部 ----------

    def _n(self, text: str) -> int:
        return len(self.encode(text))

    def _label(self, speaker: str) -> str:
        """模型自己的轮次用 `<resp>`（无冒号）；其他人用 `名字：`。"""
        if speaker == self.self_label:
            return self.cue
        return f'{speaker}{self.colon}'

    # ---------- 写入 ----------

    def append(self, speaker: str, text: str) -> None:
        """加入一个说话人的一段话（会按句拆开），并把窗口收到尺寸内。"""
        sents = [s for s in split_line(text, max_len=self.max_sentence_len) if s.strip()]
        if sents:
            self._win.extend([(speaker, s) for s in sents])

    def commit(self, text: str) -> None:
        """把模型**生成的回复**记入日志（标记 `<resp>` 由 `_label` 补，函数只存正文）。

        `emit_eos=True`（默认）时在本轮**最后一句**末尾贴 `<eos>`，贴完**重新收窗**。
        """
        before = len(self._entries)
        self.append(self.self_label, text)
        if self.emit_eos and len(self._entries) > before:
            spk, sent = self._entries[-1]
            self._entries[-1] = (spk, sent + EOS)
            # ★ 必须重新收窗：`<eos>` 是在 append() 的 enforce_window() **之后**贴上去的，
            #   会让最后一句话变长。漏了这一步就会出现「context_len() > window 但
            #   overflow 仍是 False」这种自相矛盾的状态。
            self.enforce_window()

    def rename(self, old: str, new: str) -> int:
        """把某说话人的历史标签整体改名（「对象A → 名字」的绑定）。

        返回被改名的句子数。用户已定：**允许绑定后再改名**（更像真人，也给了 TA 改口的能力）。
        `old == new` 时不做任何事、返回 0。
        """
        if old == new:
            return 0
        n = 0
        out: List[Entry] = []
        for spk, sent in self._entries:
            if spk == old:
                spk, n = new, n + 1
            out.append((spk, sent))
        self._entries[:] = out
        return n

    def enforce_window(self) -> None:
        """按句滑窗：从头部弹出**整句**，直到模型输入（日志 + 末尾 `<resp>`）放得进窗口。

        ★ 实现已搬到 `training/segmentation.py::ContextWindow`（全项目统一）；
          本方法只是转发。规则细节（只在句子边界弹、至少保留 1 句、无限窗口不测量）
          见那边的 docstring。
        """
        self._win.enforce()

    # ---------- 渲染 ----------

    def _render_entries(self, entries: Sequence[Entry]) -> str:
        """渲染给定的记录列表（`render()` 的**纯粹**版本：不读 self 的窗口状态）。

        独立出来是为了给 `ContextWindow` 当测量回调 —— 它要按"假设只留这几条"来量长度。
        """
        lines: List[str] = []
        if self.group_turns:
            cur_spk, buf = None, ''
            for spk, sent in entries:
                if spk == cur_spk:
                    buf += sent
                else:
                    if cur_spk is not None:
                        lines.append(self._label(cur_spk) + buf)
                    cur_spk, buf = spk, sent
            if cur_spk is not None:
                lines.append(self._label(cur_spk) + buf)
        else:
            lines = [self._label(spk) + sent for spk, sent in entries]
        return '\n'.join(lines)

    def render(self) -> str:
        """渲染**日志**（持久记录）—— 模型自己的轮次带 `<resp>` 前缀。"""
        return self._render_entries(self._entries)

    def _context_len_of(self, entries: Sequence[Entry]) -> int:
        """模型输入（日志 + 末尾 `<resp>`）的 token 数 —— `ContextWindow` 的测量回调。"""
        body = self._render_entries(entries)
        return self._n(f'{body}\n{self.cue}' if body else self.cue)

    def prompt(self) -> str:
        """喂给模型、请它说话的输入 ＝ **日志 + 末尾一行 `<resp>`**。

        ★ 与日志是**同一条流**：`prompt()` 只是日志再往前推一格。
          于是 `prompt_k + target_k == 日志_{k+1}`（**零替换**）——
          这是选「乙」而不是「甲」的全部意义，见 `test_stream_is_strictly_contiguous`。
        """
        body = self.render()
        return f'{body}\n{self.cue}' if body else self.cue

    # ---------- 度量 ----------

    def transcript_len(self) -> int:
        """日志的 token 数（不含末尾待生成的 `<resp>`）。"""
        return self._n(self.render())

    def context_len(self) -> int:
        """模型输入的 token 数（含末尾 `<resp>`）—— 窗口约束的就是它。"""
        return self._context_len_of(self._entries)

    def history(self) -> List[Entry]:
        return list(self._entries)

    def reset(self) -> None:
        self._win.reset()

    # ---------- loss 区间 ----------

    def loss_token_spans(self, text: str | None = None) -> List[Tuple[int, int]]:
        """返回日志里**该算 loss** 的 token 区间（半开区间 `[start, end)`）。

        规则：每段 `<resp>` **之后**、到对应的 `<eos>`（含）为止 —— 即"模型自己说的话"。
        提示词（别人的话 + `<resp>`）不算 loss。

        实现不写死 id：它先 `encode(cue)` / `encode(EOS)` 拿到 token 序列再在流里搜，
        所以真 tokenizer（`<resp>`=[140]、`<eos>`=[128]）和测试用的假编码器都能用。
        """
        ids = list(self.encode(self.render() if text is None else text))
        cue_ids = list(self.encode(self.cue))
        eos_ids = list(self.encode(EOS))
        spans: List[Tuple[int, int]] = []
        i = 0
        while i < len(ids):
            if ids[i:i + len(cue_ids)] == cue_ids:
                k = i + len(cue_ids)
                while k < len(ids) and ids[k:k + len(eos_ids)] != eos_ids:
                    k += 1
                if k < len(ids):
                    spans.append((i + len(cue_ids), k + len(eos_ids)))
                    i = k + len(eos_ids)
                    continue
            i += 1
        return spans

    # ---------- 杂项 ----------

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return (f'DialogueStream(句子={len(self._entries)}, 窗口={self.window}, '
                f'输入={self.context_len()}, 已弹出={self.dropped}, '
                f'溢出={self.overflow})')


def parse_log(
    text: str,
    encode: Callable[[str], Sequence[int]],
    window: int = 10 ** 9,
    **kwargs,
) -> DialogueStream:
    """把 `render()` 产出的日志**解析回** `DialogueStream`。

    这是"造完语料要能读回来验证"的入口 —— 没有它，"格式对不对"只能靠肉眼。
    等价性判据（有测试钉住）：`parse_log(s.render(), ...).render() == s.render()`。

    识别两类行：
      - `<resp>...`            → 模型自己的轮次（`speaker = self_label`）
      - `名字：...`            → 其他人
      - 其余（含空行）         → 续行，接到上一条的文本后面

    ⚠ 默认 `window` 取极大值：解析时**不能**再弹句子，否则"读回来"和"写出去"不等价。
      要按真实窗口裁剪，显式传 `window`。
    """
    cue = kwargs.get('cue', TURN_CUE)
    colon = kwargs.get('colon', '：')
    stream = DialogueStream(encode, window, **kwargs)
    for line in text.split('\n'):
        if not line.strip():
            continue
        if line.startswith(cue):
            # `<eos>` 由**分句器**自动粘在上一句上（`training/segmentation.py` 规则二：
            # 只由机制符/空白构成的片段并入前一句），所以这里**不需要**任何特殊处理。
            stream.append(stream.self_label, line[len(cue):])
        elif colon in line:
            speaker, body = line.split(colon, 1)
            stream.append(speaker, body)
        elif stream._entries:
            # 续行：接到上一条后面（保持原来的说话人）
            spk, sent = stream._entries[-1]
            stream._entries[-1] = (spk, sent + line)
    return stream


def read_corpus(path: str) -> List[str]:
    """读一份对话语料文件（空行分隔的若干段日志）→ 每段原文。

    ★ 本项目对话语料的**统一输入**：`build_*` 产出、`prepare.py` 消费、评估脚本读回，
      都走 `read_corpus` / `write_corpus`，**不要再各自手写 `split('\\n\\n')`** ——
      口径一旦分叉，就会重现 `AGENTS §5.11` 那种"两个数字不同源"的事故。
    """
    with open(path, encoding='utf-8') as f:
        text = f.read()
    # ★ 末尾那个换行会让最后一段多带一个 `\n`，必须剥掉才能与 `write_corpus` 往返等价
    return [b.strip('\n') for b in text.split('\n\n') if b.strip()]


def write_corpus(path: str, logs: Sequence[str]) -> int:
    """把若干段日志写成一份语料文件（段间空行、末尾一个换行）。返回写入段数。"""
    blocks = [b.strip('\n') for b in logs if b and b.strip()]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(blocks) + '\n')
    return len(blocks)


def iter_training_samples(
    script: Sequence[Entry],
    encode: Callable[[str], Sequence[int]],
    window: int = 256,
    **kwargs,
) -> Iterator[Tuple[str, str]]:
    """把一段剧本滚成训练样本：每个「模型轮」产出一条 `(prompt, reply)`。

    `script` ＝ `[(speaker, text), ...]`。`speaker == SELF_LABEL`（`自己`）的条目
    表示**这是模型该说的内容**：先产出当时的 `(prompt, reply)`，再记入日志。

    产出：
      - `prompt` ＝ 日志 + 末尾 `<resp>`（模型输入，**不含**待生成的回复）
      - `reply`  ＝ 训练目标 ＝ **内容 + `<eos>`**（不含 `<resp>`，那是 harness 喂的）

    ★★ **严格连续**（有测试钉住）：`prompt_k + reply_k == 日志_{k+1}`，逐字符相等、
    **没有任何替换**。所以整条流可以直接拼成一个 bin，用「`<resp>` → `<eos>`」定区间
    算 loss（`DialogueStream.loss_token_spans()`）。

    ⚠ 样本之间的**窗口状态是连续的**（同一条流滚下去）——这正是要训的东西：
    模型得在「已经有若干轮历史、头部可能已被弹掉」的状态下接话。
    """
    stream = DialogueStream(encode, window, **kwargs)
    for speaker, text in script:
        if speaker == SELF_LABEL:
            yield stream.prompt(), text + (EOS if stream.emit_eos else '')
            stream.commit(text)
        else:
            stream.append(speaker, text)
