"""对话流与上下文窗口管理 —— 「按句滑窗 + `自己：` + `<你该说话了>`」（v1）。

设计来源：用户 2026-09-14 提出。四条规则：

1. **按句分布** —— 对话流是一串「句子」，每句带说话人标签；
2. **滑窗按句弹出** —— 总长超过上下文窗口时，从**头部弹出整句**，直到新的句子放得下；
3. **`自己：` 自动补齐** —— 模型自己的轮次在**持久记录**里带 `自己：` 前缀；
4. **`<你该说话了>` cue** —— 轮到模型时，喂给它的上下文末尾加一行 cue；
   模型**只生成内容**（不生成自己的标签），生成完把内容以 `自己：` 记回流里。

★ 规则 3 与 4 的**关键区分**（本模块存在的理由）：

    【持久记录 transcript】  A：你好，我是李华。\\n自己：你好。
    【模型看到的 prompt】    A：你好，我是李华。\\n<你该说话了>

   ⇒ prompt **绝不带悬空的 `自己：` 结尾**。以标签结尾＝把模型自己的轮次变成
     匿名「待填槽位」（`B：` 的老毛病），模型就永远学不会「我是谁」。
      见 `tests/test_dialogue_stream.py::test_prompt_never_ends_with_self_label`。

★ 与已有 `<eos>` / `<cont>` 的关系（**互补，不冲突**）：

   - `<cont>`（id 130，见 `data/chinese/prepare.py::annotate_replies`）＝
     「我说完了、**请对方继续**」——是**轮末**标记，**由模型生成**；
   - `<你该说话了>`＝「**该你说了**」——是**轮首**提示，**由外部（harness）插入**。

   所以一次完整往返大致是：
       prompt（以 cue 结尾） → 模型生成 `你好。<cont>` → 记成 `自己：你好。<cont>`
       → 对方说话 → prompt 再次以 cue 结尾 …

★ `cue` 是**单 token**（`id=140`，2026-09-14 落地，见 `TURN_CUE_ID`）：

   机制符区间的第一个预留位 `<res0>` 被**改名**成 `<你该说话了>` 并登记为 special，
   用法与 `<eos>`(128) / `<cont>`(130) 同族 ⇒ `encode(cue) == [140]`（改前是 7 个 token）。

   ★ **安全性已实测**（这是能这么改的前提）：
     - 词表仍是 **8192**；id `0..139` 逐位未变；**汉字区 `384..8191` 逐位未变**；
     - 12 个语料源各取 5KB，编码**逐位相同**（改前 vs 改后）；
     - 所有 char-level bin（`train_char` / `v2` / `v3_{lang,know,dlg}` + 5 个 val）
       在 id `140..383` 区间**零命中**（只有废弃的旧 BPE `train.bin` 有命中，它属另一套 tokenizer）。
     ⇒ **既有 bin 与 checkpoint 全部继续有效，不需要重建任何数据。**

   注意：`Tokenizer.decode` 默认 `skip_special_tokens=True`，所以 `decode([140])` 返回 `''`；
   要拿回文本得 `decode([140], skip_special_tokens=False)`（`<eos>` 一直也是这行为）。

★ 本模块**不含任何人格内容**，是纯机制 —— 人格定义在仓库外（`~/datasets/persona/`）。
"""
from __future__ import annotations

import importlib.util
import os
from typing import Callable, Iterator, List, Sequence, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SELF_LABEL = '自己'
TURN_CUE = '<你该说话了>'
TURN_CUE_ID = 140  # 机制符区间第一个预留位（原 <res0>）被改名为 cue；见模块 docstring


def _load_split_line():
    """复用 `data/chinese/split_sentences.py::split_line`，**不重写分句器**。

    `data/chinese/` 不是包（无 `__init__.py`），仓库里也没有任何地方 import 它，
    所以这里按**路径**加载。文件被挪走/改名时，
    `test_reuses_repo_sentence_splitter` 会立刻失败，不会静默退化成「不按句切」。
    """
    path = os.path.join(_ROOT, 'data', 'chinese', 'split_sentences.py')
    spec = importlib.util.spec_from_file_location('_nanoseek_split_sentences', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'找不到分句器: {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.split_line


split_line = _load_split_line()

Entry = Tuple[str, str]  # (speaker, sentence)


class DialogueStream:
    """按句滑窗的对话流 + 双视图渲染（持久记录 / 模型 prompt）。

    参数
    ----
    encode : Callable[[str], Sequence[int]]
        **必须传真实 tokenizer 的 encode** —— 窗口是按 token 算的，不是按字符。
        测试里传 `lambda s: list(s)` 就能拿到「1 字 = 1 token」的已知答案。
    window : int
        上下文窗口（token）。**约束的是模型输入**（含 cue），不是生成出来的回复。
    self_label : str
        模型自己的持久标签，默认 `自己`。
    cue : str
        轮首提示，默认 `<你该说话了>`。
    colon : str
        标签后冒号，默认全角 `：`（与仓库既有语料一致）。
        ⚠ 用户 2026-09-14 一条消息里写过半角 `A:`，**尚未最终定**，故做成参数。
    max_sentence_len : int
        交给 `split_line` 的超长句二次切阈值。
    group_turns : bool
        False（默认）＝**一句一行**，与用户的样例一致，也让「按句分布」显式可见；
        True ＝ 把连续同一说话人的句子合并成一行（标签更少）。
        ⚠ 尚未定，故做成参数。
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
        group_turns: bool = False,
    ) -> None:
        self.encode = encode
        self.window = int(window)
        self.self_label = self_label
        self.cue = cue
        self.colon = colon
        self.max_sentence_len = max_sentence_len
        self.group_turns = group_turns
        self._entries: List[Entry] = []
        self.dropped = 0        # 累计弹出的**整句**数（不是 token 数）
        self.overflow = False   # 单句本身就超窗 ⇒ 软上限放行（否则会死循环）

    # ---------- 内部 ----------

    def _n(self, text: str) -> int:
        return len(self.encode(text))

    def _label(self, speaker: str) -> str:
        return f'{speaker}{self.colon}'

    # ---------- 写入 ----------

    def append(self, speaker: str, text: str) -> None:
        """加入一个说话人的一段话（会按句拆开），并把窗口收到尺寸内。"""
        for sent in split_line(text, max_len=self.max_sentence_len):
            if sent.strip():
                self._entries.append((speaker, sent))
        self.enforce_window()

    def commit(self, text: str) -> None:
        """把模型**生成的内容**以 `自己：` 记入持久记录（不含标签，标签这里补）。"""
        self.append(self.self_label, text)

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
        self._entries = out
        return n

    def enforce_window(self) -> None:
        """从头部弹出**整句**，直到模型输入（含 cue）放得进窗口。

        - 只在**句子边界**弹 ⇒ 永远不会留下半句；
        - 至少保留 1 句 ⇒ 单句超窗时不死循环，而是置 `overflow=True`（软上限）。
        """
        while len(self._entries) > 1 and self.context_len() > self.window:
            self._entries.pop(0)
            self.dropped += 1
        self.overflow = self.context_len() > self.window

    # ---------- 渲染 ----------

    def render(self, *, cue: bool = False) -> str:
        """渲染对话流。`cue=True` 即**模型看到的 prompt**。"""
        lines: List[str] = []
        if self.group_turns:
            cur_spk, buf = None, ''
            for spk, sent in self._entries:
                if spk == cur_spk:
                    buf += sent
                else:
                    if cur_spk is not None:
                        lines.append(self._label(cur_spk) + buf)
                    cur_spk, buf = spk, sent
            if cur_spk is not None:
                lines.append(self._label(cur_spk) + buf)
        else:
            lines = [self._label(spk) + sent for spk, sent in self._entries]
        if cue:
            lines.append(self.cue)
        return '\n'.join(lines)

    def prompt(self) -> str:
        """喂给模型、请它说话的上下文：持久记录 + 末尾一行 cue。

        ★ 末尾**永远**是 cue，**不是** `自己：` —— 见模块 docstring。
        """
        return self.render(cue=True)

    # ---------- 度量 ----------

    def transcript_len(self) -> int:
        """持久记录的 token 数（不含 cue）。"""
        return self._n(self.render())

    def context_len(self) -> int:
        """模型输入的 token 数（**含 cue**）—— 窗口约束的就是它。"""
        return self._n(self.prompt())

    def history(self) -> List[Entry]:
        return list(self._entries)

    def reset(self) -> None:
        self._entries.clear()
        self.dropped = 0
        self.overflow = False

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return (f'DialogueStream(句子={len(self._entries)}, 窗口={self.window}, '
                f'输入={self.context_len()}, 已弹出={self.dropped}, '
                f'溢出={self.overflow})')


def iter_training_samples(
    script: Sequence[Entry],
    encode: Callable[[str], Sequence[int]],
    window: int = 256,
    **kwargs,
) -> Iterator[Tuple[str, str]]:
    """把一段剧本滚成训练样本：每个「模型轮」产出一条 `(prompt, reply)`。

    `script` ＝ `[(speaker, text), ...]`。`speaker == SELF_LABEL`（`自己`）的条目
    表示**这是模型该说的内容**：先产出当时的 `(prompt, reply)`，再以 `自己：` 记入流
    （模拟真实回放）。非 `自己` 的条目直接加入流。

    产出：`prompt` **以 cue 结尾**（这就是训练输入），`reply` 是**不含标签**的原文
    （训练目标；终止符 `<eos>`/`<cont>` 由上游 `prepare.py` 那一层负责，本模块不管）。

    ⚠ 样本之间的**窗口状态是连续的**（同一条流滚下去）——这正是要训的东西：
    模型得在「已经有若干轮历史、头部可能已被弹掉」的状态下接话。
    """
    stream = DialogueStream(encode, window, **kwargs)
    for speaker, text in script:
        if speaker == SELF_LABEL:
            yield stream.prompt(), text
            stream.commit(text)
        else:
            stream.append(speaker, text)
