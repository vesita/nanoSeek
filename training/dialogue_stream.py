"""对话流与上下文窗口管理 —— 「按句滑窗 + `自己：` + `<resp>`」（v1）。

设计来源：用户 2026-09-14 提出。四条规则：

1. **按句分布** —— 对话流是一串「句子」，每句带说话人标签；
2. **滑窗按句弹出** —— 总长超过上下文窗口时，从**头部弹出整句**，直到新的句子放得下；
3. **`自己：` 自动补齐** —— 模型自己的轮次在**持久记录**里带 `自己：` 前缀；
4. **`<resp>` cue** —— 轮到模型时，喂给它的上下文末尾加一行 cue；
   模型**只生成内容**（不生成自己的标签），生成完把内容以 `自己：` 记回流里。

★ 规则 3 与 4 的**关键区分**（本模块存在的理由）：

    【持久记录 transcript】  A：你好，我是李华。\\n自己：你好。
    【模型看到的 prompt】    A：你好，我是李华。\\n<resp>

   ⇒ prompt **绝不带悬空的 `自己：` 结尾**。以标签结尾＝把模型自己的轮次变成
     匿名「待填槽位」（`B：` 的老毛病），模型就永远学不会「我是谁」。
      见 `tests/test_dialogue_stream.py::test_prompt_never_ends_with_self_label`。

★ 与已有 `<eos>` / `<cont>` 的关系（**互补，不冲突**）：

   - `<cont>`（id 130，见 `data/chinese/prepare.py::annotate_replies`）＝
     「我说完了、**请对方继续**」——是**轮末**标记，**由模型生成**；
   - `<resp>`＝「**该你说了**」——是**轮首**提示，**由外部（harness）插入**。

   所以一次完整往返大致是：
       prompt（以 cue 结尾） → 模型生成 `你好。<cont>` → 记成 `自己：你好。<cont>`
       → 对方说话 → prompt 再次以 cue 结尾 …

★ `cue` 是**单 token**（`id=140`，2026-09-14 落地，见 `TURN_CUE_ID`）：

   机制符区间的第一个预留位 `<res0>` 被**改名**成 `<resp>` 并登记为 special，
   用法与 `<eos>`(128) / `<cont>`(130) 同族 ⇒ `encode(cue) == [140]`（改前是 7 个 token）。

   ★ **为什么用新位（140）而不回收退休的 `<cont>`（130）** —— 2026-09-14 定：
     回收是**零收益**（词表大小与 id 数都不变，130 和 140 都照样占位），
     代价却是污染既有数据：id 130 在既有 bin 里共 **1,003,220** 个 `<cont>`
     （v2 344,673 / know 174,405 / dlg 163,989 / 旧 char 320,153）—— 改名会让这
     100 万 token 的含义变化。而且它违反 `train_tokenizer.py:57`
     「命名机制符顺序即 id，**绝不更改**」；B 段 ckpt 还在这 16 万个 `<cont>`
     上训过 1 epoch（强先验），换名是主动和已学关联对撞。**全新 id 零先验，严格更优。**

   ★ 名字纯属可读性取舍：当天先叫 `<你该说话了>`（自解释），随后改成 `<resp>`
     （短、与 `<eos>`/`<cont>`/`<pad>` 同族）——**同一个 id，纯改名**，
     token 数不变，旧数据与 ckpt 不受影响。

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
TURN_CUE = '<resp>'
TURN_CUE_ID = 140  # 机制符区间第一个预留位（原 <res0>）被改名为 cue；见模块 docstring
EOS = '<eos>'

# ★ 2026-09-14 用户决定：**合并 `<cont>`、保留 `<eos>`**。
#   - `<cont>`（id 130）在新格式里**不再生成** —— 它的语义（"我说完但请对方继续"）
#     已被轮次机制取代：cue 标记「该你说了」，下一个说话人标签标记「我接上了」。
#   - `<eos>`（id 128）**保留**，贴在每轮模型回复末尾（与仓库既有约定一致，
#     见 `prepare.py:248`）。
#   ⚠ **token 本身不能删**：既有 bin 里 `<cont>` 有 50 万个（v2 344,673 /
#     v3_dlg 163,989），删了旧 bin 就解不出来。只是"新数据不再生成"。
#   ★ 附带好处：`prepare.py::annotate_replies` **只认 `用户：`/`模型：`**（`:222`），
#     非匹配行**原样保留**（`:251-255`）⇒ 我们的 `对象A：`/`自己：` 格式**完全绕过**
#     那套标注（也不做 70% `A：/B：` 改写、不加引号、不走 `_should_continue`），
#     **`prepare.py` 一个字都不用改**。


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
        轮首提示，默认 `<resp>`。
    colon : str
        标签后冒号，默认全角 `：`（与仓库既有语料一致）。
        ⚠ 用户 2026-09-14 一条消息里写过半角 `A:`，**尚未最终定**，故做成参数。
    max_sentence_len : int
        交给 `split_line` 的超长句二次切阈值。
    group_turns : bool
        True（默认）＝把连续同一说话人的句子**合并成一行**。
        ⚠ 默认值是 True 而不是 False，理由是**训练目标要与渲染一致**：
        模型一次生成的是"整轮回复"，若渲染成"一句一行"，`prompt + target`
        就和下一轮的 prompt 对不上了（多出了重复标签）。见
        `tests/test_dialogue_stream.py::test_samples_are_self_consistent`。
        内部仍**按句存储**，所以窗口照样在句子边界弹出。
    emit_eos : bool
        True（默认）＝每轮模型回复末尾追加 `<eos>`（`<cont>` 已按用户决定废弃）。
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
        """把模型**生成的内容**以 `自己：` 记入持久记录（标签这里补）。

        `emit_eos=True`（默认）时，在本轮**最后一句**末尾追加 `<eos>` ——
        与仓库既有约定一致（`<eos>` 直接贴在整条回复末尾，见 `prepare.py:248`）。
        `<cont>` 已按用户 2026-09-14 的决定**不再生成**。
        """
        before = len(self._entries)
        self.append(self.self_label, text)
        if self.emit_eos and len(self._entries) > before:
            spk, sent = self._entries[-1]
            self._entries[-1] = (spk, sent + EOS)
            # ★ 必须**重新收窗**：`<eos>` 是在 `append()` 的 `enforce_window()` **之后**
            #   才贴上去的，会让最后一句话变长 5 个 token。漏了这一步就会出现
            #   「context_len() > window 但 overflow 仍是 False」这种自相矛盾的状态。
            #   （这个 bug 靠改名把 cue 从 7 字变 6 字、挪动了测试里的长度算术才暴露出来。）
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

    产出：`prompt` **以 cue 结尾**（这就是训练输入），`reply` 是**训练目标** ——
    等于「内容 + `<eos>`」（`emit_eos=True` 时）。

    ★ **自洽不变量**（有测试钉着）：把第 k 条样本的 `prompt + reply` 拼起来，
    必须是第 k+1 条 `prompt` 的前缀。**这条不成立就说明训练数据和推理时的上下文
    不是同一种形态**，模型会学到一套、用时另一套。
    见 `tests/...::test_samples_are_self_consistent`。

    ⚠ 终止符由**本模块自己插**：`prepare.py::annotate_replies` 只认
    `用户：`/`模型：`，我们的 `对象A：`/`自己：` 格式会被它**原样放行**（`:251-255`），
    所以**不能指望上游补** `<eos>`。

    ⚠ 样本之间的**窗口状态是连续的**（同一条流滚下去）——这正是要训的东西：
    模型得在「已经有若干轮历史、头部可能已被弹掉」的状态下接话。
    """
    stream = DialogueStream(encode, window, **kwargs)
    for speaker, text in script:
        if speaker == SELF_LABEL:
            prompt = stream.prompt()
            # 训练目标 = 模型实际要生成的东西：内容 + 终止符（`<cont>` 已废弃）
            yield prompt, text + (EOS if stream.emit_eos else '')
            stream.commit(text)
        else:
            stream.append(speaker, text)
