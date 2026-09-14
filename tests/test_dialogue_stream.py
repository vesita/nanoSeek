"""对话流框架（`training/dialogue_stream.py`）的测试。

★ **已知答案对照**：窗口/弹出相关的测试用**假编码器** `lambda s: list(s)`
  （1 字符 = 1 token），于是长度、弹几句全部可以手算 —— 判据不是"跑通就算过"，
  而是能对上具体数字。并且成对给出「大窗口必须不弹 / 小窗口必须弹」，
  避免"它无论如何都弹"也能通过的空测试。

★ 另有一组用**真 tokenizer** 的测试，钉住「`<resp>` 是**单 token**（id=140）」。
  如果哪天有人重建 tokenizer 而把 cue 丢了，这组会立刻红。
"""
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from training.dialogue_stream import (  # noqa: E402
    EOS,
    SELF_LABEL,
    TURN_CUE,
    TURN_CUE_ID,
    DialogueStream,
    iter_training_samples,
    split_line,
)

W = lambda s: list(s)  # noqa: E731  # 假编码器：1 字符 = 1 token


def make(window: int = 256, **kw) -> DialogueStream:
    return DialogueStream(W, window, **kw)


def strip_cue(prompt: str) -> str:
    """把 prompt 末尾那行 cue 去掉，得到"持久记录"形态。"""
    assert prompt.endswith('\n' + TURN_CUE)
    return prompt[: -len('\n' + TURN_CUE)]


# ---------------------------------------------------------------- 分句器复用


def test_reuses_repo_sentence_splitter():
    """证明真的复用了 `data/chinese/split_sentences.py`，而不是静默退化成不切句。"""
    assert split_line('a。b。') == ['a。', 'b。']
    assert split_line('你好，我是李华。') == ['你好，我是李华。']  # 逗号不是句末


# ---------------------------------------------------------------- 双视图渲染


def test_render_matches_the_designed_flow():
    """用户 2026-09-14 给的那个具体流程，逐字对上。"""
    s = make(window=256)
    s.append('A', '你好，我是李华。')
    # 模型看到的 prompt：持久记录 + 末尾一行 cue
    assert s.prompt() == 'A：你好，我是李华。\n<resp>'
    # 持久记录里还没有模型的话
    assert s.render() == 'A：你好，我是李华。'
    # 模型生成「你好」之后，以 `自己：` 记回持久记录，并按约定贴 <eos>
    s.commit('你好')
    assert s.render() == 'A：你好，我是李华。\n自己：你好<eos>'


def test_emit_eos_can_be_disabled():
    """用户原话里那一步的**字面**形态（不带 `<eos>`）—— 关掉 `emit_eos` 即可复现。"""
    s = make(window=256, emit_eos=False)
    s.append('A', '你好，我是李华。')
    s.commit('你好')
    assert s.render() == 'A：你好，我是李华。\n自己：你好'


def test_cont_is_never_emitted():
    """★ 用户 2026-09-14 决定：**合并 `<cont>`、保留 `<eos>`**。

    新格式里 `<cont>` 不再出现（它的语义已被轮次机制取代），但**token 本身仍在词表里**
    —— 既有 bin 里有 50 万个它，删了旧 bin 就解不出来。
    """
    s = make(window=256)
    s.append('对象A', '在吗？')
    s.commit('在的！你说。')
    s.append('对象A', '好的。')
    s.commit('嗯嗯，我听着。')
    text = s.render()
    assert '<eos>' in text
    assert '<cont>' not in text


def test_prompt_never_ends_with_self_label():
    """★ 本框架存在的理由：prompt 以 cue 结尾，**绝不**带悬空的 `自己：`。

    以标签结尾＝把模型自己的轮次变成一个匿名「待填槽位」（`B：` 的老毛病），
    模型就永远学不会「我是谁」。
    """
    s = make(window=40)
    for i in range(12):
        s.append('A', f'第{i}句闲聊。')
        assert s.prompt().splitlines()[-1] == TURN_CUE
        assert not s.prompt().endswith(f'{SELF_LABEL}：')
        s.commit(f'回第{i}句。')
        assert s.prompt().splitlines()[-1] == TURN_CUE
        assert not s.prompt().endswith(f'{SELF_LABEL}：')

    # 负向对照：模拟"以标签结尾"的旧写法，确认判据能抓住它（判据非恒真）
    bad = 'A：你好。\n自己：'
    assert bad.splitlines()[-1] != TURN_CUE
    assert bad.endswith(f'{SELF_LABEL}：')


# ---------------------------------------------------------------- 窗口


def test_window_enforcement_is_not_vacuous():
    """已知答案对照：**大窗口必须不弹**，小窗口必须弹。

    只有"小窗口弹了"不足以说明判据有效 —— 可能它无论如何都弹。
    """
    def build(window: int) -> DialogueStream:
        s = make(window=window)
        for i in range(20):
            s.append('A', f'句子{i}。')
        return s

    big, small = build(10_000), build(30)
    assert big.dropped == 0 and len(big) == 20, '大窗口不该弹出任何句子'
    assert small.dropped > 0 and len(small) < 20, '小窗口必须弹出句子'
    assert small.context_len() <= small.window or small.overflow


def test_context_never_exceeds_window():
    s = make(window=37)
    for i in range(30):
        s.append('A', f'这是第{i}个句子。')
        assert s.context_len() <= s.window or s.overflow
        s.commit(f'收到第{i}个。')
        assert s.context_len() <= s.window or s.overflow


def test_pops_only_whole_sentences():
    """弹出的必须是**整句** —— 不允许半句残留在流里。"""
    s = make(window=25)
    all_sents = [f'完整句子{i}。' for i in range(30)]
    for t in all_sents:
        s.append('A', t)
    kept = [sent for _, sent in s.history()]
    assert kept, '不该被清空（至少保留 1 句）'
    for sent in kept:
        assert sent in all_sents, f'残留了非整句: {sent!r}'
    assert s.dropped + len(kept) == len(all_sents), '弹出的必须是整句、不许丢字符'


def test_overflow_flag_matches_reality():
    """★ 回归：`overflow` 必须**随时**等于 `context_len() > window`。

    曾经的 bug：`commit()` 贴 `<eos>`（+5 token）后不再收窗 ⇒ 输入超窗而 `overflow`
    仍是 `False`，即「说没超，实际超了」的自相矛盾状态。本判据直接钉住这个不变量，
    比"长度断言"更早发现问题。
    """
    s = make(window=30)
    for i in range(25):
        s.append('A', f'第{i}句。')
        assert s.overflow == (s.context_len() > s.window), 'append 之后标记必须真实'
        s.commit(f'回第{i}句。')
        assert s.overflow == (s.context_len() > s.window), '★ commit 之后标记必须仍然真实'


def test_single_oversized_sentence_does_not_loop():
    """单句就超窗 ⇒ 软上限放行 + `overflow` 标记，**绝不死循环**。"""
    s = make(window=5)
    long_sent = '这是一个远远超过窗口长度的句子。'
    s.append('A', long_sent)
    assert s.overflow is True
    assert s.history() == [('A', long_sent)]


# ---------------------------------------------------------------- 自己：与绑定


def test_commit_uses_self_label():
    s = make()
    s.append('对象A', '我叫李华。')
    s.commit('你好呀。')
    assert s.history() == [('对象A', '我叫李华。'), (SELF_LABEL, f'你好呀。{EOS}')]


def test_rename_binds_a_name():
    """「对象A → 名字」的绑定：改完之后历史标签整体换成名字。"""
    s = make()
    s.append('对象A', '你好。')
    s.commit('你好呀。')
    s.append('对象A', '我是李华。')
    n = s.rename('对象A', '李华')
    assert n == 2, '应该改名 2 句'
    assert ('李华', '你好。') in s.history()
    assert all(spk != '对象A' for spk, _ in s.history())
    assert '李华：' in s.render()
    # 用户已定：允许绑定后再改名
    assert s.rename('李华', '小李') == 2
    assert s.rename('小李', '小李') == 0, 'old == new 应是 no-op'


# ---------------------------------------------------------------- 训练样本


def test_iter_training_samples_shapes():
    script = [
        ('对象A', '你好，我是李华。'),
        (SELF_LABEL, '你好呀。'),
        ('对象A', '你叫什么？'),
        (SELF_LABEL, '我叫小寻。'),
    ]
    out = list(iter_training_samples(script, W, window=60))
    assert len(out) == 2

    p0, r0 = out[0]
    assert p0 == '对象A：你好，我是李华。\n<resp>'
    assert r0 == f'你好呀。{EOS}', '训练目标 = 内容 + <eos>（不含标签）'

    p1, r1 = out[1]
    assert f'自己：你好呀。{EOS}' in p1, '上一轮必须已以 `自己：` 记入'
    assert p1.endswith(TURN_CUE)
    assert r1 == f'我叫小寻。{EOS}'


def test_prompt_is_rebuilt_with_self_label_substituting_the_cue():
    """★ cue 是**临时**的：下一轮 prompt 里，末尾 cue 被 `自己：` **取代**。

    这正是用户说的"自动补齐 `自己：` 这个前缀"。
    ⇒ 训练数据是 **(prompt, target) 对**，不是一条连续 token 流
      （prompt_{k+1} 不是 prompt_k + target_k 的简单拼接，中间发生过替换）。
      这条不变量是拼接式写数据**必错**的地方，所以钉下来。
    """
    script = [
        ('对象A', '你好，我是李华。'),
        (SELF_LABEL, '你好呀。我是小寻。'),
        ('对象A', '你多大了？'),
        (SELF_LABEL, '还在长个儿呢。'),
    ]
    (p0, r0), (p1, _) = list(iter_training_samples(script, W, window=200))

    # 正确的不变量：持久记录(p0) + `自己：` + r0 是 p1 的前缀（r0 已自带 <eos>）
    rebuilt = strip_cue(p0) + f'\n{SELF_LABEL}：' + r0
    assert p1.startswith(rebuilt), f'\n  rebuilt={rebuilt!r}\n  p1     ={p1!r}'
    # 负向对照：朴素拼接（不做替换）**必须不成立** —— 证明这条判据在真干活
    assert not p1.startswith(strip_cue(p0) + '\n' + r0)
    # cue 出现过，但持久记录里没有 cue
    assert TURN_CUE in p0 and TURN_CUE not in rebuilt


def test_group_turns_option():
    """默认 `group_turns=True`：同一说话人的连续句子渲染成一行（内部仍按句存，便于滑窗）。"""
    grouped = make()
    grouped.append('A', '你好。我是李华。')
    assert grouped.history() == [('A', '你好。'), ('A', '我是李华。')], '内部按句存'
    assert grouped.render() == 'A：你好。我是李华。', '渲染时合并'

    per_sent = make(group_turns=False)
    per_sent.append('A', '你好。我是李华。')
    assert per_sent.history() == [('A', '你好。'), ('A', '我是李华。')]
    assert per_sent.render() == 'A：你好。\nA：我是李华。'


# ---------------------------------------------------------------- 钉住 tokenizer


def test_turn_cue_is_a_single_token():
    """★ 钉住 tokenizer 事实：`<resp>` 是**单 token**（id=140），不是 7 个字符。

    tokenizer 重建后若丢了 cue，这里会立刻红 —— 否则会静默退化成 7 token，
    悄悄吃掉窗口（256 窗口里每轮白扔 6 token）。
    """
    from tokenizers import Tokenizer

    path = os.path.join(_ROOT, 'data', 'chinese', 'char_tokenizer.json')
    tok = Tokenizer.from_file(path)

    ids = tok.encode(TURN_CUE, add_special_tokens=False).ids
    assert ids == [TURN_CUE_ID], f'cue 应为单 token [{TURN_CUE_ID}]，实际 {ids}'
    # 与既有机制符同族：默认 decode 跳过 special，要拿回文本得显式关掉
    assert tok.decode([TURN_CUE_ID], skip_special_tokens=False) == TURN_CUE
    assert tok.decode([TURN_CUE_ID]) == ''
    assert tok.decode([128], skip_special_tokens=False) == '<eos>'  # 行为与既有的一致
    assert tok.get_vocab_size() == 8192, '词表大小不能变（变了所有 bin 都失效）'


def test_tokenizer_source_matches_the_built_artifact():
    """★ 词表**生成脚本**与**现盘产物**必须一致 —— 否则重建一次就静默漂移。

    `data/chinese/char_tokenizer.json` 被 `.gitignore` 忽略（`data/*/char_tokenizer.json`），
    所以**能进版本的真相是生成脚本里的 `mech_names`**。这条把两者钉在一起：
    「顺序即 id」⇒ `mech_names[i]` 必须恰好落在 id `128 + i`。
    """
    import ast
    import io
    import json

    src = io.open(os.path.join(_ROOT, 'data', 'chinese', 'train_tokenizer.py'),
                  encoding='utf-8').read()
    names = None
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Assign)
                and any(getattr(t, 'id', None) == 'mech_names' for t in node.targets)):
            names = ast.literal_eval(node.value)
    assert names, '没能从 train_tokenizer.py 里解析出 mech_names'
    assert names[12] == '<resp>', f'mech_names[12] 应为 <resp>（→ id 140），实际 {names[12]!r}'

    vocab = json.load(io.open(
        os.path.join(_ROOT, 'data', 'chinese', 'char_tokenizer.json'), encoding='utf-8')
    )['model']['vocab']
    for i, n in enumerate(names):
        assert vocab.get(n) == 128 + i, (
            f'mech_names[{i}]={n!r} 应在 id {128 + i}，实际 {vocab.get(n)} —— '
            f'生成脚本与现盘词表已漂移，重建一次就会改掉 id 语义')


@pytest.mark.parametrize('text', [
    '你好，我是李华。',
    '今天天气不错。',
    'A：你好。\n自己：你好。',
])
def test_ordinary_text_encoding_unchanged_by_cue(text):
    """加了 cue 之后，普通文本仍编成**逐字符**（汉字区 id 没被挪动）。"""
    from tokenizers import Tokenizer

    path = os.path.join(_ROOT, 'data', 'chinese', 'char_tokenizer.json')
    tok = Tokenizer.from_file(path)
    assert len(tok.encode(text, add_special_tokens=False).ids) == len(text)
