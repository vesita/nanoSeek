"""对话流框架（`training/dialogue_stream.py`）的测试。

★ **已知答案对照**：窗口/弹出相关的测试用**假编码器** `lambda s: list(s)`
  （1 字符 = 1 token），于是长度、弹几句、loss 区间下标全部可以手算 ——
  判据不是"跑通就算过"，而是能对上具体数字。并且成对给出「大窗口必须不弹 /
  小窗口必须弹」，避免"它无论如何都弹"也能通过的空测试。

★ 另有一组用**真 tokenizer** 的测试，钉住「`<resp>` 是**单 token**（id=140）」，
  以及「词表生成脚本与现盘产物不许漂移」。
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


# ---------------------------------------------------------------- 分句器复用


def test_reuses_repo_sentence_splitter():
    """证明真的复用了 `data/chinese/split_sentences.py`，而不是静默退化成不切句。"""
    assert split_line('a。b。') == ['a。', 'b。']
    assert split_line('你好，我是李华。') == ['你好，我是李华。']  # 逗号不是句末


# ---------------------------------------------------------------- 单流渲染


def test_render_matches_the_designed_flow():
    """用户 2026-09-14 设计的那条流，逐字对上（「乙」：单流、`<resp>` 内联）。"""
    s = make(window=256)
    s.append('A', '你好，我是李华。')
    # 日志：还没有模型的话；模型输入＝日志 + 末尾一行 <resp>
    assert s.render() == 'A：你好，我是李华。'
    assert s.prompt() == 'A：你好，我是李华。\n<resp>'
    # 模型生成「你好」之后，记回日志（`<resp>` 前缀 + 末尾 `<eos>`）
    s.commit('你好')
    assert s.render() == 'A：你好，我是李华。\n<resp>你好<eos>'
    assert s.prompt() == 'A：你好，我是李华。\n<resp>你好<eos>\n<resp>'


def test_prompt_of_empty_stream_is_just_the_cue():
    """空流的模型输入就是孤零零一个 `<resp>`（不该带前导换行）。"""
    assert make().prompt() == '<resp>'


def test_resp_absorbs_the_self_label():
    """★ `自己：` 的角色被 `<resp>` 吸收 —— 1 token 取代 3 token，且没有冗余标记。

    `SELF_LABEL`（`'自己'`）只是**内部哨兵**，永远不该出现在渲染结果里。
    """
    s = make()
    s.append('对象A', '在吗？')
    s.commit('在的。')
    out = s.render()
    assert '<resp>' in out
    assert '自己：' not in out and SELF_LABEL not in out


def test_emit_eos_can_be_disabled():
    s = make(emit_eos=False)
    s.append('A', '你好，我是李华。')
    s.commit('你好')
    assert s.render() == 'A：你好，我是李华。\n<resp>你好'


def test_cont_is_never_emitted():
    """★ 用户 2026-09-14 决定：**合并 `<cont>`、保留 `<eos>`**。

    新格式里 `<cont>` 不再出现（语义已被轮次机制取代），但 **token 本身仍在词表里**
    —— 既有 bin 里有 100 万个它，删了旧 bin 就解不出来。
    """
    s = make()
    s.append('对象A', '在吗？')
    s.commit('在的！你说。')
    s.append('对象A', '好的。')
    s.commit('嗯嗯，我听着。')
    text = s.render()
    assert '<eos>' in text
    assert '<cont>' not in text


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


def test_single_oversized_sentence_does_not_loop():
    """单句就超窗 ⇒ 软上限放行 + `overflow` 标记，**绝不死循环**。"""
    s = make(window=5)
    long_sent = '这是一个远远超过窗口长度的句子。'
    s.append('A', long_sent)
    assert s.overflow is True
    assert s.history() == [('A', long_sent)]


# ---------------------------------------------------------------- 名字绑定


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
    assert r0 == f'你好呀。{EOS}', '训练目标 = 内容 + <eos>（不含 <resp>，那是 harness 喂的）'

    p1, r1 = out[1]
    assert f'<resp>你好呀。{EOS}' in p1, '上一轮必须已内联进日志'
    assert p1.endswith('\n' + TURN_CUE)
    assert r1 == f'我叫小寻。{EOS}'


def test_stream_is_strictly_contiguous():
    """★★ 选「乙」的全部意义：`prompt_k + reply_k` **逐字符等于**下一段日志的前缀。

    这是**零替换**的严格连续性。选「甲」时 cue 会被 `自己：` 替换掉，
    所以数据只能是 (prompt, target) 对、喂不进「一条长流 + 按终止符掩码」的现有管线。
    「乙」把它变成连续流。
    """
    script = [
        ('对象A', '你好，我是李华。'),
        (SELF_LABEL, '你好呀。我是小寻。'),
        ('对象A', '你多大了？'),
        (SELF_LABEL, '还在长个儿呢。'),
    ]
    (p0, r0), (p1, r1) = list(iter_training_samples(script, W, window=200))

    assert p1.startswith(p0 + r0), 'prompt_k + reply_k 必须是 prompt_{k+1} 的前缀'
    # 负向对照：若中间插了任何东西（比如旧的"替换"写法），前缀关系就断了
    assert not p1.startswith(p0 + r0 + '\n' + r0), '判据要能区分出多余内容'


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


# ---------------------------------------------------------------- loss 区间


def test_loss_token_spans_cover_only_the_models_own_words():
    """★ loss 区间 = `<resp>` 之后到 `<eos>`（含）；别人的话与 `<resp>` 都不算。

    用假编码器（1 字 = 1 token）可以手算下标：
        `A：你好。\\n<resp>你好<eos>`
         0123456 789...                → cue 在 [6,12)，eos 结束于 19
    """
    s = make()
    s.append('A', '你好。')
    s.commit('你好')
    log = s.render()
    assert log == 'A：你好。\n<resp>你好<eos>'

    spans = s.loss_token_spans()
    assert spans == [(12, 19)], spans
    ids = list(s.encode(log))
    covered = ''.join(''.join(ids[a:b]) for a, b in spans)
    assert covered == f'你好{EOS}'
    # 别人的话与 <resp> 都落在 loss 之外
    assert 'A：' not in covered
    assert TURN_CUE not in covered


def test_loss_token_spans_empty_when_nothing_committed():
    s = make()
    s.append('A', '你好。')
    assert s.loss_token_spans() == []


# ---------------------------------------------------------------- 钉住 tokenizer


def test_turn_cue_is_a_single_token():
    """★ 钉住 tokenizer 事实：`<resp>` 是**单 token**（id=140），不是 6 个字符。

    tokenizer 重建后若丢了它，这里会立刻红 —— 否则会静默退化成多 token，
    悄悄吃掉窗口，而且 loss 区间的搜索也会错。
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
    assert names[12] == TURN_CUE, f'mech_names[12] 应为 {TURN_CUE!r}（→ id 140），实际 {names[12]!r}'

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
    # ⚠ 这里**不能**放含 <resp>/<eos> 的文本 —— 它们是 special，会被编成 **1** 个 token，
    #   那条测的是"普通文本仍逐字符"，混进 special 就自相矛盾了。
    'A：你好。\nB：你好。',
])
def test_ordinary_text_encoding_unchanged_by_cue(text):
    """加了 cue 之后，普通文本仍编成**逐字符**（汉字区 id 没被挪动）。"""
    from tokenizers import Tokenizer

    path = os.path.join(_ROOT, 'data', 'chinese', 'char_tokenizer.json')
    tok = Tokenizer.from_file(path)
    assert len(tok.encode(text, add_special_tokens=False).ids) == len(text)
