"""`training/masking.py::build_assistant_mask` 的测试。

## 为什么这个模块值得重点测
`build_assistant_mask` 决定**哪些 token 计入 loss**。它出错不会报错、不会崩，
只会让 loss 曲线看起来"正常但学不会"——本项目已经在这类静默错误上栽过多次
（旧 val 切分错、mask 标记 id 从 117/119 改成 128/130 后文档没跟上等）。

它是**纯张量逻辑**、无依赖、毫秒级，是"低成本高收益"测试的典型。

## 两类测试
1. 手写用例：把**意图**写死在断言里（哪一行该算 loss）。
2. 随机对照：用一份朴素的 Python 参考实现跑 300 组随机输入，逐位对齐。
   向量化实现（cummin + flip）非常容易写错边界，随机对照才能覆盖到。
"""
import random

import pytest
import torch

from training.masking import build_assistant_mask

NL = 0        # 换行
EOS = 128     # <eos> 收尾
CONT = 130    # <cont> 待续
REPLY = (EOS, CONT)
SEP = (0,)


def _ref_mask(rows, reply_ids=REPLY):
    """朴素 Python 参考实现：逐行、逐字符，不向量化、不用 cummin。

    规则（与 masking.py 文档一致）：一行里出现终止符 → 从**行首**到该行
    **最后一个**终止符（含）的 token 全部计入 loss；行内终止符之后的 token
    不计入；换行本身永远不计入；没有终止符的行整行忽略。
    """
    out = []
    for row in rows:
        m = [False] * len(row)
        start = 0
        # i 走到 len(row) 是为了处理"最后一行没有换行结尾"的情况
        for i in range(len(row) + 1):
            if i == len(row) or row[i] == NL:
                line = row[start:i]
                term = [j for j, t in enumerate(line) if t in reply_ids]
                if term:
                    for j in range(term[-1] + 1):
                        m[start + j] = True
                start = i + 1
        out.append(m)
    return out


def _pad_nl(rows):
    """把可变长行用**行尾换行**补齐到同一长度。

    换行是语义中性的填充：它只是把最后一行在同一个位置截断，不会让任何
    额外 token 计入 loss（`test_trailing_newline_padding_is_neutral` 钉住了这点）。
    """
    T = max(len(r) for r in rows)
    return [list(r) + [NL] * (T - len(r)) for r in rows]


def _check(rows, reply_ids=REPLY):
    rows = _pad_nl(rows)                      # torch.tensor 要求矩形
    y = torch.tensor(rows, dtype=torch.long)
    got = build_assistant_mask(y, reply_ids, SEP)
    exp = torch.tensor(_ref_mask(rows, reply_ids), dtype=torch.bool)
    assert got.shape == y.shape, f"mask 形状 {tuple(got.shape)} != y {tuple(y.shape)}"
    assert got.dtype == torch.bool
    assert torch.equal(got, exp), (
        f"与参考实现不一致\n  y   ={rows}\n  got ={got.tolist()}\n  exp ={exp.tolist()}")


# --------------------------------------------------------------------------
# 1) 手写用例：把意图写死
# --------------------------------------------------------------------------
def test_single_model_line_all_counted():
    """一行 `正文<eos>\\n`：正文与 <eos> 计入，换行不计入。"""
    _check([[5, 6, 7, EOS, NL]])


def test_user_line_excluded_model_line_included():
    """两行：第一行是用户（无终止符）→ 全排除；第二行是模型 → 全计入。"""
    _check([[10, 11, NL, 20, 21, EOS, NL]])


def test_no_terminator_means_all_excluded():
    _check([[10, 11, NL, 20, 21]])


def test_cont_behaves_like_eos():
    _check([[10, 11, NL, 20, 21, CONT, NL]])


def test_line_with_only_terminator_is_counted():
    """空回复 + 终止符：终止符本身必须计入（否则"学会闭嘴"这个信号就丢了）。"""
    _check([[5, NL, EOS, NL, 7, 8, NL]])


def test_terminator_mid_line_excludes_the_tail():
    """终止符之后的同行 token 不计入（这是当前实现的**既定行为**，钉住它）。

    真实数据里终止符总在行尾，所以这是边界行为；钉住是为了以后改动时
    能立刻看出"行为变了"，而不是悄悄改变 loss 口径。
    """
    y = torch.tensor([[5, EOS, 9, NL]])
    m = build_assistant_mask(y, REPLY, SEP)
    assert m.tolist() == [[True, True, False, False]]


def test_last_terminator_on_a_line_wins():
    """一行有多个终止符时，计入到**最后一个**（cummin 语义的自然结果）。"""
    y = torch.tensor([[1, EOS, 2, EOS, 3, NL]])
    m = build_assistant_mask(y, REPLY, SEP)
    assert m.tolist() == [[True, True, True, True, False, False]]


def test_two_model_lines_both_counted():
    _check([[1, EOS, NL, 2, 3, CONT, NL]])


def test_terminator_at_very_first_position():
    _check([[EOS, NL, 5, 6, NL]])


def test_terminator_at_very_last_position_no_trailing_newline():
    _check([[5, 6, NL, 7, EOS]])


def test_empty_and_degenerate_inputs():
    """空张量 / 零宽张量不能崩。"""
    assert build_assistant_mask(torch.zeros((3, 0), dtype=torch.long), REPLY, SEP).shape == (3, 0)
    assert build_assistant_mask(torch.zeros((0, 5), dtype=torch.long), REPLY, SEP).shape == (0, 5)
    # 全是换行
    _check([[NL, NL, NL]])
    # 单 token
    _check([[EOS]])
    _check([[NL]])
    _check([[7]])


def test_reply_ids_none_entries_are_ignored():
    """`reply_ids` 里混入 None（配置缺 key 时会拿到 None）不应崩，且等价于去掉。"""
    y = torch.tensor([[1, EOS, NL]])
    a = build_assistant_mask(y, (None, EOS), SEP)
    b = build_assistant_mask(y, (EOS,), SEP)
    assert torch.equal(a, b)


def test_no_reply_ids_masks_everything_out():
    y = torch.tensor([[1, 2, NL, 3, 4]])
    assert not build_assistant_mask(y, (), SEP).any()


def test_sep_ids_is_accepted_and_ignored():
    """`sep_ids` 是历史参数，当前实现忽略它（换行边界自动检测）—— 钉住这个契约。"""
    y = torch.tensor([[1, EOS, NL]])
    assert torch.equal(build_assistant_mask(y, REPLY, (0,)),
                       build_assistant_mask(y, REPLY, (999,)))


# --------------------------------------------------------------------------
# 2) 随机对照：向量化实现最容易错的就是边界拼接
# --------------------------------------------------------------------------
def test_random_rows_match_reference():
    """400 组随机行（含空行、连续换行、行首/行尾终止符）逐位对齐参考实现。"""
    rng = random.Random(20260910)
    rows = []
    for _ in range(400):
        n = rng.randint(1, 24)
        row = []
        for _ in range(n):
            r = rng.random()
            if r < 0.22:
                row.append(NL)
            elif r < 0.34:
                row.append(rng.choice(REPLY))
            else:
                row.append(rng.randint(1, 200))
        rows.append(row)
    _check(rows)


def test_trailing_newline_padding_is_neutral():
    """行尾补换行不改变原前缀上的 mask —— `_check` 依赖这个性质，单独钉住。"""
    rows = [[5, 6, EOS], [7, 8, NL, 9, CONT, NL, 3]]
    padded = _pad_nl(rows)
    for r, p in zip(rows, padded):
        a = build_assistant_mask(torch.tensor([r], dtype=torch.long), REPLY, SEP)[0]
        b = build_assistant_mask(torch.tensor([p], dtype=torch.long), REPLY, SEP)[0]
        assert torch.equal(a, b[:len(r)])
        assert not b[len(r):].any(), "填充位不该被计入 loss"


def test_random_batch_equals_row_by_row():
    """批内各行必须完全独立：整批结果 == 逐行单独调用结果的拼接。

    这条专门抓"cummin/flip 沿错维度"这类 bug —— 那种 bug 只在 B>1 时暴露。
    """
    rng = random.Random(7)
    rows = _pad_nl([[rng.choice([NL, EOS, CONT, rng.randint(1, 50)])
                     for _ in range(rng.randint(1, 20))] for _ in range(12)])
    y = torch.tensor(rows, dtype=torch.long)
    batch = build_assistant_mask(y, REPLY, SEP)
    for i, row in enumerate(rows):
        one = build_assistant_mask(torch.tensor([row], dtype=torch.long), REPLY, SEP)[0]
        assert torch.equal(batch[i], one), f"第 {i} 行批处理与单行不一致"


def test_permuting_batch_permutes_mask():
    """打乱 batch 顺序，结果必须同步打乱 —— 再抓一次"串行依赖"类 bug。"""
    rng = random.Random(99)
    rows = _pad_nl([[rng.choice([NL, EOS, CONT, rng.randint(1, 50)]) for _ in range(16)]
                    for _ in range(8)])
    y = torch.tensor(rows, dtype=torch.long)
    m = build_assistant_mask(y, REPLY, SEP)
    perm = torch.randperm(len(rows))
    m2 = build_assistant_mask(y[perm], REPLY, SEP)
    assert torch.equal(m[perm], m2)


# --------------------------------------------------------------------------
# 3) 词表 id 回归：文档说的 128/130 必须与真实 tokenizer 一致
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_documented_terminator_ids_match_real_tokenizer():
    """PROJECT_STATE / masking.py 都写 `<eos>=128, <cont>=130`。

    旧文档曾写 117/119（已作废）。这条测试直接把文档绑到真词表上。
    """
    import json
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'data', 'chinese', 'char_tokenizer.json')
    if not os.path.exists(path):
        pytest.skip("char_tokenizer.json 不存在（换机器/未下载数据）")
    with open(path, encoding='utf-8') as f:
        vocab = json.load(f)['model']['vocab']
    ids = dict(vocab) if isinstance(vocab, dict) else {t: i for t, i in vocab}
    assert ids.get('<eos>') == EOS, f"<eos> 实际是 {ids.get('<eos>')}，文档写的是 {EOS}"
    assert ids.get('<cont>') == CONT, f"<cont> 实际是 {ids.get('<cont>')}，文档写的是 {CONT}"
