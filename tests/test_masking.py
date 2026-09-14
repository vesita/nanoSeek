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
import itertools
import random

import pytest
import torch

from training.masking import build_assistant_mask, build_resp_span_mask

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
    """dev-notes/83 / masking.py 都写 `<eos>=128, <cont>=130`。

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


# ==========================================================================
# 4) build_resp_span_mask：「单流 + <resp>」格式的精确掩码（2026-09-14 新增）
# ==========================================================================
# 规则：**`<resp>` 之后 → 对应 `<eos>`（含）** 算 loss。与 build_assistant_mask
#（行内含终止符 ⇒ 整行）的唯一区别是**不含 `<resp>` 本身** —— 它是 harness 喂的，
# 模型不该生成它。见 training/dialogue_stream.py::loss_token_spans。
RESP = 140
TOPIC = 141
RESP_IDS = (RESP,)
EOS_IDS = (EOS,)


def _ref_resp_mask(rows, resp_ids=RESP_IDS, eos_ids=EOS_IDS):
    """朴素参考实现：**逐字照抄** `masking.py::build_resp_span_mask` docstring 里的
    四条判据，O(n²) 但 n≤24，只求"肉眼可核对"。

    ⚠ **别"顺手优化"成一次扫描的状态机。** 第一版就是这么写的，而它错了：
    状态机在遇到 `<resp>` 时立刻开始给后续 token 打标，**根本不知道 `<eos>` 会不会来**，
    于是在**未闭合**的 `<resp>` 上会把尾巴整段标成 True
    （例：`<resp>a<eos><resp>b` ⇒ 它给 `b` 打标，实现正确答案是 False）。
    那次是**参考写错了，不是实现错** —— 消融时先怀疑参考。
    """
    out = []
    for row in rows:
        n = len(row)
        m = [False] * n
        for t in range(n):
            # 判据 1+2：t 之前存在 `<resp>`（prev_resp < t 由 j in range(t) 保证）
            prev_resp = -1
            for j in range(t):
                if row[j] in resp_ids:
                    prev_resp = j
            if prev_resp < 0:
                continue
            # 判据 3：自那个 `<resp>` 之后（严格早于 t）还没遇到 `<eos>`
            prev_eos = -1
            for j in range(t):
                if row[j] in eos_ids:
                    prev_eos = j
            if prev_eos > prev_resp:
                continue
            # 判据 4：t 之后（含 t）存在 `<eos>` —— 没闭合的回复整段不算
            next_eos = n
            for j in range(t, n):
                if row[j] in eos_ids:
                    next_eos = j
                    break
            if next_eos >= n:
                continue
            m[t] = True
        out.append(m)
    return out


def _check_resp(rows):
    rows = _pad_nl(rows)
    y = torch.tensor(rows, dtype=torch.long)
    got = build_resp_span_mask(y, RESP_IDS, EOS_IDS)
    exp = torch.tensor(_ref_resp_mask(rows), dtype=torch.bool)
    assert got.shape == y.shape and got.dtype == torch.bool
    if not torch.equal(got, exp):
        # ⚠ 只打印**第一处不一致的行** —— 400 行批处理时整表 dump 会把日志淹没
        #   （实测一次失败吐了 14KB，把真正的信息挤没了）。
        bad = (got != exp).any(dim=1).nonzero().flatten().tolist()
        i = bad[0]
        cols = (got[i] != exp[i]).nonzero().flatten().tolist()
        raise AssertionError(
            f"{len(bad)}/{len(rows)} 行与参考实现不一致；第一处 row[{i}]：\n"
            f"  y   ={rows[i]}\n"
            f"  got ={got[i].tolist()}\n"
            f"  exp ={exp[i].tolist()}\n"
            f"  不一致列={cols}")


def _authority_spans(ids, cue=RESP_IDS, eos=EOS_IDS):
    """`DialogueStream.loss_token_spans()` 的算法逐字复刻（顺序扫描）。

    ★ 这是 loss 口径的**权威定义**：`build_resp_span_mask` 只是它的向量化实现，
    两者必须对**所有**输入给出同一个 mask（含相邻 `<resp>` 这种退化输入）。
    """
    out = []
    i = 0
    while i < len(ids):
        if tuple(ids[i:i + len(cue)]) == tuple(cue):
            k = i + len(cue)
            while k < len(ids) and tuple(ids[k:k + len(eos)]) != tuple(eos):
                k += 1
            if k < len(ids):
                out.append((i + len(cue), k + len(eos)))
                i = k + len(eos)
                continue
        i += 1
    return out


def _ref_from_spans(rows, cue=RESP_IDS, eos=EOS_IDS):
    out = []
    for row in rows:
        m = [False] * len(row)
        for a, b in _authority_spans(row, cue, eos):
            for j in range(a, b):
                m[j] = True
        out.append(m)
    return out


def test_resp_span_excludes_resp_but_includes_the_rest():
    """★ 核心契约：`<resp>你好。<eos>` ⇒ `<resp>` **不算**，正文与 `<eos>` 算。"""
    y = torch.tensor([[RESP, 5, 6, EOS, NL]])
    m = build_resp_span_mask(y, RESP_IDS, EOS_IDS)
    assert m.tolist() == [[False, True, True, True, False]]


# 穷举用的短行字母表（含 `<topic>`：它落在区间内要算 loss）
_ALPHA = (NL, EOS, RESP, TOPIC, 7)


def test_resp_span_two_references_agree_exhaustively():
    """两个**独立写法**的参考实现，在全部长度 ≤7 的短行上必须逐位一致。

    四判据版（`_ref_resp_mask`）和权威顺序扫描版（`_ref_from_spans`）是两条不同的
    思路；它们互相钉住，才不会出现"参考实现只是把被测实现换个写法抄一遍"的空对照。
    """
    for L in range(1, 8):
        for row in itertools.product(_ALPHA, repeat=L):
            row = list(row)
            a = _ref_resp_mask([row])[0]
            b = _ref_from_spans([row])[0]
            assert a == b, f"两个参考在 row={row} 上分歧：四判据={a} 权威={b}"
    # 顺带把**实现**也穷举一遍（L≤6，15625 行一次过批处理）——随机对照会漏掉
    # 概率极低的组合（相邻 cue 就是这么漏掉的，随机种子里 400 行没抽到）。
    rows = _pad_nl([list(r) for L in range(1, 7)
                    for r in itertools.product(_ALPHA, repeat=L)])
    y = torch.tensor(rows, dtype=torch.long)
    got = build_resp_span_mask(y, RESP_IDS, EOS_IDS)
    exp = torch.tensor(_ref_from_spans(rows), dtype=torch.bool)
    bad = (got != exp).any(dim=1).nonzero().flatten().tolist()
    assert not bad, (f"{len(bad)} 行不一致，第一处 row={rows[bad[0]]}\n"
                     f"  got={got[bad[0]].tolist()}\n  权威={exp[bad[0]].tolist()}")


def test_resp_span_consecutive_cue_matches_authority():
    """★ **相邻 `<resp>`** ⇒ 第二个 `<resp>` **算 loss**（`[F,T,T]`，不是 `[F,F,T]`）。

    权威定义 `DialogueStream.loss_token_spans` 是顺序扫描：第一个 `<resp>` 的扫描
    一路找到 `<eos>`，中间那个 `<resp>` 只是**正文**（它不是 `<eos>`），于是落进区间。
    第一版向量化实现用"含自身"的 `prev_resp` + `prev_resp < t` 来实现"cue 自身不算"，
    在相邻 cue 上就分叉了（它给 `[F,F,T]`）—— **这次是实现的错，不是参考的错**。

    真实流里 `append/commit` 不会产出相邻 cue，所以这是**口径一致性**测试而非行为测试；
    留着它是因为"两条路径换一个输入就分叉"正是最该被钉住的一类 bug。
    """
    for row in ([RESP, RESP, EOS],
                [RESP, RESP, 5, EOS],
                [RESP, RESP, RESP, 5, EOS],
                [RESP, RESP, 5, EOS, NL, RESP, 6]):
        got = build_resp_span_mask(torch.tensor([row]), RESP_IDS, EOS_IDS)[0].tolist()
        exp = _ref_from_spans([row])[0]
        assert got == exp, f"row={row}\n  got={got}\n  权威={exp}"
    # 把关键那一格写死，防止两个参考被同时改错
    assert build_resp_span_mask(torch.tensor([[RESP, RESP, EOS]]),
                                RESP_IDS, EOS_IDS)[0].tolist() == [False, True, True]


def test_resp_span_vs_eos_line_only_differ_on_resp():
    """★★ **负向对照**：两种模式的差异**恰好只**在 `<resp>` 上。

    这条同时证明：① 新规则确实修掉了"把 `<resp>` 算进 loss"；② 其余位置两者一致
    （否则就不是"修一处"，而是换了口径）。
    """
    y = torch.tensor([[RESP, 5, 6, EOS, NL, 9, 9, NL, RESP, 7, EOS]])
    old = build_assistant_mask(y, (EOS, CONT), SEP)
    new = build_resp_span_mask(y, RESP_IDS, EOS_IDS)
    assert old.tolist()[0] == [True, True, True, True, False, False, False, False, True, True, True]
    assert new.tolist()[0] == [False, True, True, True, False, False, False, False, False, True, True]
    # ⚠ `old != new` 形状是 (B, T)，`.nonzero()` 返回的是**行/列下标对**，
    #   直接 flatten 会得到 [0,0,0,8]（[[0,0],[0,8]] 展平）而不是 [0,8]。
    #   这里只关心第 0 行的列下标。
    diff = (old != new)[0].nonzero().flatten().tolist()
    assert diff == [0, 8], f'差异应只在下标 0/8（两个 <resp>），实际 {diff}'


def test_resp_span_user_line_excluded():
    """用户行（没有 `<resp>`）整行排除 —— 即使在两轮模型回复之间。"""
    _check_resp([[RESP, 1, EOS, NL, 10, 11, NL, RESP, 2, EOS]])


def test_resp_span_without_closing_eos_counts_nothing():
    """没有闭合的 `<eos>` ⇒ 整段不算 loss（防半截回复）。"""
    y = torch.tensor([[RESP, 5, 6, NL, RESP, 7]])
    assert not build_resp_span_mask(y, RESP_IDS, EOS_IDS).any()


def test_resp_span_eos_without_resp_counts_nothing():
    y = torch.tensor([[5, 6, EOS, NL]])
    assert not build_resp_span_mask(y, RESP_IDS, EOS_IDS).any()


def test_resp_span_topic_inside_the_span_is_counted():
    """`<topic>` 落在区间内 ⇒ 算 loss（模型要学会自己输出它）。"""
    y = torch.tensor([[RESP, TOPIC, 5, EOS, NL]])
    assert build_resp_span_mask(y, RESP_IDS, EOS_IDS).tolist() == [[False, True, True, True, False]]


def test_resp_span_empty_line_reply():
    """`<resp><eos>`（空回复）⇒ 只有 `<eos>` 算 loss。"""
    y = torch.tensor([[RESP, EOS, NL]])
    assert build_resp_span_mask(y, RESP_IDS, EOS_IDS).tolist() == [[False, True, False]]


def test_resp_span_degenerate_inputs():
    assert build_resp_span_mask(torch.zeros((3, 0), dtype=torch.long), RESP_IDS, EOS_IDS).shape == (3, 0)
    assert build_resp_span_mask(torch.zeros((0, 5), dtype=torch.long), RESP_IDS, EOS_IDS).shape == (0, 5)
    _check_resp([[RESP]])
    _check_resp([[EOS]])
    _check_resp([[NL]])


def test_resp_span_ignores_none_ids():
    y = torch.tensor([[RESP, 1, EOS]])
    assert torch.equal(build_resp_span_mask(y, (None, RESP), EOS_IDS),
                       build_resp_span_mask(y, RESP_IDS, EOS_IDS))


def test_resp_span_random_rows_match_reference():
    """400 组随机行逐位对齐参考实现（顺带覆盖 `<topic>`、连续 resp、空行）。"""
    rng = random.Random(20260914)
    rows = []
    for _ in range(400):
        row = []
        for _ in range(rng.randint(1, 24)):
            r = rng.random()
            if r < 0.25:
                row.append(NL)
            elif r < 0.35:
                row.append(EOS)
            elif r < 0.45:
                row.append(RESP)
            elif r < 0.50:
                row.append(TOPIC)
            else:
                row.append(rng.randint(1, 200))
        rows.append(row)
    _check_resp(rows)


def test_resp_span_batch_is_row_independent():
    """批内各行独立（抓 cummax/cummin 沿错维度这类只在 B>1 暴露的 bug）。"""
    rng = random.Random(11)
    rows = _pad_nl([[rng.choice([NL, EOS, RESP, TOPIC, rng.randint(1, 50)])
                     for _ in range(rng.randint(1, 20))] for _ in range(12)])
    y = torch.tensor(rows, dtype=torch.long)
    batch = build_resp_span_mask(y, RESP_IDS, EOS_IDS)
    for i, row in enumerate(rows):
        one = build_resp_span_mask(torch.tensor([row], dtype=torch.long), RESP_IDS, EOS_IDS)[0]
        assert torch.equal(batch[i], one), f"第 {i} 行批处理与单行不一致"
