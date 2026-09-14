"""`training/segmentation.py`（全项目统一分句器）的测试。

两组重点：

1. **与旧实现逐字等价**（`test_matches_legacy_behaviour_on_plain_text`）——
   `prepare.py --split-sentences` 依赖它，换实现不许改行为。测试里就地重写了
   旧算法当**参考实现**，对同一批样例双向对拍。
2. **机制符的原子性与归属**（`test_*special*`）—— 2026-09-14 实测事故：
   `split_line('我是小寻。<eos>')` 曾返回 `['我是小寻。', '<eos>']`，导致对话流
   按句滑窗时**把 `<eos>` 单独弹掉** = 真坏数据。配了**负向对照**证明判据非恒真。
"""
import importlib
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.segmentation import (  # noqa: E402
    SPECIAL_RE,
    ContextWindow,
    StreamSegmenter,
    is_special_only,
    iter_sentences,
    normalize_text,
    prepare_natural_text,
    process_file,
    run_cli,
    split_line,
    split_text,
)

# ---------------------------------------------------------------- 参考实现（旧算法原样重写）
_LEGACY_SENT = re.compile(r"[。！？…?!]|(?<!\d)\.(?!\d)")
_LEGACY_SUB = "，,；;、"


def legacy_split_line(line: str, max_len: int = 80):
    """旧 `data/chinese/split_sentences.py::split_line` 的**逐字**副本（作参考）。

    ★ 它是"没有机制符概念"的实现 —— 正因如此才能用来证明新规则的负向对照。
    """
    toks = re.split(f"({_LEGACY_SENT.pattern})", line)
    sentences, buf = [], ""
    for i, t in enumerate(toks):
        if t == "":
            continue
        if i % 2 == 1:
            buf += t
            sentences.append(buf)
            buf = ""
        else:
            buf += t
    if buf.strip():
        sentences.append(buf)
    if not sentences:
        return [line] if line.strip() else []
    out = []
    for s in sentences:
        if len(s) <= max_len:
            out.append(s)
            continue
        segs, cur = [], ""
        for ch in s:
            cur += ch
            if ch in _LEGACY_SUB:
                segs.append(cur)
                cur = ""
        if cur:
            segs.append(cur)
        out.extend(segs if segs else [s])
    return out


PLAIN_SAMPLES = [
    'a。b。',
    '你好，我是李华。',                       # 逗号不是句末
    '你好。我是小寻。你呢？',
    '我还没去拿呢',                           # 无标点
    '', '   ',                                # 空行 / 空白行
    '。你好',                                 # 以标点开头
    '你好。',                                 # 以标点结尾
    '3.14 是圆周率。',                        # 小数点保护
    'U.S. 是缩写。',
    '这是一句特别长长长长长长长长长长长长长长长长长长长长的句子，里面有逗号，还有分号；快切吧。',
    '！？…?! 连着来。',
    '       前面有空格。后面也有。      ',
]


@pytest.mark.parametrize('line', PLAIN_SAMPLES)
def test_matches_legacy_behaviour_on_plain_text(line):
    """★ 不含机制符时，新实现与旧实现必须**逐字相同**（换实现不许改行为）。"""
    assert split_line(line) == legacy_split_line(line), f'与旧实现不一致: {line!r}'


@pytest.mark.parametrize('max_len', [1, 5, 12, 80])
def test_matches_legacy_behaviour_across_max_len(max_len):
    """超长句二次切的阈值也要对得上（`max_len` 是公开参数，会被调）。"""
    for line in PLAIN_SAMPLES:
        assert split_line(line, max_len) == legacy_split_line(line, max_len), (line, max_len)


# ---------------------------------------------------------------- 机制符：规则一（不切进去）


def test_no_split_inside_a_special_token():
    """机制符**内部**的句末标点（半角 `.`）不算边界。"""
    assert split_line('<x.y>没事。') == ['<x.y>没事。']
    # 负向对照：`。` 在 token **外面**时本来就是真边界，必须照切 ——
    # 免得把"规则一"写成了"见到 `<` 就整行不切"的空规则。
    assert split_line('<think>第一步。</think>第二步。') == ['<think>第一步。', '</think>第二步。']


def test_special_regex_covers_the_real_tokens():
    for tok in ['<eos>', '<unk>', '<cont>', '<pad>', '<bos>', '<sep>',
                '<think>', '<answer>', '<resp>', '<你该说话了>']:
        assert SPECIAL_RE.fullmatch(tok), tok


# ---------------------------------------------------------------- 机制符：规则二（不单独成句）


@pytest.mark.parametrize('tail', ['<eos>', '<resp>', '<cont>'])
def test_special_token_stays_attached_to_its_sentence(tail):
    """★ `<eos>` 这类**后缀标记**必须粘在它终止的那一句上。"""
    assert split_line(f'我是小寻。{tail}') == [f'我是小寻。{tail}']


def test_the_bug_this_fixes_is_real():
    """★★ **负向对照**：证明"规则二"在真干活，而不是恒真的空测试。

    旧实现（无合并规则）确实会切出两段 —— 这正是"`<eos>` 被单独弹掉"的事故来源。
    """
    line = '我是小寻。<eos>'
    assert legacy_split_line(line) == ['我是小寻。', '<eos>'], '旧实现本该切出两段'
    assert split_line(line) == ['我是小寻。<eos>'], '新实现必须粘住'


def test_resp_prefix_is_part_of_the_first_sentence():
    """`<resp>` 是**前缀**标记，不该单独成句，也不算"只有机制符"。"""
    assert split_line('<resp>你好。我是小寻。<eos>') == ['<resp>你好。', '我是小寻。<eos>']


def test_special_only_detection():
    assert is_special_only('<eos>') is True
    assert is_special_only('<eos><resp>') is True
    assert is_special_only('  <eos>  ') is True
    assert is_special_only('<eos>好') is False
    assert is_special_only('好') is False
    assert is_special_only('') is True


def test_leading_special_only_piece_is_kept():
    """整行只有机制符时不能把首段丢掉（没有"前一句"可并）。"""
    assert split_line('<eos>') == ['<eos>']


# ---------------------------------------------------------------- split_text


def test_split_text_preserves_blank_lines_and_turn_structure():
    text = '用户：你好。我是李华。\n模型：你好呀。<eos>\n\n用户：在吗？\n'
    out = split_text(text)
    assert out.split('\n') == [
        '用户：你好。', '我是李华。',     # 行内分句
        '模型：你好呀。<eos>',           # 机制符粘住
        '',                              # 空行保留
        '用户：在吗？',
        '',
    ]


def test_failed_output_is_removed_but_special_suffix_kept():
    """超长句二次切时，末尾的 `<eos>` 必须跟着最后一段走。"""
    long = '啊' * 100 + '，' + '嗯' * 100 + '。<eos>'
    out = split_line(long, max_len=10)
    assert out[-1].endswith('。<eos>'), out
    assert ''.join(out) == long


# ---------------------------------------------------------------- 壳转发


def test_shim_reexports_the_canonical_module():
    """`data/chinese/split_sentences.py` 必须**原样转发** canonical 实现。

    `prepare.py:61` 用的是裸模块名 `from split_sentences import split_text`，
    所以这个壳必须继续存在，且转出的得是**同一个函数对象**（不是又抄一份实现）。
    """
    dc = str(ROOT / 'data' / 'chinese')
    if dc not in sys.path:
        sys.path.insert(0, dc)
    shim = importlib.import_module('split_sentences')
    from training import segmentation
    assert shim.split_text is segmentation.split_text
    assert shim.split_line is segmentation.split_line
    assert shim.SENT_PAT is segmentation.SENT_PAT


def test_prepare_py_import_line_still_works():
    """把 `prepare.py` 真正加载一次 —— 证明它的裸模块名导入没被打断。"""
    dc = ROOT / 'data' / 'chinese'
    if not (dc / 'prepare.py').exists():
        pytest.skip('prepare.py 不存在')
    if str(dc) not in sys.path:
        sys.path.insert(0, str(dc))
    spec = importlib.util.spec_from_file_location('_prepare_import_probe', dc / 'prepare.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.split_text is split_text, 'prepare.py 拿到的必须就是 canonical 实现'


# ============================================================================
# 流式（增量）分句
# ============================================================================


def test_streaming_splits_user_input_into_sentences():
    """用户输入自动分句：多句输入 → 逐句吐出。"""
    seg = StreamSegmenter()
    out = seg.feed('你好。今天天气不错。要出去走走吗？')
    out += seg.flush()
    assert out == ['你好。', '今天天气不错。', '要出去走走吗？'], out


def test_streaming_emits_before_input_ends():
    """真流式：边界一确定就吐，不必等输入结束（`pending` 是压着的尾巴）。"""
    seg = StreamSegmenter()
    assert seg.feed('第一句。第二句。') == ['第一句。']
    assert seg.pending == '第二句。'
    assert seg.flush() == ['第二句。']
    assert seg.pending == ''


def test_streaming_newline_is_a_boundary():
    """换行也是分句标识符（同时充当轮次/段落分隔）。

    ⚠ 末尾那个空串来自 `flush()`（输入以换行收尾 = 最后有一个空行），不在 `feed` 里。
    """
    seg = StreamSegmenter()
    assert seg.feed('一\n二\n\n三\n') == ['一', '二', '', '三']
    assert seg.flush() == ['']


def test_streaming_char_by_char_attaches_eos():
    """★ 最极端的流式：**逐字**喂，`<eos>` 跨 5 次 feed 到达，也必须粘在上一句。"""
    seg = StreamSegmenter()
    out = []
    for ch in '我是小寻。<eos>\n第二句。\n':
        out += seg.feed(ch)
    out += seg.flush()
    assert out == ['我是小寻。<eos>', '第二句。', ''], out


def test_streaming_hold_is_necessary():
    """★★ **负向对照**：关掉 `hold_last` 后，`<eos>` **必须**被切成独立一段。

    逐字喂时，`hold_last=False` 会在句号处立刻吐出 `'我是小寻。'`；随后到达的 `<eos>`
    只能单独成句 —— 这证明那条"压一段"不是多余的小心，而是流式下不复发老 bug 的**唯一**原因。
    """
    seg = StreamSegmenter(hold_last=False)
    out = []
    for ch in '我是小寻。<eos>\n':
        out += seg.feed(ch)
    out += seg.flush()
    assert '我是小寻。' in out and '<eos>' in out, f'负向对照应复现旧 bug，实际 {out}'
    assert '我是小寻。<eos>' not in out


def test_streaming_without_hold_does_not_emit_per_character():
    """★ 回归（2026-09-14 修的真 bug）：`hold_last=False` **不等于**"见到什么吐什么"。

    末尾不是句边界的片段是"还没写完的文本"，无论 `hold_last` 如何都必须留在缓冲区。
    早先漏了这条 ⇒ 逐字喂时 `'我'` 都被当成一句吐出来。
    """
    seg = StreamSegmenter(hold_last=False)
    fed = []
    for ch in '你好世界':                     # 还没有句号
        fed += seg.feed(ch)
    assert fed == [], f'句号之前不该吐出任何东西，实际 {fed}'
    assert seg.pending == '你好世界'
    fed += seg.feed('。')                     # 句号一到，`hold_last=False` 立刻吐
    assert fed == ['你好世界。']


def test_open_token_tail_is_held_not_split():
    """未闭合的 `<` 尾巴：不能把前面的句子先发出去（否则 `<eos>` 只能单独成句）。"""
    seg = StreamSegmenter()
    assert seg.feed('我是小寻。<') == []
    assert seg.pending == '我是小寻。<'
    assert seg.feed('eos>') == []
    assert seg.feed('\n') == ['我是小寻。<eos>']


def test_open_token_tail_that_never_closes_is_released():
    """`<` 后面跟了空格 ⇒ 它不是机制符（只是普通文本）；前面那句必须能正常吐出来。

    ⚠ 空格本身是内容，会被保留（分句器不做有损改写）—— 所以按带空格的实况断言。
    """
    seg = StreamSegmenter()
    out = seg.feed('我是小寻。< 这是小于号')
    out += seg.feed('。')
    out += seg.flush()
    assert '我是小寻。' in out, f'`<` 后面是空格 ⇒ 不是机制符，前面的句子该正常吐出：{out}'
    # 注：`。` 与 `<` 之间那个**纯空白**残余会被丢弃 —— 这是 `split_line` 的既有行为
    #（纯空白收尾不构成句子），与旧实现逐字一致，不是本次引入的。
    assert out[-1] == '< 这是小于号。', out


def test_iter_sentences_over_a_chunk_stream():
    """跨块的句子要能拼起来（`'今天'` + `'不错。'` → `'今天不错。'`）。"""
    chunks = ['你好。', '今天', '不错。', '\n走了。']
    assert list(iter_sentences(chunks)) == ['你好。', '今天不错。', '走了。']


def test_split_text_equals_the_streamer_without_hold():
    """批量入口必须与"流式但压段关掉"等价 —— 一套引擎，两个用法。"""
    for text in ['你好。世界。', '一\n二\n\n三\n', '我是小寻。<eos>\n下一句。\n', '']:
        seg = StreamSegmenter(hold_last=False)
        assert split_text(text) == '\n'.join(seg.feed(text) + seg.flush()), text


# ============================================================================
# 自然文本 → 训练可用文本
# ============================================================================


def test_normalize_text_known_answers():
    assert normalize_text('a\r\nb') == 'a\nb'
    assert normalize_text('a\rb') == 'a\nb'
    assert normalize_text('a\x00b\x07c') == 'abc'          # 去控制符
    assert normalize_text('行尾空格   \n下一行') == '行尾空格\n下一行'
    assert normalize_text('a\n\n\n\nb') == 'a\n\nb'        # 3+ 换行 → 段落分隔
    assert normalize_text('\n\na\n\n') == 'a'              # 去首尾空行
    # ★ 负向对照：句内空格必须**留着**（代码/中英混排靠它）
    assert normalize_text('for i in range(3):') == 'for i in range(3):'


def test_prepare_natural_text_keeps_blocks_and_splits_sentences():
    """核心契约：**一句一行，空行仍是块分隔符**（丢了空行整本书会变成一个样本）。"""
    raw = '第一段。第二句。\n\n第二段。这里还有一句。\n'
    out = prepare_natural_text(raw)
    assert out == '第一段。\n第二句。\n\n第二段。\n这里还有一句。'
    assert [b for b in out.split('\n\n')] == ['第一段。\n第二句。', '第二段。\n这里还有一句。']


def test_prepare_natural_text_is_idempotent():
    """跑两遍必须一样（否则"数据集过一遍"会因重复处理而漂移）。"""
    raw = '一段。两句。\n\n下一段。\n'
    once = prepare_natural_text(raw)
    assert prepare_natural_text(once) == once


def test_prepare_natural_text_min_chars_drops_noise_but_keeps_blank_lines():
    raw = '好的。\n嗯。\n\n这一段足够长，应当保留下来。\n'
    out = prepare_natural_text(raw, min_chars=3)     # '好的。'=3 字保留，'嗯。'=2 字丢弃
    lines = out.split('\n')
    assert '嗯。' not in lines, '过短行应被丢弃'
    assert '好的。' in lines
    assert '' in lines, '空行（块边界）永远保留'


def test_prepare_natural_text_handles_book_like_text():
    """书籍式长文本：多段、长句、无空行的连续文本都要能处理。"""
    para = '他走进屋子。' * 60          # 一句都没有分隔也只是一行
    raw = f'{para}\n\n第二段。很短。\n\n\n第三段。\n'
    out = prepare_natural_text(raw, max_len=20)
    blocks = [b for b in out.split('\n\n') if b.strip()]
    assert len(blocks) == 3, blocks
    long_lines = [ln for ln in out.split('\n') if ln.strip()]
    assert all(len(ln) <= 40 for ln in long_lines), '超长句必须被二次切'


def test_process_file_and_run_cli(tmp_path, capsys):
    src = tmp_path / 'book.txt'
    src.write_text('第一句。第二句。\n\n另一段。\n', encoding='utf-8')
    st = process_file(str(src))
    assert st['dst'].endswith('book_seg.txt')
    assert st['blocks'] == 2
    assert (tmp_path / 'book_seg.txt').read_text(encoding='utf-8').startswith('第一句。\n第二句。')

    rc = run_cli(['--file', str(src), '--out', str(tmp_path / 'o.txt')])
    assert rc == 0
    assert '→' in capsys.readouterr().out
    assert (tmp_path / 'o.txt').exists()


# ============================================================================
# 上下文管理（按句滑窗）
# ============================================================================


def test_context_window_pops_whole_units_until_it_fits():
    """用户定的规则：超预算就**从头部弹出整单元**，直到放得下。"""
    win = ContextWindow(measure=lambda us: sum(len(u) for u in us), budget=6)
    win.extend(['aaa', 'bbb', 'ccc', 'ddd'])          # 12 > 6
    # 弹 aaa → 9 > 6；再弹 bbb → 6 ≤ 6 停
    assert win.units == ['ccc', 'ddd'], win.units
    assert win.dropped == 2
    assert win.size() <= win.budget
    assert not win.overflow


def test_context_window_keeps_at_least_one_unit_and_flags_overflow():
    """单个单元本身就超预算 ⇒ 不能弹空、也不能死循环，而是置 `overflow`（软上限）。"""
    win = ContextWindow(measure=lambda us: sum(len(u) for u in us), budget=3)
    win.push('这是一个远超预算的单元')
    assert len(win) == 1 and win.overflow is True
    assert win.dropped == 0


def test_context_window_keep_min_rejects_zero():
    with pytest.raises(ValueError):
        ContextWindow(measure=len, budget=10, keep_min=0)


def test_context_window_unlimited_budget_never_measures():
    """★ 无穷预算**完全不测量** —— 否则解析整份语料是 O(n²)（实测 60s+ 超时）。"""
    calls = []

    def measure(units):
        calls.append(len(units))
        return sum(len(u) for u in units)

    win = ContextWindow(measure=measure, budget=ContextWindow.NO_LIMIT)
    win.extend(['a', 'b', 'c'] * 50)
    win.push('再来一个')
    assert calls == [], f'无预算限制时不该测量，实际调了 {len(calls)} 次'
    assert len(win) == 151 and win.overflow is False


def test_context_window_finite_budget_does_measure():
    """负向对照：有限预算**必须**照旧测量并裁剪（别把上面的优化做成空开关）。"""
    calls = []

    def measure(units):
        calls.append(len(units))
        return sum(len(u) for u in units)

    win = ContextWindow(measure=measure, budget=20)
    win.extend(['aaaa', 'bbbb', 'cccc', 'dddd', 'eeee', 'ffff'])
    assert calls, '有限预算必须测量'
    assert win.dropped > 0
    assert win.size() <= win.budget


def test_context_window_units_is_the_live_list():
    """`units` 是**活的** —— 调用方（`DialogueStream.commit`）要就地改最后一个单元。"""
    win = ContextWindow(measure=len, budget=1000)
    win.extend(['a', 'b'])
    win.units[-1] = win.units[-1] + '<eos>'
    assert win.units == ['a', 'b<eos>']


def test_context_window_reset_clears_state():
    win = ContextWindow(measure=lambda us: sum(len(u) for u in us), budget=5)
    win.extend(['aaa', 'bbb', 'ccc'])
    assert win.dropped > 0
    win.reset()
    assert len(win) == 0 and win.dropped == 0 and win.overflow is False


def test_dialogue_stream_delegates_context_management():
    """★ `DialogueStream` 的上下文管理**必须**走 `ContextWindow`（别再自己实现一套）。"""
    from training.dialogue_stream import DialogueStream
    ds = DialogueStream(lambda s: list(s), 40)
    assert isinstance(ds._win, ContextWindow), 'DialogueStream 必须组合 ContextWindow'
    for i in range(20):
        ds.append('A', f'第{i}句话。')
    assert ds.dropped == ds._win.dropped
    assert ds.overflow == ds._win.overflow
    assert ds.context_len() <= ds.window or ds.overflow


def test_run_cli_rejects_missing_file(capsys):
    assert run_cli(['--file', '/nonexistent/zzz.txt']) == 1
    assert '❌' in capsys.readouterr().out
