"""`data/chinese/clean_corpus.py` 的测试 —— **每条规则都要有对照/已知答案**。

为什么要这么严（本仓库最贵的教训在 `AGENTS.md §5.4`）：清洗工具最坏的失败模式是
"规则看起来生效、其实一个 block 也没动"——阈值调了没反应，人会以为"语料很干净"。
所以这里每个规则都配一条 **松/严对照**：松阈值必须放行、严阈值必须命中，
两条结果必须**不同**。只断言"跑完不报错"的测试等于没测。

覆盖的失败模式：
* 去重做成**按行**去重（会把不同 block 里相同的 `用户：` 行删掉）
* 去重顺序不确定（文件/块顺序变了 → 输出不可复现）
* rep3 / K-gram 规则**误伤非对话 block**（c4_zh/wikipedia 占大头）
* K-gram 规则只统计不判定（阈值变了结果不变 = 空测试）
* `--apply` 之外偷偷写数据文件，或反过来改了 `--src`
"""
import os
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
DC = ROOT / 'data' / 'chinese'


@pytest.fixture(scope='module')
def clean(load_module_from_path):
    if not (DC / 'clean_corpus.py').exists():
        pytest.skip('data/chinese/clean_corpus.py 不存在')
    if str(DC) not in sys.path:
        sys.path.insert(0, str(DC))
    return load_module_from_path(DC / 'clean_corpus.py', 'clean_corpus_under_test')


# ==========================================================================
# 夹具
# ==========================================================================
def write_corpus(root: pathlib.Path, mapping: dict) -> pathlib.Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, text in mapping.items():
        (root / name).write_text(text, encoding='utf-8')
    return root


def blocks_of(path: pathlib.Path) -> list:
    return [b for b in path.read_text(encoding='utf-8').split('\n\n') if b.strip()]


def run_once(clean, tmp_path, mapping, extra=(), apply=True, name='run'):
    """在临时 src/dst 上跑一次清洗，返回 (src, dst, 结构化结果)。"""
    src = write_corpus(tmp_path / f'{name}_src', mapping)
    dst = tmp_path / f'{name}_dst'
    argv = ['--src', str(src), '--dst', str(dst), '--ngram-sketch-bits', '10', *extra]
    if apply:
        argv.append('--apply')
    result = clean.run(clean.parse_args(argv), log=lambda *a, **k: None)
    return src, dst, result


# ==========================================================================
# 1) 纯函数：已知答案
# ==========================================================================
def test_rep_n_known_values(clean):
    """已知答案：口径 = 「落在重复类型里的 n-gram 数 / 总 n-gram 数」，且先去空白。

    ⚠ 这不是 CTRL 的 `1 - 不同/总数`。两者不等价（前者恒 ≥ 后者），换成后者上面
    这些期望值全部会变（'aaaa' 会从 1.0 变成 0.5）—— 那正是 2026-09-13 抓到的口径分叉。
    """
    assert clean.rep_n('', 3) == 0.0
    assert clean.rep_n('abcd', 3) == 0.0                  # 两个 3-gram，各出现一次
    assert clean.rep_n('aaaa', 3) == 0.0                  # 太短不判（与 eval 侧同一条守卫）
    assert clean.rep_n('a' * 40, 3) == pytest.approx(1.0)
    # 'ababab'：4 个 3-gram = aba,bab,aba,bab，全部属于重复类型
    assert clean.rep_n('ababab', 3) == pytest.approx(1.0)
    assert clean.rep_n('哈' * 40, 3) > 0.9
    # 中间态：一半左右落在重复类型里
    assert 0.3 < clean.rep_n('abcde' * 2, 3) < 0.99


def test_rep_n_matches_eval_side_implementation(clean):
    """★ 口径锁定：`clean_corpus.rep_n` 必须与评估侧 `ngram_repetition` 逐位一致。

    2026-09-13 实测踩过：清洗器原实现是 CTRL 口径（1 - 不同/总数），评估侧是
    「重复类型里的 gram 数/总数」，同一个 0.3 在两处含义不同（multi_turn 回复
    均值 0.202 vs 0.104）。阈值是用评估侧对照标定的，所以清洗端必须对齐，
    否则阈值搬不过来、且两边结论会互相打架。
    ⚠ 这条测试**故意直接调用评估侧函数**做对拍：只断言几个常量值挡不住公式被人换掉。
    """
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    'inference', 'scripts'))
    from eval_dialogue import ngram_repetition  # noqa: PLC0415

    samples = ['abcd', 'aaaa', 'ababab', 'abcde' * 2, '春眠不觉晓处处闻啼鸟春眠不觉晓',
               'a b a b a b', 'abab\nab', '这是一段普通的中文回答，没有明显重复。']
    for s in samples:
        assert clean.rep_n(s, 3) == pytest.approx(ngram_repetition(s, 3)), f'口径分叉：{s!r}'
    # 反向对照：证明这条对拍不是"两边都恒等"的平凡通过
    assert clean.rep_n('aaaa', 3) != pytest.approx(1 - 1 / 2)  # CTRL 口径会给 0.5


def test_window_hashes_are_position_independent(clean):
    """★ 同一短语出现在不同位置必须拿到同一个哈希。

    这是 K-gram 短语规则能工作的**前提**：如果哈希带绝对位置，同一句套话
    在每条回复里都会得到不同哈希，频次统计就永远是 1 —— 规则静默失效。
    """
    codes = clean.char_codes('前缀甲乙丙丁今天天气真不错啊后缀戊己庚辛')
    hashes = clean.window_hashes(codes, 4)
    # '今天天气真不错啊' 在 codes 里的起始下标 = 6
    phrase = clean.char_codes('今天天气真不错啊')
    phrase_hashes = clean.window_hashes(phrase, 4)
    assert list(hashes[6:6 + phrase_hashes.size]) == list(phrase_hashes)
    # 换到另一个位置（前面加若干字符），同一短语的哈希不变
    moved = clean.window_hashes(clean.char_codes('零一二三四五六七八九' + '今天天气真不错啊'), 4)
    assert list(moved[10:10 + phrase_hashes.size]) == list(phrase_hashes)


def test_split_replies_includes_multiline_continuation(clean):
    block = '用户：问\n模型：第一行\n第二行续\n用户：再问\n模型：答复'
    assert clean.split_replies(block) == ['第一行\n第二行续', '答复']


def test_has_model_reply_only_for_tagged_lines(clean):
    assert clean.has_model_reply('用户：a\n模型：b')
    assert not clean.has_model_reply('他提到模型：这个词但不是标签')
    assert not clean.has_model_reply('红楼梦第一回')


def test_normalized_hash_ignores_whitespace(clean):
    assert clean.normalized_hash('用户：你好\n模型：世界') == \
        clean.normalized_hash('用户： 你好\n\n模型：世 界')


# ==========================================================================
# 2) --dedup-blocks：全局 + 块级（不是行级）+ 确定性
# ==========================================================================
def test_dedup_keeps_first_occurrence_globally(clean, tmp_path):
    b1 = '用户：你好\n模型：世界'
    b3 = '用户：你好\n模型：再见'      # 与 b1 共享一行，但不是同一个 block
    src, dst, res = run_once(clean, tmp_path, {
        'a.txt': f'{b1}\n\n{b1}\n\n{b3}',
        'b.txt': f'{b1}\n\n用户：另一个\n模型：另一个回复',
    }, extra=['--dedup-blocks'])

    assert blocks_of(dst / 'a.txt') == [b1, b3], '块内重复应只留首次出现'
    assert blocks_of(dst / 'b.txt') == ['用户：另一个\n模型：另一个回复'], '跨文件重复也要去'
    assert res['rule_hits'][clean.RULE_DEDUP] == 2


def test_dedup_is_not_line_level_dedup(clean, tmp_path):
    """★ 反面对照：共享同一行 `用户：你好` 的两个**不同** block 都必须留下。

    如果实现退化成"按行去重"，第二个 block 的 `用户：你好` 会被删掉。
    """
    b1 = '用户：你好\n模型：世界'
    b2 = '用户：你好\n模型：再见'
    _, dst, _ = run_once(clean, tmp_path, {'a.txt': f'{b1}\n\n{b2}'},
                         extra=['--dedup-blocks'])
    out = blocks_of(dst / 'a.txt')
    assert out == [b1, b2]
    assert out[1].count('用户：你好') == 1, '第二个 block 的对话行必须原样保留'


def test_dedup_is_deterministic_across_runs(clean, tmp_path):
    mapping = {'b.txt': 'X\n\nX\n\nY', 'a.txt': 'X\n\nZ'}   # 故意让 b 在 a 前被创建
    _, dst1, _ = run_once(clean, tmp_path, mapping, extra=['--dedup-blocks'], name='r1')
    _, dst2, _ = run_once(clean, tmp_path, mapping, extra=['--dedup-blocks'], name='r2')
    for fn in ('a.txt', 'b.txt'):
        assert (dst1 / fn).read_bytes() == (dst2 / fn).read_bytes()
    # 文件名排序 ⇒ a.txt 先出现，X 留在 a.txt，b.txt 里的 X 全丢
    assert blocks_of(dst1 / 'a.txt') == ['X', 'Z']
    assert blocks_of(dst1 / 'b.txt') == ['Y']


def test_dedup_off_keeps_duplicates(clean, tmp_path):
    """对照：不传 --dedup-blocks 时重复块必须原样保留（规则只能显式开）。"""
    _, dst, _ = run_once(clean, tmp_path, {'a.txt': 'A\n\nA'})
    assert blocks_of(dst / 'a.txt') == ['A', 'A']


# ==========================================================================
# 3) --max-reply-rep3：只丢重度重复的回复
# ==========================================================================
REP_BAD = '哈' * 40
REP_GOOD = '一只猫走进了一家书店，问店员有没有关于鱼的书'


def test_rep3_drops_only_the_repetitive_reply(clean, tmp_path):
    good = f'用户：讲个笑话\n模型：{REP_GOOD}'
    bad = f'用户：讲个笑话\n模型：{REP_BAD}'
    _, dst, res = run_once(clean, tmp_path, {'a.txt': f'{good}\n\n{bad}'},
                           extra=['--max-reply-rep3', '0.3'])
    assert blocks_of(dst / 'a.txt') == [good], '只有重度重复的那条该被丢'
    assert res['rule_hits'][clean.RULE_REP3] == 1


def test_rep3_contrast_rule_off_keeps_everything(clean, tmp_path):
    """★ 对照：同样语料不传规则时两条都留下 —— 证明上一条是规则干的。"""
    mapping = {'a.txt': f'用户：讲个笑话\n模型：{REP_GOOD}\n\n用户：讲个笑话\n模型：{REP_BAD}'}
    _, dst, res = run_once(clean, tmp_path, mapping)
    assert len(blocks_of(dst / 'a.txt')) == 2
    assert res['rule_hits'].get(clean.RULE_REP3, 0) == 0


def test_rep3_any_reply_triggers_block_drop(clean, tmp_path):
    """契约：块内**任一**回复重度重复就丢整块（不是"平均"或"全部"）。"""
    block = f'用户：a\n模型：{REP_GOOD}\n用户：b\n模型：{REP_BAD}'
    _, dst, _ = run_once(clean, tmp_path, {'a.txt': block}, extra=['--max-reply-rep3', '0.3'])
    assert blocks_of(dst / 'a.txt') == []


REP_MID = 'abcde' * 2    # rep3 = 0.75，落在 0.3 与 0.99 之间（口径见 test_rep_n_known_values）


def test_rep3_contrast_threshold_actually_bites(clean, tmp_path):
    """★ 松/严对照：同一 block，阈值 0.99 放行、0.3 丢弃。

    用一条 rep3≈0.78 的**中间态**回复：它必须随阈值改变命运，
    否则说明阈值根本没接进判定（空测试）。
    """
    block = f'用户：a\n模型：{REP_MID}'
    assert 0.3 < clean.rep_n(REP_MID, 3) < 0.99
    _, dst_loose, _ = run_once(clean, tmp_path, {'a.txt': block},
                               extra=['--max-reply-rep3', '0.99'], name='loose')
    _, dst_strict, _ = run_once(clean, tmp_path, {'a.txt': block},
                                extra=['--max-reply-rep3', '0.3'], name='strict')
    assert blocks_of(dst_loose / 'a.txt') == [block], '松阈值必须放行'
    assert blocks_of(dst_strict / 'a.txt') == [], '严阈值必须命中'


# ==========================================================================
# 4) --min-reply-chars
# ==========================================================================
def test_min_reply_chars_drops_only_when_all_replies_short(clean, tmp_path):
    all_short = '用户：a\n模型：嗯\n用户：b\n模型：好'
    mixed = '用户：a\n模型：嗯\n用户：b\n模型：这是一条足够长的回复，用来验证"只要有一条够长就保留"'
    _, dst, res = run_once(clean, tmp_path, {'a.txt': f'{all_short}\n\n{mixed}'},
                           extra=['--min-reply-chars', '10'])
    assert blocks_of(dst / 'a.txt') == [mixed]
    assert res['rule_hits'][clean.RULE_MIN] == 1


def test_min_reply_chars_never_drops_nondialogue(clean, tmp_path):
    """非对话 block 没有回复 ⇒ "所有回复都短"是空洞真，**不能**据此丢它。"""
    plain = '红楼梦第一回甄士隐梦幻识通灵'
    _, dst, _ = run_once(clean, tmp_path, {'c4_zh.txt': plain},
                         extra=['--min-reply-chars', '1000'])
    assert blocks_of(dst / 'c4_zh.txt') == [plain]


# ==========================================================================
# 5) K-gram 短语规则：已知答案 + 松/严对照
# ==========================================================================
TRAP = '今天天气真不错啊'          # 8 字，被 3 条不同回复复用
K = 4


def _phrase_corpus():
    return {
        'a.txt': '\n\n'.join([
            f'用户：今天怎么样\n模型：{TRAP}，我们出去走走吧',
            '用户：聊点别的\n模型：其实我最近在看一本关于海洋生物的书，挺有意思',
            f'用户：还有呢\n模型：{TRAP}，顺便买点水果回来',
        ]),
        'b.txt': '\n\n'.join([
            f'用户：继续\n模型：{TRAP}，下午一起去公园散步',
            '用户：好\n模型：程序里有个递归函数写错了，边界条件没处理好',
        ]),
    }


NORMAL_BLOCKS = {
    '其实我最近在看一本关于海洋生物的书，挺有意思',
    '程序里有个递归函数写错了，边界条件没处理好',
}


def test_ngram_rule_drops_trap_blocks_keeps_normal(clean, tmp_path):
    _, dst, res = run_once(clean, tmp_path, _phrase_corpus(), extra=[
        '--max-ngram-freq', '2', '--ngram-size', str(K), '--max-ngram-cover', '0.2'])
    assert res['rule_hits'][clean.RULE_NGRAM] == 3, '三条嵌套话的回复都该被丢'
    kept = blocks_of(dst / 'a.txt') + blocks_of(dst / 'b.txt')
    assert len(kept) == 2
    for body in NORMAL_BLOCKS:
        assert any(body in b for b in kept), f'正常回复被误伤：{body}'
    assert all(TRAP not in b for b in kept)


def test_ngram_rule_contrast_loose_thresholds_keep_everything(clean, tmp_path):
    """★ 对照 A：频次阈值调到极松（100）⇒ heavy 集合为空 ⇒ 一条都不丢。"""
    _, dst, res = run_once(clean, tmp_path, _phrase_corpus(), extra=[
        '--max-ngram-freq', '100', '--ngram-size', str(K), '--max-ngram-cover', '0.2'],
        name='loose_freq')
    assert res['coverage']['heavy'] == 0
    assert res['rule_hits'].get(clean.RULE_NGRAM, 0) == 0
    assert len(blocks_of(dst / 'a.txt')) == 3 and len(blocks_of(dst / 'b.txt')) == 2


def test_ngram_rule_contrast_loose_cover_keeps_everything(clean, tmp_path):
    """★ 对照 B：覆盖率阈值调到 0.99 ⇒ 套话块覆盖率只有 ~0.5 ⇒ 一条都不丢。"""
    _, dst, res = run_once(clean, tmp_path, _phrase_corpus(), extra=[
        '--max-ngram-freq', '2', '--ngram-size', str(K), '--max-ngram-cover', '0.99'],
        name='loose_cover')
    assert res['coverage']['heavy'] > 0, 'heavy 集合非空但覆盖率阈值放行 —— 才是真对照'
    assert res['rule_hits'].get(clean.RULE_NGRAM, 0) == 0
    assert len(blocks_of(dst / 'a.txt')) == 3


def test_ngram_rule_coverage_drops_and_rises_back(clean, tmp_path):
    """报告里的覆盖率必须真的动：清洗前 > 0，清洗后 = 0。"""
    _, _, res = run_once(clean, tmp_path, _phrase_corpus(), extra=[
        '--max-ngram-freq', '2', '--ngram-size', str(K), '--max-ngram-cover', '0.2'],
        name='cov')
    assert res['coverage']['enabled'] is True
    assert res['coverage']['before'] > 0.2
    assert res['coverage']['after'] == 0.0


def test_ngram_rule_requires_both_thresholds(clean, tmp_path):
    """只给一半参数必须报错，不能静默变成空操作。"""
    src = write_corpus(tmp_path / 's', {'a.txt': '用户：a\n模型：b'})
    for extra in (['--max-ngram-freq', '2'], ['--max-ngram-cover', '0.2']):
        argv = ['--src', str(src), '--dst', str(tmp_path / 'd'),
                '--ngram-sketch-bits', '10', *extra]
        with pytest.raises(SystemExit):
            clean.run(clean.parse_args(argv), log=lambda *a, **k: None)


def test_ngram_rule_survives_pathological_sketch_collisions(clean, tmp_path):
    """★ 草图只有 4 位（16 个桶）时碰撞爆炸，但**判定必须仍然正确**。

    这证明最终用的是第 2 遍的**精确计数**，而不是草图的估计值：
    若拿估计值判定，碰撞高估会让正常文本也被误杀。
    """
    _, dst, res = run_once(clean, tmp_path, _phrase_corpus(), extra=[
        '--max-ngram-freq', '2', '--ngram-size', str(K), '--max-ngram-cover', '0.2',
        '--ngram-sketch-bits', '4'], name='collide')
    assert res['rule_hits'][clean.RULE_NGRAM] == 3
    kept = blocks_of(dst / 'a.txt') + blocks_of(dst / 'b.txt')
    assert len(kept) == 2, f'碰撞把正常块也误杀了：{[k[:12] for k in kept]}'


def test_ngram_candidate_cap_is_a_loud_failure(clean, tmp_path):
    """候选上限撞了必须**报错**，不能静默降级（防 OOM 的最后一环）。"""
    src = write_corpus(tmp_path / 's', _phrase_corpus())
    argv = ['--src', str(src), '--dst', str(tmp_path / 'd'), '--ngram-sketch-bits', '4',
            '--max-ngram-freq', '1', '--ngram-size', str(K), '--max-ngram-cover', '0.2',
            '--ngram-max-candidates', '1']
    with pytest.raises(SystemExit):
        clean.run(clean.parse_args(argv), log=lambda *a, **k: None)


# ==========================================================================
# 6) 非对话 block：默认原样保留（对照开关才动它）
# ==========================================================================
PLAIN_REPEAT = '哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈'


def test_nondialogue_blocks_survive_all_dialogue_rules(clean, tmp_path):
    """★ c4_zh/wikipedia/四大名著占大头 —— 默认规则一条都不许碰它们。"""
    mapping = {
        'c4_zh.txt': f'红楼梦第一回甄士隐梦幻识通灵\n\n{PLAIN_REPEAT}',
        'wikipedia_cn.txt': '量子力学是物理学的一个分支',
        'a.txt': '用户：讲个笑话\n模型：嗯',
    }
    _, dst, res = run_once(clean, tmp_path, mapping, extra=[
        '--max-reply-rep3', '0.3', '--min-reply-chars', '1000',
        '--max-ngram-freq', '2', '--ngram-size', str(K), '--max-ngram-cover', '0.01',
    ])
    assert blocks_of(dst / 'c4_zh.txt') == ['红楼梦第一回甄士隐梦幻识通灵', PLAIN_REPEAT]
    assert blocks_of(dst / 'wikipedia_cn.txt') == ['量子力学是物理学的一个分支']
    assert blocks_of(dst / 'a.txt') == []          # 对话块被 min-reply-chars 丢掉
    assert res['rule_hits'][clean.RULE_MIN] == 1


def test_filter_nondialogue_rep3_contrast(clean, tmp_path):
    """★ 对照开关：加了 --filter-nondialogue-rep3，重复的非对话块才会被丢。"""
    mapping = {'c4_zh.txt': f'红楼梦第一回甄士隐梦幻识通灵\n\n{PLAIN_REPEAT}'}
    _, dst_off, res_off = run_once(clean, tmp_path, mapping,
                                   extra=['--max-reply-rep3', '0.3'], name='off')
    _, dst_on, res_on = run_once(clean, tmp_path, mapping,
                                 extra=['--max-reply-rep3', '0.3',
                                        '--filter-nondialogue-rep3'], name='on')
    assert len(blocks_of(dst_off / 'c4_zh.txt')) == 2
    assert res_off['rule_hits'].get(clean.RULE_REP3_NONDIALOGUE, 0) == 0
    assert blocks_of(dst_on / 'c4_zh.txt') == ['红楼梦第一回甄士隐梦幻识通灵']
    assert res_on['rule_hits'][clean.RULE_REP3_NONDIALOGUE] == 1


NONDIALOGUE_MID = '春眠不觉晓处处闻啼鸟春眠不觉晓'   # rep3 ≈ 0.462，落在 0.3 与 0.6 之间


def test_nondialogue_max_rep3_default_matches_reply_threshold_bitwise(clean, tmp_path):
    """★ 向后兼容：不给 `--nondialogue-max-rep3` 时，必须与显式给同一个值**逐位一致**。"""
    mapping = {'c4_zh.txt': f'红楼梦第一回甄士隐梦幻识通灵\n\n{NONDIALOGUE_MID}',
               'a.txt': f'用户：a\n模型：{REP_MID}'}
    base = ['--max-reply-rep3', '0.3', '--filter-nondialogue-rep3']
    _, d_default, _ = run_once(clean, tmp_path, mapping, extra=base, name='g_default')
    _, d_explicit, _ = run_once(clean, tmp_path, mapping,
                                extra=[*base, '--nondialogue-max-rep3', '0.3'],
                                name='g_explicit')
    for fn in ('a.txt', 'c4_zh.txt'):
        assert (d_default / fn).read_bytes() == (d_explicit / fn).read_bytes()


def test_nondialogue_max_rep3_is_independent_of_reply_threshold(clean, tmp_path):
    """★ 新参数的核心契约：G 与 F 各自独立，两类块听各自的阈值。

    动机（父 agent 实测）：rep3 随长度单调上升（50 字 0.024 → 1600 字 0.131），
    对话回复均长 65 字、c4_zh 块均长 998 字，一个阈值不可能同时对两者正确。
    """
    dialogue = f'用户：a\n模型：{REP_MID}'
    mapping = {'c4_zh.txt': NONDIALOGUE_MID, 'a.txt': dialogue}
    assert 0.3 < clean.rep_n(NONDIALOGUE_MID, 3) < 0.6
    assert 0.3 < clean.rep_n(REP_MID, 3) < 0.9

    # F 严(0.3) → 对话块丢；G 松(0.6) → 长非对话块留
    _, dst_a, res_a = run_once(clean, tmp_path, mapping, name='gA', extra=[
        '--max-reply-rep3', '0.3', '--filter-nondialogue-rep3',
        '--nondialogue-max-rep3', '0.6'])
    assert blocks_of(dst_a / 'a.txt') == []
    assert blocks_of(dst_a / 'c4_zh.txt') == [NONDIALOGUE_MID]
    assert res_a['rule_hits'].get(clean.RULE_REP3_NONDIALOGUE, 0) == 0
    assert res_a['rule_hits'][clean.RULE_REP3] == 1

    # F 松(0.9) → 对话块留；G 严(0.3) → 长非对话块丢
    _, dst_b, res_b = run_once(clean, tmp_path, mapping, name='gB', extra=[
        '--max-reply-rep3', '0.9', '--filter-nondialogue-rep3',
        '--nondialogue-max-rep3', '0.3'])
    assert blocks_of(dst_b / 'a.txt') == [dialogue]
    assert blocks_of(dst_b / 'c4_zh.txt') == []
    assert res_b['rule_hits'][clean.RULE_REP3_NONDIALOGUE] == 1
    assert res_b['rule_hits'].get(clean.RULE_REP3, 0) == 0


def test_nondialogue_max_rep3_is_inert_without_filter_flag(clean, tmp_path):
    """没有 `--filter-nondialogue-rep3` 时 G 是死旋钮 —— 非对话块一个都不许动。"""
    _, dst, res = run_once(clean, tmp_path, {'c4_zh.txt': NONDIALOGUE_MID},
                           extra=['--max-reply-rep3', '0.9',
                                  '--nondialogue-max-rep3', '0.01'])
    assert blocks_of(dst / 'c4_zh.txt') == [NONDIALOGUE_MID]
    assert res['rule_hits'].get(clean.RULE_REP3_NONDIALOGUE, 0) == 0


def test_nondialogue_max_rep3_works_with_reply_rule_off(clean, tmp_path):
    """F=0（对话 rep3 关闭）但 G>0 + filter flag ⇒ 非对话过滤仍要生效。"""
    _, dst, res = run_once(clean, tmp_path, {'c4_zh.txt': NONDIALOGUE_MID},
                           extra=['--filter-nondialogue-rep3',
                                  '--nondialogue-max-rep3', '0.3'])
    assert blocks_of(dst / 'c4_zh.txt') == []
    assert res['rule_hits'][clean.RULE_REP3_NONDIALOGUE] == 1


# ==========================================================================
# 7) dry-run / --apply / 不碰 src / 报告
# ==========================================================================
def test_dry_run_writes_only_report(clean, tmp_path):
    src, dst, res = run_once(clean, tmp_path, {'a.txt': 'A\n\nA'},
                             extra=['--dedup-blocks'], apply=False)
    assert (src / 'a.txt').read_text(encoding='utf-8') == 'A\n\nA', 'src 必须一字不改'
    assert not (dst / 'a.txt').exists(), 'dry-run 不许写数据文件'
    assert (dst / 'CLEANING_REPORT.md').exists(), 'dry-run 也要留报告'
    assert res['report'].endswith('CLEANING_REPORT.md')


def test_apply_writes_cleaned_files_and_never_touches_src(clean, tmp_path):
    original = 'A\n\nA\n\nB'
    src, dst, _ = run_once(clean, tmp_path, {'a.txt': original}, extra=['--dedup-blocks'])
    assert (src / 'a.txt').read_text(encoding='utf-8') == original
    assert blocks_of(dst / 'a.txt') == ['A', 'B']


def test_apply_mirrors_files_that_are_fully_kept(clean, tmp_path):
    """没被清洗的文件也要写进 dst —— 否则 prepare --source-dir 会漏掉整个来源。"""
    src, dst, _ = run_once(clean, tmp_path, {'a.txt': 'X\n\nY', 'c4_zh.txt': 'Z'},
                           extra=['--dedup-blocks'])
    assert (dst / 'a.txt').exists() and (dst / 'c4_zh.txt').exists()


def test_dst_equal_src_is_refused(clean, tmp_path):
    src = write_corpus(tmp_path / 's', {'a.txt': 'A'})
    with pytest.raises(SystemExit):
        clean.run(clean.parse_args(['--src', str(src), '--dst', str(src)]),
                  log=lambda *a, **k: None)
    assert (src / 'a.txt').read_text(encoding='utf-8') == 'A'


def test_report_has_per_source_and_rule_columns(clean, tmp_path):
    _, dst, _ = run_once(clean, tmp_path, _phrase_corpus(), extra=[
        '--dedup-blocks', '--max-ngram-freq', '2', '--ngram-size', str(K),
        '--max-ngram-cover', '0.2'], name='rep')
    text = (dst / 'CLEANING_REPORT.md').read_text(encoding='utf-8')
    for token in ('a.txt', 'b.txt', 'dedup_blocks', 'max_ngram_cover', '覆盖率', '--apply'):
        assert token in text, f'报告缺 {token}'
