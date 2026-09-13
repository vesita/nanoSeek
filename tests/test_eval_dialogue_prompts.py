"""`inference/scripts/eval_dialogue.py` 的 prompt 样式与轮次识别。

**为什么值得单独测**：这个脚本产出的是**定性质量结论**（d1/d2/turns/rep），
而它的 prompt 格式必须和模型训练语料一致 —— 不一致就是喂 OOD 输入，
表现为 `turns` 假性归零，且会让"见过这套标签"的模型获得**虚假主场优势**，
足以把 v1/v2 的结论评反。2026-09-11 之前它一直用 v1 的 `用户：/模型：`，
而主线语料（v2）早已去标签改成 `A：/B：`（`TECH_DEBT` P1）。

这类 bug **不报错**，只是静默给出错误结论 —— 所以必须有测试钉住。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load():
    """按路径加载（该脚本不在包内，且文件名不是合法模块名的常见形态）。"""
    spec = importlib.util.spec_from_file_location(
        'eval_dialogue_under_test', ROOT / 'inference' / 'scripts' / 'eval_dialogue.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['eval_dialogue_under_test'] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def ed():
    return _load()


# ==========================================================================
# 样式表本身
# ==========================================================================

def test_default_style_is_ab_matching_v2_corpus(ed):
    """★ 核心回归：默认样式必须是 v2 语料的 `A：/B：`，不能退回旧的 `用户：/模型：`。"""
    assert ed.DEFAULT_STYLE == 'ab'
    assert ed.DIALOGUE_PROMPTS is ed.PROMPT_STYLES['ab']


def test_both_styles_available_and_same_length(ed):
    """两套样式都要在（v1 基座仍需用旧样式评），且条数一致便于横向比较。"""
    assert set(ed.PROMPT_STYLES) == {'ab', 'user-model'}
    lens = {len(v) for v in ed.PROMPT_STYLES.values()}
    assert len(lens) == 1, f'两套 prompt 条数不一致：{lens}'


def test_styles_do_not_leak_each_others_markers(ed):
    """★ 样式之间不能串味 —— 否则等于混着眼，谁都评不准。"""
    for p in ed.PROMPT_STYLES['ab']:
        assert '用户' not in p and '模型' not in p, f'ab 样式里混进了旧标签：{p!r}'
    for p in ed.PROMPT_STYLES['user-model']:
        assert 'A：' not in p and 'B：' not in p, f'user-model 样式里混进了新标签：{p!r}'


def test_ab_prompts_end_with_b_cue(ed):
    """`A：...\\nB：` 才是在"等模型接话"；只给 A 行会让它继续自说自话。"""
    for p in ed.PROMPT_STYLES['ab']:
        assert p.endswith('B：'), f'ab prompt 没有以 B：结束：{p!r}'
        assert '\n' in p, f'ab prompt 缺少换行分隔：{p!r}'


# ==========================================================================
# 轮次识别（同时兼容两套约定）
# ==========================================================================

def test_detects_ab_alternation(ed):
    r = ed.dialogue_turn_structure('好的，我理解。\nA：那我该怎么办\nB：先试着放松。')
    assert r['style_detected'] == 'ab'
    assert r['has_structure'] is True


def test_detects_user_model_alternation(ed):
    """旧的 `用户：/模型：` 也要能识别 —— 否则评 v1 会得到假性 turns=0。"""
    r = ed.dialogue_turn_structure('我理解。\n用户：那我该怎么办\n模型：先放松。')
    assert r['style_detected'] == 'user-model'
    assert r['has_structure'] is True


def test_fragments_have_no_structure(ed):
    r = ed.dialogue_turn_structure('今天天气不错，我想出去走走。')
    assert r['turns'] == 0
    assert r['has_structure'] is False
    assert r['style_detected'] == 'none'


def test_one_sided_repetition_is_not_structure(ed):
    """★ 单侧重复（`B：` 刷屏）不是对话结构 —— 正是"复读坍塌"的样子。

    旧实现用 `user+model >= 2 and has_model`，会把这种坍塌判成"有结构"。
    """
    r = ed.dialogue_turn_structure('B：好\nB：好\nB：好')
    assert r['has_structure'] is False
    assert r['turns'] == 3


def test_format_drift_is_visible(ed):
    """★ 用 A：提示、却生成 `用户：` 时要能被看出来（style_detected 会报 user-model）。

    这就是本函数"同时看两套"的价值：只看一套会把"结构对、标签换了"误判成"没结构"。
    """
    r = ed.dialogue_turn_structure('用户：你好\n模型：你也好')
    assert r['style_detected'] == 'user-model'     # 与实际调用方给的 'ab' 不符 → 漂移
    assert r['has_structure'] is True              # 但结构本身是有的


def test_halfwidth_colon_also_counted(ed):
    """语料里两种冒号都出现过，识别不能只认全角。"""
    assert ed.dialogue_turn_structure('A: 你好\nB: 你也好')['style_detected'] == 'ab'
