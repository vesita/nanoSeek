"""`scripts/intent_probe.py`（意图跟随探针）的测试。

三组重点：

1. **判据的已知答案对照**（`test_selftest_controls_all_pass`）—— 每组意图的"参考好回复"
   必须 pass、"参考坏回复"必须 fail。判据若连好/坏都分不开，就是恒真的空测试。
2. **两个实测假阳性的回归钉** —— 2026-09-15 建探针时，靠"读被翻转的样本"（AGENTS §5.9）
   抓到的两个真错误，各钉一条，防止再犯：
   - 裸 `[二两]` 会命中"**两**种因素"；
   - 裸 `难受` 会命中**提示词回显**（`A：我失恋了，很难受。`）。
3. **样本文件覆盖** —— 探针的 14 组意图必须能覆盖 `analysis/*_ood_prompts.txt` 里的提示词，
   否则 `--from-file` 会静默漏掉整组（那样报告会"看起来正常"但其实少了行）。
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.intent_probe import (  # noqa: E402
    INTENTS,
    PROMPT_TO_INTENT,
    _strip_echo,
    parse_samples,
    score,
    selftest,
)

BY_KEY = {it.key: it for it in INTENTS}


def test_selftest_controls_all_pass():
    """★ 判据的已知答案对照：14/14 组的"参考好回复"pass、"参考坏回复"fail。"""
    assert selftest() == 0, '有意图组的判据分不开好/坏回复，先修判据'


def test_bare_measure_word_is_not_an_arithmetic_answer():
    """★ 回归（假阳性 #1）：`两种因素` 不是"1+1 等于 2"。"""
    it = BY_KEY['arith']
    assert score('两种因素', it)[0] is False
    assert score('差距大小都是两种因素，但好处在于我们看到每一个人', it)[0] is False
    # 负向对照：真正的答案形态必须仍然 pass
    assert score('1+1等于2。', it)[0] is True
    assert score('1+1等于二。', it)[0] is True
    assert score('答案是 2', it)[0] is True


def test_prompt_echo_alone_is_not_comforting():
    """★ 回归（假阳性 #2）：复读提示词 `我失恋了，很难受。` 不是安慰。"""
    it = BY_KEY['comfort']
    assert score('我失恋了，很难受。', it)[0] is False, '提示词本身不该被判成安慰'
    assert score('A：我失恋了，很难受。', it)[0] is False
    # 负向对照：真安慰必须 pass
    assert score('我理解这种难受，先别急着逼自己好起来。', it)[0] is True


def test_strip_echo_removes_the_prompt_line_but_keeps_the_reply():
    p = '我失恋了，很难受。'
    assert _strip_echo(f'A：{p}', p) == ''
    assert _strip_echo(f'{p}\nB：我理解。', p) == 'B：我理解。'
    assert _strip_echo(f'A：{p}\nB：我理解你的难受。', p) == 'B：我理解你的难受。'
    # 不加前缀的普通文本不受影响
    assert _strip_echo('我理解你的难受。', p) == '我理解你的难受。'


def test_intents_cover_the_archived_sample_files():
    """探针的意图表必须覆盖归档样本里的提示词（否则 --from-file 会静默漏整组）。"""
    prompts = {it.prompt for it in INTENTS}
    assert len(prompts) == len(INTENTS), 'INTENTS 里有重复提示词，PROMPT_TO_INTENT 会互相覆盖'
    for name in ('B_ood_prompts.txt', 'know2_ood_prompts.txt'):
        path = ROOT / 'analysis' / name
        if not path.exists():
            continue
        parsed = parse_samples(str(path))
        assert parsed, f'{name} 一条都没解析出来'
        # 归档文件里出现的提示词，除"附"节（###）外都应能在意图表里找到
        missing = [p for p in parsed if p not in PROMPT_TO_INTENT]
        assert not missing, f'{name} 有意图表未覆盖的提示词: {missing}'
