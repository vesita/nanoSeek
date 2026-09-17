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
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.intent_probe import (  # noqa: E402
    INTENTS,
    PROMPT_TO_INTENT,
    SAMPLE_FAIL_PREFIX,
    _strip_echo,
    is_sample_failure,
    main,
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


def test_sample_failure_is_never_scored_as_a_wrong_answer(monkeypatch):
    """★★ 回归钉（2026-09-16 实盘踩到）：采样器崩溃**不得**被当成"模型答错"。

    现场：本探针用子进程调 `sample_py.py`，而 gfx1030 需要
    `HSA_OVERRIDE_GFX_VERSION=10.3.0 HSA_ENABLE_SDMA=0`。调用者没 export 时子进程
    HIP 直接崩（rc=1），28 条样本全变成 `<采样失败 …>`，然后**被当作模型输出打分**
    ⇒ 报告会给出"0/28"这个**假结论**（把基础设施故障读成能力结论）。

    判据（能区分好坏两种实现，不是恒真）：
    - 哨兵串必须被 `is_sample_failure()` 认出来；
    - `main(--out-dir …)` 在**每条都失败**时必须**非零退出**，且**不写报告文件**。
    """
    assert is_sample_failure(f'{SAMPLE_FAIL_PREFIX} rc=1：HIP error>')
    assert not is_sample_failure('我叫小寻，很高兴认识你。'), '正常回复被误判成失败'

    def boom(out_dir, prompt, seed, **kw):
        return f'{SAMPLE_FAIL_PREFIX} rc=1：HIP error>'

    monkeypatch.setattr('scripts.intent_probe.sample_with_existing_entry', boom)
    out_md = ROOT / 'out' / '_intent_probe_failguard.md'
    out_md.parent.mkdir(parents=True, exist_ok=True)
    if out_md.exists():
        out_md.unlink()
    rc = main(['--out-dir', 'out/_eval_nonexistent', '--out', str(out_md)])
    assert rc == 3, f'采样全失败时必须非零退出（拿到 rc={rc}）—— 否则会静默出 0/N 假报告'
    assert not out_md.exists(), '采样失败时**不许**写出报告文件（写了就会被当成结论读）'


def test_out_dir_mode_strips_prompt_echo_before_scoring(monkeypatch, tmp_path):
    """★★ 回归钉（2026-09-16 实盘踩到）：`--out-dir` 路径也必须先剥提示词回显。

    现场：`sample_py.py` 在 `--- 生成 ---` 之后会**先把提示词回显一行**。于是
    `makers` 的期望词是 `研发`，而提示词正是「你是谁**研发**的？」⇒ 模型什么都不用答对，
    只要回显就"达标"。实测把 6/28 抬成了假成绩（去回显后 4/28）。
    2026-09-15 只修了 `--from-file`（`parse_samples` 里剥），`--out-dir` 漏了。

    判据（能区分好坏实现）：让采样器**回显提示词 + 一句不含任何期望词的废话**，
    修复后该组必须判 **fail**；未修复则会因回显里的 `研发` 判 pass。
    """
    def echoing_sampler(out_dir, prompt, seed, **kw):
        return f'A：{prompt}\nB：今天天气不错。'

    monkeypatch.setattr('scripts.intent_probe.sample_with_existing_entry', echoing_sampler)
    jf = tmp_path / 'echo.json'
    rc = main(['--out-dir', 'out/_eval_nonexistent', '--json', str(jf)])
    assert rc == 0, f'正常采样路径应成功（rc={rc}）'
    per = json.loads(jf.read_text(encoding='utf-8'))['out/_eval_nonexistent']
    assert per['makers'][0]['ok'] is False, (
        '提示词回显未被剥掉：makers 的期望词「研发」命中了提示词本身（假阳性复现）')
    assert '你是谁研发的' not in per['makers'][0]['text'], '存档文本里仍带提示词回显'
    total = sum(1 for v in per.values() for s in v if s['ok'])
    assert total == 0, f'回显型废话不该有任何一个达标，却拿到 {total}'
