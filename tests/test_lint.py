"""lint 门禁：把「一定是 bug」的静态规则变成一条测试。

## 背景
`scripts/ngram_capacity_probe.py` 里我把模块级常量从 `GATES` 改名成 `GATE_NAMES`，
却漏改了报告段的 3 处引用 —— 结果整段 4 分钟的评测跑完，在**最后打印结果时**
才抛 `NameError`，前面的结果全丢。同一类错误在项目里还出现过：`_ds` 重构成
`pick_bin_names` 后残留引用。

这类"名字写错/重复定义"是纯静态可判定的，用 lint 拦下来是**零成本**的，
不该靠"跑一遍试试"。

## 为什么门禁开得这么窄
只选这四条：`E9`（语法/运行期错误）、`F821`（未定义名）、`F811`（重复定义）、
`F823`（局部变量未绑定）。2026-09-10 全项目实测合计只有 **1 处**真实违规，
所以可以设零容忍。

故意不开 `F401`（88 处未使用导入）、`F841`（29 处未使用变量）、行宽/导入排序 ——
那些噪声会让门禁变成"人人加 noqa"的摆设。想清理时单独跑：
    ruff check --select F401 --fix
"""
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

# 本项目自己的代码。故意不含 data/external（第三方语料仓库）。
TARGETS = [
    'model', 'training', 'data/chinese', 'local', 'tests',
    'inference', 'scripts', 'nano_arith', 'analysis', 'dev_scripts',
    'pre-research', 'cli.py',
]


def _ruff(*args):
    return subprocess.run(
        [sys.executable, '-m', 'ruff', *args],
        cwd=ROOT, capture_output=True, text=True)


def test_ruff_is_available():
    r = _ruff('--version')
    if r.returncode != 0:
        pytest.skip("ruff 未安装（uv sync --group dev）")


def test_no_undefined_or_redefined_names():
    """★ 零容忍门禁：未定义名 / 重复定义 / 未绑定局部变量。

    `training/train.py` 里踩过的真事：`config_keys` 快照之后定义的全局会被
    静默改回默认值。虽然那是语义问题（lint 抓不到），但同一类"名字"错误
    （比如重构后残留旧名）正是这四条规则覆盖的。
    """
    targets = [t for t in TARGETS if (ROOT / t).exists()]
    if not targets:
        pytest.skip("目标目录都不存在")
    r = _ruff('check', '--no-cache', '--output-format', 'concise', *targets)
    if r.returncode == 0:
        return
    assert False, (
        "ruff 门禁未通过（E9/F821/F811/F823）。这些是**真 bug**，不是风格问题：\n"
        + (r.stdout or '') + (r.stderr or ''))


def test_ruff_config_is_present_and_narrow():
    """门禁规则必须显式写死在 pyproject 里，且只含"一定是 bug"的规则。

    如果有人为了让自己代码过而把规则放宽，这条会红 —— 门禁的存在本身就是
    要拦住"临时放宽"。
    """
    text = (ROOT / 'pyproject.toml').read_text(encoding='utf-8')
    assert '[tool.ruff.lint]' in text
    assert 'select = ["E9", "F821", "F811", "F823"]' in text, \
        "ruff 规则集被改动了 —— 如果是有意放宽，请同时更新 tests/test_lint.py 的说明"


def test_vendored_data_is_excluded_from_lint():
    """第三方语料仓库（data/external 等，上万个文件）必须排除在门禁之外。

    否则 `pytest` 会因为别人仓库里的代码变红，门禁立刻失去可信度。
    """
    text = (ROOT / 'pyproject.toml').read_text(encoding='utf-8')
    for d in ('data/external', 'out', '.venv'):
        assert d in text, f"{d} 没有被 extend-exclude 排除"
