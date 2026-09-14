"""`scripts/per_source_ce_probe.py` 的「源对齐对照」选择逻辑。

**为什么值得单独测**：这个探针是 warm start 之后**唯一可信的验收尺子**
（`v3_*` 的 val 基本被 v2 见过）。它自带三条对照，其中「源对齐对照」
（`token/raw_chars` ≈ 1.0）原本**硬编码了 manifest_v2 的三个源名**——
换 manifest（`v3_dlg` / `v3_know` 只有 9 个源、源名集不同）时会直接 `KeyError`
**崩在对照上**，于是"B 段在自己 val 上到底怎样"这个问题整轮拿不到答案
（2026-09-14 实测踩到，见 `analysis/per_source_ce_B_ownval.txt` 早期版本）。

这类 bug 的危险在于它**看起来像"探针跑不了这个 manifest"**，
从而把"尺子坏了"误读成"这个问题没法测"。所以钉住三件事：
1. 有 v2 对照源时，口径不变（还是那三个，历史存档可比）；
2. 没有时**退化**到该 manifest 自己的前 3 个源，**绝不 KeyError**；
3. 任何情况下返回的源名都**确实存在于 spans 里**（判据非恒真：故意喂一个
   源名不在 spans 的 manifest，返回的列表必须不含它）。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        'per_source_ce_probe_under_test', ROOT / 'scripts' / 'per_source_ce_probe.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['per_source_ce_probe_under_test'] = mod
    spec.loader.exec_module(mod)
    return mod


probe = _load()
V2_CTRL = list(probe.V2_ALIGN_CONTROLS)


def _mk(order, raw_names=None):
    spans = {n: (0, 10) for n in order}
    raw = {n: 10 for n in (order if raw_names is None else raw_names)}
    return spans, raw


def test_v2_manifest_keeps_the_archived_controls():
    """v2 三个对照源都在时必须原样返回，且不给退化说明（保证与历史存档同口径）。"""
    order = V2_CTRL + ['红楼梦.txt']
    spans, raw = _mk(order)
    names, note = probe.align_control_names(spans, order, raw)
    assert names == V2_CTRL
    assert note == ''


def test_v3_manifest_falls_back_without_keyerror():
    """v3_dlg 那种源名集：必须退化成自己的前 3 个源，并给出说明。"""
    order = ['belle_multiturn.txt', 'dailychat_dialogue.txt', 'escov_zh.txt', 'glm_dialogue.txt']
    spans, raw = _mk(order)
    names, note = probe.align_control_names(spans, order, raw)
    assert names == order[:3]
    assert 'v2 对照源名' in note
    # 关键：返回的每个名字都能在 spans 里查到（旧实现死在这里）
    for n in names:
        assert n in spans


def test_returned_names_are_always_lookupable():
    """非恒真判据：喂一个**没有 raw、也不在 spans** 的源名，它绝不能出现在返回值里。"""
    order = ['a.txt', 'b.txt', 'c.txt']
    spans = {n: (0, 10) for n in order}
    spans['c4_zh.txt'] = (0, 10)          # 在 spans 里
    raw = {'a.txt': 10, 'b.txt': 10}      # 但没有 raw ⇒ 不能拿它做对照
    names, note = probe.align_control_names(spans, order, raw)
    assert 'c4_zh.txt' not in names
    for n in names:
        assert n in spans and n in raw
    assert note  # 走的是退化路径，必须留痕


def test_empty_manifest_yields_empty_controls_but_says_so():
    """连退化候选都没有时：返回空表并留说明，而不是静默通过（空对照必须可见）。"""
    names, note = probe.align_control_names({}, [], {})
    assert names == []
    assert note
