"""`scripts/val_train_contamination_probe.py` 的哈希与"查得到/查不到"判据。

**为什么值得单独测**：这个审计脚本的产出会直接决定"B 段在 v3_dlg val 上的改善"
能不能被记为能力提升。它有两个**静默**失效方式：
1. 滚动哈希算错（分块与整段不一致）⇒ 明明在训练集里也查不到 ⇒ 污染率假性为 0
   ⇒ 结论会**偏向"B 真学会了"**（正是最想相信的那个方向）。
2. 查询侧与训练侧用了不一致的哈希（比如漏传全流偏移 `i0`）⇒ 同样查不到。

所以钉三件事：与朴素实现逐位一致、分块/整段一致、"
**把同一段打乱后必须查不到**"（判据非恒真）。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        'contam_under_test', ROOT / 'scripts' / 'val_train_contamination_probe.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['contam_under_test'] = mod
    spec.loader.exec_module(mod)
    return mod


ct = _load()
N = 8


def _naive(tokens, i, n=N):
    """朴素定义：h(i) = BASE^(n-1) * sum_k t[i+k] * BASE_INV^(i+k)  (mod 2^64)。"""
    s = 0
    for k in range(n):
        s += int(tokens[i + k]) * pow(ct.BASE_INV, i + k, ct.M64)
    return (pow(ct.BASE, n - 1, ct.M64) * s) % ct.M64


def test_matches_naive_reference():
    rng = np.random.default_rng(0)
    toks = rng.integers(0, 2075, size=50).astype(np.uint16)
    h = ct.chunk_hashes(toks, N, 0)
    assert len(h) == len(toks) - N + 1
    for i in range(len(h)):
        assert int(h[i]) == _naive(toks, i), f'位置 {i} 与朴素实现不一致'


def test_chunked_equals_whole():
    """分块 + 全局偏移 i0 必须与整段结果逐位相同（否则跨 chunk 的窗口全查不到）。"""
    rng = np.random.default_rng(1)
    toks = rng.integers(0, 2075, size=60).astype(np.uint16)
    whole = ct.chunk_hashes(toks, N, 0)
    # chunk1 覆盖 token 0..39（窗口起点 0..32），chunk2 从 token 30 起（起点 30..52）
    # ⇒ 重叠窗口起点 30..32，两边必须逐位相同
    a = ct.chunk_hashes(toks[:40], N, 0)
    b = ct.chunk_hashes(toks[30:], N, 30)
    assert len(a) == 33 and len(b) == 23
    assert np.array_equal(a[30:], b[:3])
    assert np.array_equal(a, whole[: len(a)])
    assert int(ct.window_hash(toks[17:17 + N], N, 17)) == int(whole[17])


def test_shuffle_changes_hash_and_scan_distinguishes(tmp_path):
    """端到端：种进 bin 的窗口必须查得到，打乱后必须查不到。"""
    rng = np.random.default_rng(7)
    plant = rng.integers(0, 2075, size=N).astype(np.uint16)
    body = rng.integers(0, 2075, size=5000).astype(np.uint16)
    body[2000:2000 + N] = plant
    bin_path = tmp_path / 'train.bin'
    body.tofile(bin_path)

    sh = plant.copy()
    rng.shuffle(sh)
    q = np.array([int(ct.window_hash(plant, N, 2000)),
                  int(ct.window_hash(sh, N, 2000)),
                  int(ct.window_hash(rng.integers(0, 2075, size=N).astype(np.uint16), N, 100))],
                 dtype=np.uint64)
    hits = ct.scan_hits(str(bin_path), N, q, chunk=1024)
    assert hits[0], '种进去的窗口必须命中（假阴性）'
    assert not hits[1], '打乱后的窗口不该命中（假阳性）'
    assert not hits[2], '随机窗口不该命中（假阳性）'
