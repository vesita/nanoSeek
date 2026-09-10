"""`scripts/ngram_capacity_probe.py` 的测试：多项式哈希、分块流式读取、槽位对齐。

## 为什么这个模块必须有测试
它是 NDB v7 的**容量上限测量工具**。它算错不会崩 —— 只会给一个偏乐观或偏悲观的
覆盖率，然后我们据此决定「NDB 值不值得做」。这个项目已经为"测量工具本身有 bug"
付出过代价（前缀采样低估覆盖率 5.4×；位置对齐错位导致查表错位）。

## 重点：槽位对齐（off-by-L）
`hashes(tok,L)[i]` 的上下文是 `tok[i:i+L]`，预测 `tok[i+L]`，所以放进"逐位置"
数组时应落在下标 `i+L-1`。这个模块历史上写错过两次。下面的
`test_lookup_alignment_toy_table` 用手写小表把这条钉死，并且**验证错误的写法
确实会被区分出来**（否则这条测试就是空测试）。
"""
import pathlib

import numpy as np
import torch
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
NG_PATH = ROOT / 'local' / 'ngram_capacity_full.py'
MULT = np.uint64(1000003)


@pytest.fixture(scope='module')
def ng(load_module_from_path):
    """按路径加载探针脚本（local/ 不在包里，且顶层有 sys.path/env 副作用）。"""
    if not NG_PATH.exists():
        pytest.skip(f"{NG_PATH} 不存在")
    return load_module_from_path(NG_PATH, 'ngram_capacity_full_under_test')


def write_bin(path, toks):
    np.asarray(toks, dtype=np.uint16).tofile(path)
    return str(path)


# ==========================================================================
# 1) hashes：长度、已知答案、与切片自洽
# ==========================================================================
@pytest.mark.parametrize("L", [1, 2, 4, 6, 8, 12])
def test_hashes_length_is_len_minus_L(ng, L):
    tok = np.arange(50, dtype=np.int64)
    assert len(ng.hashes(tok, L)) == 50 - L


def test_hashes_known_answer_manual_polynomial(ng):
    """手算多项式哈希（含 uint64 回绕），逐位对照。

    这是最基础的正确性锚点：把 h = Σ tok[j]·MULT^(L-1-j) 手写一遍，
    确保实现里的循环顺序、初值、类型（uint64 回绕）都没错。
    """
    tok = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.int64)
    L = 3
    for i in range(len(tok) - L):
        h = np.uint64(0)
        for j in range(L):
            h = h * MULT + np.uint64(tok[i + j])
        assert ng.hashes(tok, L)[i] == h, f"第 {i} 个哈希与手算不符"


def test_hashes_window_equals_hash_of_that_slice(ng):
    """`hashes(tok,L)[i]` 必须等于把窗口 `tok[i:i+L]` 单独拿出来算的哈希。

    这条把"上下文到底是哪一段"从注释变成可执行断言。
    """
    rng = np.random.default_rng(0)
    tok = rng.integers(1, 500, size=120).astype(np.int64)
    for L in (2, 5, 9):
        h = ng.hashes(tok, L)
        for i in range(len(h)):
            # 至少给 L+1 个 token，hashes 才有输出（它返回 len-L 个哈希）
            assert h[i] == ng.hashes(tok[i:i + L + 1], L)[0]


def test_hashes_is_deterministic_and_order_sensitive(ng):
    tok = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    assert np.array_equal(ng.hashes(tok, 3), ng.hashes(tok, 3))
    # 调换顺序必须改变哈希（否则 n-gram 就退化成 bag-of-token）
    assert not np.array_equal(ng.hashes(tok, 3), ng.hashes(tok[::-1], 3))


# ==========================================================================
# 2) ★ 槽位对齐：用手写表抓 off-by-L
# ==========================================================================
def _toy_tokens(n=200):
    return np.array([(i * 29 + i * i * 3) % 61 + 2 for i in range(n)], dtype=np.int64)


def test_lookup_alignment_toy_table(ng):
    """★ 逐位置下标 = 上下文结尾下标 = i+L-1。写成 i+L 就整体错一位。

    刻意**不用** build_top1 建表（避免"用被测代码测被测代码"），
    而是手写一张小表，再按文档约定放哈希、查续写、与真实下一个 token 比对。
    """
    L, M = 4, 1 << 12
    tok = _toy_tokens(200)
    h = ng.hashes(tok, L) % M

    table = np.full(M, -1, dtype=np.int64)
    for i in range(len(h)):
        table[h[i]] = tok[i + L]                 # 手写建表：槽 → 该后缀的下一个 token

    vh = ng.hashes(tok, L)
    slot = np.full(len(tok), -1, dtype=np.int64)
    slot[L - 1: L - 1 + len(vh)] = vh % M        # ← 文档约定的放置方式

    for j in range(L - 1, len(tok) - 1):
        assert table[slot[j]] == tok[j + 1], f"位置 {j} 查表得到了错的续写"


def test_off_by_one_placement_is_actually_worse(ng):
    """**空测试防护**：确认错位写法在这份语料上真的更差。

    如果两种写法都能全对，上面那条测试就什么都没测到 —— 本项目的
    `gradient_checkpointing` 就出过这种事故（两个 A/B 臂其实完全一样）。
    """
    L, M = 4, 1 << 12
    tok = _toy_tokens(200)
    vh = ng.hashes(tok, L)
    table = np.full(M, -1, dtype=np.int64)
    for i in range(len(vh)):
        table[(vh[i] % M)] = tok[i + L]

    good = np.full(len(tok), -1, dtype=np.int64)
    good[L - 1: L - 1 + len(vh)] = vh % M
    bad = np.full(len(tok), -1, dtype=np.int64)
    bad[L: L + len(vh)] = vh % M                  # 错位一位

    def acc(slot):
        return sum(1 for j in range(L, len(tok) - 1) if table[slot[j]] == tok[j + 1])

    n = len(range(L, len(tok) - 1))
    assert acc(good) == n, "正确写法应当全对"
    assert acc(bad) < n, "错位写法居然也全对 —— 这条对照测不出 off-by-L，需要换语料"


def test_next_token_index_matches_context_end(ng):
    """把"上下文结尾 i+L-1 ↔ 目标 i+L"写成显式断言，二者必须自洽。"""
    tok = _toy_tokens(60)
    for L in (2, 3, 6):
        h = ng.hashes(tok, L)
        for i in range(len(h)):
            end = i + L - 1
            assert np.array_equal(tok[end - L + 1: end + 1], tok[i:i + L])
            assert tok[end + 1] == tok[i + L]


# ==========================================================================
# 3) iter_chunks：流式分块
# ==========================================================================
def test_iter_chunks_sequential_reconstructs_file(ng, tmp_path):
    tok = np.arange(1, 1001, dtype=np.int64) % 800 + 1
    p = write_bin(tmp_path / 'c.bin', tok)
    pieces, offs = [], []
    for chunk, off in ng.iter_chunks(p, chunk=137, L_max=8, seed=None, sample_runs=None):
        pieces.append(chunk)
        offs.append(off)
    assert np.array_equal(np.concatenate(pieces), tok), "顺序分块拼不回原文件"
    assert offs[0] == 0 and offs == sorted(offs), "偏移量必须是递增且从 0 开始"


def test_iter_chunks_sampled_total_and_seed(ng, tmp_path):
    tok = np.arange(1, 5001, dtype=np.int64) % 4000 + 1
    p = write_bin(tmp_path / 'c.bin', tok)
    a = [c for c, _ in ng.iter_chunks(p, chunk=1000, L_max=8, seed=42, sample_runs=3000)]
    b = [c for c, _ in ng.iter_chunks(p, chunk=1000, L_max=8, seed=42, sample_runs=3000)]
    assert sum(len(c) for c in a) == 3000, "采样总量必须正好等于 sample_runs"
    assert all(np.array_equal(x, y) for x, y in zip(a, b)), "同 seed 必须可复现"
    c = [x for x, _ in ng.iter_chunks(p, chunk=1000, L_max=8, seed=7, sample_runs=3000)]
    assert not all(np.array_equal(x, y) for x, y in zip(a, c)), "换 seed 结果应当不同"


def test_iter_chunks_sampled_runs_are_contiguous(ng, tmp_path):
    """采样必须是**连续 run**（保留后缀结构），不是逐 token 随机抽。"""
    tok = np.arange(1, 5001, dtype=np.int64)
    p = write_bin(tmp_path / 'c.bin', tok)
    for chunk, off in ng.iter_chunks(p, chunk=500, L_max=4, seed=1, sample_runs=1500):
        assert np.array_equal(chunk, tok[off:off + len(chunk)]), "分块不是连续区间"


# ==========================================================================
# 4) build_top1 / build_bitmaps：端到端对照
# ==========================================================================
def _deterministic_corpus(n):
    """周期 61 的确定性序列：同一 L-gram 的续写唯一，可在语料内自洽验证。"""
    return np.array([(i * i * 7 + 13) % 61 + 2 for i in range(n)], dtype=np.int64)


def test_build_top1_recovers_continuations_on_deterministic_corpus(ng, tmp_path):
    """同一语料上建表再查表：命中率应当接近 1（唯一的损失来源是哈希碰撞）。"""
    L, M = 6, 1 << 20
    tok = _deterministic_corpus(6000)
    p = write_bin(tmp_path / 'c.bin', tok)

    # 先确认这份语料在 M 下不碰撞，避免测试依赖运气
    h = ng.hashes(tok, L) % M
    ctx_slot = {}
    for i in range(len(h) - 1):
        ctx_slot.setdefault(bytes(tok[i:i + L].astype(np.int32)), h[i])
    assert len(set(ctx_slot.values())) == len(ctx_slot), "测试语料在 M 下发生了碰撞，换 M"

    g_tok, g_cnt = ng.build_top1(p, L, M, chunk=1000, sample_runs=None)
    pred = g_tok[h]
    truth = tok[L:L + len(h)]
    assert (g_cnt[h] > 0).all(), "有槽位没被填上"
    acc = float((pred == truth).mean())
    assert acc > 0.99, f"同语料回查命中率只有 {acc:.3f}"


def test_build_bitmaps_self_coverage_is_full(ng, tmp_path):
    """用自己的语料当 val：覆盖率必须 ~100%。"""
    Ls, Ms = [6], [1 << 16]
    tok = _deterministic_corpus(4000)
    p = write_bin(tmp_path / 'c.bin', tok)
    bits, n_seen = ng.build_bitmaps(p, Ls, Ms, chunk=1000, sample_runs=None, L_max=6)
    assert n_seen == len(tok)
    h = ng.hashes(tok, 6)
    cov = bits[(6, 1 << 16)][h % (1 << 16)].mean()
    assert cov > 0.99, f"自覆盖只有 {cov:.4f}"


def test_build_bitmaps_random_control_is_near_zero(ng, tmp_path):
    """随机对照：随机 token 的后缀几乎不可能落在周期性语料的槽里。

    这条是"覆盖率数字确实来自真实匹配"的反证 —— 没有它，
    build_bitmaps 全返回 True 也能骗过上面那条自覆盖测试。
    """
    tok = _deterministic_corpus(4000)
    p = write_bin(tmp_path / 'c.bin', tok)
    bits, _ = ng.build_bitmaps(p, [6], [1 << 16], chunk=1000, sample_runs=None, L_max=6)
    rng = np.random.default_rng(0)
    rnd = rng.integers(2, 63, size=20000).astype(np.int64)
    h = ng.hashes(rnd, 6)
    cov = bits[(6, 1 << 16)][h % (1 << 16)].mean()
    assert cov < 0.05, f"随机序列覆盖率 {cov:.4f} 偏高，说明位图判定有问题"


# ==========================================================================
# 5) load_val
# ==========================================================================
def test_load_val_shape_and_range(ng, tmp_path):
    tok = np.arange(1, 3001, dtype=np.int64) % 2000 + 1
    p = write_bin(tmp_path / 'v.bin', tok)
    x = ng.load_val(p, block=64, n_windows=8, seed=0)
    assert tuple(x.shape) == (8, 64)
    assert x.dtype == torch.int64, f"load_val 应返回 int64 torch 张量，实际 {x.dtype}"
    assert int(x.min()) >= 0 and int(x.max()) < 65536


def test_load_val_is_seed_reproducible(ng, tmp_path):
    tok = np.arange(1, 3001, dtype=np.int64) % 2000 + 1
    p = write_bin(tmp_path / 'v.bin', tok)
    a = ng.load_val(p, 64, 8, seed=3)
    b = ng.load_val(p, 64, 8, seed=3)
    assert (a == b).all()
