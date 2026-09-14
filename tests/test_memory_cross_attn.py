# -*- coding: utf-8 -*-
"""`model/memory_cross_attn.py`（神经元级长程读接口 / RETRO-lite）的测试。

## 为什么这个文件是 2026-09-15 才有的
这个模块此前**一条测试都没有** —— 对它的"验证"全是跑脚本看输出。
代价很快就来了：`write_online` 里那个 `write_gate` 被当成"可学习写门控"用了好几轮，
**实际拿不到任何梯度**（`forward` 读路径从不引用它、`write_online` 是 `@torch.no_grad()`），
所以它永远停在初值、是个恒真掩码。**"它在 `parameters()` 里"是恒真判据，不是证据。**
教训见 `AGENTS.md §5.10`：判据必须能区分好坏两种口径。

## 本文件的核心是这条：库很小时 forward 不能出 NaN
`k = min(top_k, 库里的活跃条数)`，而跨槽 z 分数用 `sim_used.std(dim=-1)` 算。
`std`（correction=1）在**单元素**上是 NaN，且 `0 * nan = nan`
⇒ 库只有 1 条时，**即使 `sim_gain=0` 也会把整个 forward 污染成 NaN**。
实测（修复前）：`live=1 → isnan=True`，`live≥2 → 干净`。
这条只在"空库 + 在线写"的路径上会咬人（预构建库 n 很大，没事），
但那正是"让模型自己填库"要走的路 ⇒ 必须先修，且必须钉住。
"""
import pytest
import torch

from model.memory_cross_attn import MemoryCrossAttention

D = 32          # n_embd
CH = 16         # chunk
CAP = 32        # 库容量
V = 40          # 逻辑词表（只用于造 logits/targets）


def make_ndb(*, live, top_k=4, att_sim_gain=0.0, seed=0, identical=False):
    """小而完整的读接口。`live=None` = 预构建库（整库活跃）。"""
    g = torch.Generator().manual_seed(seed)
    keys = (torch.ones(CAP, D) if identical else torch.randn(CAP, D, generator=g))
    return MemoryCrossAttention(
        n_embd=D, n_head=4, chunk=CH, top_k=top_k, gate_init=0.1,
        live=live, store_keys=keys, att_sim_gain=att_sim_gain)


# ==========================================================================
# 1) ★ 回归：任何活跃条数都不能产出 NaN
# ==========================================================================
@pytest.mark.parametrize("live", [1, 2, 3, 4, 8, 16, CAP])
@pytest.mark.parametrize("lam", [0.0, 2.0])          # 2.0 = 计划里要用的 λ
def test_forward_is_finite_for_every_live_count(live, lam):
    """★ 修复前 `live=1` 这条会失败（输出全 NaN）。

    λ 取 0 也要测：NaN 的传播是 `0 * nan = nan`，`sim_gain=0` **不构成** 保护。
    """
    ndb = make_ndb(live=live, att_sim_gain=lam)
    h = torch.randn(2, CH, D)
    out = ndb(h)
    assert out.shape == h.shape
    assert torch.isfinite(out).all(), (
        f"live={live} λ={lam}: forward 出现非有限值 —— "
        f"跨槽 z 分数在 k=min(top_k,live)<2 时退化了")


def test_identical_store_entries_do_not_produce_nan():
    """已知答案的退化输入：库里所有条目**完全相同** ⇒ 跨槽 std = 0。

    此时 z 分数的分母是 `0 + 1e-6`，分子是 0 ⇒ 结果应当是 0 而不是 NaN。
    这条与 `live=1` 是**两个不同的退化原因**，要分别钉住。
    """
    ndb = make_ndb(live=CAP, identical=True, att_sim_gain=2.0)
    out = ndb(torch.randn(2, CH, D))
    assert torch.isfinite(out).all()


def test_empty_store_forward_is_an_exact_noop():
    """空库（在线写的第一笔之前）必须是**精确的 0**，不能拿全零槽去做注意力。"""
    ndb = make_ndb(live=0)
    h = torch.randn(2, CH, D)
    out = ndb(h)
    assert torch.equal(out, torch.zeros_like(h))
    assert not out.requires_grad, "空库的 no-op 不该带计算图（上游梯度应为 0）"


def test_prebuilt_store_uses_whole_store():
    """`live=None` = 预构建库：活跃条数是整库（与旧行为逐位一致的那条路径）。"""
    ndb = make_ndb(live=None)
    assert ndb._n_live() == CAP
    assert torch.isfinite(ndb(torch.randn(2, CH, D))).all()


# ==========================================================================
# 2) 在线写：规则式，且返回的账要对得上
# ==========================================================================
def test_write_online_fills_store_and_reports_counts():
    ndb = make_ndb(live=0)
    h = torch.randn(4, 256, D)
    tgt = torch.randint(0, V, (4, 256))
    logits = torch.randn(4, 256, V)

    n, info = ndb.write_online(h, tgt, logits, quantile=0.02)

    assert n > 0, "应当写下最惊讶的那批 chunk"
    assert info["surprise_written"] == n
    assert info["live"] == n == ndb._n_live()
    assert ndb.written_total == n
    # 写进去的是**键**（chunk 均值），不是别的空间的东西
    written = ndb.store[:n].float()
    assert torch.isfinite(written).all()
    assert not torch.allclose(written, torch.zeros_like(written)), "不该写进全零键"


def test_write_online_writes_nothing_when_all_targets_masked():
    """★ 负向对照：全是 `ignore_index` 时，**一条都不该写**。

    人格层里"对方说的话"就落在 loss 掩码外。若这里写进去了，
    等于把对方的话记成"自己答不上来的东西"。
    """
    ndb = make_ndb(live=0)
    h = torch.randn(2, CH * 2, D)
    tgt = torch.full((2, CH * 2), -100)
    logits = torch.randn(2, CH * 2, V)

    n, info = ndb.write_online(h, tgt, logits, quantile=0.02)

    assert n == 0
    assert info["skip"] == "no_valid_target"
    assert ndb._n_live() == 0


def test_write_online_quantile_zero_writes_every_valid_chunk():
    """`quantile=0` = 关掉规则筛选 ⇒ 所有有效 chunk 都写。用来钉住"筛选强度由参数控制"。"""
    ndb = make_ndb(live=0)
    B, T = 2, CH * 2
    h = torch.randn(B, T, D)
    tgt = torch.randint(0, V, (B, T))
    logits = torch.randn(B, T, V)

    n, _ = ndb.write_online(h, tgt, logits, quantile=0.0)

    assert n == (B * T) // CH, "quantile=0 时应写下全部 chunk"


def test_write_online_is_capped_by_store_capacity():
    """环形库：写超容量时不越界，`live` 封顶在容量上。"""
    ndb = make_ndb(live=0, top_k=1)
    h = torch.randn(8, CH, D)          # 8 个 chunk > CAP=32? 否 ⇒ 多写几批
    tgt = torch.randint(0, V, (8, CH))
    logits = torch.randn(8, CH, V)
    for _ in range(6):                 # 6 * 8 = 48 > CAP=32
        ndb.write_online(h, tgt, logits, quantile=0.0)

    assert ndb._n_live() == CAP
    assert ndb.write_full_events > 0, "写满过就应该记到 write_full_events"
    assert torch.isfinite(ndb(torch.randn(2, CH, D))).all()


# ==========================================================================
# 3) 反向传播：读侧的参数必须真的拿到梯度（"在 parameters() 里"不算证据）
# ==========================================================================
def test_read_side_params_receive_gradient():
    """已知答案对照：读门控 / λ / wq 必须有非零梯度，否则它们同样是空壳。

    ★ 这条是本文件存在的第二个理由：写门控就是栽在"没测梯度"上的。
    """
    ndb = make_ndb(live=CAP, att_sim_gain=0.0, seed=1)
    # wo 是零初始化（记忆起点是 no-op）⇒ 先给它一个非零值，否则整条支路梯度为 0，
    # 测出来的是"wo 为零"这个已知事实，而不是"参数不可学"。
    torch.nn.init.normal_(ndb.wo.weight, std=0.02)

    h = torch.randn(2, CH, D)
    ndb(h).pow(2).mean().backward()

    for name in ("gate_proj.weight", "sim_gain", "wq.weight", "wo.weight"):
        p = dict(ndb.named_parameters())[name]
        assert p.grad is not None, f"{name} 没有梯度 ⇒ 它不是可学参数"
        assert torch.isfinite(p.grad).all()
