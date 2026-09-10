"""Δ 插值数学的测试（`scripts/ngram_capacity_probe.py` 的纯函数部分）。

## 为什么单独拆出来测
这段代码在一天里犯了两个**只有跑起来才会暴露**的错误：
1. 第一版把「§12.3 口径」和「置信度门控」混成一个数，两处语义被偷换；
2. 第二版忘了把模型输出的 `(B,T,V)` 展平成 `(N,V)`，`gather` 直接抛
   `Index tensor must have the same number of dimensions`（跑到一半才炸）。

两条都不是"想清楚就能避免"的，而是**必须有小张量对照**的。所以这里把
`gate_factors` / `nll_under_retrieval` 抽成纯函数，用几十个 token 的假数据
在毫秒级验证，而不是靠跑 10 分钟的真实评测。

## 口径说明（重要）
- `A_ungated` = `ngram_sample_capacity.py` 的原始口径（无门控，只按覆盖屏蔽）
- `B_oracle` 用到了标签 y，**推理时不可实现**，只是离线收益上界
- `C_uncert` / `D_disagree` 只用推理时可获得的信息，**可部署**
"""
import pathlib

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parent.parent
NG_PATH = ROOT / 'local' / 'ngram_capacity_full.py'


@pytest.fixture(scope='module')
def ng(load_module_from_path):
    return load_module_from_path(NG_PATH, 'ngram_capacity_full_delta_under_test')


def make_batch(ng, n=64, v=32, seed=0, cover_rate=0.5, hit_rate=0.6):
    """造一个自洽的假 batch：lp / p_y / p_max / tok_ng / ok / hit_y 全部互相一致。

    `hit_rate` 取 1.0 / 0.0 时是**确定性**的全命中 / 全不命中（不是概率抽样）——
    否则 v=32 时随机 tok_ng 会有 ~3% 的概率恰好撞对，"全错"场景就名不副实。
    """
    g = torch.Generator().manual_seed(seed)
    lp = F.log_softmax(torch.randn(n, v, generator=g), dim=-1)
    y = torch.randint(0, v, (n,), generator=g)
    ok = torch.rand(n, generator=g) < cover_rate
    tok_ng = torch.randint(0, v, (n,), generator=g)
    if hit_rate >= 1.0:
        tok_ng = y.clone()                       # 全命中
    elif hit_rate <= 0.0:
        tok_ng = (y + 1) % v                     # 全不命中（保证 ≠ y）
    else:
        hit_mask = torch.rand(n, generator=g) < hit_rate
        tok_ng = torch.where(hit_mask, y, tok_ng)
    tok_ng = torch.where(ok, tok_ng, torch.full_like(tok_ng, -1))
    p_y = lp.gather(1, y.unsqueeze(1)).squeeze(1).exp()
    p_max = lp.max(dim=-1).values.exp()
    hit_y = (tok_ng == y).float()
    return dict(lp=lp, y=y, ok=ok, tok_ng=tok_ng, p_y=p_y, p_max=p_max, hit_y=hit_y)


def reference_nll_covered(lp, y, tok_ng, ok, lam):
    """§12.3 的**朴素全矩阵**实现（先 clone 整个 (N,V) 分布再改），作为基准。

    刻意写得慢而直白：只在覆盖位置把 λ 混进 onehot，其余保持模型分布。
    """
    p = lp.exp().clone()
    idx = torch.nonzero(ok, as_tuple=True)[0]
    if len(idx):
        tgt = tok_ng[idx]
        p[idx] = p[idx] * (1 - lam)
        p[idx, tgt] += lam
    return -torch.log(p.clamp_min(1e-9)).gather(1, y.unsqueeze(1)).squeeze(1)


# ==========================================================================
# 1) nll_under_retrieval 的代数
# ==========================================================================
def test_zero_lambda_is_exactly_the_baseline(ng):
    """λ=0 必须逐位等于 -log p_y（无库基线）—— 否则所有 Δ 都被基线偏差污染。"""
    b = make_batch(ng)
    gate = torch.ones_like(b['p_y'])
    got = ng.nll_under_retrieval(b['p_y'], b['hit_y'], gate, 0.0)
    exp = -torch.log(b['p_y'].clamp_min(1e-9))
    assert torch.equal(got, exp)


def test_zero_gate_is_exactly_the_baseline(ng):
    """gate=0（未覆盖位置）同样必须退化为基线。"""
    b = make_batch(ng)
    got = ng.nll_under_retrieval(b['p_y'], b['hit_y'], torch.zeros_like(b['p_y']), 0.9)
    assert torch.equal(got, -torch.log(b['p_y'].clamp_min(1e-9)))


def test_analytic_value_when_retrieval_hits(ng):
    """命中时 p_new = p_y·(1−λ) + λ，解析值可手算。"""
    p_y = torch.tensor([0.25])
    hit_y = torch.tensor([1.0])
    gate = torch.tensor([1.0])
    for lam in (0.1, 0.3, 0.7):
        got = ng.nll_under_retrieval(p_y, hit_y, gate, lam)
        exp = -torch.log(torch.tensor(p_y.item() * (1 - lam) + lam))
        assert got.item() == pytest.approx(exp.item(), abs=1e-6)


def test_analytic_value_when_retrieval_misses(ng):
    """未命中时 p_new = p_y·(1−λ)—— NLL 变差，这正是"错检索有代价"。"""
    p_y = torch.tensor([0.25])
    hit_y = torch.tensor([0.0])
    gate = torch.tensor([1.0])
    for lam in (0.1, 0.5):
        got = ng.nll_under_retrieval(p_y, hit_y, gate, lam)
        exp = -torch.log(torch.tensor(p_y.item() * (1 - lam)))
        assert got.item() == pytest.approx(exp.item(), abs=1e-6)
        assert got.item() > -np.log(p_y.item())     # 比基线差


def test_larger_lambda_monotonically_helps_when_always_right(ng):
    p_y = torch.tensor([0.2])
    hit_y = torch.tensor([1.0])
    gate = torch.tensor([1.0])
    nlls = [ng.nll_under_retrieval(p_y, hit_y, gate, l).item() for l in (0.0, 0.2, 0.5, 1.0)]
    assert nlls == sorted(nlls, reverse=True), "全命中时 λ 越大 NLL 应越小"


def test_output_shape_and_finiteness(ng):
    b = make_batch(ng)
    gate = torch.ones_like(b['p_y'])
    got = ng.nll_under_retrieval(b['p_y'], b['hit_y'], gate, 0.5)
    assert got.shape == b['p_y'].shape and torch.isfinite(got).all()


# ==========================================================================
# 2) ★ A_ungated 必须与 §12.3 的朴素全矩阵实现等价
# ==========================================================================
def test_ungated_gate_matches_reference_formula(ng):
    """新实现的 O(N) 写法 vs §12.3 的 O(N·V) 朴素写法，必须逐位一致。

    这条是"改写没有偷换语义"的唯一保证。第一版改写就死在这里：
    它把门控从 `1−p_model[tgt]` 换成了 `1−p_y`，λ 大时 4.5e-2 的偏差。
    """
    b = make_batch(ng)
    gates = ng.gate_factors(b['lp'], b['p_y'], b['p_max'], b['tok_ng'], b['ok'])
    for lam in (0.0, 0.05, 0.2, 0.5):
        got = ng.nll_under_retrieval(b['p_y'], b['hit_y'], gates['A_ungated'], lam)
        exp = reference_nll_covered(b['lp'], b['y'], b['tok_ng'], b['ok'], lam)
        assert torch.allclose(got, exp, atol=1e-6), \
            f"λ={lam} 时与 §12.3 参考实现不符，最大差 {(got - exp).abs().max().item():.3e}"


def test_ungated_gate_is_exactly_the_coverage_mask(ng):
    b = make_batch(ng)
    gates = ng.gate_factors(b['lp'], b['p_y'], b['p_max'], b['tok_ng'], b['ok'])
    assert torch.equal(gates['A_ungated'], b['ok'].float())


# ==========================================================================
# 3) 门控因子本身
# ==========================================================================
def test_all_gates_are_in_unit_interval(ng):
    b = make_batch(ng)
    for name, g in ng.gate_factors(b['lp'], b['p_y'], b['p_max'], b['tok_ng'], b['ok']).items():
        assert (g >= 0).all() and (g <= 1).all(), f"{name} 超出 [0,1]"


def test_all_gates_vanish_where_slot_is_empty(ng):
    """槽为空的位置一律不注入 —— 这是所有口径共同的底线。"""
    b = make_batch(ng, cover_rate=0.3)
    assert (~b['ok']).any(), "测试数据里得有未覆盖位置"
    for name, g in ng.gate_factors(b['lp'], b['p_y'], b['p_max'], b['tok_ng'], b['ok']).items():
        assert torch.equal(g[~b['ok']], torch.zeros(int((~b['ok']).sum()))), f"{name} 没归零"


def test_three_named_gates_are_distinct(ng):
    """四个口径不能互相恒等，否则"分开报"没有意义。"""
    b = make_batch(ng)
    g = ng.gate_factors(b['lp'], b['p_y'], b['p_max'], b['tok_ng'], b['ok'])
    pairs = [(a, c) for i, a in enumerate(ng.GATE_NAMES) for c in ng.GATE_NAMES[i + 1:]]
    for a, c in pairs:
        assert not torch.allclose(g[a], g[c]), f"{a} 与 {c} 完全相同，口径没有区分度"


def test_p_tgt_uses_retrieved_token_not_target(ng):
    """`D_disagree` 依赖的 p_tgt 必须是**检索到**的 token 的概率。

    这正是第一版偷换的那个量（写成了 p_y）。检索命中时 p_tgt == p_y，
    未命中时必须不同 —— 用这个性质把它钉住。
    """
    b = make_batch(ng)
    p_tgt = b['lp'].gather(1, b['tok_ng'].clamp(min=0).unsqueeze(1)).squeeze(1).exp()
    hit = (b['tok_ng'] == b['y']) & b['ok']
    assert hit.any() and (~hit & b['ok']).any()
    assert torch.allclose(p_tgt[hit], b['p_y'][hit], atol=1e-7)
    assert not torch.allclose(p_tgt[~hit & b['ok']], b['p_y'][~hit & b['ok']])


def test_oracle_gate_shrinks_as_model_gets_confident(ng):
    """B_oracle = (1−p_y)⁺：模型越确信，注入越少。"""
    b = make_batch(ng)
    g = ng.gate_factors(b['lp'], b['p_y'], b['p_max'], b['tok_ng'], b['ok'])['B_oracle']
    conf = b['p_y'] > 0.5
    if conf.any() and (~conf).any():
        assert g[conf].max() <= g[~conf].max()


# ==========================================================================
# 4) ★ 形状回归：第一版就是在这里炸的
# ==========================================================================
def test_gate_factors_rejects_3d_input(ng):
    """模型输出是 (B,T,V)；忘了 reshape 就必须**立刻报错**，而不是跑到一半崩。

    第一版直接在 (B,T,V) 上调 gather，报的是
    `Index tensor must have the same number of dimensions as input tensor` ——
    错误信息完全没提"忘了展平"，排查成本很高。这里显式给出可读的报错。
    """
    b = make_batch(ng, n=8, v=16)
    lp3 = b['lp'].reshape(2, 4, 16)
    with pytest.raises(ValueError, match="二维"):
        ng.gate_factors(lp3, b['p_y'], b['p_max'], b['tok_ng'], b['ok'])


def test_flat_logprob_row_matches_manual_log_softmax(ng):
    """(B,T,V) → (N,V) 展平后，每行的 log_softmax 必须与手工计算一致。"""
    g = torch.Generator().manual_seed(3)
    lg = torch.randn(2, 4, 16, generator=g)
    flat = F.log_softmax(lg.reshape(-1, lg.size(-1)), dim=-1)
    for b in range(2):
        for t in range(4):
            row = lg[b, t]
            assert torch.allclose(flat[b * 4 + t], F.log_softmax(row, dim=-1), atol=1e-6)


# ==========================================================================
# 5) 端到端小场景：全对的库应当显著降 CE
# ==========================================================================
def test_perfect_retrieval_lowers_ce(ng):
    """构造"检索总是对"的极限场景：λ>0 时 CE 必须下降，且随 λ 单调。

    这条把各个零件的方向性串起来 —— 单看公式都对、但符号反了的情况，
    只有端到端的"效果方向"能抓到。
    """
    b = make_batch(ng, n=256, v=32, cover_rate=1.0, hit_rate=1.0, seed=5)
    gates = ng.gate_factors(b['lp'], b['p_y'], b['p_max'], b['tok_ng'], b['ok'])
    base = (-torch.log(b['p_y'].clamp_min(1e-9))).mean().item()
    ces = []
    for lam in (0.0, 0.1, 0.3, 0.5):
        nll = ng.nll_under_retrieval(b['p_y'], b['hit_y'], gates['A_ungated'], lam)
        ces.append(nll.mean().item())
    assert ces[0] == pytest.approx(base, abs=1e-9)
    assert ces == sorted(ces, reverse=True), f"λ 增大 CE 反而上升：{ces}"


def test_garbage_retrieval_raises_ce(ng):
    """反向对照：检索全错时 λ>0 必须让 CE **变差**。

    没有这条，"λ 越大越好"就可能来自实现里的符号错误（比如把该惩罚的项
    当成了奖励项）。正反两个方向都测，才能确定公式方向正确。
    """
    b = make_batch(ng, n=256, v=32, cover_rate=1.0, hit_rate=0.0, seed=6)
    assert b['hit_y'].sum().item() == 0
    gates = ng.gate_factors(b['lp'], b['p_y'], b['p_max'], b['tok_ng'], b['ok'])
    base = (-torch.log(b['p_y'].clamp_min(1e-9))).mean().item()
    ce = ng.nll_under_retrieval(b['p_y'], b['hit_y'], gates['A_ungated'], 0.5).mean().item()
    assert ce > base, "检索全错时 CE 应当变差"
