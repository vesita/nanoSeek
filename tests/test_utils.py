"""`model/utils.py` 的测试：LSE 残差 / RMSNorm / RoPE / Sinkhorn / Newton-Schulz。

## 为什么这些值得测
全是**数值原语**：写错了不会报错，只会让训练慢慢变差或悄悄失稳。
其中 Newton-Schulz 正交化最危险 —— 它是 Muon 的核心，占单步 21.7% 时间，
本项目还专门为它调过系数（`muon_ns_steps=7, muon_ns_aggressive=4`）。
阈值全部按**实测值**设定（见各测试的 docstring），不凭直觉写。
"""
import math

import pytest
import torch

from model.utils import (
    RMSNorm,
    apply_rotary_pos_emb,
    logsumexp_residual,
    precompute_rope_freqs,
    rotate_half,
    sinkhorn_knopp,
    zeropower_via_newtonschulz,
    zeropower_via_newtonschulz_split,
)


# ==========================================================================
# 工具
# ==========================================================================
def orth_residual(X):
    """正交化质量：在**窄侧**算 ‖QᵀQ − I‖_F / √n。

    ⚠ 必须在窄侧算。第一次测量时我在高矩阵上用了 QᵀQ（宽侧），量到的是
    "行正交"而不是我们要的列正交，得出了错误结论。这是本项目真实踩过的坑。
    """
    X = X.float()
    if X.size(0) >= X.size(1):
        G, n = X.T @ X, X.size(1)
    else:
        G, n = X @ X.T, X.size(0)
    return ((G - torch.eye(n, dtype=G.dtype)) .norm() / n ** 0.5).item()


def ill_conditioned(rows=64, cols=32, rank=8, noise=0.05, seed=0):
    """模拟真实动量矩阵：低秩（外积）+ 小噪声 —— Muon 实际遇到的就是这种。"""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(rows, rank, generator=g) @ torch.randn(rank, cols, generator=g)
    return A + noise * torch.randn(rows, cols, generator=g)


# ==========================================================================
# 1) LSE 残差
# ==========================================================================
def test_lse_residual_equals_logaddexp():
    """LSE(x,y) 的数学定义就是 logaddexp(x,y)（实现只是做了 max 平移保稳定）。"""
    torch.manual_seed(0)
    x, y = torch.randn(4, 5, 8), torch.randn(4, 5, 8)
    assert torch.allclose(logsumexp_residual(x, y), torch.logaddexp(x, y), atol=1e-6)


def test_lse_residual_is_bounded_by_max_plus_log2():
    """核心卖点：有界收缩。max(x,y) ≤ LSE ≤ max(x,y)+ln2。

    这是它相对「线性相加」的全部意义 —— 结果永远压在最强的分支附近，
    不会随层数无限涨大（SwiGLU 极端值就是被线性相加逐层放大的）。
    """
    torch.manual_seed(1)
    x, y = torch.randn(1000) * 20, torch.randn(1000) * 20
    out = logsumexp_residual(x, y)
    mx = torch.maximum(x, y)
    assert (out >= mx - 1e-5).all()
    assert (out <= mx + math.log(2) + 1e-5).all()


def test_lse_residual_symmetric_and_idempotent():
    torch.manual_seed(2)
    x, y = torch.randn(64), torch.randn(64)
    assert torch.allclose(logsumexp_residual(x, y), logsumexp_residual(y, x), atol=1e-6)
    # f(x, x) = x + ln2
    assert torch.allclose(logsumexp_residual(x, x), x + math.log(2), atol=1e-6)


def test_lse_residual_numerically_stable_at_extremes():
    """±1e4 不能溢出成 inf/NaN —— 这正是它替代朴素 log(exp+exp) 的理由。"""
    x = torch.tensor([1e4, -1e4, 0.0, 1e30])
    y = torch.tensor([-1e4, 1e4, 0.0, -1e30])
    out = logsumexp_residual(x, y)
    assert torch.isfinite(out).all(), out
    assert out[0] == pytest.approx(1e4, rel=1e-6)


def test_lse_residual_gradient_is_finite_at_extremes():
    x = torch.tensor([1e4], requires_grad=True)
    y = torch.tensor([-1e4], requires_grad=True)
    logsumexp_residual(x, y).backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(y.grad).all()
    assert x.grad.item() == pytest.approx(1.0)     # 只有占优分支拿梯度
    assert y.grad.item() == pytest.approx(0.0, abs=1e-6)


# ==========================================================================
# 2) RMSNorm
# ==========================================================================
def test_rmsnorm_output_has_unit_rms():
    torch.manual_seed(0)
    x = torch.randn(8, 16, 32)
    out = RMSNorm(32)(x)
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


def test_rmsnorm_is_scale_invariant():
    """RMSNorm 只按 RMS 缩放 → 输入整体乘常数，输出不变（eps 允许微小偏差）。"""
    torch.manual_seed(1)
    x = torch.randn(4, 8, 16) * 100
    n = RMSNorm(16)
    assert torch.allclose(n(x * 7.0), n(x), atol=1e-3)


def test_rmsnorm_weight_scales_linearly():
    x = torch.randn(2, 3, 8)
    n = RMSNorm(8)
    with torch.no_grad():
        n.weight.mul_(3.0)
    assert torch.allclose(n(x), 3.0 * RMSNorm(8)(x), atol=1e-5)


def test_rmsnorm_zero_input_is_zero_not_nan():
    n = RMSNorm(8)
    out = n(torch.zeros(2, 3, 8))
    assert torch.isfinite(out).all()
    assert torch.equal(out, torch.zeros_like(out))


def test_rmsnorm_preserves_shape_and_dtype():
    """权重与输入同 dtype 时（真实训练的情形），输出 dtype 不变。"""
    for dt in (torch.float32, torch.bfloat16):
        n = RMSNorm(8).to(dt)
        out = n(torch.randn(2, 5, 8, dtype=dt))
        assert out.shape == (2, 5, 8) and out.dtype == dt


def test_rmsnorm_upcasts_when_weight_dtype_differs():
    """陷阱说明（不是 bug）：`x * self.weight` 走类型提升，权重 fp32 + 输入 bf16 → 输出 fp32。

    半精度训练时若忘了把 module 一起 `.to(dtype)`，前向会悄悄多出 fp32 计算。
    钉住这个行为，免得以后误以为是随机 bug。
    """
    n = RMSNorm(8)                                   # 默认 fp32 权重
    out = n(torch.randn(2, 5, 8, dtype=torch.bfloat16))
    assert out.dtype == torch.float32


# ==========================================================================
# 3) RoPE
# ==========================================================================
def test_rotate_half_contract():
    """rotate_half([a, b]) = [-b, a]（前后半段互换并给后半段取负）。"""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert torch.equal(rotate_half(x), torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))


def test_precompute_rope_freqs_shapes_and_origin():
    cos, sin = precompute_rope_freqs(head_dim=8, seq_len=16)
    assert cos.shape == (16, 8) and sin.shape == (16, 8)
    # t=0 时旋转角为 0 → cos=1, sin=0（位置 0 的向量不该被旋转）
    assert torch.allclose(cos[0], torch.ones(8), atol=1e-6)
    assert torch.allclose(sin[0], torch.zeros(8), atol=1e-6)
    # 每个二维平面内 cos²+sin²=1
    assert torch.allclose(cos ** 2 + sin ** 2, torch.ones(16, 8), atol=1e-5)


def test_rope_dot_product_depends_only_on_relative_position():
    """RoPE 的**定义性质**：<f(q,m), f(k,n)> 只依赖 n−m。

    即 S[m,n] 必须是 Toeplitz 矩阵：S[m,n] == S[0,n−m]。

    ⚠ 前提：q/k 必须是**同一个向量**复制到各位置。若每个位置用各自的随机向量
    （直觉上更"自然"的写法），点积里混进了不同向量的差异，性质不成立 ——
    第一版测试就是这么写错的（代码没问题，是期望写错了）。

    这条能一次性抓住频率表算错、rotate_half 配对错、只旋转 q 没旋转 k 等
    几乎所有 RoPE 实现错误。
    """
    torch.manual_seed(0)
    T, D = 16, 8
    qv, kv = torch.randn(1, 1, 1, D), torch.randn(1, 1, 1, D)
    q = qv.expand(1, T, 1, D).contiguous()     # 同一向量铺满所有位置
    k = kv.expand(1, T, 1, D).contiguous()
    cos, sin = precompute_rope_freqs(D, T)
    qr, kr = apply_rotary_pos_emb(q, k, cos, sin)
    S = torch.einsum('bthd,bshd->bts', qr, kr)[0]

    # (a) 对角线恒定 = <q,k>：位置 m 的 q 与位置 m 的 k 夹角不含相对位移
    diag = torch.diagonal(S)
    assert torch.allclose(diag, diag[0].expand(T), atol=1e-5), f"对角线不恒定：{diag.tolist()}"

    # (b) Toeplitz：S[m,n] == S[0,n−m]
    for m in range(T):
        for n in range(m, T):
            assert S[m, n].item() == pytest.approx(S[0, n - m].item(), abs=1e-5), \
                f"相对位置性质在 (m={m}, n={n}) 处破了"


def test_rope_rotation_matrix_is_orthogonal_with_det_one():
    """逐位置重建旋转矩阵 R_t，确认 RᵀR=I 且 det=+1（是真旋转，不是反射）。"""
    T, D = 6, 8
    cos, sin = precompute_rope_freqs(D, T)
    for t in range(T):
        R = torch.zeros(D, D)
        for j in range(D):
            e = torch.zeros(1, T, 1, D)
            e[0, t, 0, j] = 1.0
            er, _ = apply_rotary_pos_emb(e, e.clone(), cos, sin)
            R[:, j] = er[0, t, 0]
        assert torch.allclose(R.T @ R, torch.eye(D), atol=1e-5), f"t={t} 不是正交矩阵"
        assert torch.linalg.det(R).item() == pytest.approx(1.0, abs=1e-4), f"t={t} det≠+1"


def test_rope_preserves_norm_of_single_vector():
    """旋转是等距变换：单向量旋转后 L2 范数不变。"""
    torch.manual_seed(1)
    T, D = 8, 8
    q = torch.randn(1, T, 1, D)
    cos, sin = precompute_rope_freqs(D, T)
    qr, _ = apply_rotary_pos_emb(q, q.clone(), cos, sin)
    assert torch.allclose(qr.norm(dim=-1), q.norm(dim=-1), atol=1e-5)


def test_rope_preserves_shape():
    T, D = 12, 16
    q = torch.randn(2, T, 3, D)
    cos, sin = precompute_rope_freqs(D, T)
    qr, kr = apply_rotary_pos_emb(q, q.clone(), cos, sin)
    assert qr.shape == q.shape and kr.shape == q.shape


# ==========================================================================
# 4) Sinkhorn-Knopp
# ==========================================================================
def test_sinkhorn_rows_and_cols_sum_to_one():
    torch.manual_seed(0)
    M = sinkhorn_knopp(torch.randn(6, 6), n_iter=100)
    assert torch.allclose(M.sum(-1), torch.ones(6), atol=1e-6)
    assert torch.allclose(M.sum(-2), torch.ones(6), atol=1e-6)
    assert (M > 0).all(), "双重随机矩阵要求元素严格正"


def test_sinkhorn_is_spectral_norm_bounded_by_one():
    """mHC 的稳定性保证：B 是双重随机矩阵 ⇒ ‖B‖₂ ≤ 1 ⇒ 残差变换非扩张。"""
    torch.manual_seed(1)
    for _ in range(5):
        M = sinkhorn_knopp(torch.randn(8, 8), n_iter=200)
        assert torch.linalg.matrix_norm(M, ord=2).item() <= 1.0 + 1e-5


def test_sinkhorn_fixes_doubly_stochastic_input():
    """已经是双重随机的矩阵取 log 再过 Sinkhorn，应当（近似）还原。"""
    n = 5
    P = torch.full((n, n), 1.0 / n)
    assert torch.allclose(sinkhorn_knopp(P.log(), n_iter=50), P, atol=1e-5)


# ==========================================================================
# 5) Newton-Schulz 正交化
# ==========================================================================
def _classic_reference(G, steps=10, eps=1e-7):
    """`zeropower_via_newtonschulz(aggressive=0)` 的手写基准实现。"""
    X = G.float()
    was_tall = X.size(0) > X.size(1)
    if was_tall:
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        XX = X @ X.T
        X = 2.0 * X - 1.5 * (XX @ X) + 0.5 * (XX @ XX @ X)
    return X.T if was_tall else X


def test_ns_aggressive_zero_is_bitwise_identical_to_classic():
    """**向后兼容契约**：`aggressive=0` 必须与旧实现逐位一致。

    项目里已有大量 checkpoint 是在旧实现下训出来的；只要这个契约破了，
    "继续训同一个模型"就不再可比。所以用 torch.equal 而不是 allclose。
    """
    torch.manual_seed(0)
    for shape in [(32, 32), (64, 32), (32, 64), (16, 8)]:
        G = torch.randn(*shape)
        got = zeropower_via_newtonschulz(G, steps=10, aggressive=0)
        exp = _classic_reference(G, steps=10).to(G.dtype)
        assert torch.equal(got, exp), f"shape={shape} 与经典实现不逐位相等"


def test_ns_does_not_mutate_input():
    """绝不能原地改 G —— 优化器传进来的是 `p.grad`，改了就等于污染梯度。"""
    torch.manual_seed(0)
    G = torch.randn(32, 32)
    G0 = G.clone()
    zeropower_via_newtonschulz(G, steps=5, aggressive=2)
    assert torch.equal(G, G0), "输入被就地修改了"


@pytest.mark.parametrize("shape", [(64, 32), (32, 64), (32, 32), (1, 32), (32, 1)])
def test_ns_preserves_shape(shape):
    """高/宽矩阵都要能原样返回 —— 历史 bug 是转置后忘了转回。"""
    G = torch.randn(*shape)
    assert zeropower_via_newtonschulz(G).shape == shape


@pytest.mark.parametrize("dt", [torch.float32, torch.bfloat16, torch.float16])
def test_ns_preserves_dtype(dt):
    assert zeropower_via_newtonschulz(torch.randn(16, 16, dtype=dt)).dtype == dt


def test_ns_zero_matrix_is_finite():
    """全零矩阵不能产出 NaN（除 eps 保护）。"""
    out = zeropower_via_newtonschulz(torch.zeros(8, 8))
    assert torch.isfinite(out).all()


def test_ns_classic10_is_good_on_well_conditioned():
    """实测：良态方阵 median 残差 3.4e-5 → 阈值 1e-3 有 ~30× 余量。"""
    residuals = [orth_residual(zeropower_via_newtonschulz(torch.randn(32, 32),
                                                          steps=10, aggressive=0))
                 for _ in range(20)]
    assert torch.tensor(residuals).median().item() < 1e-3


def test_ns_cutting_steps_is_catastrophic():
    """**别砍步数**：经典 5 步实测残差 0.27（良态）/ 0.86（病态）。

    这条钉住"当初想把 ns_steps 砍到 5 省时间"这个已被否决的方案，
    防止以后有人再试一次。
    """
    torch.manual_seed(0)
    r5 = torch.tensor([orth_residual(zeropower_via_newtonschulz(torch.randn(32, 32),
                                                                steps=5, aggressive=0))
                       for _ in range(20)]).median().item()
    assert r5 > 0.1, f"经典 5 步残差只有 {r5:.3e}？与实测的 0.27 差太远，重新核对"


def test_ns_mixed_4_aggressive_3_classic_is_at_least_as_good_as_classic10():
    """当前配置（steps=7, aggressive=4）的正当性依据。

    实测（良态 1.6e-6 vs 3.4e-5 / 病态 4.3e-2 vs 6.1e-2）：**7 步不但省 30%
    计算，正交化质量还更好**。如果哪天这个不等式反过来，说明实现或系数被改坏了。
    """
    torch.manual_seed(0)
    well = [torch.randn(32, 32) for _ in range(20)]
    ill = [ill_conditioned(seed=i) for i in range(20)]
    for name, mats in (("良态方阵", well), ("病态动量", ill)):
        r_mixed = torch.tensor([orth_residual(zeropower_via_newtonschulz(
            m, steps=7, aggressive=4)) for m in mats]).median().item()
        r_c10 = torch.tensor([orth_residual(zeropower_via_newtonschulz(
            m, steps=10, aggressive=0)) for m in mats]).median().item()
        assert r_mixed <= r_c10, f"{name}: 混相7步 {r_mixed:.3e} 反而比经典10步 {r_c10:.3e} 差"


def test_ns_aggressive_count_changes_result():
    """`aggressive` 必须真的起作用（否则配置项是摆设）。"""
    torch.manual_seed(0)
    G = torch.randn(32, 32)
    a = zeropower_via_newtonschulz(G, steps=7, aggressive=0)
    b = zeropower_via_newtonschulz(G, steps=7, aggressive=4)
    assert not torch.equal(a, b)


def test_ns_aggressive_is_clamped_to_step_range():
    """aggressive 越界要被夹住，而不是索引越界或跑满全程。"""
    torch.manual_seed(0)
    G = torch.randn(32, 32)
    all_aggr = zeropower_via_newtonschulz(G, steps=5, aggressive=5)
    assert torch.equal(zeropower_via_newtonschulz(G, steps=5, aggressive=999), all_aggr)
    classic = zeropower_via_newtonschulz(G, steps=5, aggressive=0)
    assert torch.equal(zeropower_via_newtonschulz(G, steps=5, aggressive=-5), classic)
    # aggressive=steps 与全经典必须不同（否则说明夹取把激进全吃掉了）
    assert not torch.equal(all_aggr, classic)


# ==========================================================================
# 6) Newton-Schulz（Muon Split，按注意力头分块）
# ==========================================================================
def test_ns_split_preserves_shape_both_layouts():
    for shape, head_first in [((64, 48), True), ((48, 64), False)]:
        n_heads = 4
        G = torch.randn(*shape)
        out = zeropower_via_newtonschulz_split(G, steps=10, n_heads=n_heads,
                                               head_first=head_first)
        assert out.shape == shape


def test_ns_split_each_head_is_orthogonal():
    """分头正交化的定义：**每个头**的切片各自正交，而不是整块。"""
    torch.manual_seed(0)
    n_heads, d, n = 4, 16, 48
    G = torch.randn(n_heads * d, n)
    out = zeropower_via_newtonschulz_split(G, steps=10, n_heads=n_heads, head_first=True)
    for h in range(n_heads):
        r = orth_residual(out.view(n_heads, d, n)[h])
        assert r < 1e-4, f"第 {h} 头残差 {r:.3e} 太大"


def test_ns_split_differs_from_whole_matrix():
    """分头与整块必须给出不同结果，否则 muon_split 这个开关是空的。

    本项目已有一次"空测试"事故（gradient_checkpointing 被上游短路，两个 A/B 臂
    其实完全一样），所以凡是"开关型"配置都要有一条'它确实改变了行为'的测试。
    """
    torch.manual_seed(0)
    G = torch.randn(64, 48)
    whole = zeropower_via_newtonschulz(G, steps=10)
    split = zeropower_via_newtonschulz_split(G, steps=10, n_heads=4, head_first=True)
    assert not torch.equal(whole, split)


def test_ns_split_aggressive_zero_matches_classic_split():
    """split 版本的 aggressive=0 同样要保持向后兼容。"""
    torch.manual_seed(0)
    G = torch.randn(64, 48)
    out = zeropower_via_newtonschulz_split(G, steps=10, n_heads=4, head_first=True, aggressive=0)
    # 头内等价于对每个头单独跑经典 NS
    ref = torch.stack([
        _classic_reference(G.view(4, 16, 48)[h], steps=10).to(G.dtype)
        for h in range(4)
    ]).reshape(G.shape)
    assert torch.allclose(out, ref, atol=1e-6)


def test_ns_split_dtype_and_input_preserved():
    torch.manual_seed(0)
    G = torch.randn(64, 48, dtype=torch.bfloat16)
    G0 = G.clone()
    out = zeropower_via_newtonschulz_split(G, steps=10, n_heads=4)
    assert out.dtype == torch.bfloat16
    assert torch.equal(G, G0), "split 版也不能就地修改输入"
