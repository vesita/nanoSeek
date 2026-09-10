"""`model/optimizer.py`（Muon / MuonAdamW）的测试。

## 为什么值得测
Muon 占单步时间的 **21.7%**，是第二大开销；而且它的更新规则里叠了三层东西
（动量 → Nesterov → Newton-Schulz 正交化 → 权重衰减插值），任何一层写反都不会
报错，只会让训练"看起来在跑但不收敛"。本项目还专门为它调过步数/系数
（`ns_steps=7, ns_aggressive=4`），这些配置项必须证明**真的接上了**。

## 测试策略
用手写的参考实现逐项对照，而不是只断言"参数变了"——后者对任何错误都成立。
"""

import pytest
import torch

from model.optimizer import Muon, MuonAdamW
from model.utils import zeropower_via_newtonschulz


def make_param(shape, seed=0, requires_grad=True):
    g = torch.Generator().manual_seed(seed)
    p = torch.nn.Parameter(torch.randn(*shape, generator=g))
    p.requires_grad = requires_grad
    return p


def set_grad(p, seed):
    g = torch.Generator().manual_seed(seed)
    p.grad = torch.randn(*p.shape, generator=g)


# ==========================================================================
# 1) 更新规则：与手写参考实现逐步对照
# ==========================================================================
def test_matrix_update_matches_manual_reference():
    """一个完整 step 必须等于手写的：
    动量 → Nesterov → NS 正交化 → 权重衰减插值 → 梯度下降。
    """
    lr, momentum, wd, steps, aggr = 1e-2, 0.9, 0.1, 7, 4
    p = make_param((16, 8), seed=1)
    set_grad(p, seed=2)
    p0, g0 = p.detach().clone(), p.grad.clone()

    opt = Muon([{'params': [p], 'weight_decay': wd}], lr=lr, momentum=momentum,
               nesterov=True, ns_steps=steps, ns_aggressive=aggr)
    opt.step()

    buf = g0.clone()                                   # 第一步步动量 = g
    upd = g0 + momentum * buf                          # Nesterov
    upd = zeropower_via_newtonschulz(upd, steps=steps, aggressive=aggr)
    upd = (1 - wd) * upd + wd * p0                     # 权重衰减插值
    expect = p0 - lr * upd
    assert torch.allclose(p.detach(), expect, atol=1e-6), \
        f"与手写参考最大差 {(p.detach() - expect).abs().max().item():.3e}"


def test_second_step_uses_accumulated_momentum():
    """第二步的动量必须是 m = momentum·m_prev + g，不能每步重置。"""
    lr, momentum = 0.1, 0.9
    p = make_param((8, 8), seed=3)
    set_grad(p, seed=4)
    opt = Muon([{'params': [p]}], lr=lr, momentum=momentum, nesterov=True,
               ns_steps=5, ns_aggressive=0)
    opt.step()
    g1 = p.grad.clone()
    buf1 = opt.state[p]['momentum_buffer'].clone()
    assert torch.allclose(buf1, g1), "第一步动量缓冲应等于梯度"

    set_grad(p, seed=5)
    g2 = p.grad.clone()
    opt.step()
    buf2 = opt.state[p]['momentum_buffer']
    assert torch.allclose(buf2, momentum * g1 + g2, atol=1e-6), "动量没有累积"


def test_nesterov_off_uses_buffer_directly():
    lr, momentum = 0.1, 0.9
    p = make_param((8, 8), seed=6)
    set_grad(p, seed=7)
    p0, g0 = p.detach().clone(), p.grad.clone()
    opt = Muon([{'params': [p]}], lr=lr, momentum=momentum, nesterov=False,
               ns_steps=5, ns_aggressive=0)
    opt.step()
    upd = zeropower_via_newtonschulz(g0, steps=5, aggressive=0)   # 非 Nesterov：直接用 buf
    assert torch.allclose(p.detach(), p0 - lr * upd, atol=1e-6)


def test_zero_weight_decay_is_pure_muon():
    lr = 1e-3
    p = make_param((12, 6), seed=8)
    set_grad(p, seed=9)
    p0, g0 = p.detach().clone(), p.grad.clone()
    opt = Muon([{'params': [p]}], lr=lr, momentum=0.9, nesterov=False,
               ns_steps=5, ns_aggressive=0)
    opt.step()
    # nesterov=False → buf = g；wd=0 → 不掺原参数
    expect = p0 - lr * zeropower_via_newtonschulz(g0, steps=5, aggressive=0)
    assert torch.allclose(p.detach(), expect, atol=1e-6)


def test_lr_scales_update_linearly():
    p1, p2 = make_param((8, 8), seed=10), make_param((8, 8), seed=10)
    set_grad(p1, 11); set_grad(p2, 11)
    p0 = p1.detach().clone()
    Muon([{'params': [p1]}], lr=1e-2, momentum=0.9, ns_steps=5).step()
    Muon([{'params': [p2]}], lr=2e-2, momentum=0.9, ns_steps=5).step()
    d1, d2 = (p1.detach() - p0), (p2.detach() - p0)
    # 容差按 float32 精度定：更新量 ~1e-2，而 p0 ~O(1)，做差会留下 ~1e-7 的舍入残差
    assert torch.allclose(d2, 2 * d1, atol=2e-6), \
        f"lr 与更新量不成正比，最大差 {(d2 - 2 * d1).abs().max().item():.3e}"


# ==========================================================================
# 2) 1D 参数：不做正交化，退化为纯动量
# ==========================================================================
def test_1d_param_falls_back_to_plain_momentum():
    """bias / norm 权重没有"方向"可正交化，必须退化为纯动量更新。

    如果实现把它们也丢进 NS，会直接触发 `assert G.ndim == 2`。
    """
    lr = 0.1
    p = make_param((8,), seed=12)
    set_grad(p, 13)
    p0, g0 = p.detach().clone(), p.grad.clone()
    Muon([{'params': [p]}], lr=lr, momentum=0.9, nesterov=True, ns_steps=5).step()
    upd = g0 + 0.9 * g0                    # Nesterov，无正交化
    assert torch.allclose(p.detach(), p0 - lr * upd, atol=1e-7)


def test_3d_param_is_rejected_by_ns_not_silently_ignored():
    """>=3D 参数走这个优化器会触发 NS 的 ndim 断言 —— 钉住这个失败模式。

    `configure_optimizers` 正是靠这条把 3D 参数（kv 记忆的 mem_persist）
    分给 AdamW，而不是让 Muon 崩掉。
    """
    p = make_param((2, 4, 4), seed=14)
    set_grad(p, 15)
    opt = Muon([{'params': [p]}], lr=0.1, ns_steps=5)
    with pytest.raises(AssertionError):
        opt.step()


# ==========================================================================
# 3) split_heads：按注意力头分块正交化
# ==========================================================================
def test_split_heads_routes_to_split_orthogonalization():
    """`split_heads` 命中的参数必须走 split 版，结果与整块不同。

    本项目出过一次"空测试"事故（两个 A/B 臂其实完全一样），
    所以凡"开关型"配置都要有一条'它确实改变了行为'的测试。
    """
    from model.utils import zeropower_via_newtonschulz_split
    lr, steps = 1.0, 10
    p = make_param((32, 16), seed=16)          # 2 头 × 16
    set_grad(p, 17)
    g0 = p.grad.clone()
    p0 = p.detach().clone()

    opt = Muon([{'params': [p]}], lr=lr, momentum=0.9, nesterov=False,
               ns_steps=steps, split_heads={id(p): (2, True)})
    opt.step()
    expect_split = p0 - lr * zeropower_via_newtonschulz_split(g0, steps=steps,
                                                             n_heads=2, head_first=True)
    expect_whole = p0 - lr * zeropower_via_newtonschulz(g0, steps=steps)
    assert torch.allclose(p.detach(), expect_split, atol=1e-6), "没走 split 版"
    assert not torch.allclose(expect_split, expect_whole), "split 与整块必须不同"


# ==========================================================================
# 4) ns_steps / ns_aggressive 配置项必须真的接上
# ==========================================================================
def test_ns_steps_and_aggressive_are_wired_through():
    """配置里写 `ns_steps=7, ns_aggressive=4` 时，实际调用必须用这两个值。

    做法：注入一个**记账用**的 orthogonalization_fn，把收到的 kwargs 记下来。
    这是唯一能证明"配置真的传到了 NS"的方法 —— 只比较参数变化无法区分。
    """
    seen = {}

    def spy(g, steps=10, eps=1e-7, aggressive=0):
        seen.update(steps=steps, aggressive=aggressive)
        return g

    p = make_param((8, 8), seed=18)
    set_grad(p, 19)
    opt = Muon([{'params': [p]}], lr=0.1, ns_steps=7, ns_aggressive=4,
               orthogonalization_fn=spy)
    opt.step()
    assert seen == {'steps': 7, 'aggressive': 4}, f"实际收到 {seen}"


def test_default_aggressive_is_zero_for_backward_compat():
    """不传 `ns_aggressive` 时必须默认 0（= 与旧 checkpoint 逐位兼容）。"""
    seen = {}

    def spy(g, steps=10, eps=1e-7, aggressive=0):
        seen['aggressive'] = aggressive
        return g

    p = make_param((8, 8), seed=20)
    set_grad(p, 21)
    Muon([{'params': [p]}], lr=0.1, orthogonalization_fn=spy).step()
    assert seen['aggressive'] == 0


# ==========================================================================
# 5) 状态管理：checkpoint 续训
# ==========================================================================
def test_state_dict_roundtrip_preserves_trajectory():
    """存/读 optimizer state 后，后续更新轨迹必须一致（= 续训可复现）。

    必须传 `deepcopy`：`torch.optim.Optimizer.load_state_dict` 在 dtype/device
    已匹配时**不拷贝张量**，会把源 state_dict 里的 buffer 直接挂进目标优化器
    （见 `test_load_state_dict_aliases_source_tensors`）。真实续训路径里
    state 来自 `torch.load`，源优化器已销毁，所以不存在别名问题；
    但测试里两个优化器同时活着，不 deepcopy 就会互相踩。
    """
    import copy
    p1, p2 = make_param((8, 8), seed=22), make_param((8, 8), seed=22)
    set_grad(p1, 23); set_grad(p2, 23)
    o1 = Muon([{'params': [p1]}], lr=0.1, momentum=0.9, ns_steps=5, ns_aggressive=2)
    o2 = Muon([{'params': [p2]}], lr=0.1, momentum=0.9, ns_steps=5, ns_aggressive=2)
    o1.step()
    o2.step()
    o2.load_state_dict(copy.deepcopy(o1.state_dict()))
    set_grad(p1, 24); set_grad(p2, 24)
    o1.step(); o2.step()
    assert torch.equal(p1.detach(), p2.detach()), "恢复 state 后轨迹分叉"


def test_load_state_dict_aliases_source_tensors():
    """★ 记录一个 PyTorch 陷阱：`load_state_dict` 不拷贝已匹配 dtype 的张量。

    现象：`o2.load_state_dict(o1.state_dict())` 之后，两个优化器的
    `momentum_buffer` 是**同一个张量**；随后 `o1.step()` 原地更新缓冲，
    `o2` 的状态被一起改掉。

    为什么重要：本项目用异步 checkpoint 线程保存训练状态。只要有人写出
    "把在线优化器的 state_dict 交给另一个优化器"这类代码，就会踩到；
    而它不报错，只让两条训练轨迹悄悄分叉。钉住这个行为，避免以后误判成
    Muon 的实现 bug（我第一次就是这么误判的，差 2e-3 以为是逻辑错）。
    """
    import copy
    p1, p2 = make_param((8, 8), seed=40), make_param((8, 8), seed=40)
    set_grad(p1, 41); set_grad(p2, 41)
    o1 = Muon([{'params': [p1]}], lr=0.1, ns_steps=5)
    o2 = Muon([{'params': [p2]}], lr=0.1, ns_steps=5)
    o1.step()
    o2.load_state_dict(o1.state_dict())            # 不 deepcopy
    b1, b2 = o1.state[p1]['momentum_buffer'], o2.state[p2]['momentum_buffer']
    assert b1.data_ptr() == b2.data_ptr(), \
        "PyTorch 行为变了：load_state_dict 现在会拷贝 —— 可以删掉这条测试了"

    o2.load_state_dict(copy.deepcopy(o1.state_dict()))   # deepcopy 后不再别名
    b2 = o2.state[p2]['momentum_buffer']
    assert b1.data_ptr() != b2.data_ptr()
    assert torch.equal(b1, b2), "值仍然要一致"


def test_skips_params_without_grad():
    """grad 为 None 的参数必须跳过（冻结层不会产生梯度）。"""
    p = make_param((8, 8), seed=25)
    p.grad = None
    p0 = p.detach().clone()
    opt = Muon([{'params': [p]}], lr=0.1, ns_steps=5)
    opt.step()
    assert torch.equal(p.detach(), p0)
    assert 'momentum_buffer' not in opt.state.get(p, {})


# ==========================================================================
# 6) MuonAdamW 组合优化器
# ==========================================================================
def test_muonadamw_dispatches_to_both():
    pm, pa = make_param((8, 8), seed=26), make_param((8,), seed=27)
    set_grad(pm, 28); set_grad(pa, 29)
    pm0, pa0 = pm.detach().clone(), pa.detach().clone()
    muon = Muon([{'params': [pm]}], lr=0.1, ns_steps=5)
    adamw = torch.optim.AdamW([{'params': [pa]}], lr=0.1)
    opt = MuonAdamW(muon, adamw)
    opt.step()
    assert not torch.equal(pm.detach(), pm0), "Muon 组没被更新"
    assert not torch.equal(pa.detach(), pa0), "AdamW 组没被更新"


def test_muonadamw_param_groups_is_concatenation():
    pm, pa = make_param((4, 4)), make_param((4,))
    muon = Muon([{'params': [pm]}], lr=0.1)
    adamw = torch.optim.AdamW([{'params': [pa]}], lr=0.1)
    opt = MuonAdamW(muon, adamw)
    assert opt.param_groups == muon.param_groups + adamw.param_groups


def test_muonadamw_zero_grad():
    pm, pa = make_param((4, 4)), make_param((4,))
    set_grad(pm, 1); set_grad(pa, 2)
    opt = MuonAdamW(Muon([{'params': [pm]}], lr=0.1),
                    torch.optim.AdamW([{'params': [pa]}], lr=0.1))
    opt.zero_grad()
    assert pm.grad is None and pa.grad is None


def test_muonadamw_state_dict_roundtrip():
    """组合优化器的 state 往返（同样要 deepcopy，理由见 Muon 版测试）。"""
    import copy
    pm1, pm2 = make_param((4, 4), seed=30), make_param((4, 4), seed=30)
    pa1, pa2 = make_param((4,), seed=31), make_param((4,), seed=31)
    set_grad(pm1, 32); set_grad(pm2, 32); set_grad(pa1, 33); set_grad(pa2, 33)
    o1 = MuonAdamW(Muon([{'params': [pm1]}], lr=0.1, ns_steps=5),
                   torch.optim.AdamW([{'params': [pa1]}], lr=0.1))
    o2 = MuonAdamW(Muon([{'params': [pm2]}], lr=0.1, ns_steps=5),
                   torch.optim.AdamW([{'params': [pa2]}], lr=0.1))
    o1.step(); o2.step()
    o2.load_state_dict(copy.deepcopy(o1.state_dict()))
    set_grad(pm1, 34); set_grad(pm2, 34); set_grad(pa1, 35); set_grad(pa2, 35)
    o1.step(); o2.step()
    assert torch.equal(pm1.detach(), pm2.detach())
    assert torch.equal(pa1.detach(), pa2.detach())


def test_muonadamw_state_dict_has_both_halves():
    pm, pa = make_param((4, 4)), make_param((4,))
    opt = MuonAdamW(Muon([{'params': [pm]}], lr=0.1),
                    torch.optim.AdamW([{'params': [pa]}], lr=0.1))
    sd = opt.state_dict()
    assert set(sd.keys()) == {'muon', 'adamw'}
    with pytest.raises(KeyError):
        opt.load_state_dict({'muon': sd['muon']})      # 缺一半必须报错，不能静默


# ==========================================================================
# 7) 与真实模型的接线
# ==========================================================================
def test_tiny_model_split_heads_is_populated(tiny_model):
    """`muon_split=True` 的模型必须真的产出非空 split_heads。

    否则 `--muon_split` 就是摆设：配置打开了、代码里一个头都没分。
    """
    opt = tiny_model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), 'cpu')
    assert isinstance(opt, MuonAdamW)
    assert len(opt.muon.split_heads) > 0, "muon_split 打开却没有分任何头"
    for spec in opt.muon.split_heads.values():
        n_heads, head_first = spec
        assert isinstance(n_heads, int) and n_heads > 0
        assert isinstance(head_first, bool)


def test_tiny_model_muon_param_groups_are_nonempty(tiny_model):
    opt = tiny_model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), 'cpu')
    assert len(opt.muon.param_groups[0]['params']) > 0
    # 每个 Muon 参数都必须是 2D（1D/3D 必须被分给 AdamW，否则 step() 会崩）
    for p in opt.muon.param_groups[0]['params']:
        assert p.dim() == 2, f"Muon 组里混进了 {p.dim()}D 参数"
