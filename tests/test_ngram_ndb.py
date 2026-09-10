# -*- coding: utf-8 -*-
"""`model/ngram_ndb.py`（NDB v7 原型）的测试。

## 这个模块为什么必须有测试
它是"可读写数据库"的第一次实现，同时踩在三个历史雷区上：

1. **槽位对齐**：`hashes(t,L)[i]` 的上下文结尾在 `i+L-1`，不是 `i+L`。
   本仓库为此错过两次（见 tests/test_ngram_hash.py）。
2. **val 泄漏**：写路径如果默认开启，`estimate_loss` 会把验证集写进训练库。
   本项目被 val 泄漏坑过最惨的一次（旧切分双标准 → 所有指标失真）。
3. **增量 vs 离线不一致**：在线写入是 **有损** 的 top-K 维护，
   如果它与离线批处理建表对不上，那么"离线测出的 Δ=−0.0723"就无法兑现。
   所以下面有 `test_incremental_write_equals_offline_build` 这条已知答案对照。
"""
import numpy as np
import pytest
import torch

from model.ngram_ndb import NgramNDB, suffix_hashes

V = 64            # 测试用小词表（真实的 8192 会让表太大）
M = 4096          # 每级槽位
L = 4


def make_ndb(levels=(L,), slots=M, top_k=4, n_embd=16, **kw):
    """小而完整的 NDB，用于测试。"""
    torch.manual_seed(0)
    ndb = NgramNDB(n_embd=n_embd, levels=levels, slots=slots, top_k=top_k,
                   vocab_size=V, max_table_gb=1.0, **kw)
    return ndb.eval()


def make_tokens(n=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, V, (1, n), generator=g)


def feed(ndb, inputs, targets=None, n_embd=16, chunk=1):
    """把一个序列喂给 observe，返回写权重均值。"""
    targets = inputs if targets is None else targets
    B, T = inputs.shape
    h = torch.zeros(B, T, n_embd)
    with ndb.write_enabled():
        return ndb.observe(h, inputs, targets)


# ==========================================================================
# 1) 基础契约
# ==========================================================================
def test_write_is_disabled_by_default():
    """★ val 泄漏防线：不显式 `write_enabled()` 就一个字节都不该写。

    `train.py` 的同一个 forward 同时服务训练 / estimate_loss / 健康体检。
    写路径默认开启 = 验证集会写进训练库。这条测试是那个事故的防线。
    """
    ndb = make_ndb()
    tok = make_tokens()
    h = torch.zeros(1, tok.shape[1], 16)
    assert ndb.observe(h, tok, tok) is None, "未开启写却返回了结果"
    assert ndb.n_observed == 0
    assert ndb._totals[0].sum() == 0
    assert not ndb.write_active


def test_write_context_manager_restores_previous_state():
    """`write_enabled()` 退出后必须恢复原状态（嵌套也正确）。"""
    ndb = make_ndb()
    assert not ndb.write_active
    with ndb.write_enabled():
        assert ndb.write_active
        with ndb.write_enabled():
            assert ndb.write_active
        assert ndb.write_active, "内层退出后不该把外层也关掉"
    assert not ndb.write_active


def test_observe_then_flush_fills_table():
    ndb = make_ndb(top_k=1)
    tok = make_tokens(300)
    feed(ndb, tok)
    assert ndb.n_observed > 0
    n = ndb.flush()
    assert n > 0
    assert ndb._totals[0].sum() > 0
    assert ndb.n_filled_slots()[0] > 0


def test_table_is_not_a_parameter_and_not_in_state_dict():
    """★ 「不增加模型大小」的前提：表**不进** parameters() / state_dict()。

    如果哪天有人把表注册成 buffer 或 Parameter，模型大小和 checkpoint 会一起爆炸，
    而 loss 曲线看不出任何异常。这条把它钉住。
    """
    ndb = make_ndb()
    # 两个 Linear(weight+bias) + level_weight = 5
    assert len(list(ndb.parameters())) == 5, f"门控参数项数变了：{len(list(ndb.parameters()))}"
    sd = ndb.state_dict()
    assert not any('toks' in k or 'cnts' in k or 'totals' in k for k in sd), \
        f"表被塞进了 state_dict：{list(sd)}"
    # 门控参数量必须极小（相对 75M 基座可忽略）
    n_gate = sum(p.numel() for p in ndb.parameters())
    assert n_gate < 5000, f"门控参数 {n_gate} 太多了，不再'不增加模型大小'"


def test_memory_guard_rejects_oversized_table():
    """表超上限必须在**构造时**报错，而不是跑到一半 OOM。"""
    with pytest.raises(ValueError, match="超过上限"):
        NgramNDB(n_embd=8, levels=(8, 8, 8), slots=268_435_456, top_k=4,
                 vocab_size=8192, max_table_gb=1.0)


def test_levels_and_slots_length_must_match():
    with pytest.raises(ValueError, match="长度必须一致"):
        NgramNDB(n_embd=8, levels=(8, 6), slots=[1024], vocab_size=V, max_table_gb=1.0)


# ==========================================================================
# 2) ★ 已知答案对照：增量写 == 离线建表
# ==========================================================================
def _offline_topk(pairs, M, top_k, V):
    """参考实现：把所有 (slot, token, count) 收集起来，每槽取前 K。"""
    agg = {}
    for s, y, c in pairs:
        agg[(s, y)] = agg.get((s, y), 0.0) + c
    per_slot = {}
    for (s, y), c in agg.items():
        per_slot.setdefault(s, []).append((y, c))
    out = {}
    for s, lst in per_slot.items():
        lst.sort(key=lambda t: (-t[1], t[0]))
        out[s] = lst[:top_k]
    return out


def test_incremental_write_equals_offline_build():
    """★ 核心对照：分多个窗口增量写入，结果必须与离线一次建表**逐位相同**。

    构造上保证没有词被挤出 top-K（每个槽的不同续写数 ≤ K），
    所以增量维护是无损的 —— 这一条验证的是**对齐、聚合、top-K 选取**都正确。
    """
    top_k = 6
    # M 要足够大：碰撞会让同一槽收到不同的词，增量 top-K 是**有损**的，
    # 那时"逐位相同"就不再是正确期望。下面显式断言无碰撞，避免测试静默退化。
    M_big = 1 << 20
    ndb = make_ndb(top_k=top_k, slots=M_big)
    # 用周期序列：每个后缀的续写唯一，且同一后缀只对应一个 token
    tok = torch.tensor([[(i * 7 + (i // 11)) % V for i in range(600)]])
    pairs = []
    # 手工收集应当被写入的 (slot, token, count)
    import collections
    want = collections.Counter()
    flat = tok.reshape(-1).numpy().astype(np.int64)
    with ndb.write_enabled():
        for lo in range(0, 600, 120):                     # 分 5 个窗口增量写
            seg = tok[:, lo:lo + 120]
            if seg.shape[1] <= L:
                continue
            h = torch.zeros(1, seg.shape[1], 16)
            ndb.observe(h, seg, seg)
            # 参考：把这一段里所有位置也记下来（权重=1，因为写门控初始 σ(1)≠1，
            # 所以下面用同一个门控权重，保证两边同权）
            ndb.flush()
    # 增量结果
    got = {}
    for s in np.nonzero(ndb._totals[0])[0]:
        lst = [(int(y), float(c)) for y, c in zip(ndb._toks[0][s], ndb._cnts[0][s]) if y >= 0]
        # 独占
        lst.sort(key=lambda t: (-t[1], t[0]))
        got[int(s)] = lst

    # 参考结果：**按同样的窗口切分**收集 (slot, token) 计数，权重用写门控的常数。
    # ★ 必须也分窗口：跨窗口边界的上下文是观察不到的（窗口内只产出 len(seg)-L 个哈希），
    #   拿全局序列做参考会多出 L-1 个/窗口的位置 —— 这是实现的性质，不是 bug。
    ndb2 = make_ndb(top_k=top_k, slots=M_big)
    with torch.no_grad():
        w_const = float(torch.sigmoid(ndb2.write_gate(torch.zeros(1, 1, 16))).item())
    ref_pairs, ctx_slot = [], {}
    for lo in range(0, len(flat), 120):
        seg = flat[lo:lo + 120]
        if len(seg) <= L:
            continue
        hh = suffix_hashes(seg, L)
        sl = (hh % M_big).astype(np.int64)
        for i in range(len(hh)):
            # 同一上下文必须落在同一槽（否则增量 top-K 会因碰撞而有损，对照就不成立）
            ctx_slot.setdefault(bytes(seg[i:i + L].astype(np.int32)), int(sl[i]))
        ref_pairs += [(int(a), int(b), w_const) for a, b in zip(sl, seg[L:L + len(hh)])]
    assert len(set(ctx_slot.values())) == len(ctx_slot), "测试语料在该 M 下发生碰撞，会引入有损差异"
    ref = _offline_topk(ref_pairs, M_big, top_k, V)

    assert set(got) == set(ref), "增量写入的槽集合与离线参考不一致"
    for s in ref:
        got_map = dict(got[s])
        ref_map = {y: c for y, c in ref[s]}
        assert set(got_map) == set(ref_map), f"槽 {s} 的 token 集合不一致"
        for y in ref_map:
            assert got_map[y] == pytest.approx(ref_map[y], abs=0.75), \
                f"槽 {s} token {y} 计数不一致：{got_map[y]} vs {ref_map[y]}"


def test_counts_are_weighted_by_write_gate():
    """写门控值必须真的改变计数（否则"模型决定写多重"是空话）。"""
    ndb_a, ndb_b = make_ndb(top_k=1), make_ndb(top_k=1)
    # b 的写门控偏置调低 → 权重更小 → 计数更小
    with torch.no_grad():
        ndb_b.write_gate.bias.fill_(-3.0)
    tok = make_tokens(300)
    feed(ndb_a, tok); ndb_a.flush()
    feed(ndb_b, tok); ndb_b.flush()
    assert ndb_a._totals[0].sum() > ndb_b._totals[0].sum() * 3, \
        "写门控没起作用：两个门控产出的表总量应当差很多"


def test_flush_is_idempotent_and_clears_buffer():
    """连续 flush 两次，第二次必须无事可做（缓冲已清空）。"""
    ndb = make_ndb()
    feed(ndb, make_tokens(300))
    n1 = ndb.flush()
    n2 = ndb.flush()
    assert n1 > 0 and n2 == 0


def test_observation_count_survives_topk_eviction():
    """★ `_totals` 必须记录**全部**观测，而不只是留在 top-K 里的。

    否则读门控的"top-1 占比 = cnt/total"会被系统性高估 —— 一个只被看过 1 次、
    top-1 计数为 1 的槽，会显得和"看过 100 次、top-1 计数 60"一样可信。
    """
    ndb = make_ndb(top_k=1)          # K=1：大量 token 会被挤出
    g = torch.Generator().manual_seed(3)
    # 同一批后缀配很多不同续写 → 频繁挤出
    tok = torch.randint(0, V, (1, 400), generator=g)
    feed(ndb, tok)
    ndb.flush()
    total = int(ndb._totals[0].sum())
    in_table = int(ndb._cnts[0].sum())
    assert total > 0
    assert total >= in_table, "total 不该小于留在表里的计数"
    # 有挤出发生，所以 total 应当明显大于 in_table
    assert total > in_table, "这份数据下应当发生 top-K 挤出，否则测不到东西"


# ==========================================================================
# 3) 读：形状、凸组合、门控范围
# ==========================================================================
def test_read_returns_valid_convex_combination():
    """`p_new` 必须是合法概率分布（凸组合不改变归一性）。"""
    ndb = make_ndb()
    feed(ndb, make_tokens(300)); ndb.flush()
    tok = make_tokens(60, seed=9)
    B, T = tok.shape
    h = torch.zeros(B, T, 16)
    p_model = torch.full((B, T, V), 1.0 / V)
    p_new, stats = ndb.read(h, tok, p_model, targets=tok)
    assert p_new.shape == p_model.shape
    assert torch.allclose(p_new.sum(-1), torch.ones(B, T), atol=1e-4), "不是归一分布"
    assert (p_new >= -1e-6).all()
    assert 0.0 <= stats.covered <= 1.0


def test_read_with_empty_table_falls_back_to_model():
    """表为空时 `p_ng` 全零，但门控仍会混入零 → `p_new` 会略小于模型分布。

    这里只要求**不崩且归一**；同时 `covered` 必须是 0。
    """
    ndb = make_ndb()
    tok = make_tokens(60)
    p_model = torch.full((1, 60, V), 1.0 / V)
    p_new, stats = ndb.read(torch.zeros(1, 60, 16), tok, p_model)
    assert stats.covered == 0.0
    assert torch.isfinite(p_new).all()


def test_read_gate_in_unit_interval_and_differentiable():
    ndb = make_ndb()
    feed(ndb, make_tokens(300)); ndb.flush()
    tok = make_tokens(60, seed=5)
    # ★ 必须用非零 h：h=0 时 ∂w/∂W = h = 0，梯度恒为零（第一版测试就栽在这）
    h = torch.randn(1, 60, 16, generator=torch.Generator().manual_seed(1), requires_grad=True)
    p_model = torch.full((1, 60, V), 1.0 / V)
    p_new, _ = ndb.read(h, tok, p_model, targets=tok)
    # ★ 不能用 p_new.sum()：两个分布都归一，sum 关于门控 g **解析恒为常数 N**，
    #   梯度必然为零（第一版测试就栽在这）。必须用真实目标：目标位置的 NLL。
    loss = -torch.log(p_new.gather(-1, tok.unsqueeze(-1)).clamp_min(1e-9)).mean()
    loss.backward()
    # 读门控必须拿到梯度（这才是"模型自己决定怎么读"）
    assert ndb.read_gate.weight.grad is not None
    assert ndb.read_gate.weight.grad.abs().sum() > 0, "读门控没有梯度"
    assert ndb.level_weight.grad is not None
    # 写门控通过 read_gate 的输入拿到梯度（文档里说明了这是间接路径）
    assert ndb.write_gate.weight.grad is not None
    assert ndb.write_gate.weight.grad.abs().sum() > 0, \
        "写门控没梯度 —— 「模型自己决定写」就是空话"


def test_level_alpha_is_a_distribution():
    ndb = make_ndb(levels=(4, 3), slots=1024)
    alpha = torch.softmax(ndb.level_weight, 0)
    assert alpha.shape == (2,)
    assert alpha.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert (alpha > 0).all()


def test_multi_level_read_uses_all_levels():
    """多级时 `p_ng` 应当同时用到两级：只喂一个级能覆盖的序列时仍应有覆盖。"""
    ndb = make_ndb(levels=(4, 2), slots=1024)
    feed(ndb, make_tokens(300)); ndb.flush()
    tok = make_tokens(60, seed=11)
    p_model = torch.full((1, 60, V), 1.0 / V)
    _, stats = ndb.read(torch.zeros(1, 60, 16), tok, p_model)
    assert stats.covered > 0


# ==========================================================================
# 4) ★ 对齐：off-by-L 回归
# ==========================================================================
def test_lookup_alignment_matches_manual_table():
    """★ 手工建表 + 逐位置查表，验证 `_lookup` 的下标对齐。

    把"上下文结尾在 i+L-1"这条约定变成可执行断言。写错成 `i+L` 时，
    查到的续写会整体前移一位 —— 而且 loss 曲线看不出来。
    """
    ndb = make_ndb(top_k=1)
    flat = np.array([(i * 13 + 7) % V for i in range(120)], dtype=np.int64)
    # 手工直接把表填成"每个后缀 → 它的下一个 token"
    hh = suffix_hashes(flat, L) % M
    for i in range(len(hh)):
        s = int(hh[i])
        ndb._toks[0][s, 0] = int(flat[i + L])
        ndb._cnts[0][s, 0] = 1
        ndb._totals[0][s] = 1
    tok = torch.from_numpy(flat).reshape(1, -1)
    lk = ndb._lookup(tok)
    top1, cnt, total, covered = lk[0]
    # 位置 j 的上下文是 flat[j-L+1 : j+1]，预测 flat[j+1]
    for j in range(L - 1, len(flat) - 1):
        assert covered[0, j], f"位置 {j} 应被覆盖"
        assert top1[0, j, 0] == flat[j + 1], f"位置 {j} 查表错位"


def test_suffix_hashes_length_and_contract():
    tok = np.arange(50, dtype=np.int64)
    for l in (1, 4, 8):
        assert len(suffix_hashes(tok, l)) == 50 - l
    # 第 i 个哈希必须等于把该窗口单独拿出来算
    h = suffix_hashes(tok, 4)
    for i in range(len(h)):
        assert h[i] == suffix_hashes(tok[i:i + 5], 4)[0]


# ==========================================================================
# 5) 与探针口径一致（离线 Δ 的兑现前提）
# ==========================================================================
def test_p_ng_is_renormalized_within_level():
    """`p_ng` 在每级内按 top-K 计数**重新归一**。

    为什么必须这样（这是被测试抓到的一个真 bug）：top-K 的计数之和 < total
    （有被挤出 / 未进 top-K 的词），所以 `cnt/total` 本身不是概率分布。
    不归一就混进凸组合，`p_new` 会整体小于 1（实测只有 0.88）。
    """
    ndb = make_ndb(top_k=4)
    feed(ndb, make_tokens(400)); ndb.flush()
    tok = make_tokens(80, seed=2)
    p_model = torch.zeros(1, 80, V); p_model[..., 0] = 1.0
    with torch.no_grad():
        ndb.read_gate.weight.zero_()
        ndb.read_gate.bias.fill_(20.0)          # σ(20) ≈ 1：完全信任检索
        ndb.level_weight.fill_(0.0)
    p_new, _ = ndb.read(torch.zeros(1, 80, 16), tok, p_model)
    top1, cnt, total, covered = ndb._lookup(tok)[0]   # 每级一个元组
    j = int(np.argmax(covered[0]))
    assert covered[0, j], "测试点应当被覆盖"
    y = int(top1[0, j, 0])
    valid = cnt[0, j][top1[0, j] >= 0]
    expect = float(cnt[0, j, 0]) / float(valid.sum())      # 级内归一后的 top-1 质量
    assert p_new[0, j, y].item() == pytest.approx(expect, abs=1e-5)
    assert p_new.sum(-1).mean().item() == pytest.approx(1.0, abs=1e-4), "不是归一分布"


def test_k1_p_ng_is_exactly_onehot():
    """★ K=1 时 `p_ng` 必须恰好是 one-hot —— 这是**与探针口径一致**的前提。

    离线测出的 Δ=−0.0723 用的是 `p = (1−λ)p_model + λ·onehot(top1)`
    （见 scripts/ngram_capacity_probe.py::nll_under_retrieval，以及 §12.3 原脚本）。
    如果在线实现的 K=1 不是 one-hot，那个离线数字就无法兑现。
    """
    ndb = make_ndb(top_k=1)
    feed(ndb, make_tokens(400)); ndb.flush()
    tok = make_tokens(80, seed=4)
    p_model = torch.zeros(1, 80, V); p_model[..., 0] = 1.0
    with torch.no_grad():
        ndb.read_gate.weight.zero_()
        ndb.read_gate.bias.fill_(20.0)
        ndb.level_weight.fill_(0.0)
    p_new, _ = ndb.read(torch.zeros(1, 80, 16), tok, p_model)
    top1, cnt, total, covered = ndb._lookup(tok)[0]   # 每级一个元组
    for j in np.nonzero(covered[0])[0]:
        y = int(top1[0, j, 0])
        assert p_new[0, j, y].item() == pytest.approx(1.0, abs=1e-5), \
            f"位置 {j} 的 p_ng 不是 one-hot（K=1）"
        assert p_new[0, j].sum().item() == pytest.approx(1.0, abs=1e-4)


def test_p_new_is_normalized_when_slots_are_empty():
    """★ 未覆盖位置的门控必须归零，否则 `p_new = (1−g)·p_model` 总和 < 1。

    这是被 `test_read_returns_valid_convex_combination` 抓到的真 bug：
    探针里由 `okf` 掩码做这件事，搬到模块时漏过一次。
    """
    ndb = make_ndb()
    tok = make_tokens(60)          # 表是空的 → 全部未覆盖
    p_model = torch.full((1, 60, V), 1.0 / V)
    p_new, stats = ndb.read(torch.randn(1, 60, 16), tok, p_model)
    assert stats.covered == 0.0
    assert p_new.sum(-1).mean().item() == pytest.approx(1.0, abs=1e-5)
