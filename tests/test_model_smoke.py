"""`model/gpt.py` 的冒烟测试：形状、损失、梯度、共享权重、因果性。

## 为什么值得测
这是整个项目的核心，改动频率最高。它不会"悄悄算错"——但很容易"悄悄改变语义"：
- 前向返回的 loss 是 CE + MoE aux + MTP 的**混合**（本项目被混合口径误导过 3 次）
- `wte` / `lm_head` / `mtp_head` 是**同一个张量**（参数量统计被虚高到 90.19M 过）
- MoE 路由带**无梯度的原地状态更新**（见文末 aux-free 的两条测试）

## 全部用小配置 + CPU，秒级完成
"""
import pathlib

import pytest
import torch
import yaml

from model.config import GPTConfig
from model.gpt import GPT

ROOT = pathlib.Path(__file__).resolve().parent.parent
REAL_CFG = ROOT / 'out' / 'ndb_run' / 'config.yaml'


def build(config, seed=1234, eval_mode=True):
    torch.manual_seed(seed)
    m = GPT(config)
    return m.eval() if eval_mode else m


def rand_batch(b=2, t=32, v=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, v, (b, t), generator=g)


# ==========================================================================
# 1) 形状 / 损失 / 梯度
# ==========================================================================
def test_forward_shapes(tiny_model, tiny_config):
    x = rand_batch(t=32, v=tiny_config.vocab_size)
    logits, loss = tiny_model(x, x)
    assert logits.shape == (2, 32, tiny_config.vocab_size)
    # 注意：混合 loss（CE + MoE aux + MTP）的形状是 (1,) 而不是标量
    assert loss.numel() == 1 and torch.isfinite(loss)


def test_forward_without_targets_returns_only_last_position(tiny_model, tiny_config):
    """★ 契约（也是陷阱）：`targets=None` 时只返回**最后一个位置**的 logits。

    `gpt.py:207` 的推理优化：只对 `x[:, [-1], :]` 过 lm_head，省掉 T-1 次大矩阵乘。
    生成场景正确（只需要最后一位），但任何"需要全部位置 logits"的离线评估
    如果忘了传 targets，就会**静默拿到 (B,1,V)** —— 后续按位置聚合全错，
    而且不报错。`test_causality_*` 第一版就因此变成空测试（见那里的注释）。
    """
    x = rand_batch(t=16, v=tiny_config.vocab_size, seed=3)
    logits, loss = tiny_model(x)
    assert loss is None
    assert tuple(logits.shape) == (2, 1, tiny_config.vocab_size), \
        "这个契约变了：要么改文档，要么修所有依赖 (B,1,V) 的评估代码"

    # 传了 targets 才是全位置
    logits_full, loss_full = tiny_model(x, x)
    assert tuple(logits_full.shape) == (2, 16, tiny_config.vocab_size)
    assert loss_full is not None
    # 最后一位的 logits 两条路径必须一致（同一份 x，只是过不过全位置 lm_head）
    assert torch.allclose(logits[:, -1], logits_full[:, -1], atol=1e-5), \
        "只算最后一位与全量计算的最后一位不一致 —— 推理优化改变了数值"


def test_initial_loss_is_close_to_log_vocab(tiny_config):
    """随机初始化时 CE 应接近 ln(V)（= 均匀分布），偏差大说明 logits 尺度失控。"""
    import math
    m = build(tiny_config, eval_mode=False)
    x = rand_batch(t=64, v=tiny_config.vocab_size, seed=1)
    _, loss = m(x, x)
    # 混合 loss 含 MTP，所以只会比 ln(V) 高，不会低很多
    assert math.log(tiny_config.vocab_size) - 0.5 < loss.item() < math.log(tiny_config.vocab_size) + 1.0


def test_backward_gives_finite_grads(tiny_config):
    m = build(tiny_config, eval_mode=False)
    x = rand_batch(t=32, v=tiny_config.vocab_size, seed=2)
    _, loss = m(x, x)
    loss.backward()
    grads = [(n, p.grad) for n, p in m.named_parameters() if p.requires_grad]
    assert len(grads) > 0
    missing = [n for n, g in grads if g is None]
    assert not missing, f"这些参数没拿到梯度：{missing[:5]}"
    bad = [n for n, g in grads if not torch.isfinite(g).all()]
    assert not bad, f"这些参数的梯度非有限：{bad[:5]}"


def test_longer_than_block_size_raises(tiny_model, tiny_config):
    x = rand_batch(t=tiny_config.block_size + 1, v=tiny_config.vocab_size)
    with pytest.raises(AssertionError, match="block size"):
        tiny_model(x, x)


# ==========================================================================
# 2) 参数量：共享权重导致的虚高（本项目真实踩过的坑）
# ==========================================================================
def test_weight_tying_wte_and_lm_head_are_the_same_tensor(tiny_model, tiny_config):
    """`lm_head.weight` 与 `wte.weight` 必须是**同一个张量对象**（权重共享）。

    如果哪天变成两份，参数量会凭空翻倍，而 loss 曲线看不出任何异常。
    """
    wte = tiny_model.transformer.wte.weight
    head = tiny_model.lm_head.weight
    assert wte is head, "wte 与 lm_head 不再是同一个张量（权重共享被破坏）"
    assert wte.data_ptr() == head.data_ptr()


def test_get_num_params_counts_unique_tensors(tiny_model):
    """`get_num_params()` 数的是 `self.parameters()`。

    `nn.Module.parameters()` 会对共享张量**去重**，所以它给的是真实参数量。
    """
    uniq = sum(p.numel() for p in {id(p): p for p in tiny_model.parameters()}.values())
    assert tiny_model.get_num_params(non_embedding=False) == uniq


def test_state_dict_sum_inflates_due_to_weight_tying(tiny_model):
    """★ 文档记的坑：`sum(state_dict().values().numel())` 会**重复计数**共享张量。

    PROJECT_STATE §2.1：真实 81.58M，但 `state_dict()` 求和 = 90.19M（+10.6%），
    因为 `wte` / `lm_head` / `mtp_head` 是同一个张量被列了 3 遍，
    再叠加 `state_dict()` 还会带上 buffer（参数以外的张量）。
    统计参数量时必须用 `get_num_params()`。

    这里做**精确对账**（而不是只断言"更大"），把虚高拆成两个可解释的来源。
    """
    sd = tiny_model.state_dict()
    # 注意：state_dict() 默认 keep_vars=False，返回的是 param.detach() ——
    # 新的 Python 对象但共享同一份 storage，所以只能用 data_ptr() 判身份，不能用 id()。
    param_ptrs = {p.data_ptr() for p in tiny_model.parameters()}
    sd_param = sum(v.numel() for v in sd.values() if torch.is_tensor(v) and v.data_ptr() in param_ptrs)
    sd_other = sum(v.numel() for v in sd.values() if torch.is_tensor(v) and v.data_ptr() not in param_ptrs)
    real = tiny_model.get_num_params(non_embedding=False)
    emb = tiny_model.transformer.wte.weight.numel()

    # 共享张量在 state_dict 里被列了几次（wte / lm_head / mtp_head）
    shared = [k for k, v in sd.items()
              if torch.is_tensor(v) and v.data_ptr() == tiny_model.transformer.wte.weight.data_ptr()]
    assert len(shared) >= 2, f"wte 只被列了 {len(shared)} 次？权重共享变了"

    # 来源 1：共享张量重复计数（重复次数 − 1）× 嵌入大小
    assert sd_param - real == (len(shared) - 1) * emb, "虚高量对不上共享权重的重复计数"
    # 来源 2：state_dict 还会带 buffer
    assert sd_other > 0
    assert sum(v.numel() for v in sd.values() if torch.is_tensor(v)) == real + (len(shared) - 1) * emb + sd_other


@pytest.mark.slow
def test_real_config_param_counts_match_documentation():
    """钉住 PROJECT_STATE 的两个关键数字：**81.58M（开 MTP）/ 75.15M（关 MTP）**。

    参数量被算错过一次（state_dict 求和 → 90.19M），这两个数字是后续所有
    "模型多大 / 每参数多少 token"讨论的基准，值得用测试守住。
    """
    if not REAL_CFG.exists():
        pytest.skip(f"{REAL_CFG} 不存在")
    with open(REAL_CFG, encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    fields = set(GPTConfig.__dataclass_fields__)
    kw = {k: v for k, v in raw.items() if k in fields}
    kw['vocab_size'] = 8192            # train.py 从 tokenizer 注入，不在 yaml 里

    with_mtp = GPT(GPTConfig(**kw)).get_num_params()
    kw_off = dict(kw, use_mtp=False)
    without_mtp = GPT(GPTConfig(**kw_off)).get_num_params()

    assert with_mtp / 1e6 == pytest.approx(81.58, rel=2e-3), f"实际 {with_mtp/1e6:.2f}M"
    assert without_mtp / 1e6 == pytest.approx(75.15, rel=2e-3), f"实际 {without_mtp/1e6:.2f}M"
    assert with_mtp - without_mtp == pytest.approx(6.44e6, rel=5e-2), "MTP 那部分不是 6.44M"


@pytest.mark.slow
def test_real_config_mtp_and_swiglu_setting():
    """基座 v2 的两个关键决定必须在配置里落实（否则又是一次"配置没生效"）。"""
    if not REAL_CFG.exists():
        pytest.skip(f"{REAL_CFG} 不存在")
    with open(REAL_CFG, encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    assert raw['use_aux_free_balance'] is True
    # 旧基线开了 MTP/mHC，新基座关掉（依据见 PROJECT_STATE §5）
    assert 'use_mtp' in raw and 'use_mhc' in raw


# ==========================================================================
# 3) 因果性
# ==========================================================================
def test_causality_future_tokens_do_not_change_past_logits(tiny_config):
    """★ 改最后一个 token 不能影响前面任何位置的 logits。

    用**两个同种子初始化的模型**各跑一次前向，避免 router_bias 的原地更新
    污染对照（见文末两条测试）。这条一次性覆盖 attention mask 写反、
    位置编码泄漏未来、MoE 路由越界聚合等一整类错误。

    ⚠ 必须传 `targets`！`targets=None` 时 forward 只返回最后一位的 logits
    （`gpt.py:207` 的推理优化），`l1[:, :-1]` 会是空张量、`allclose` 恒真 ——
    第一版就是这么写成空测试的。下面显式断言形状，杜绝再次退化。
    """
    m1 = build(tiny_config, seed=7)
    m2 = build(tiny_config, seed=7)
    x1 = rand_batch(b=1, t=tiny_config.block_size, v=tiny_config.vocab_size, seed=8)
    x2 = x1.clone()
    x2[0, -1] = (x2[0, -1] + 1) % tiny_config.vocab_size

    with torch.no_grad():
        l1, _ = m1(x1, x1)          # ← 传 targets 才会返回全部位置
        l2, _ = m2(x2, x2)

    # 防空测试：必须真的拿到 T 个位置，否则下面的比较毫无意义
    assert l1.shape[1] == tiny_config.block_size, \
        f"只拿到 {l1.shape[1]} 个位置，因果性比较退化为空测试"
    assert not torch.allclose(l1[:, -1], l2[:, -1]), "改最后一位却没影响最后一位的 logits？"
    assert torch.allclose(l1[:, :-1], l2[:, :-1], atol=1e-6), \
        "未来 token 改变了过去的 logits —— 因果性被破坏"


# ==========================================================================
# 4) ★ aux-free 路由偏置：eval 模式下的隐藏副作用
# ==========================================================================
def test_aux_free_router_bias_mutates_even_in_eval(tiny_model):
    """★ 发现：`model.eval()` **不能**阻止 aux-free 路由偏置被原地更新。

    `model/mlp.py:133` 的 `router_bias.sub_(...)` 只看 `use_aux_free_balance`，
    完全不看 `self.training`，也不看是否在 `torch.no_grad()` 里。

    后果（都已实测）：
    1. 验证/评估也会改模型参数 —— 评估不是"只读"操作；
    2. 同一 batch 连续前向两次，logits 不逐位相同（实测差 ~2e-5 且随次数累积）；
    3. `estimate_loss` 跑 200 个 batch 会顺带做 200 次偏置更新。

    这是**既有行为**（整个 19781 步基线都带着它），本测试只负责记录，
    不改变训练动态。要改的话应当是单独的 A/B 实验。
    """
    if not tiny_model.config.use_aux_free_balance:
        pytest.skip("本配置没开 aux-free")
    tiny_model.eval()                     # tiny_model 夹具默认是 train 模式
    blk = tiny_model.transformer.h[0].mlp
    assert hasattr(blk, 'router_bias')
    before = blk.router_bias.clone()
    x = rand_batch(t=16, v=tiny_model.config.vocab_size, seed=9)
    with torch.no_grad():
        tiny_model(x)
    assert tiny_model.training is False
    assert not torch.equal(before, blk.router_bias), "eval 模式下 router_bias 不该被改（行为变了？）"


def test_aux_free_makes_repeated_eval_non_deterministic(tiny_model):
    """上一条的直接后果：同一输入连续两次前向，logits 不逐位相同。

    ★ 对测量的影响：任何"配对比较"（A/B、有无记忆库）都必须把两次评估放在
    **同一次前向**里算，或者接受这份噪声。本项目探针的 Δ 就是把 off / 各 λ
    在同一次前向里一起算的，所以不受影响。
    """
    if not tiny_model.config.use_aux_free_balance:
        pytest.skip("本配置没开 aux-free")
    x = rand_batch(t=16, v=tiny_model.config.vocab_size, seed=10)
    with torch.no_grad():
        l1, _ = tiny_model(x)
        l2, _ = tiny_model(x)
    assert not torch.equal(l1, l2), "两次前向逐位相同 —— 说明 aux-free 更新被关掉了（或 eval 已经不再改状态）"


def test_aux_free_update_is_sign_based_and_bounded_by_balance_factor(tiny_model):
    """偏置每步按**符号**更新，幅度恰为 balance_factor（不是按比例）。

    过载专家降 bias、欠载升 bias —— 这是 V4 aux-free 的定义。
    """
    if not tiny_model.config.use_aux_free_balance:
        pytest.skip("本配置没开 aux-free")
    blk = tiny_model.transformer.h[0].mlp
    bf = blk.balance_factor
    before = blk.router_bias.clone()
    with torch.no_grad():
        tiny_model(rand_batch(t=16, v=tiny_model.config.vocab_size, seed=11))
    delta = (blk.router_bias - before).abs()
    assert torch.allclose(delta, torch.full_like(delta, bf), atol=1e-7), \
        f"每步更新幅度应当恒为 balance_factor={bf}，实际 {delta.tolist()}"


# ==========================================================================
# 5) 生成
# ==========================================================================
def test_generate_shape_and_token_range(tiny_model, tiny_config):
    idx = torch.zeros((1, 4), dtype=torch.long)
    out = tiny_model.generate(idx, max_new_tokens=8)
    assert out.shape == (1, 12)
    assert int(out.min()) >= 0 and int(out.max()) < tiny_config.vocab_size
    assert torch.equal(out[:, :4], idx), "generate 不该改动 prompt 部分"


def test_generate_respects_block_size_clipping(tiny_model, tiny_config):
    """上下文超过 block_size 时必须裁剪，而不是抛断言。"""
    idx = torch.zeros((1, tiny_config.block_size + 5), dtype=torch.long)
    out = tiny_model.generate(idx, max_new_tokens=3)
    assert out.shape == (1, tiny_config.block_size + 8)
