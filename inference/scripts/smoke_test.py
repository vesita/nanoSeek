"""
快速冒烟测试：验证 model.py 的全部保留架构配置都能正常前向传播 + 反向传播，并对比参数量。
架构（MoE / 共享专家 / aux-free / √softplus / MLA / MTP / CSA / CSA+HCA）+
结构设计升级（Attention Sinks / mHC / Lightning Indexer / Hash 路由）。
Muon 单测混相 Newton-Schulz 正交化和训练 loss 下降。

固定架构（无需配置）：RMSNorm + SwiGLU；RoPE 由 use_rope 开关控制。

用法（从项目根目录）：
    uv run python inference/scripts/smoke_test.py
"""
import sys
from pathlib import Path

# 脚本在 inference/scripts/ 子目录里，Python 默认不会把项目根目录加进模块搜索路径。
# 这里把根目录插到 sys.path 开头，才能 `from model import ...`。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import math

import torch
from model import GPTConfig, GPT, MoE


def make_model(**overrides):
    cfg = GPTConfig(
        vocab_size=65, block_size=64, n_layer=2, n_head=2, n_embd=64,
        dropout=0.0, bias=False, use_rope=True, **overrides,
    )
    return GPT(cfg)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# 各种保留架构的开关组合
CASES = [
    ("MoE", dict(use_moe=True, n_experts=4, n_top_k=2)),
    ("共享专家", dict(use_moe=True, n_experts=4, n_top_k=2, use_shared_expert=True)),
    ("aux-free", dict(use_moe=True, n_experts=4, n_top_k=2, use_aux_free_balance=True)),
    ("√softplus", dict(use_moe=True, n_experts=4, n_top_k=2, use_sqrtsoftplus=True)),
    ("MLA", dict(use_mla=True, kv_lora_rank=32, qk_rope_head_dim=8)),
    ("MTP", dict(use_mtp=True, n_mtp=1)),
    ("CSA均池", dict(use_csa=True, csa_compress=16, csa_topk=2, csa_window=32, use_csa_learnable=False)),
    ("CSA可学习", dict(use_csa=True, csa_compress=16, csa_topk=2, csa_window=32)),
    ("CSA+HCA", dict(use_csa=True, csa_compress=16, csa_topk=2, csa_window=32, use_hca=True)),
    # --- V4 结构设计升级 ---
    ("AttnSink", dict(use_csa=True, csa_compress=16, csa_topk=2, csa_window=32, use_attn_sink=True)),
    ("mHC", dict(use_mhc=True, hc_mult=4)),
    ("mHC+CSA", dict(use_mhc=True, hc_mult=4, use_csa=True, csa_compress=16, csa_topk=2, csa_window=32, use_hca=True)),
    ("LightIndex", dict(use_csa=True, csa_compress=16, csa_topk=2, csa_window=32, use_lightning_indexer=True)),
    ("Hash路由", dict(use_moe=True, n_experts=4, n_top_k=2, num_hash_layers=1)),
    # --- 计算图重排（拓扑实验 A）---
    ("ffn_attn", dict(block_order="ffn_attn")),
    ("ffn_attn+mHC", dict(block_order="ffn_attn", use_mhc=True, hc_mult=4)),
    # --- 稀疏注意力布线（拓扑实验 B）---
    ("稀疏布线", dict(no_attn_layers=[0, 2, 4])),
    ("稀疏布线+mHC", dict(no_attn_layers=[0, 2, 4], use_mhc=True, hc_mult=4)),
    # --- 显式记忆 token（拓扑实验 C）---
    ("记忆token", dict(n_memory_tokens=16)),
    ("记忆token+mHC", dict(n_memory_tokens=16, use_mhc=True, hc_mult=4)),
    # --- 对数放缩残差（LSE 数值连接实验）---
    ("LSE残差", dict(use_lse_residual=True)),
    ("LSE残差+CSA", dict(use_lse_residual=True, use_csa=True, csa_compress=16, csa_topk=2, csa_window=32)),
    # --- 对数放缩门控混合（LSE gate-mix v2）---
    ("LSE门控", dict(use_lse_gate=True)),
    ("LSE门控+CSA", dict(use_lse_gate=True, use_csa=True, csa_compress=16, csa_topk=2, csa_window=32)),
    ("LSE门控+mHC", dict(use_lse_gate=True, use_mhc=True, hc_mult=4)),
    ("全特性", dict(use_mhc=True, hc_mult=4, use_attn_sink=True,
                    use_moe=True, n_experts=4, n_top_k=2, use_shared_expert=True,
                    use_aux_free_balance=True, num_hash_layers=1,
                    use_csa=True, csa_compress=16, csa_topk=2, csa_window=32,
                    use_hca=True, use_lightning_indexer=True)),
]

x = torch.randint(0, 65, (4, 64))
y = torch.randint(0, 65, (4, 64))

for name, overrides in CASES:
    model = make_model(**overrides)
    logits, loss = model(x, y)
    loss.backward()
    # 检查所有梯度都没有 NaN（常见 bug：位置编码维度对不上 / 广播错位）
    nan_free = all(not torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
    print(f"{name:>10}: 参数 {count_params(model):>7,} | loss {loss.item():.4f} | 梯度无 NaN = {nan_free}")
    assert nan_free, f"{name} 出现 NaN 梯度！"

# MoE 负载均衡辅助损失（Switch 式）：随机路由器不可能恰好均匀，损失应非零
moe = MoE(GPTConfig(n_embd=64, n_experts=4, n_top_k=2))
moe(torch.randn(4, 16, 64))
print(f"MoE 辅助损失 = {moe.aux_loss.item():.4f}（应非零，随机路由器不可能恰好均匀）")
assert moe.aux_loss.item() > 0

# aux-free 偏置修正：过载专家 bias 应下降、欠载专家上升，且不再产生辅助损失
moe_af = MoE(GPTConfig(n_embd=64, n_experts=4, n_top_k=2, use_aux_free_balance=True))
bias_before = moe_af.router_bias.clone()
for _ in range(10):
    moe_af(torch.randn(4, 16, 64))
print(f"aux-free 偏置更新后：max|Δbias| = {(moe_af.router_bias - bias_before).abs().max().item():.4f}，aux_loss = {moe_af.aux_loss.item()}")
assert (moe_af.router_bias - bias_before).abs().max().item() > 0, "aux-free 偏置没有更新！"
assert moe_af.aux_loss.item() == 0.0, "aux-free 模式不应产生辅助损失！"

# 验证 mHC 的 Sinkhorn-Knopp 投影：结果必须是双重随机矩阵（行和列和都为 1、非负）
from model import sinkhorn_knopp
import torch.nn.functional as F
B_raw = torch.randn(4, 4)
B_ds = sinkhorn_knopp(F.softplus(B_raw))
row_ok = (B_ds.sum(dim=-1) - 1).abs().max().item() < 1e-4
col_ok = (B_ds.sum(dim=-2) - 1).abs().max().item() < 1e-4
nonneg = (B_ds >= 0).all().item()
print(f"Sinkhorn 双重随机：行和误差 {B_ds.sum(dim=-1).sub(1).abs().max().item():.2e} | "
      f"列和误差 {B_ds.sum(dim=-2).sub(1).abs().max().item():.2e} | 非负 = {nonneg}")
assert row_ok and col_ok and nonneg, "Sinkhorn 投影没有生成双重随机矩阵！"

# 验证 SwiGLU Clamp：极端输入下，开钳制后的输出必须大幅收敛
from model import SwiGLU
glu = SwiGLU(GPTConfig(n_embd=64, dropout=0.0, bias=False, swiglu_clamp=10.0))
big_in = torch.randn(4, 16, 64) * 100  # 故意塞一个极端大的输入
with torch.no_grad():
    out_clamped = glu(big_in).abs().max().item()          # 开钳制 forward
    glu.config.swiglu_clamp = 0.0
    out_unclamped = glu(big_in).abs().max().item()        # 同权重、关钳制 forward
    glu.config.swiglu_clamp = 10.0
print(f"极端输入下：开钳制 |out| max = {out_clamped:.3f}，关钳制 = {out_unclamped:.3f}")
assert out_unclamped > 10.0, "对照组输出太小，测试没有意义（输入不够极端）"
assert out_clamped < out_unclamped / 10, "SwiGLU Clamp 没有有效压制极端输出！"

# 验证对数放缩门控混合（LSE gate-mix v2）：α=sigmoid(raw_gate) 初始 0.5、可学习、且
# 在 α=0（纯 LSE）时退化为有界残差。构造 Block 直接检查 gate 参数与梯度。
gate_model = make_model(use_lse_gate=True)
gate_blocks = [b for b in gate_model.transformer.h]
assert all(hasattr(b, "raw_gate") for b in gate_blocks), "gate-mix 的每个 Block 应有 raw_gate！"
alpha0 = torch.sigmoid(gate_blocks[0].raw_gate).item()
print(f"LSE 门控初始 α = {alpha0:.3f}（应 = 0.5，线性/LSE 均衡起点）")
assert abs(alpha0 - 0.5) < 1e-4, "raw_gate 初始 0 → sigmoid ≈ 0.5"
# 可学习：训练几步后 raw_gate 应有梯度且能更新
opt = gate_model.configure_optimizers(weight_decay=0.1, learning_rate=1e-2, betas=(0.9, 0.99), device_type='cpu')
gx = torch.randint(0, 65, (4, 64)); gy = torch.randint(0, 65, (4, 64))
for _ in range(5):
    opt.zero_grad(); _, l = gate_model(gx, gy); l.backward(); opt.step()
assert all(b.raw_gate.grad is not None and not torch.isnan(b.raw_gate.grad).any() for b in gate_blocks), \
    "raw_gate 应有干净梯度！"
at = torch.sigmoid(gate_blocks[0].raw_gate).item()
print(f"训练 5 步后 α 更新到 {at:.3f}（≠0.5，确实在学）")
assert abs(at - 0.5) > 1e-6, "raw_gate 未通过训练更新，α 应发生移动"
# 与 mHC 兼容、与纯 v1 残差互斥
assert make_model(use_lse_gate=True, use_mhc=True), "gate-mix 应与 mHC 共存"
try:
    make_model(use_lse_residual=True, use_lse_gate=True)
    raise SystemExit("应触发 v1/v2 互斥断言")
except AssertionError:
    pass
print("LSE 门控：α 可学习且与 mHC 兼容、与纯 v1 互斥 ✅")

# 验证对数放缩残差（LSE）：有界收缩 + 软选择偏置 + 数值稳定（负输入不溢出）
from model import logsumexp_residual
big_a = torch.randn(4, 16, 80) * 100          # 极端大的输入
big_b = torch.randn(4, 16, 80) * 100
out_lse = logsumexp_residual(big_a, big_b)
# 性质 1（有界）：|LSE - max(a,b)| ≤ log2，且 LSE 远小于线性相加的极端值
bound_err = (out_lse - torch.maximum(big_a, big_b)).abs().max().item()
add_max = (big_a + big_b).abs().max().item()
print(f"LSE 残差：|LSE - max| max = {bound_err:.3f}（应 ≤ log2≈0.693）| "
      f"LSE max = {out_lse.abs().max().item():.3f} vs 线性相加 max = {add_max:.3f}")
assert bound_err <= 0.693 + 1e-4, "LSE 残差超出 soft-max 上界（max+log2），有界性失效！"
assert out_lse.abs().max().item() < add_max, "LSE 残差没有抑制极端值（应比线性相加温和）"
# 性质 2（软选择）：当一分支远大于另一支时，输出 ≈ 大分支（不含被抑制分支的贡献）
a_small = torch.randn(4, 16, 80) * 1.0
b_neg = torch.full_like(a_small, -1e3)       # 极端负 = exp(·)→0，被完全抑制
sel = logsumexp_residual(a_small, b_neg)
print(f"LSE 软选择：大分支主导时 |LSE - a| max = {(sel - a_small).abs().max().item():.4f}（应≈0）")
assert (sel - a_small).abs().max().item() < 1e-6, "LSE 在大分支主导时应完全抑制被抑制分支！"
# 性质 3（梯度）：logaddexp 处处可微，残差变小不影响反向传播
a_g = a_small.clone().requires_grad_(True)
y_g = logsumexp_residual(a_g, a_small * 0.5)
y_g.sum().backward()
assert a_g.grad is not None and not torch.isnan(a_g.grad).any(), "LSE 残差梯度出现 NaN！"
print("LSE 残差：梯度干净（logaddexp 处处可微）")

# 验证 Muon 优化器：先单测混相 Newton-Schulz 正交化，再端到端训练几步
from model import Muon, zeropower_via_newtonschulz

# 固定种子：NS 正交化误差对输入矩阵敏感，随机矩阵偶发超阈值导致冒烟 flaky
torch.manual_seed(1337)
G = torch.randn(16, 16)
Q = zeropower_via_newtonschulz(G, steps=10)   # 8 激进 + 2 经典
orth_err = (Q.T @ Q - torch.eye(16)).abs().max().item()
print(f"混相 Newton-Schulz 正交化误差 = {orth_err:.2e}（应接近 0）")
assert orth_err < 1e-3, "Newton-Schulz 没有把矩阵正交化！"

muon_model = make_model(use_muon=True)
opt = muon_model.configure_optimizers(weight_decay=0.1, learning_rate=1e-2, betas=(0.9, 0.99), device_type='cpu')
muon_x = torch.randint(0, 65, (4, 64))
muon_y = torch.randint(0, 65, (4, 64))
losses = []
for _ in range(10):
    opt.zero_grad()
    _, loss = muon_model(muon_x, muon_y)
    loss.backward()
    opt.step()
    losses.append(loss.item())
print(f"Muon 10 步 loss：{' → '.join(f'{l:.3f}' for l in losses)}")
assert all(not math.isnan(l) for l in losses), "Muon 训练出现 NaN！"
assert losses[-1] < losses[0], "Muon 训练 loss 没有下降！"

print("\n冒烟测试通过 ✅")
