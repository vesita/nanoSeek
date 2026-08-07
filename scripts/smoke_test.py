"""
快速冒烟测试：验证 model.py 的各种架构配置（原始 GPT-2 / modern 三件套 /
MoE / MLA / MTP / 全开组合）都能正常前向传播 + 反向传播，并对比参数量。

用法（从项目根目录）：
    uv run python scripts/smoke_test.py
"""
import sys
from pathlib import Path

# 脚本在 scripts/ 子目录里，Python 默认不会把项目根目录加进模块搜索路径。
# 这里把根目录插到 sys.path 开头，才能 `from model import ...`。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math

import torch
from torch.nn import functional as F
from model import GPTConfig, GPT, MoE


def make_model(**overrides):
    cfg = GPTConfig(
        vocab_size=65, block_size=64, n_layer=2, n_head=2, n_embd=64,
        dropout=0.0, bias=False, **overrides,
    )
    return GPT(cfg)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# 各种架构的开关组合
CASES = [
    ("原始", {}),
    ("现代", dict(use_rmsnorm=True, use_rope=True, use_swiglu=True)),
    ("Clamp", dict(use_rmsnorm=True, use_rope=True, use_swiglu=True,
                   swiglu_clamp=10.0)),
    ("MoE", dict(use_rmsnorm=True, use_rope=True, use_swiglu=True,
                 use_moe=True, n_experts=4, n_top_k=2)),
    ("预判路由", dict(use_rmsnorm=True, use_rope=True, use_swiglu=True,
                   use_moe=True, n_experts=4, n_top_k=2,
                   use_anticipatory_routing=True)),
    ("MLA", dict(use_rmsnorm=True, use_rope=True, use_swiglu=True,
                 use_mla=True, kv_lora_rank=32, qk_rope_head_dim=8)),
    ("MTP", dict(use_rmsnorm=True, use_rope=True, use_swiglu=True,
                 use_mtp=True, n_mtp=1)),
    ("mHC", dict(use_rmsnorm=True, use_rope=True, use_swiglu=True,
                 use_mhc=True)),
    ("全开", dict(use_rmsnorm=True, use_rope=True, use_swiglu=True,
                 use_moe=True, n_experts=4, n_top_k=2,
                 use_mla=True, kv_lora_rank=32, qk_rope_head_dim=8,
                 use_mtp=True, n_mtp=1)),
    ("CSA", dict(use_rmsnorm=True, use_rope=True, use_swiglu=True,
                 use_csa=True, csa_compress=16, csa_topk=2, csa_window=32)),
    ("CSA+HCA", dict(use_rmsnorm=True, use_rope=True, use_swiglu=True,
                     use_csa=True, csa_compress=16, csa_topk=2, csa_window=32,
                     use_hca=True)),
]

x = torch.randint(0, 65, (4, 64))
y = torch.randint(0, 65, (4, 64))

for name, overrides in CASES:
    model = make_model(**overrides)
    logits, loss = model(x, y)
    loss.backward()
    # 检查所有梯度都没有 NaN（常见 bug：位置编码维度对不上 / 广播错位）
    nan_free = all(not torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
    print(f"{name:>4}: 参数 {count_params(model):>7,} | loss {loss.item():.4f} | 梯度无 NaN = {nan_free}")
    assert nan_free, f"{name} 出现 NaN 梯度！"

# 直接验证 MoE 的负载均衡辅助损失确实非零（会被加进总损失里）
moe = MoE(GPTConfig(n_embd=64, n_experts=4, n_top_k=2))
moe(torch.randn(4, 16, 64))
print(f"MoE 辅助损失 = {moe.aux_loss.item():.4f}（应非零，随机路由器不可能恰好均匀）")
assert moe.aux_loss.item() > 0

# 直接验证 SwiGLU Clamp：同一套权重，只切换钳制开关，观察 forward 输出差异。
# 钳制作用在「门控乘积」上（异常值产生于此），c_proj 线性层会重新组合——
# 所以用行为测试：开钳制后，极端输入引发的输出必须大幅收敛。
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

# 验证 Muon 优化器（V4）：先单测 Newton-Schulz 正交化，再端到端训练几步
from model import Muon, zeropower_via_newtonschulz

G = torch.randn(16, 16)
Q = zeropower_via_newtonschulz(G, steps=20)
orth_err = (Q.T @ Q - torch.eye(16)).abs().max().item()
print(f"Newton-Schulz 正交化误差 = {orth_err:.2e}（应接近 0）")
assert orth_err < 1e-3, "Newton-Schulz 没有把矩阵正交化！"

muon_model = make_model(use_rmsnorm=True, use_rope=True, use_swiglu=True, use_muon=True)
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

# 验证 mHC：混合矩阵必须是双随机矩阵（行和 = 列和 = 1、非负）
from model import sinkhorn_knopp
m = sinkhorn_knopp(torch.randn(2, 2))   # 用默认 5 次迭代
row_sum = m.sum(dim=1)
col_sum = m.sum(dim=0)
print(f"Sinkhorn 投影：行和 {row_sum.tolist()}，列和 {col_sum.tolist()}，全部非负 = {(m >= 0).all().item()}")
assert (m >= 0).all().item(), "双随机矩阵必须非负！"
assert torch.allclose(row_sum, torch.ones(2), atol=1e-3), "行和必须为 1！"
assert torch.allclose(col_sum, torch.ones(2), atol=1e-3), "列和必须为 1！"

mhc_model = make_model(use_rmsnorm=True, use_rope=True, use_swiglu=True, use_mhc=True)
logits, loss = mhc_model(x, y)
loss.backward()
assert not math.isnan(loss.item()), "mHC 前向 loss 出现 NaN！"
# 验证 mHC 模型能正常训练几步、混合矩阵仍在双随机流形上
mhc_opt = mhc_model.configure_optimizers(weight_decay=0.1, learning_rate=1e-2, betas=(0.9, 0.99), device_type='cpu')
for _ in range(3):
    mhc_opt.zero_grad()
    _, loss = mhc_model(x, y)
    loss.backward()
    mhc_opt.step()
for blk in mhc_model.transformer.h:
    mm = sinkhorn_knopp(blk.mix, iters=3)
    assert torch.allclose(mm.sum(dim=1), torch.ones(2), atol=1e-3), "训练后混合矩阵不再满足行和=1！"
    assert torch.allclose(mm.sum(dim=0), torch.ones(2), atol=1e-3), "训练后混合矩阵不再满足列和=1！"
print("mHC 双流模型训练 3 步后，各层混合矩阵仍保持双随机 ✅")

print("\n冒烟测试通过 ✅")
