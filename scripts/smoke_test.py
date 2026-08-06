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

import torch
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
    ("MoE", dict(use_rmsnorm=True, use_rope=True, use_swiglu=True,
                 use_moe=True, n_experts=4, n_top_k=2)),
    ("MLA", dict(use_rmsnorm=True, use_rope=True, use_swiglu=True,
                 use_mla=True, kv_lora_rank=32, qk_rope_head_dim=8)),
    ("MTP", dict(use_rmsnorm=True, use_rope=True, use_swiglu=True,
                 use_mtp=True, n_mtp=1)),
    ("全开", dict(use_rmsnorm=True, use_rope=True, use_swiglu=True,
                 use_moe=True, n_experts=4, n_top_k=2,
                 use_mla=True, kv_lora_rank=32, qk_rope_head_dim=8,
                 use_mtp=True, n_mtp=1)),
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

print("\n冒烟测试通过 ✅")
