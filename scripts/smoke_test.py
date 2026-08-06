"""
快速冒烟测试：验证魔改后的 model.py 在「原始 GPT-2」和「modern(DeepSeek风格)」两种
配置下都能正常前向传播 + 反向传播，并对比两者的参数量。

用法（从项目根目录）：
    uv run python scripts/smoke_test.py
"""
import sys
from pathlib import Path

# 脚本在 scripts/ 子目录里，Python 默认不会把项目根目录加进模块搜索路径。
# 这里把根目录插到 sys.path 开头，才能 `from model import ...`。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from model import GPTConfig, GPT


def make_model(**overrides):
    cfg = GPTConfig(
        vocab_size=65, block_size=64, n_layer=2, n_head=2, n_embd=64,
        dropout=0.0, bias=False, **overrides,
    )
    return GPT(cfg)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# 原始 GPT-2 结构（默认三个开关全关）
orig = make_model()
# modern 结构（三个开关全开）
modern = make_model(use_rmsnorm=True, use_rope=True, use_swiglu=True)

print(f"original params: {count_params(orig):>8,}  (non-embedding {orig.get_num_params():,})")
print(f"modern   params: {count_params(modern):>8,}  (non-embedding {modern.get_num_params():,})")

x = torch.randint(0, 65, (4, 64))
y = torch.randint(0, 65, (4, 64))

for name, model in [("original", orig), ("modern", modern)]:
    logits, loss = model(x, y)
    loss.backward()
    # 检查所有梯度都没有 NaN（常见 bug：位置编码维度对不上 / 广播错位）
    nan_free = all(not torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
    print(f"{name:>8}: logits {tuple(logits.shape)}, loss {loss.item():.4f}, gradients NaN-free = {nan_free}")

print("\nSMOKE TEST PASSED ✅")
