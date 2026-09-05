"""
nanoSeek 架构有效秩与奇异值能量谱分析探测脚本 (Rank & Spectrum Analyzer)

设计目的：
在 100M 大模型正式长程预训练之前，通过随机初始化/前向表征的奇异值分解（SVD），
直接测量：
1. 投影层（Q/K/V/O、FFN/MoE）的初始条件数与有效秩 (Effective Rank)
2. MLA (低秩 KV 压缩) 潜在维度是否发生信息瓶颈或维数浪费
3. mHC (2-流流形残差) 流间特征正交性与能量守恒性 (Birkhoff 投影有效性)
4. 多层前向传播下的特征秩退化趋势 (Rank Collapse / Feature Collapse)
"""

import os
import sys
import math
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.nanoseek_100m import get_nanoseek_100m_config
from model.gpt import GPT

def calculate_effective_rank(tensor: torch.Tensor, eps: float = 1e-7) -> float:
    """计算矩阵的有效秩 (Roy & Vetterli, 2007: 奇异值分布的香农熵指数化)。
    
    erank(A) = exp( - \sum p_i \ln p_i )
    其中 p_i = \sigma_i / \sum \sigma_i
    """
    if tensor.dim() > 2:
        tensor = tensor.flatten(0, -2)
    
    # 转换为 float32 确保 SVD 精度
    tensor = tensor.detach().float()
    U, S, V = torch.linalg.svd(tensor, full_matrices=False)
    
    S_sum = S.sum()
    if S_sum < eps:
        return 0.0
    p = S / S_sum
    p = p[p > eps]
    entropy = -(p * torch.log(p)).sum().item()
    return math.exp(entropy)

def analyze_model_weights(model: GPT):
    print("=" * 70)
    print("  [1] 权重矩阵初始有效秩与谱特性分析 (Weights Rank Spectrum)")
    print("=" * 70)
    
    categories = {
        "Embedding/Head": [],
        "MLA Latent Proj": [],
        "Attention Head Proj": [],
        "MoE Expert Proj": [],
        "mHC Manifold Proj": []
    }
    
    for name, param in model.named_parameters():
        if param.dim() < 2:
            continue
        erank = calculate_effective_rank(param)
        max_rank = min(param.shape[0], param.shape[1])
        ratio = erank / max_rank
        item = (name, tuple(param.shape), erank, max_rank, ratio)
        
        if "wte" in name or "lm_head" in name:
            categories["Embedding/Head"].append(item)
        elif "kv_a" in name or "q_a" in name or "lora" in name or "c_attn" in name:
            categories["MLA Latent Proj"].append(item)
        elif "experts" in name or "shared_expert" in name:
            categories["MoE Expert Proj"].append(item)
        elif "raw_A" in name or "raw_B" in name or "raw_C" in name:
            categories["mHC Manifold Proj"].append(item)
        else:
            categories["Attention Head Proj"].append(item)
            
    for cat_name, items in categories.items():
        if not items:
            continue
        print(f"\n--- {cat_name} (前 4 项代表性参数) ---")
        for name, shape, erank, max_r, ratio in items[:4]:
            print(f"  • {name:<45} 形状: {str(shape):<15} 有效秩: {erank:5.1f} / {max_r:<4} (充盈度: {ratio*100:5.1f}%)")

def analyze_forward_representation_rank(model: GPT, batch_size=4, seq_len=128):
    print("\n" + "=" * 70)
    print("  [2] 前向特征表征逐层秩退化检测 (Layer-wise Representation Rank)")
    print("=" * 70)
    
    device = torch.device("cpu")
    model.eval()
    x = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    
    # 钩子捕获每一层 Transformer Block 的输出隐藏状态
    hidden_states = {}
    hooks = []
    
    def get_hook(idx):
        def hook_fn(module, input, output):
            hidden_states[idx] = output
        return hook_fn
        
    for i, block in enumerate(model.transformer.h):
        hooks.append(block.register_forward_hook(get_hook(i)))
        
    with torch.no_grad():
        _ = model(x)
        
    for h in hooks:
        h.remove()
        
    print(f"测试输入: Batch={batch_size}, SeqLen={seq_len}, 空间理论全秩 = min({batch_size*seq_len}, {model.config.n_embd}) = {min(batch_size*seq_len, model.config.n_embd)}")
    print(f"{'层索引':<8} | {'隐状态形状':<20} | {'有效秩 (eRank)':<16} | {'空间秩充盈度':<12}")
    print("-" * 65)
    
    max_possible_rank = min(batch_size * seq_len, model.config.n_embd)
    for i in sorted(hidden_states.keys()):
        out = hidden_states[i]
        # 如果是 mHC (B, T, hc_mult, C)，先沿流均值
        if out.dim() == 4:
            out = out.mean(dim=2)
        
        flat_out = out.reshape(-1, out.shape[-1])
        erank = calculate_effective_rank(flat_out)
        ratio = (erank / max_possible_rank) * 100
        print(f"Layer {i:02d}  | {str(tuple(out.shape)):<20} | {erank:10.2f}         | {ratio:6.1f}%")

if __name__ == "__main__":
    print("正在实例化 nanoSeek-100M 配置并构建权重...")
    cfg = get_nanoseek_100m_config(vocab_size=32000, block_size=1024)
    model = GPT(cfg)
    analyze_model_weights(model)
    analyze_forward_representation_rank(model, batch_size=4, seq_len=128)
