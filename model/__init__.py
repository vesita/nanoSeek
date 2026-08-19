"""模型核心包：从拆分后的模块重导出所有公开符号。

让外部可以继续 `from model import GPT, GPTConfig`（内部按模块拆分）：
- config.py    ：GPTConfig
- utils.py     ：RMSNorm / RoPE / LSE 残差 / Sinkhorn / Newton-Schulz
- attention.py ：CausalSelfAttention
- mlp.py       ：SwiGLU / MoE
- block.py     ：Block / MTPModule
- optimizer.py ：Muon / MuonAdamW
- gpt.py       ：GPT（模型主体）
"""
from .config import GPTConfig
from .utils import (
    RMSNorm,
    logsumexp_residual,
    precompute_rope_freqs,
    apply_rotary_pos_emb,
    sinkhorn_knopp,
    zeropower_via_newtonschulz,
)
from .attention import CausalSelfAttention
from .mlp import SwiGLU, MoE
from .block import Block, MTPModule
from .optimizer import Muon, MuonAdamW
from .gpt import GPT

__all__ = [
    "GPT",
    "GPTConfig",
    "CausalSelfAttention",
    "SwiGLU",
    "MoE",
    "Block",
    "MTPModule",
    "Muon",
    "MuonAdamW",
    "RMSNorm",
    "logsumexp_residual",
    "precompute_rope_freqs",
    "apply_rotary_pos_emb",
    "sinkhorn_knopp",
    "zeropower_via_newtonschulz",
]