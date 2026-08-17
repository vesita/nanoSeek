"""模型核心包：重导出 model.py 的所有公开符号。

让外部可以继续 `from model import GPT, GPTConfig`（内部文件在 model/model.py）。
"""
from .model import (
    GPT,
    GPTConfig,
    MoE,
    SwiGLU,
    Muon,
    zeropower_via_newtonschulz,
    sinkhorn_knopp,
    logsumexp_residual,
)

__all__ = [
    "GPT",
    "GPTConfig",
    "MoE",
    "SwiGLU",
    "Muon",
    "zeropower_via_newtonschulz",
    "sinkhorn_knopp",
    "logsumexp_residual",
]
