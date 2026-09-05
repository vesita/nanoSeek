"""
Compact Transformer Model for Arithmetic (nano_arith).

Features:
- Configurable layers (e.g. 2 to 4), hidden dim (e.g. 64 to 128), heads (e.g. 4)
- Lightweight parameter count (~50k to 300k params)
- Rotary Position Embeddings (RoPE) or standard learnable embeddings
- RMSNorm + SwiGLU / GeLU MLP
- Built-in activation hooks for neuron & attention map inspection
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ModelConfig:
    vocab_size: int = 18
    d_model: int = 64
    n_layers: int = 3
    n_heads: int = 4
    d_ff: int = 192  # 3 * d_model
    max_seq_len: int = 64
    dropout: float = 0.0
    bias: bool = False

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 64, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:seq_len, :], self.sin_cached[:seq_len, :]

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k shape: [B, H, S, D]
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, S, D]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        
        # Stored attention weights for inspection
        self.last_attn_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, C = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        
        attn = F.softmax(scores, dim=-1)
        self.last_attn_weights = attn.detach()
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, C)
        return self.out_proj(out)

class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        # SwiGLU: w1 * silu(w3) -> w2
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias)
        
        # Stored neuron activations for inspection
        self.last_intermediate: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w1(x))
        inter = gate * self.w3(x)
        self.last_intermediate = inter.detach()
        return self.w2(inter)

class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin, mask)
        x = x + self.ffn(self.norm2(x))
        return x

class NanoArithTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rope = RotaryEmbedding(dim=cfg.d_model // cfg.n_heads, max_seq_len=cfg.max_seq_len)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.embed.weight
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        B, S = input_ids.shape
        cos, sin = self.rope(input_ids, S)
        
        # Causal mask: upper triangular with -inf
        mask = torch.full((S, S), float("-inf"), device=input_ids.device)
        mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)
        
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
            
        x = self.norm_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            # Shift so tokens predict the next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.cfg.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
            
        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int = 10, eos_id: int = 15) -> torch.Tensor:
        # prompt_ids: [B, S]
        out = prompt_ids.clone()
        for _ in range(max_new_tokens):
            logits = self(out)["logits"][:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            out = torch.cat([out, next_token], dim=-1)
            if (next_token == eos_id).all():
                break
        return out

if __name__ == "__main__":
    cfg = ModelConfig()
    model = NanoArithTransformer(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"NanoArith Model initialized: {total_params:,} parameters")
    x = torch.randint(0, 16, (2, 16))
    out = model(x, labels=x)
    print("Forward pass successful. Loss:", out["loss"].item())
