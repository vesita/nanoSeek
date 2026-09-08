# -*- coding: utf-8 -*-
"""Logits 级残差神经数据库 (ResidualNeuralDB)。

设计理念（解决上一代在第6层 hidden 注入时导致语义破坏的根因）：
1. 注入位置改为【最后一层 Logits 输出端】：
   - 彻底避免在骨干网络中间层（第6层 mHC）污染模型的流形表征与句法连贯性；
   - 外部数据库的作用定位为：【词表概率分布的自适应纠偏 / 外部事实偏置】。
2. 写入机制改为【高惊讶（High-Surprise）错题本模式】：
   - 只有当模型的局部预测 cross-entropy loss > surprise_threshold 时才触发写入；
   - 写入的 Key: 最终隐层状态 h (前文语义表征)；
   - 写入的 Value: 真实目标 token 的残差编码（即正确答案对应的 embedding 向量）。
3. 检索与注入：
   - 检索 Query: 当前 token 的隐层状态 h；
   - 检索出的 Value 经解调投影成 vocab_size (8192) 维的 logits 偏置量 delta_logits；
   - 最终 logits = model_logits + gate * delta_logits。
4. 全程维护用户核心原则：
   - 记忆库纯 no_grad，占用极低显存，不产生优化器状态；
   - 接口网络参数量极小（仅两个线性投影与可学习自适应门控）。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.product_key_memory import ProductKeyMemory


class ResidualNeuralDB(nn.Module):
    def __init__(
        self,
        n_embd=512,
        vocab_size=8192,
        key_dim=128,
        sub_keys=1024,      # 1024^2 = 1M 槽
        top_k=16,
        surprise_threshold=3.0,  # 只有单字 loss > 3.0 (高惊讶/预测困难) 处才写库
        temperature=0.1,
        write_lr=0.2,
        device=None,
        dtype=torch.bfloat16
    ):
        super().__init__()
        self.n_embd = n_embd
        self.vocab_size = vocab_size
        self.key_dim = key_dim
        self.surprise_threshold = surprise_threshold
        self.dtype = dtype
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 1. 外部 no_grad 记忆库 (1M 槽, 存储 key_dim 维度的残差特征)
        self.memory = ProductKeyMemory(
            key_dim=key_dim,
            sub_keys=sub_keys,
            top_k=top_k,
            value_dim=key_dim,
            write_lr=write_lr,
            temperature=temperature,
            device=self.device,
            dtype=dtype
        )

        # 2. 检索接口 (Read Interface)
        # h (n_embd) -> query (key_dim)
        self.q_proj = nn.Linear(n_embd, key_dim, bias=False)
        self.q_norm = nn.LayerNorm(key_dim)
        
        # 读注入门控 (冷启动初始化为偏负，默认少干扰)
        self.read_gate = nn.Linear(n_embd, 1, bias=True)
        nn.init.constant_(self.read_gate.bias, -2.0)  # sigmoid(-2) ~ 0.11
        
        # 检索出的 memory_value (key_dim) -> delta_logits (vocab_size)
        # 将记忆直接转化为词表概率层面的偏置
        self.logits_proj = nn.Linear(key_dim, vocab_size, bias=False)

        # hidden 注入路径（倒数第二层）：memory_value (key_dim) -> delta_hidden (n_embd)
        # 仅在 hidden 注入实验（train_residual_db_hidden.py）使用；logits 路径仍走 logits_proj。
        self.hidden_proj = nn.Linear(key_dim, n_embd, bias=False)

        # 3. 写入接口 (Write Interface)
        # h (n_embd) -> key (key_dim)
        self.k_proj = nn.Linear(n_embd, key_dim, bias=False)
        self.k_norm = nn.LayerNorm(key_dim)
        
        # 写入内容是正确 token embedding 压缩至 key_dim 的目标语义
        self.target_enc = nn.Linear(n_embd, key_dim, bias=False)

        # P2: 写入时排除无信息的目标 token id（<unk>=129 生僻字占位, <eos>/<cont> 分隔符, -100 mask）
        self.ignorable_ids = frozenset({128, 129, 130, -100})

        self.to(self.device).to(dtype)

    def forward_and_correct(self, h, logits, targets=None, token_embeddings=None):
        """
        参数:
            h: 最终隐层输出 (B, T, n_embd)
            logits: 模型原本的未归一化对数概率 (B, T, vocab_size)
            targets: 监督训练标签 (B, T), 仅在训练时提供
            token_embeddings: 模型的词嵌入层权重 (vocab_size, n_embd), 用于提取目标 token 向量
        返回:
            corrected_logits: 纠偏后的 logits (B, T, vocab_size)
            info: 诊断信息 dict (门控大小、高惊讶写入率等)
        """
        B, T, _ = h.shape
        h_flat = h.reshape(B * T, self.n_embd)

        # ---- 1. 读路径 (从记忆库检索并纠偏 Logits) ----
        q = self.q_norm(self.q_proj(h_flat)).float()
        mem_vals, slot_ids, weights = self.memory.retrieve(q) # (B*T, key_dim)
        
        # 将检索内容投影为 logits 偏置
        delta_logits = self.logits_proj(mem_vals).reshape(B, T, self.vocab_size)
        
        # 自适应注入门控
        gate = torch.sigmoid(self.read_gate(h_flat)).reshape(B, T, 1)
        corrected_logits = logits + (gate * delta_logits).to(logits.dtype)

        info = {
            "gate_mean": gate.mean().item(),
            "delta_norm": delta_logits.norm(dim=-1).mean().item(),
            "surprise_written": 0
        }

        # ---- 2. 写路径 (高惊讶错题本模式，仅在训练阶段且有 targets 时触发) ----
        #   P0: value 用「目标嵌入 - 基座预测嵌入」的残差，而非目标嵌入本身。
        #   P2: 排除 <unk>/<eos>/<cont> 等无信息目标，避免记忆被生僻字/分隔符淹没。
        if self.training and targets is not None and token_embeddings is not None:
            with torch.no_grad():
                # 计算逐 token 的 CrossEntropyLoss (未加权)
                orig_losses = F.cross_entropy(
                    logits.reshape(-1, self.vocab_size),
                    targets.reshape(-1),
                    reduction="none"
                ).reshape(B, T)

                # 识别“高惊讶”样本 (模型预测出现明显困难的位置)
                surprise_mask = orig_losses > self.surprise_threshold

                # P2: 排除无信息目标（<unk>占位符、<eos>/<cont> 分隔符、-100 mask 位）
                ignorable = self.ignorable_ids
                is_informative = ~torch.isin(
                    targets, torch.tensor(sorted(ignorable), device=targets.device)
                )
                surprise_mask = surprise_mask & is_informative

                n_surprises = surprise_mask.sum().item()
                info["surprise_written"] = n_surprises

                if n_surprises > 0:
                    # 提取高惊讶位置的上下文隐层作为 Key
                    h_surprise = h[surprise_mask] # (N_s, n_embd)
                    k_surprise = self.k_norm(self.k_proj(h_surprise)).float()

                    # 真实正确 token 的 embedding 表示
                    target_tokens = targets[surprise_mask] # (N_s,)
                    target_emb = F.embedding(target_tokens, token_embeddings) # (N_s, n_embd)

                    # P0: 基座“预测”的 token（argmax），其嵌入作为预测基线
                    pred_tokens = logits.argmax(dim=-1)[surprise_mask]   # (N_s,)
                    pred_emb = F.embedding(pred_tokens, token_embeddings)  # (N_s, n_embd)
                    # 残差 = 目标嵌入 - 基座预测嵌入（基座对→≈0；基座错→大）
                    residual = target_emb - pred_emb
                    val_surprise = self.target_enc(residual)   # (N_s, key_dim)

                    # 写入外部 no_grad 记忆库 (强度与惊讶程度正相关)
                    surprise_weights = torch.clamp((orig_losses[surprise_mask] - self.surprise_threshold) / 2.0, 0.2, 1.0)
                    self.memory.write(k_surprise, val_surprise, weight=surprise_weights)

        return corrected_logits, info
