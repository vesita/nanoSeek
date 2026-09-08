# -*- coding: utf-8 -*-
"""接口网络：读写分离的 ReadInterface / WriteInterface。

桥接模型 hidden state 和外部 no_grad 记忆库。这是**唯一参与梯度**的部分。

架构（读写分离，DNC/记忆增强网络式）：
  · ReadInterface  (读接口)   — 决定"该回忆什么"
      - encode_query: bf16 hidden -> f32 query（供记忆库检索打分）
      - judge_read:  bf16 hidden -> 0-1 门控（该token是否值得查记忆）
      - decode:      bf16 value -> bf16 增量（注入回模型）
  · WriteInterface (写接口)   — 决定"该记住什么"
      - encode_key:  bf16 hidden -> f32 key（写入记忆库的检索键）
      - encode_val:  bf16 hidden -> bf16 value（写入记忆库的内容）
      - gate_write:  bf16 hidden -> 0-1 门控（该token是否值得记）

两者的梯度路径天然分离：
  · ReadInterface 吃"记忆帮助预测正确 token"的梯度（经 decode 注入回模型）
  · WriteInterface 吃"记住这个 token 对将来有用"的梯度（经 encode 反馈）
这与外部记忆库 no_grad 形成完整闭环。

与模型基础类型同构（bf16）；检索 query 转 f32 提升检索质量。
"""
import torch
import torch.nn as nn


class ReadInterface(nn.Module):
    """读接口：encode query（检索）+ judge（是否读）+ decode（注入回模型）。"""
    def __init__(self, n_embd, key_dim=None, top_k=8):
        super().__init__()
        self.n_embd = n_embd
        self.top_k = top_k
        self.key_dim = key_dim or max(n_embd // 4, 64)
        # query 编码（降维到检索空间）
        self.q_proj = nn.Linear(n_embd, self.key_dim, bias=False)
        self.q_norm = nn.LayerNorm(self.key_dim)
        # 是否读的门控（surprise）
        self.judge_read = nn.Linear(n_embd, 1, bias=True)
        nn.init.constant_(self.judge_read.bias, -0.5)  # 冷启动默认少读
        # value 解码回模型维度
        self.v_proj = nn.Linear(self.key_dim, n_embd, bias=False)

    def encode_query(self, h):
        """bf16 hidden -> f32 query（检索打分）。h: (..., n_embd)"""
        return self.q_norm(self.q_proj(h)).float()

    def judge(self, h):
        """bf16 hidden -> (...,1) 0-1 读门控。"""
        return torch.sigmoid(self.judge_read(h))

    def decode(self, value):
        """bf16 value (..., key_dim) -> bf16 增量 (..., n_embd)。"""
        return self.v_proj(value)


class WriteInterface(nn.Module):
    """写接口：encode key（检索键）+ encode value（内容）+ gate（是否写）。"""
    def __init__(self, n_embd, key_dim=None):
        super().__init__()
        self.n_embd = n_embd
        self.key_dim = key_dim or max(n_embd // 4, 64)
        # key 编码（写入记忆库的检索键，与 ReadInterface 同 key_dim 空间）
        self.k_proj = nn.Linear(n_embd, self.key_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.key_dim)
        # value 编码（写入内容，记忆库 key_dim 维）
        self.v_proj = nn.Linear(n_embd, self.key_dim, bias=False)
        # 是否写的门控
        self.gate_write = nn.Linear(n_embd, 1, bias=True)
        nn.init.constant_(self.gate_write.bias, -0.5)  # 冷启动默认少写

    def encode_key(self, h):
        """bf16 hidden -> f32 query（写入的检索键）。h: (..., n_embd)"""
        return self.k_norm(self.k_proj(h)).float()

    def encode_val(self, h):
        """bf16 hidden -> bf16 value（写入内容）。"""
        return self.v_proj(h)

    def gate(self, h):
        """bf16 hidden -> (...,1) 0-1 写门控。"""
        return torch.sigmoid(self.gate_write(h))
