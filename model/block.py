"""Block：Transformer 块（标准单流 / mHC 4 流并行残差）+ MTPModule。"""
import dataclasses
import torch
import torch.nn as nn
from torch.nn import functional as F

from .utils import RMSNorm, logsumexp_residual, sinkhorn_knopp
from .attention import CausalSelfAttention
from .mlp import SwiGLU, MoE


class Block(nn.Module):

    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.config = config
        self.use_mhc = config.use_mhc
        # 对数放缩残差 v1：与 mHC 互斥（mHC 已有自己的 4 流残差拓扑，v1 只服务标准路径）。
        assert not (config.use_mhc and config.use_lse_residual), \
            "use_lse_residual 与 use_mhc 互斥：LSE 对数域残差 v1 只作用在标准单流路径！"
        self.use_lse_residual = config.use_lse_residual
        # 对数放缩门控混合 v2：非 4 流拓扑，与 mHC 不互斥，但 v1 纯残差是它的退化特例。
        assert not (config.use_lse_residual and config.use_lse_gate), \
            "use_lse_residual（纯硬替换）与 use_lse_gate（门控混合）只能开一个——gate-mix 已含 v1"
        self.use_lse_gate = config.use_lse_gate
        # 稀疏注意力布线：本层是否跳过注意力子层（只跑 FFN）。
        self.skip_attn = layer_idx in config.no_attn_layers
        # 稀疏注意力布线：skip 层的注意力槽替换为 FFN（hidden_scale=3 → 57,600 参数 ≈
        # CausalSelfAttention 57,604，仅差 attn_sink 的 4 参数/层）→ 规模严格补回基线，
        # 只剩"无注意力"这一个拓扑变量，且所有参数真正参与训练（无冻结废参数）。
        # 布线从 A F 变 F F（FFN 变换 ×2）。
        self.attn = SwiGLU(config, hidden_scale=3) if self.skip_attn else CausalSelfAttention(config)
        # 固定架构：RMSNorm + 残差；FFN 用 MoE（可选）或 SwiGLU
        self.ln_1 = RMSNorm(config.n_embd)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MoE(config, use_hash=layer_idx < config.num_hash_layers) \
            if config.use_moe else SwiGLU(config)

        if self.use_mhc:
            # mHC 超连接：4 流并行残差。每流宽度仍为 n_embd，子层 F 只跑 1 次。
            # 两组 A/B/C（attn 子层 + FFN 子层），见 _mhc_forward。
            hc = config.hc_mult
            self.hc_mult = hc
            self.raw_A_attn = nn.Parameter(torch.zeros(1, hc))       # → sigmoid（有界非负）
            self.raw_B_attn = nn.Parameter(torch.zeros(hc, hc))      # → softplus → sinkhorn（双重随机）
            self.raw_C_attn = nn.Parameter(torch.zeros(hc, 1))       # → sigmoid
            self.raw_A_ffn = nn.Parameter(torch.zeros(1, hc))
            self.raw_B_ffn = nn.Parameter(torch.zeros(hc, hc))
            self.raw_C_ffn = nn.Parameter(torch.zeros(hc, 1))
            # 初始化：A/C 全 0 → sigmoid=0.5（各流等权）；B 全 0 → exp(0)=1 → sinkhorn
            # 后均匀混合（对称起点，不扭曲初始行为）。

        # v2 门控混合：可学习标量 α = sigmoid(raw_gate)，起点 sigmoid(0)=0.5（线性/LSE 均衡）。
        # 每层一个标量，attn/ffn 两个子层残差共享同一个 α（见 forward 的 res 定义）。
        if self.use_lse_gate:
            self.raw_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        if self.use_mhc:
            return self._mhc_forward(x)
        # skip_attn 层的 attn 槽已是宽 FFN（SwiGLU），走同样双子层路径，无特殊分支。
        # 计算图重排：默认 attn→ffn；block_order="ffn_attn" 时先 FFN 后注意力。
        # norm 与子层绑定不可拆：attn 永远用 ln_1，mlp 永远用 ln_2，只调换两段顺序。
        # 残差合并：三种模式
        #   0) 线性加法           : x + F(x)
        #   1) 纯 LSE (v1)        : LSE(x, F(x))
        #   2) 门控混合 (v2)      : α·x + (1-α)·LSE(x, F(x))，α = sigmoid(raw_gate) 可学习
        # 门控混合保留线性主通道（学力）∩ LSE 分支（抗尖刺/抗过度自信）。
        if self.use_lse_residual:
            res = lambda a, b: logsumexp_residual(a, b)
        elif self.use_lse_gate:
            alpha = torch.sigmoid(self.raw_gate)
            res = lambda a, b: alpha * a + (1.0 - alpha) * logsumexp_residual(a, b)
        else:
            res = lambda a, b: a + b
        if self.config.block_order == "ffn_attn":
            x = res(x, self.mlp(self.ln_2(x)))
            x = res(x, self.attn(self.ln_1(x)))
        else:
            x = res(x, self.attn(self.ln_1(x)))
            x = res(x, self.mlp(self.ln_2(x)))
        return x

    def _mhc_forward(self, x):
        """mHC 4-copy：X' = B·X + C·F(A·X)。

        x: (B, T, hc, d)  4 个并行残差流。
        每步：A（1×hc，sigmoid）把 4 流压成 1 流给子层 F；子层只算 1 次；
        C（hc×1，sigmoid）把子层输出展开回 4 流；B（hc×hc，双重随机）混合残差流。
        默认先注意力后 FFN；block_order="ffn_attn" 时对调。
        """
        hc = self.hc_mult
        # skip_attn 层的 attn 槽已是宽 FFN，is_attn=True 子层照样跑（SwiGLU 变换），
        # 只是不涉及注意力；两组 A/B/C 都参与训练。
        if self.config.block_order == "ffn_attn":
            x = self._mhc_sublayer(x, hc, is_attn=False)  # FFN 先
            x = self._mhc_sublayer(x, hc, is_attn=True)   # 注意力后
        else:
            x = self._mhc_sublayer(x, hc, is_attn=True)   # 注意力先
            x = self._mhc_sublayer(x, hc, is_attn=False)  # FFN 后
        return x

    def _mhc_sublayer(self, x, hc, is_attn):
        """mHC 单个子层：A 压流 → 子层 F 跑 1 次 → C 展开 → B 混合残差。
        is_attn=True 取 attn 组 A/B/C + ln_1/attn；False 取 ffn 组 + ln_2/mlp。
        两组权重各自跟随所属子层，重排顺序时无需 remap。
        """
        A = torch.sigmoid(self.raw_A_attn if is_attn else self.raw_A_ffn)
        h_in = (x * A.view(1, 1, hc, 1)).sum(dim=2)            # (B, T, d)
        h_out = (self.attn(self.ln_1(h_in)) if is_attn
                 else self.mlp(self.ln_2(h_in)))               # 子层只跑 1 次
        C = torch.sigmoid(self.raw_C_attn if is_attn else self.raw_C_ffn)
        delta = h_out.unsqueeze(2) * C.view(1, 1, hc, 1)       # (B, T, hc, d)
        B_ds = sinkhorn_knopp(F.softplus(
            self.raw_B_attn if is_attn else self.raw_B_ffn))   # (hc, hc) 双重随机
        return torch.einsum('bthd,hc->btcd', x, B_ds) + delta  # 残差混合 + 子层增量

    def get_moe_aux_loss(self):
        """如果本块是 MoE，返回其辅助损失；否则返回 None（GPT 据此累加）。"""
        if isinstance(self.mlp, MoE):
            return self.mlp.get_aux_loss()
        return None

    def get_indexer_loss(self):
        """本块注意力的 Lightning Indexer 辅助损失（未启用返回 None）。"""
        return self.attn.get_indexer_loss()


class MTPModule(nn.Module):
    """MTP：多 token 预测模块（DeepSeek-V3 核心）。

    普通 GPT 在位置 t 只预测 t+1；MTP 额外提供一个模块，用
    「位置 t 的隐藏状态 + 目标 token t+1 的嵌入」去预测 t+2。
    推理时这些模块不参与（生成没有 targets），训练信号却成倍增强——
    相当于几乎白赚一个"下一步预演"任务。
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.emb_proj    = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.norm = RMSNorm(config.n_embd)
        # 复用单层 Block 做特征融合（内部是因果 attention，正好合适）。
        # 强制关闭 mHC：MTP 的输入是单流 hidden + next_emb，不是 4 流残差。
        self.block = Block(dataclasses.replace(config, use_mhc=False))

    def forward(self, hidden, next_emb):
        # hidden:  (B, T, n_embd) 主模型在位置 t 的隐藏状态
        # next_emb: (B, T, n_embd) 目标 token t+1 的嵌入（提前剧透下一步）
        h = self.hidden_proj(hidden) + self.emb_proj(next_emb)
        h = self.norm(h)
        return self.block(h)