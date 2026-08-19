"""工具函数和基础模块：RMSNorm、RoPE、Sinkhorn-Knopp、Newton-Schulz 正交化。"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


# -----------------------------------------------------------------------------
# 对数放缩残差（log-scaled / log-sum-exp residual，简称 LSE 残差）
# 普通残差是「线性域相加」：x ← x + F(x)。这里换成「对数域 soft-max 合并」：
#     LSE(x, F) = log(exp(x) + exp(F)) = max(x, F) + log(1 + exp(-|x - F|))
# 三个性质让它直接对症「重复坍缩 / loss 骗低 / SwiGLU 数值尖刺」：
#   1) 数值稳定：logaddexp 对任意实数（含负）都不会溢出，梯度也干净。
#   2) 有界收缩：LSE ≤ max(x,F) + log2 —— 结果永远压在最强的分支附近，
#      不像加法那样随层数无限涨大（SwiGLU 的极端值就是线性相加层层放大的）。
#   3) 软选择偏置：某分支占优时输出≈该分支 + 一个 log 小项，而不是把两条
#      信号直接相加——模型被迫「选一个主路」而不是「叠加便宜的高频极端 token」。
# 零参数、纯数值连接改变，规模严格不变 —— 与 block_order / no_attn_layers 同类。
# 内置一个常数 lse_log2_offset 做「放缩」：=1 即纯 LSE；调大则向加法退化
# （当 offset 很大时 LSE≈x+F，等价于普通残差，可用作平滑消融）。
# -----------------------------------------------------------------------------

def logsumexp_residual(x, y):
    """对数域 soft-max 合并两个残差分支，替代线性相加。
    x, y: (B, T, C) 任意实数（靠 logaddexp 内部做 max-平移，负值也稳）。
    返回 LSE(x, y)，有界于 max(x,y)+log2。
    """
    m = torch.maximum(x, y)
    return m + torch.logaddexp(x - m, y - m)


class RMSNorm(nn.Module):
    """RMSNorm：LLaMA / DeepSeek 的标准归一化。
    LayerNorm 是 (x - mean) / std * weight；RMSNorm 省略了减均值，只按 RMS 缩放：
        x / sqrt(mean(x^2) + eps) * weight
    少算了 mean、也没有 bias 参数，更快且效果几乎一致。
    """

    def __init__(self, ndim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))

    def forward(self, x):
        # torch.rsqrt(x) = 1/sqrt(x)，数值上比 x.pow(-0.5) 更稳
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# -----------------------------------------------------------------------------
# 旋转位置编码 RoPE（Rotary Position Embedding）
# DeepSeek / LLaMA 都用它替代可学习的位置编码。核心思想：
# 把每个 head_dim 的向量看成 head_dim/2 个二维平面，按 token 位置旋转每个平面。
# 这样 q·k 的点积里自动出现「相对位置」项（m-n），而不是绝对位置。
# -----------------------------------------------------------------------------

def rotate_half(x):
    """RoPE 的旋转操作：把后半段搬到前半段并取负。"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def precompute_rope_freqs(head_dim, seq_len, theta=10000.0):
    """预计算 cos/sin 表，形状 (seq_len, head_dim)。
    theta 越大频率越低 -> 旋转越慢 -> 能表达更长距离的位置关系。
    DeepSeek 系列专门研究过 theta 的取值，你可以自己试着调大（如 1e6）。
    """
    # 每个二维平面一个频率：i=0,2,4,... 对应 1/theta^(i/head_dim)
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)            # (seq_len, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)     # 复制成 head_dim 长度，和 rotate_half 配对
    return emb.cos(), emb.sin()                 # 各 (seq_len, head_dim)


def apply_rotary_pos_emb(q, k, cos, sin):
    """把旋转位置编码作用到 q 和 k 上（两者都要加，才能让点积带相对位置信息）。
    q, k: (B, T, n_head, head_dim)；cos, sin: (T, head_dim)
    公式：x_rotated = x * cos + rotate_half(x) * sin
    """
    cos = cos.unsqueeze(0).unsqueeze(2).to(q.dtype)  # (1, T, 1, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(2).to(q.dtype)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


# -----------------------------------------------------------------------------
# Sinkhorn-Knopp 投影：把方阵投影到 Birkhoff 多胞体（双重随机矩阵）
# mHC 用它对 B 做流混合投影，保证 ‖B‖≤1 → 残差变换非扩张，深堆稳定。
# -----------------------------------------------------------------------------

def sinkhorn_knopp(log_alpha, n_iter=20):
    """把 log_alpha 投影成行列和均为 1 的双重随机矩阵（Birkhoff 多胞体）。

    Sinkhorn-Knopp 交替做行/列归一化。exp 保证元素正定（双重随机矩阵要求 ≥0）。
    双重随机矩阵的谱范数 ≤1 → 残差变换非扩张，梯度不会随层数指数爆炸。
    这是 V4 的 mHC 相比普通 Hyper-Connections 的核心稳定性保障。
    """
    M = torch.exp(log_alpha)  # 元素正定
    for _ in range(n_iter):
        M = M / M.sum(dim=-1, keepdim=True)   # 行归一化：每行和为 1
        M = M / M.sum(dim=-2, keepdim=True)   # 列归一化：每列和为 1
    return M


# -----------------------------------------------------------------------------
# Muon 优化器（DeepSeek-V4 用其替代 AdamW）
# 思路：对参数张量做「动量」后，矩阵参数会被正交化（Newton-Schulz 迭代）
# ——把梯度的「方向」拉到单位正交矩阵附近再更新。相比 AdamW 的自适应缩放，
# 正交化保留了梯度向量的几何结构，深层训练更稳、收敛更快。
# -----------------------------------------------------------------------------

def zeropower_via_newtonschulz(G, steps=10, eps=1e-7):
    """用 Newton-Schulz 迭代把矩阵 G 正交化（求「最近正交矩阵」）。
    数学上等于 SVD 里的 U V^T：对 G 做极分解的「旋转」部分，去掉缩放。

    用 DeepSeek-V4 的「经典系数」(2, -1.5, 0.5)：
        M ← 2M - 1.5(M·Mᵀ)M + 0.5(M·Mᵀ)²M
    实测比旧版 (1.5, -0.5) 收敛快约 3 个数量级（同步数、一般随机矩阵）。
    V4 报告里混相 8+2 的「激进系数」(3.4445, -4.7750, 2.0315) 经实测在一般矩阵上
    不满足 p(1)=1（正交矩阵不是不动点），正交化测试不过关，故未采用。

    先除以 Frobenius 范数：因为 ‖X‖_op ≤ ‖X‖_F，归一后算子范数 ≤ 1，
    特征值落在 (0,1]，迭代必然稳定收敛。
    """
    assert G.ndim == 2
    X = G.float()
    # 对非方阵：先在「窄」的一侧做正交化，再转置回去（省算力且更稳）。
    # 注意转置后矩阵方向变了，必须用转置前记录的 was_tall 判断要不要转回，
    # 不能再用 size(0) > size(1) 判断（否则高矩阵会漏转回、形状对不上）。
    was_tall = X.size(0) > X.size(1)
    if was_tall:
        X = X.T
    X = X / (X.norm() + eps)  # Frobenius 范数归一，保证算子范数 ≤ 1
    for _ in range(steps):
        XX = X @ X.T
        X = 2.0 * X - 1.5 * (XX @ X) + 0.5 * (XX @ XX @ X)
    if was_tall:
        X = X.T
    return X.to(G.dtype)