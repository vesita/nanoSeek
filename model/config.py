"""GPTConfig：模型配置数据类。"""
from dataclasses import dataclass, field


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 的 vocab_size 为 50257，向上填充到最近的 64 的倍数以提高效率
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True：在 Linear 里加 bias。False：效果略好且更快
    # 固定架构：RMSNorm + SwiGLU 硬编码。RoPE 与 wpe 二选一。
    use_rope: bool = False      # RoPE 旋转位置编码（True）或可学习位置编码 wpe（False）
    rope_theta: float = 10000.0 # RoPE 基频（可调，DeepSeek 系列对此做了很多文章）
    # --- V4 稳定性技巧：SwiGLU 输出钳制（DeepSeek-V4 报告）---
    # MoE 训练中，SwiGLU 的输出可能冒出极大的数值异常值（outlier），
    # 被路由机制逐层放大后触发 loss spike。V4 直接把门控输出钳制到 [-v, v]。
    swiglu_clamp: float = 0.0   # 钳制区间半宽；0.0 = 关闭（原始行为）
    # --- MoE 混合专家（DeepSeek-V3/V4 核心）。用 MoE 替换 FFN 层 ---
    use_moe: bool = False       # MoE 替换 MLP/SwiGLU
    n_experts: int = 8          # 路由专家总数
    n_top_k: int = 2            # 每个 token 激活的专家数
    moe_aux_weight: float = 0.01 # 负载均衡辅助损失的权重（Switch 式，use_aux_free_balance=False 时用）
    use_shared_expert: bool = False  # V4：始终激活的共享专家（output = shared(x) + Σ routed(x)）
    use_aux_free_balance: bool = False  # V4：aux-free 偏置修正替代 Switch aux loss（无需调 moe_aux_weight）
    balance_factor: float = 0.001   # aux-free 偏置每步更新幅度
    use_sqrtsoftplus: bool = False  # V4：路由打分用 √softplus(logits)*route_scale 替代 softmax
    route_scale: float = 2.5        # √softplus 打分的缩放系数（V4 默认）
    # --- MLA 多头潜在注意力（DeepSeek-V2 核心）。低秩压缩 KV + 部分 RoPE ---
    use_mla: bool = False       # 用 MLA 替换标准 KV 投影
    kv_lora_rank: int = 64      # KV 压缩后的潜在维度
    qk_rope_head_dim: int = 16  # 每头参与旋转的维数（其余维不带位置信息）
    # --- MTP 多 token 预测（DeepSeek-V3/V4 核心）。训练时额外预测未来的 token ---
    use_mtp: bool = False       # 开启多 token 预测
    n_mtp: int = 1              # 额外预测的 token 数（1 = 多预测 t+2）
    mtp_weight: float = 0.3     # MTP 损失在总 loss 中的权重（DeepSeek-V3 建议 0.3）
    # --- V4 优化器：Muon 替代 AdamW ---
    # 矩阵参数经 Newton-Schulz 正交化后更新，深层训练更稳、收敛更快。
    use_muon: bool = False      # True：矩阵参数用 Muon，embedding/lm_head/norm 用 AdamW
    muon_momentum: float = 0.95 # Muon 动量系数
    muon_ns_steps: int = 10     # Newton-Schulz 迭代次数（默认 8 激进 + 2 经典）
    # --- V4 核心：CSA/HCA 混合注意力（简化教育版）---
    # 块级 KV 压缩 + top-k 稀疏块选择 + 滑窗局部注意力 + HCA 重度压缩全局信号。
    # 核心收益：注意力开销从 O(T²) 降到 O(T·(nb + win))，这是 1M 上下文能跑起来的关键。
    use_csa: bool = False       # True：用 CSA 混合注意力（建议搭配 use_rope）
    csa_compress: int = 16      # 块大小 m：每 m 个 token 压成 1 个潜在 KV
    csa_topk: int = 4           # 每个 query 稀疏选几个压缩块（不看全部）
    csa_window: int = 64        # 滑窗：保留最近多少个原始 token（局部细节）
    use_hca: bool = False       # 加 HCA：重度压缩全局信号（无稀疏选择）
    use_csa_learnable: bool = True  # V4：可学习门控池化替代平均池化（压缩块内 m 个 token）
    # --- V4 结构设计升级（实验性，默认全关）---
    # Attention Sinks：每头一个可学习标量偏置，作为 softmax 的"垃圾桶"吸收无关注意力。
    use_attn_sink: bool = False
    # mHC 超连接：4 流并行残差（X_{l+1} = B·X_l + C·F(A·X_l)，A/C sigmoid 有界、B 双重随机）。
    use_mhc: bool = False
    hc_mult: int = 4            # 残差流数（V4 原版 = 4）
    # Lightning Indexer：学习型块选择替代 CSA 的 raw top-k（256 上下文收益有限，先搭框架）。
    use_lightning_indexer: bool = False
    # Hash 路由：前 num_hash_layers 层用 hash(token_id)%n_experts 确定性分配（0 = 禁用）。
    num_hash_layers: int = 0
    # 计算图重排（拓扑实验）：块内子层顺序。
    # "attn_ffn"（默认，标准 Pre-LN 布线）| "ffn_attn"（先 FFN 后注意力）。
    # 只交换执行顺序，不新增参数、不改权重名（attn 永远用 ln_1，mlp 永远用 ln_2）。
    block_order: str = "attn_ffn"
    # 稀疏注意力布线（拓扑实验）：这些层索引跳过注意力（只跑 FFN 子层）。
    # 参数保留但不参与前向 → 参数量严格不变，纯接线改变；空列表 = 所有层都有注意力。
    # 例：[0, 2, 4] → 6 层里第 1/3/5 层只跑 FFN，注意力稀疏布在第 2/4/6 层。
    no_attn_layers: list = field(default_factory=list)
    # 显式记忆 token（拓扑实验，Python-only）：序列前插入 K 个可学习嵌入，参与每一层
    # 注意力+FFN，作为跨 token 的长程全局工作区（直击 Sinks 训满失效/采样死循环的病根）。
    # 0 = 关闭（与老模型完全一致，checkpoint 兼容不破）。
    # 建议 K = csa_compress 的整数倍（如 16），使记忆块与压缩块边界对齐、真实 token 的
    # 块划分零偏移。参数量 = K×n_embd（K=16, d=80 → 1,280，全部参与训练）。
    n_memory_tokens: int = 0
    # --- 对数放缩残差（数值连接实验，零参数）---
    # 普通残差线性相加 x+F(x)；开启后换成对数域 soft-max 合并 LSE(x,F(x))。
    # 有界收缩（≤max+log2）+ 软选择偏置，直接压制 SwiGLU 极端值和「loss 骗低」。
    # 与 mHC 互斥（mHC 已有自己的 4 流残差拓扑，v1 只在标准路径应用）。
    use_lse_residual: bool = False
    # --- 对数放缩门控混合（LSE gate-mix，v2）---
    # 对上轮「纯 LSE 硬替换拖累学力」的修正：残差合并变成可学习标量 α 的插值
    #     x ← α·x + (1-α)·LSE(x, F(x))，α = sigmoid(raw_gate)，raw_gate 初始 0 → α=0.5
    # - α→1：偏向纯线性（保留 lin 臂的强学力 / 表征累加）
    # - α→0：偏向纯 LSE（偏 lse 臂的抗尖刺 / 抗过度自信 → 抗重复坍缩）
    # 模型自己权衡每条残差路径的软硬程度。每层 1 个标量，参数量极小（6 层 = 6 参数）。
    # 与 mHC 不互斥（gate-mix 是标准单流残差内部混合，非 mHC 那种 4 流拓扑）。
    use_lse_gate: bool = False
    # --- 注意力稳定性：QK-Norm（零/近零参数）---
    # 对 q/k 做 L2 归一化 + 每头可学习 scale，再乘 1/sqrt(head_dim)。
    # 目标：把注意力 logits 的尺度拉平，压制「过度自信 → 重复坍缩」，对标准注意力、
    # CSA、MLA 都生效。scale 初始 = sqrt(head_dim)，前向等价于原始 `q·k/sqrt(d)`
    # 的按头缩放，不会改变起点快照。
    use_qk_norm: bool = False
    # --- MoE 稳定性：Router Z-Loss ---
    # DeepSeek 系列用于稳定 MoE 路由的辅助正则：z = logsumexp(router_logits)，
    # 惩罚 z² 的平均，防止路由 logits 数值过大导致训练波动 / 专家崩溃。
    # 权重 0 = 关闭；建议从 1e-4 起步。
    z_loss_weight: float = 0.0