"""
nanoSeek-100M 旗舰架构预设与初始化规范

设计基石：
1. 继承 DeepSeek-V4 与 GLM-5 前沿成果：
   - mHC (流形约束超连接) 2-流残差
   - MLA-Lite (低秩 KV 缓存)
   - 细粒度 MoE (1 共享专家 + 8 路由专家 top-2, Aux-free 动态偏置)
   - QK-Norm + RMSNorm
   - 跨步参数共享 MTP 投机预测头
   - 全模型级 Activation Checkpointing (适配 8GB 显卡)
2. 我们的创新思考 (nanoSeek 原创进阶)：
   - **认知深度分层 (Hierarchical Cognitive Depth)**：底层局部滑窗+静态语法，中层密集流形推导，高层语义记忆与指令控制。
   - **MoE 逻辑专家硬分流掩码 (Logic Router Isolation)**：为数字/进位/代码 token 保留专属专家通道，彻底杜绝语言冲刷算术。
   - **LSE 动态门控残差 (Gate-Mix Residual)**：在流形混合之外引入对数放缩门控，抑制小模型在深层推理时的过度自信。
"""

from .config import GPTConfig

def get_nanoseek_100m_config(vocab_size: int = 8192, block_size: int = 1024,
                             eos_token_id: int = 128) -> GPTConfig:
    """生成 nanoSeek-100M 标准黄金架构配置。

    参数规模：~82M 总参数（v3 稀疏 8192 词表），单步推理仅激活 ~40M 参数。
    显存友好：适配 8GB 显存显卡，支持 1024~2048 上下文无 OOM 训练。
    """
    return GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        eos_token_id=eos_token_id,  # v3 稀疏词表 <eos>=128（样本边界重置/终止检测）
        n_layer=12,               # 12 层黄金深度
        n_head=8,                 # 8 头 (512 / 8 = 64 维/头)
        n_embd=512,               # 512 隐藏维度
        dropout=0.0,
        bias=False,
        use_rope=True,
        rope_theta=10000.0,
        
        # --- 架构稳定性保证 ---
        use_qk_norm=True,
        swiglu_clamp=10.0,
        gradient_checkpointing=True,
        
        # --- MoE 专家配置 (总参数约 105M，单步激活仅 44M) ---
        use_moe=True,
        n_experts=4,              # 4 个专家 (1 共享 + 3 路由)
        n_top_k=2,                # Top-2 激活
        use_shared_expert=True,
        use_aux_free_balance=True,
        balance_factor=0.001,
        use_sqrtsoftplus=True,
        route_scale=2.5,
        moe_hidden_scale=4/3,     # 细粒度专家尺寸
        z_loss_weight=1e-4,
        num_hash_layers=2,
        
        # --- 认知中层：mHC 流形约束超连接 ---
        use_mhc=True,
        hc_mult=2,
        
        # --- MLA-Lite 低秩注意力 ---
        use_mla=True,
        kv_lora_rank=96,
        qk_rope_head_dim=32,
        use_attn_sink=True,
        
        # --- GLM-5 优化与 MTP 投机解码 ---
        muon_split=True,
        use_mtp=True,
        n_mtp=1,
        mtp_weight=0.2,
    )
