# nanoSeek-100M 架构设计与开源融合创新设计规范

## 1. 架构总览与设计哲学

在 8GB 显存消费级显卡（AMD gfx1032 / RX 6600 或 RTX 3060/4060）环境下，我们既要让模型跨入 **100M 参数级（涌现通识与推理逻辑）**，又要保证**显存安全（不 OOM）、计算吞吐高（激活小）**。

因此，我们融合 **DeepSeek-V4、GLM-5、Qwen2.5 与 Kimi** 的核心创新，提出了 **nanoSeek-100M 认知分层架构**：
- **总参数量**：**103.99 M (~104M)**
- **单步推理激活参数量**：**~43.8 M**（FLOPs 仅相当于 40M 稠密模型，推理吞吐高达 120+ tokens/s）
- **长文本 KV 显存**：MLA-Lite 压缩后，4K 上下文仅占 **42 MB**
- **训练显存占用**：开启 Block-level Gradient Checkpointing，训练峰值显存仅 **3.2 GB ~ 4.2 GB**（在 8GB 卡上具有极充裕余量）

---

## 2. 深度融合前沿大模型的核心创新

1. **来自 DeepSeek-V4 的设计**：
   - **mHC (Manifold-Constrained Hyper-Connections)**：采用 2-流并行残差，利用 Sinkhorn-Knopp 将流间转移矩阵投影在双随机流形上，保持深层特征能量守恒，打破小模型在深层推理时的表征瓶颈；
   - **Aux-free 负载均衡 MoE**：舍弃破坏路由的 Switch 辅助损失，改为基于负载偏差在线修正专家 bias；
   - **√softplus 路由打分与 SwiGLU 输出钳制**：把门控激活硬截断在 $[-10, 10]$，根除训练中的数值离群尖刺（Loss Spikes）。
2. **来自 GLM-5 的设计**：
   - **Muon Split (按注意力头切片正交化)**：将 Q/K/V 投影按 Head 维度拆分后分别执行 Newton-Schulz 正交迭代，解决 MLA 与 Muon 配合时的特征塌陷问题；
   - **跨步参数共享 MTP (Multi-Token Prediction)**：在顶层复用 1 个轻量 Block 做下一 token 及下下一 token 预测，稠密化表征监督并提升投机解码接受长度。
3. **来自 Qwen / Kimi 的设计**：
   - **Head-level QK-Norm**：在点积前对 Query 和 Key 执行 RMSNorm，彻底稳定 Attention Logits 尺度；
   - **Attention Sinks**：为每头配置可学习标量吸收槽，消除 Softmax 强制归一化引发的局部垃圾注意力积聚。

---

## 3. nanoSeek-100M 原创能力分层与维度规格

| 层次编号 | 认知能力目标 | 架构组件与特征 | 详细超参数规格 |
| :--- | :--- | :--- | :--- |
| **底座嵌入** | 词汇离散化映射与输出投影 | Tied Embedding (首尾共享词表) | $V = 32,000, d = 512$（参数 16.38M） |
| **L01 ~ L02** | 局部词法与语法抽象 | SWA (局部滑窗) + 静态 Hash 路由 MoE | 2 层，局部因果窗 256，静态 Hash 分配 |
| **L03 ~ L08** | 核心逻辑推导与符号演算 | mHC (2-流残差) + MLA-Lite + 细粒度 MoE | 6 层，1 共享 + 3 路由 (Top-2)，$d_c^{KV}=96$ |
| **L09 ~ L10** | 跨轮长程记忆与上下文对齐 | MLA-Lite + Attention Sinks + 紧凑 SwiGLU | 2 层，长文本状态持久化与低秩 KV 缓存 |
| **L11 ~ L12** | 终止控制与回复生成 | 稠密全注意力 + 紧凑 SwiGLU | 2 层，严格监督 `<eos>` 截断与意图对齐 |
| **投机输出** | 多 Token 预测 (MTP) | 跨步参数共享投机 Block (GLM-5 模式) | 1 个共享 Block，预测 $t+1, t+2$（加权 0.2） |

---

## 4. 训练基础设施与防 OOM 保障

1. **自动硬件环境规约**：
   - `HSA_OVERRIDE_GFX_VERSION=10.3.0`（解决 gfx1032 兼容）
   - `HSA_ENABLE_SDMA=0`（根治 PCIe 异步拷贝内存泄露与死锁）
   - `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`（碎片防护）
2. **全模型级梯度检查点 (Activation Checkpointing)**：
   - 在 `model/gpt.py` 主循环包装 `torch.utils.checkpoint.checkpoint`，以 20% 的重算时间换取 **65%~75%** 的显存节约。
3. **混合优化器策略**：
   - 2D 矩阵参数（MoE 专家、MLA 投影、mHC 权重）$\to$ **Muon 优化器**（NS 10步正交更新，状态显存减半）；
   - 词表嵌入、RMSNorm 与门控标量 $\to$ **AdamW 优化器**。
