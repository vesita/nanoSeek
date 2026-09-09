# 开发笔记 70：nano_arith 专用微型算术模型与神经元进位回路探测

- **日期**：2026-03-29
- **实验目标**：剥离自然语言词表与上下文干扰，构建专用于算术的超轻量 Transformer（160k ~ 480k 参数），验证因果反转（Reverse Digits）、泛化跃迁（Grokking），并用 Mechanistic Interpretability 探针定位隐空间中的“进位神经元（Carry Neurons）”。

## 1. 背景与设计动机
通用 2.7M 模型做多位数加法有算力瓶颈（笔记 66-68）：①通用 4533 词表稀释数字嵌入分布；②**算术非因果性**——`123 + 456 = 579` 最高位输出在最前，却依赖低位所有链式进位，打破自回归单向因果流；③**因果反转方案（Reverse Digits）**——输出从个位开始（`123 + 456 = 975`），生成第一位仅需关注个位，进位作为隐状态沿序列向后传递。为此在 `nano_arith/` 搭建独立实验闭环。

## 2. 核心实验与指标
### 实验 A：2 位数加法（361k 参数，3层，d_model=96）
- **超参数**：Batch 128, AdamW (weight_decay=0.1, lr=1e-3, cosine warmup=200 steps)。
- **收敛轨迹**：Step 200 Loss 1.0311 / Val EM 5.5%；Step 400 0.4930 / 25.0%；Step 600 0.0320 / **99.5%**（典型泛化跃迁/Grokking）；Step 1200+ 0.0000 / **100.0%**。
- 耗时：AMD ROCm APU 上 3000 步仅 **44 秒**，验证集 100% 绝对精确。

### 实验 B：机制可解释性（Mechanistic Interpretability）
在 `analyze.py` 编写进位探针：500 个未见算式中统计个位产生进位（$a_0 + b_0 \ge 10$）时各层 MLP 中间神经元激活，计算点二列相关系数。
- **Layer 1 高度专化的进位神经元**：**Neuron #188** 正相关 **+0.6743**（进位时平均激活 0.696，无进位 0.341）；**Neuron #92** 负相关 **-0.6872**（进位时受强抑制）。
- 表明 360k 微型模型内部自发涌现明确的算术进位判定子回路（Sub-circuit）。

## 3. 产物与交付清单
- 代码：`nano_arith/dataset.py`（17-token 算术数据集生成器）、`model.py`（RoPE + SwiGLU + RMSNorm 精简 Transformer 及特征钩子）、`train.py`（训练与 Grokking 监控）、`analyze.py`（进位神经元探针与注意力热力图）、`README.md`（复现说明）。
- 权重：`out/nano_arith_2digit/best.pt`（100% 验证集加法准确率）。
