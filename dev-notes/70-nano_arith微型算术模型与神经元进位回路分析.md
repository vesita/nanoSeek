# 开发笔记 70：nano_arith 专用微型算术模型与神经元进位回路探测

- **日期**：2026-03-29
- **实验目标**：剥离自然语言词表与上下文干扰，构建专用于算术的超轻量 Transformer（160k ~ 480k 参数），验证因果反转（Reverse Digits）、泛化跃迁（Grokking），并通过 Mechanistic Interpretability 探针定位隐空间中的“进位神经元（Carry Neurons）”。

---

## 1. 背景与设计动机

在先前通用 2.7M 模型的强化学习与微调实验中（笔记 66-68），我们观察到通用语言模型进行多位数加法时存在明显的算力瓶颈。根本原因包括：
1. **词表与多任务干扰**：通用 4533 词表稀释了数字的嵌入分布；
2. **算术的非因果性（Non-causal Flow）**：自然加法输出（如 `123 + 456 = 579`）中，最高位输出在最前，但其值依赖于低位所有可能的链式进位，打破了自回归模型的单向因果流；
3. **因果反转方案（Reverse Digits）**：若将输出调整为从个位开始（`123 + 456 = 975`），模型在生成第一位时仅需关注个位相加，进位可以作为隐状态沿序列向后传递。

为此，我们在 `nano_arith/` 下搭建了完全独立的实验闭环。

---

## 2. 核心实验与指标

### 实验 A：2 位数加法（361k 参数，3层，d_model=96）
- **超参数**：Batch 128, AdamW (weight_decay=0.1, lr=1e-3, cosine warmup=200 steps)。
- **训练收敛轨迹**：
  - Step 200: Train Loss 1.0311 | Val EM: 5.5%
  - Step 400: Train Loss 0.4930 | Val EM: 25.0%
  - Step 600: Train Loss 0.0320 | Val EM: **99.5%** （出现典型的泛化跃迁/Grokking 现象）
  - Step 1200+: Train Loss 0.0000 | Val EM: **100.0%**
- 耗时：AMD ROCm APU 上仅需 **44 秒** 即可完成全部 3000 步训练，验证集达到 100% 绝对精确。

### 实验 B：机制可解释性分析（Mechanistic Interpretability）
为了检验模型究竟是“死记硬背”还是“学会了算法”，我们在 `analyze.py` 中编写了进位探针：
- 在 500 个未见算式中，统计个位产生进位（$a_0 + b_0 \ge 10$）时，各层 MLP 中间神经元的激活值，并计算点二列相关系数（Point-biserial Correlation）。
- **重大发现**：在 Layer 1 中发现了高度专化的**进位神经元**：
  - **Neuron #188**：正相关系数 **+0.6743**（进位时平均激活 0.696，无进位时为 0.341）。
  - **Neuron #92**：负相关系数 **-0.6872**（进位时受到强抑制）。
- 这表明 360k 参数的微型模型内部自发组织并涌现出了明确的算术进位判定子回路（Sub-circuit）。

---

## 3. 产物与交付清单

1. **子项目代码**：
   - `nano_arith/dataset.py`：17-token 算术数据集生成器。
   - `nano_arith/model.py`：RoPE + SwiGLU + RMSNorm 的精简 Transformer 及特征钩子。
   - `nano_arith/train.py`：训练与 Grokking 监控脚本。
   - `nano_arith/analyze.py`：进位神经元探针与注意力热力图可视化脚本。
   - `nano_arith/README.md`：详细实验复现说明。
2. **权重保存**：
   - `out/nano_arith_2digit/best.pt`（100% 验证集加法准确率）。
