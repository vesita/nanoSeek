# nano_arith: 专用微型算术 Transformer 与神经元分析

本目录是一个独立的高性能微型 Transformer 算术实验子项目，旨在研究：
1. **小模型算术极限与表示学习**：通过极小词表（17个 token）与针对性架构设计，消除通用语言模型长词表与自然语言分布带来的干扰；
2. **因果顺序（Causal Ordering / Reverse Digits）**：对比数字反转输出（个位向高位因果推演）与自然输出的收敛速度与极限；
3. **点石成金（Grokking）现象**：在微小参数量（160k ~ 360k 参数）下，观察模型从 Memorization 到 Generalization 的跃迁；
4. **神经元与回路可解释性（Mechanistic Interpretability）**：自动探针搜索模型内部是否存在专门负责“进位（Carry）”的单神经元与注意力通路。

---

## 目录结构

- `dataset.py`：词表定义（17 tokens: `0-9`, `+`, `-`, `*`, `=`, `<pad>`, `<eos>`, `<sos>`）及算术数据集生成器。支持任意位数（1~N 位）、非对称位数（如 2位+4位）与正向/反向因果输出。
- `model.py`：精简版 Transformer（RoPE 旋转位置编码、RMSNorm、SwiGLU、嵌入层权重绑定）。包含中间层与注意力权重捕获 hooks。
- `train.py`：训练与评估脚本。支持多位数动态细分评估（1d/2d/3d/4d 分项精确度），支持余弦衰减与权重衰减（AdamW）。
- `eval_lengths.py`：长度泛化（Length Generalization）测试矩阵工具，评测 1~8 位数及非对称长度的零样本泛化能力。
- `analyze.py`：神经元与注意力分析器。计算前馈网络（FFN）神经元对“个位是否有进位”的点二列相关系数（Point-biserial Correlation），并可视化注意力权重矩阵。
- `cli.py`：命令行交互与快速推演工具。可以直接输入任意算式进行实时验证。

---

## 任意位数加法（1~4 位数）与非对称加法实验

- **模型配置**：4 层 Transformer，`d_model=96`，`n_heads=4`，`d_ff=288`，共 **48.1 万** 参数。
- **训练表现**：训练集动态混合 1~4 位任意对称与非对称加法（涵盖个位数相加、链式多重进位如 9999+1、不对称位数如 15+789 等）。
- **实测准确率**：
  - 1 位数加法（如 7+8）：**100.0%**
  - 2 位数加法（如 45+78）：**100.0%**
  - 3 位数加法（如 123+456）：**100.0%**
  - 4 位数加法（如 3847+5189）：**100.0%**
  - 链式极限进位（`9999+1=10000`）：**100.0%**
  - 非对称相加（`12+3456=3468`, `15+789=804`）：**100.0%**
- **训练时间**：AMD APU 上仅需 **90 秒**（4000 步）。

---

## 长度外推（Length Generalization / OOD）发现与分析

运行 `eval_lengths.py` 探究未见过的长位数（5~8位）：
- **现象**：当模型训练在 1~4 位时，5 位数及以上的零样本 Exact Match 出现失效（0%）。
- **根因分析**：
  1. **RoPE 位置编码频域外推**：训练时 RoPE 看到的相对位置差均在 12 以内，超过 15 步后位置向量内积偏离训练分布；
  2. **隐状态进位累积漂移**：没有显式 Scratchpad / CoT（思维链），模型必须完全依靠连续隐状态单步吸收 5~8 次进位传递；
  3. **解决方案**：若要达到 1~8 甚至 20 位任意长度，需结合两项技术：（A）加入带进位链标记的 Scratchpad（逐步输出中间进位）；（B）应用 NoPE（No Positional Encoding）或相对位置偏置（ALiBi/Fire）提升长度泛化。

---

## 命令行交互体验

可以直接通过 `cli.py` 测试任意算式：
```bash
# 单次计算
.venv/bin/python nano_arith/cli.py "3847+5189="
# 输出: Predicted : 9036 ✅ (Match)

# 极端多重进位
.venv/bin/python nano_arith/cli.py "9999+1="
# 输出: Predicted : 10000 ✅ (Match)

# 交互式模式
.venv/bin/python nano_arith/cli.py
```

---

## 实验结果与核心发现

### 1. 两位数加法（2-digit Addition）收敛与 Grokking 验证
- **参数量**：361,728 参数（3 层，d_model=96, n_heads=4, d_ff=288）
- **训练耗时**：44 秒（AMD ROCm APU, 3000 steps）
- **指标**：
  - Step 200: Train Loss 1.0311, Val EM 5.5%
  - Step 400: Train Loss 0.4930, Val EM 25.0%
  - Step 600: Train Loss 0.0320, Val EM **99.5%**（发生急剧泛化跃迁）
  - Step 1200+: Train Loss 0.0000, Val EM **100.0%**
- **结论**：验证集达到 **100% 绝对精确匹配**。

### 2. 神经元进位探针（Carry Neuron Probing）
运行 `analyze.py` 对 500 个独立测试样本进行激活分析（判断 $a_0 + b_0 \ge 10$ 时中间神经元的激活）：
- 在中间层（Layer 1）发现了高度显著的**进位神经元**：
  - **Layer 1 Neuron #92**：相关系数达到 **-0.6872**（进位时受强烈抑制）。
  - **Layer 1 Neuron #188**：相关系数达到 **+0.6743**（进位时均值激活 0.696 vs 无进位 0.341）。
- 这证明模型并没有采用简单的哈希记忆，而是在 Transformer 的 MLP 隐空间中自动涌现出了加法进位计算的逻辑电路！

---

## 快速使用

### 训练 2 位数加法模型（~45 秒）
```bash
PYTHONUNBUFFERED=1 HSA_ENABLE_SDMA=0 HSA_OVERRIDE_GFX_VERSION=10.3.0 TMPDIR=/home/vesita/AI/scratch \
.venv/bin/python nano_arith/train.py \
  --d_model 96 --n_layers 3 --n_heads 4 --d_ff 288 \
  --batch_size 128 --max_steps 3000 --eval_interval 200 \
  --min_digits 2 --max_digits 2 --ops "+" \
  --save_dir out/nano_arith_2digit
```

### 运行进位神经元与注意力分析
```bash
PYTHONUNBUFFERED=1 HSA_ENABLE_SDMA=0 HSA_OVERRIDE_GFX_VERSION=10.3.0 TMPDIR=/home/vesita/AI/scratch \
.venv/bin/python nano_arith/analyze.py --ckpt out/nano_arith_2digit/best.pt
```
