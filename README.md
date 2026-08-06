# nanoGPT 学习魔改版

在 [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) 的基础上改造的深度学习实验项目：**用 8GB 显存的小模型，亲手复现 DeepSeek 风格的技术**。

nanoGPT 的极简设计（全部代码就 `model.py` + `train.py` 两个文件）让它成为理想的实验台——没有框架抽象，每一行都能看懂、都能改。

---

## 当前功能

**现代化模型组件**（`model.py`，全部可开关，默认关 = 原始 GPT-2 结构）：

| 配置项 | 默认 | 说明 |
|--------|------|------|
| `use_rmsnorm` | `false` | LayerNorm → RMSNorm（DeepSeek/LLaMA 标准归一化，省掉均值运算） |
| `use_rope` | `false` | 可学习位置编码 → RoPE 旋转位置编码（点积自动携带相对位置信息） |
| `use_swiglu` | `false` | GELU MLP → SwiGLU 门控前馈（同等参数下表达力更强） |
| `rope_theta` | `10000.0` | RoPE 基频，可调 |

**YAML 配置系统**（替代原版 exec Python 配置）：
- 实验配置放 `config/*.yaml`，由 `config_loader.py` 加载
- 命令行覆盖：`uv run python train.py config/xxx.yaml --n_layer=4`
- 配置值带类型检查，写错类型直接报错（比如 YAML 里 `1e-3` 会被解析成字符串，会立即被拦下）

**默认 small 模型**：4 层 / 128 维 / 256 上下文 ≈ **0.83M 参数**，RTX 5060 上约 1 分钟一轮训练，适合快速迭代实验。

**实验基础设施**：
- 冒烟测试 `scripts/smoke_test.py`：秒级验证模型前向/反向 + 参数量对比
- checkpoint 统一输出到 `out/`（git 忽略）
- 干净的 git 仓库，每个实验一个 commit，可随时回退对比

---

## 安装与环境

使用 [uv](https://github.com/astral-sh/uv) 管理依赖（项目已在 `pyproject.toml` 声明）：

```sh
uv sync          # 按 pyproject.toml 安装依赖到 .venv
uv run python scripts/smoke_test.py   # 验证环境 + 模型可用
```

环境要求：Python ≥ 3.12，PyTorch ≥ 2.0（用于 flash attention 和 torch.compile），有 NVIDIA GPU 最佳。

---

## 项目结构

```
nanoGPT/
├── model.py          # GPT 模型（魔改核心：RMSNorm/RoPE/SwiGLU 开关）
├── train.py          # 训练循环
├── sample.py         # 文本采样
├── bench.py          # 性能基准
├── config_loader.py  # YAML 配置加载器（替代原 configurator）
├── config/           # 实验配置（YAML）
│   ├── train_shakespeare_char.yaml     # 基线：英文 small（原始 GPT-2 结构，65 词表）
│   └── train_chinese.yaml              # 中文 small（西游记，4507 词表）
├── scripts/
│   └── smoke_test.py # 冒烟测试
├── data/             # 数据集与预处理脚本
└── out/              # 训练输出（git 忽略）
    ├── shakespeare-char/
    └── chinese/
```

---

## 快速开始

项目只有两个实验配置：**英文 small 基线**（做架构对比用，快）和**中文 small**（最终目标）。

**① 准备数据**（一次性）

```sh
uv run python data/shakespeare_char/prepare.py   # 英文：65 词表，约 100 万 token
uv run python data/chinese/prepare.py            # 中文《西游记》：4507 词表，约 68 万 token
```

**② 训练中文模型**

```sh
uv run python train.py config/train_chinese.yaml
```

约 5-10 分钟跑完（5000 步）。**注意**：中文初始 loss ≈ 8.4，不是英文的 ~4.2——那是 `ln(4507)` 词表难度的正常值，跨数据集比 loss 没有意义。

**③ 采样看效果**

```sh
uv run python sample.py --out_dir=out/chinese --start="悟空" --num_samples=3 --max_new_tokens=300
```

**④ 架构 A/B（在英文基线上做）**

对比「原始 GPT-2」和「modern 三件套」，用英文基线跑：速度快、loss 低、好比较。modern 用命令行覆盖开关即可，无需新建配置：

```sh
uv run python train.py config/train_shakespeare_char.yaml
uv run python train.py config/train_shakespeare_char.yaml --out_dir=out/shakespeare-char-modern --use_rmsnorm=true --use_rope=true --use_swiglu=true
```

各约 1 分钟。关键看训练途中出现的 **best val loss**（小数据集上模型会过拟合，最终 loss 会反弹，别比最后一轮）：

```sh
uv run python -c "
import torch
for d in ['out/shakespeare-char', 'out/shakespeare-char-modern']:
    ck = torch.load(f'{d}/ckpt.pt', map_location='cpu', weights_only=False)
    print(f'{d}: best_val_loss = {ck[\"best_val_loss\"]:.4f}')
"
```

**⑤ 提交实验**（每个实验一个 commit，方便回退）

```sh
git add -A && git commit -m "实验：..."
```

---

## 配置系统

配置文件是 YAML，键对应 `train.py` 里的全局变量：

```yaml
# config/train_chinese.yaml
out_dir: out/chinese
dataset: chinese
n_layer: 4
n_head: 4
n_embd: 128
learning_rate: 0.001
```

- **命令行覆盖**：`--key=value`，值和 `train.py` 默认值类型必须一致。布尔/浮点也能覆盖，比如 `--use_rmsnorm=true --use_rope=true --use_swiglu=true` 就相当于原来的 modern 配置
- **类型检查**：YAML 里的值会和 `train.py` 全局变量比对类型，不一致立刻报错。比如 PyYAML 把 `1e-3` 解析成字符串，这种坑会被当场抓住
- **浮点写法注意**：YAML 里写浮点记得带小数点（`0.001` 而不是 `1e-3`），否则会被解析成字符串

---

## 实验记录与学习路线

目标是理解 DeepSeek 的技术，全部在小模型上亲手实现验证：

| 级别 | 技术 | 状态 |
|------|------|------|
| 1 | 现代化基础组件：RMSNorm + RoPE + SwiGLU | ✅ 已实现并验证（打平基线，参数量更少） |
| 2 | MoE 混合专家（DeepSeek-V3 核心） | ⏳ 计划中 |
| 3 | MLA 多头潜在注意力（DeepSeek-V2 核心） | ⏳ 计划中 |
| 4 | MTP 多 token 预测（DeepSeek-V3） | ⏳ 计划中 |

**已跑过的实验**（在 10.7M 模型上）：基线的 best val loss 1.4733，modern 1.4764——几乎打平，但 modern 少了约 10 万参数（RoPE 删掉了位置编码表）、且更早收敛。这印证了这三件套是"结构效率"改进：同等表现、更省参数，规模放大后优势更明显。

---

## 常见操作速查

| 操作 | 命令 |
|------|------|
| 冒烟测试 | `uv run python scripts/smoke_test.py` |
| 训练 | `uv run python train.py config/<实验名>.yaml` |
| 采样 | `uv run python sample.py --out_dir=out/<实验名>` |
| 看 checkpoint 里的配置 | `torch.load('out/xxx/ckpt.pt', weights_only=False)['model_args']` |
| 提交实验 | `git add -A && git commit -m "..."` |
| 回退到上个实验 | `git checkout <上一个commit>` |

---

## 参考

- 上游项目：[karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)（MIT License）
- 原作者的零基础教学：[Zero To Hero 系列](https://karpathy.ai/zero-to-hero.html)
- 本项目借鉴的论文技术：GPT-2、RoPE（RoFormer）、SwiGLU（PaLM）、RMSNorm（LLaMA）、MoE/MLA/MTP（DeepSeek-V2/V3）
