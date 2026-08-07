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

**DeepSeek-V2/V3 核心**：

| 配置项 | 默认 | 说明 |
|--------|------|------|
| `use_moe` | `false` | FFN → MoE 混合专家（top-k 路由 + 负载均衡辅助损失） |
| `n_experts` / `n_top_k` | `8` / `2` | 专家总数 / 每 token 激活的专家数 |
| `use_mla` | `false` | 多头潜在注意力：KV 低秩压缩 + 部分 RoPE |
| `use_mtp` | `false` | 多 token 预测：训练时额外预测未来 token |

**DeepSeek-V4 新技术**（`training/config/train_chinese_v4_*.yaml` 逐项验证）：

| 配置项 | 默认 | 说明 |
|--------|------|------|
| `swiglu_clamp` | `0.0` | SwiGLU 门控输出钳制半宽（0 = 关）。V4 稳定性技巧：从源头压制被 MoE 路由放大的异常值 |
| `use_muon` | `false` | Muon 优化器替代 AdamW：矩阵参数做 Newton-Schulz 正交化 |
| `use_mhc` | `false` | 流形约束超连接替代残差：双流混合 + 双随机矩阵（谱范数=1，信号不放大） |
| `use_anticipatory_routing` | `false` | 预判路由：离散路由选择用 EMA 旧参数，与骨干更新解耦（防 loss spike） |
| `use_csa` | `false` | 压缩稀疏注意力：块级 KV 压缩 + top-k 稀疏选择 + 滑窗（O(T²)→O(T·(nb+win))） |
| `use_hca` | `false` | 重度压缩注意力：全局摘要信号（配合 use_csa 使用） |

**YAML 配置系统**（替代原版 exec Python 配置）：
- 实验配置放 `config/*.yaml`，由 `config_loader.py` 加载
- 命令行覆盖：`uv run python train.py config/xxx.yaml --n_layer=4`
- 配置值带类型检查，写错类型直接报错（比如 YAML 里 `1e-3` 会被解析成字符串，会立即被拦下）

**默认 small 模型**：4 层 / 128 维 / 256 上下文 ≈ **0.83M 参数**，RTX 5060 上约 1 分钟一轮训练，适合快速迭代实验。

**实验基础设施**：
- 冒烟测试 `inference/scripts/smoke_test.py`：秒级验证模型前向/反向 + 参数量对比
- checkpoint 统一输出到 `out/`（git 忽略）
- 干净的 git 仓库，每个实验一个 commit，可随时回退对比

---

## 安装与环境

使用 [uv](https://github.com/astral-sh/uv) 管理依赖（项目已在 `pyproject.toml` 声明）：

```sh
uv sync          # 按 pyproject.toml 安装依赖到 .venv
uv run python inference/scripts/smoke_test.py   # 验证环境 + 模型可用
```

环境要求：Python ≥ 3.12，PyTorch ≥ 2.0（用于 flash attention 和 torch.compile），有 NVIDIA GPU 最佳。

---

## 项目结构

```
nanoGPT/
├── model/                     # 模型核心
│   ├── __init__.py            # 重导出 GPT / GPTConfig 等
│   ├── model.py               # GPT 模型（V2/V3/V4 全部架构开关）
│   └── config_loader.py       # YAML 配置加载器
├── training/                  # 训练相关
│   ├── train.py               # 训练循环
│   └── config/                # 实验配置（YAML）
│       ├── train_chinese.yaml             # 快速调试用（base，非 MoE）
│       └── train_chinese_v4_csa.yaml      # 默认对话模型（CSA/HCA + MoE）
├── inference/                 # 推理与部署
│   ├── sample.py              # Python 文本采样（任何架构都能跑）
│   ├── bench.py               # 性能基准
│   ├── runtime/               # Rust 推理框架（candle，CPU 推理 + 对话 REPL）
│   │   ├── src/               # model.rs / attention.rs / main.rs / tokenizer.rs
│   │   └── scripts/convert.py # checkpoint → safetensors + tokenizer.json
│   └── scripts/               # smoke_test.py（冒烟）/ package.py（打包）
├── data/                      # 数据集脚本（数据文件按需下载，git 忽略）
│   └── chinese/
│       ├── prepare.py         # 语料 → train.bin / val.bin
│       ├── train_tokenizer.py # BPE 分词器训练
│       └── download_dialogue.py  # 魔搭对话语料下载
├── out/                       # 训练输出（git 忽略）
└── release/                   # 打包产物（git 忽略）
```

---

## 快速开始

项目只有两个实验配置：**英文 small 基线**（做架构对比用，快）和**中文 small**（最终目标）。

**① 准备数据**（一次性）

```sh
uv run python data/chinese/download_dialogue.py     # 1. 从魔搭下载中文对话语料
uv run python data/chinese/train_tokenizer.py       # 2. 训 BPE 分词器 → tokenizer.json
uv run python data/chinese/prepare.py               # 3. 编码 → train.bin / val.bin
```

**② 训练中文模型**

```sh
uv run python training/train.py training/config/train_chinese.yaml
```

约 5-10 分钟跑完（5000 步）。**注意**：BPE 词表 8000，随机初始化 loss ≈ `ln(8000)` ≈ 9.0；这个值只和词表大小挂钩，跨数据集/词表比 loss 没有意义。

**③ 采样看效果**

```sh
uv run python inference/sample.py --out_dir=out/chinese --start="悟空" --num_samples=3 --max_new_tokens=300
```

**④ 版本化续训**（详见下方「版本化连续训练」）：`--init_from=resume` 续训、`--init_from=<路径>.pt` 基于已有模型做后训练。

**⑤ 提交实验**（每个实验一个 commit，方便回退）

```sh
git add -A && git commit -m "实验：..."
```

---

## 配置系统

配置文件是 YAML，键对应 `train.py` 里的全局变量：

```yaml
# training/config/train_chinese.yaml
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
| 2 | MoE 混合专家（DeepSeek-V3 核心） | ✅ 已实现并验证 |
| 3 | MLA 多头潜在注意力（DeepSeek-V2 核心） | ✅ 已实现并验证 |
| 4 | MTP 多 token 预测（DeepSeek-V3） | ✅ 已实现并验证 |
| 5 | V4：SwiGLU Clamping（数值稳定性） | ✅ 已实现，待对比训练 |
| 6 | V4：Muon 优化器（Newton-Schulz 正交化） | ✅ 已实现，待对比训练 |
| 7 | V4：mHC 流形约束超连接 | ✅ 已实现，待对比训练 |
| 8 | V4：Anticipatory Routing（预判路由） | ✅ 已实现，待对比训练 |
| 9 | V4：CSA/HCA 压缩稀疏注意力 | ✅ 已实现（简化教育版），待对比训练 |

**数据集**：已从字符级换为 **BPE 子词分词**（魔搭中文对话语料 ~153MB，8000 词表，ByteLevel 预分词）。8000 词表专为日常中文设计：压缩率 ~1.3x（比 30000 只降 13%），但嵌入表从 2.88M 砍到 0.77M，transformer 参数占比大幅提升。数据流程：`download_dialogue.py`（魔搭下载对话）→ `train_tokenizer.py`（训 BPE，`--vocab-size` 可调）→ `prepare.py`（编码成 train.bin/val.bin）。数据文件全部 git 忽略、按需下载。

**默认模型**：`train_chinese_v4_csa.yaml`（CSA/HCA + MoE，~4.3M 参数）——项目已从"逐项对比 V4 技术"转向"持续训练一个对话模型"，其余对比配置已删除。

## 版本化连续训练

训练按"版本"推进，每版一个 `out/` 目录，靠 YOLO 式的 `best.pt`/`last.pt` 双 checkpoint 衔接：

```sh
# v0：从零初训（会存 last.pt 每次评估 + best.pt 仅在 val 变优时存）
uv run python training/train.py training/config/train_chinese_v4_csa.yaml --max_iters=10000

# 续训（同一版本继续训，恢复优化器和学习率计划）
uv run python training/train.py training/config/train_chinese_v4_csa.yaml --init_from=resume --max_iters=15000

# 后训练（基于旧版本 best.pt 开新版本，优化器/学习率重置）
uv run python training/train.py training/config/train_chinese_v4_csa.yaml \
  --init_from=out/chinese-v4-csa/best.pt --out_dir=out/chinese-v4-csa-v2 --max_iters=5000
```

- `best.pt`：val loss 最优的 checkpoint，续训/部署默认用它
- `last.pt`：每次评估时的最新状态，防止训练中断丢进度
- 进度条显示当前轮次（epoch）

---

## Rust 部署（CPU 对话推理）

`runtime/` 是一个基于 candle 的 Rust 推理框架：把训练好的 PyTorch checkpoint 转成 safetensors，用 Rust 在 CPU 上跑对话，完全不依赖 Python。

**部署流程**（训练产物 → 可对话模型）：

```sh
# 1. 转换 checkpoint（默认转换最优的 CSA 模型）
uv run python inference/runtime/scripts/convert.py --ckpt out/chinese-v4-csa/best.pt --dataset chinese

# 2. 编译 Rust 运行时
cd inference/runtime && cargo build --release

# 3. 一次性生成
./target/release/nanogpt-runtime --prompt "悟空" --max-new-tokens 100

# 4. 进入对话 REPL（输入 `退出` / `exit` / Ctrl-D 结束）
./target/release/nanogpt-runtime
```

**Rust 端支持的架构**（和 `model.py` 逐位对齐，对拍误差 ~1e-6）：

| 特性 | 状态 | 说明 |
|------|------|------|
| RMSNorm / RoPE / SwiGLU | ✅ | 基础 modern 三件套 |
| MoE（top-k 路由 + 专家 FFN） | ✅ | 含预判路由（加载 `router_slow`，推理不做 EMA 更新） |
| SwiGLU Clamp | ✅ | V4 数值稳定性钳制 |
| CSA + HCA（压缩稀疏注意力） | ✅ | 块压缩 + top-k 稀疏选择 + 滑窗 + 全局信号 |
| mHC（双流超连接） | ✅ | Sinkhorn 投影到双随机矩阵 + 记忆流解码 |
| MLA / MTP | ❌ | 零配置启用（V2/V3 遗留特性，无当前实验使用），需要时再补 |

**CLI 选项**：

| 选项 | 默认 | 说明 |
|------|------|------|
| `--model` / `--config` / `--tokenizer` | runtime/ 下三个文件 | 覆盖加载路径 |
| `--prompt "..."` | 无 | 给则一次性生成，否则进 REPL |
| `--max-new-tokens N` | 300 | 生成长度 |
| `--temperature T` / `--top-k K` | 0.8 / 200 | 采样参数 |
| `--seed N` | 1337 | 随机种子 |
| `--print-logits` | 关 | 调试：打印最后一个位置的 top-10 logits |
| `--dump-logits FILE` | 关 | 调试：把全部 logits 落盘（用于和 Python 端对拍） |

**部署注意**：
- 推理是纯 CPU，逐 token 重新前向（无 KV cache），块长 256 的小模型上足够快；放大模型后建议加 KV cache。
- 权重全部 F32，无量化；需要更低显存/内存可后续加。
- `--print-logits` / `--dump-logits` 配合 `inference/sample.py` 的 Python 输出做逐位对拍，是验证部署正确性的标准手段。

---

## 常见操作速查

| 操作 | 命令 |
|------|------|
| 冒烟测试 | `uv run python inference/scripts/smoke_test.py` |
| 训练 | `uv run python train.py config/<实验名>.yaml` |
| 采样 | `uv run python inference/sample.py --out_dir=out/<实验名>` |
| 部署到 Rust | `uv run python inference/runtime/scripts/convert.py --ckpt out/<实验>/best.pt --dataset chinese` → `cd inference/runtime && cargo run --release -- --prompt "悟空"` |
| 看 checkpoint 里的配置 | `torch.load('out/xxx/best.pt', weights_only=False)['model_args']` |
| TensorBoard 对比曲线 | `uv run tensorboard --logdir out/`（需在配置里开 `tensorboard_log: true`） |
| 提交实验 | `git add -A && git commit -m "..."` |
| 回退到上个实验 | `git checkout <上一个commit>` |

**训练体验**（实验目录只留可读文件：`best.pt` / `results.csv` / `loss_curve.png`）：
- **`results.csv`**（YOLO 式）：每个评估点一行 `step, train/loss, val/loss, lr, mfu, time`，纯文本、Excel 可直接打开、训练中断也能读到已落盘部分
- **`loss_curve.png`**：训练结束自动生成 train/val 双曲线，不用开任何工具直接看图
- **checkpoint 异步保存**（后台线程 + 原子改名），保存时训练不再卡顿
- **TensorBoard 可选**：默认关闭（避免二进制事件文件）；需要多实验曲线叠加时用 `--tensorboard_log=True` 开启，事件写到 `out/<实验>/tensorboard/` 子目录，然后用 `uv run tensorboard --logdir out/` 查看

---

## 参考

- 上游项目：[karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)（MIT License）
- 原作者的零基础教学：[Zero To Hero 系列](https://karpathy.ai/zero-to-hero.html)
- 本项目借鉴的论文技术：GPT-2、RoPE（RoFormer）、SwiGLU（PaLM）、RMSNorm（LLaMA）、MoE/MLA/MTP（DeepSeek-V2/V3）
