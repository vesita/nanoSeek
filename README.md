# nanoSeek 学习魔改版

在 [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) 的基础上改造的深度学习实验项目：**用 8GB 显存的小模型，亲手复现 DeepSeek 风格的技术**。

nanoSeek 的极简设计（全部代码就 `model.py` + `train.py` 两个文件）让它成为理想的实验台——没有框架抽象，每一行都能看懂、都能改。

---

## 📦 v0.2 发布（2026-08-19）

当前方向已达质量天花板（详见 [dev-notes/29](dev-notes/29-数据构成实验与质量天花板.md)、[dev-notes/30](dev-notes/30-5000步重复坍缩验证.md)），定版 0.2：

- **默认模型**：`out/chinese-data2-reb`（val 1.2700 @1500 步，全系列最优）
  - 1500 步 = 本架构防重复坍缩的质量保护点（更多步数 → 采样退化）
  - EOS 自吐 10/10、rep3≈0、回复短；内容为心理咨询话术风格（语料构成决定）
- **Rust 端已适配**：`inference/runtime` 新增 QK-Norm 支持 + 融合 QKV 转换拆分，
  Python/Rust 对拍通过（max err 0.003，top-5 一致）；默认温度 0.6 + 重复惩罚 1.2
- **发布包**：`release/nanoSeek-v0.2`（Linux）/ `release/nanoSeek-v0.2-windows`（Windows）
- **数据扩充记录**：Zhihu-KOL 闲聊语料（500K 条）已接入管线，供未来更大模型使用
- 历史实验已归档至 `out/archive/`、`release/archive/`

---

## 当前功能

**固定架构**（`model.py` 硬编码，不可配置）：RMSNorm + SwiGLU。

**配置项**（`model.py` / YAML 可开关）：

| 配置项 | 默认 | 说明 |
|--------|------|------|
| `use_rope` | `false` | 可学习位置编码 → RoPE 旋转位置编码 |
| `rope_theta` | `10000.0` | RoPE 基频，可调 |
| `swiglu_clamp` | `0.0` | SwiGLU 门控输出钳制半宽（V4 稳定性技巧） |

**MoE 混合专家**（DeepSeek-V3/V4）：

| 配置项 | 默认 | 说明 |
|--------|------|------|
| `use_moe` | `false` | FFN → MoE（top-k 路由 + 负载均衡） |
| `n_experts` / `n_top_k` | `8` / `2` | 路由专家数 / 每 token 激活数 |
| `use_shared_expert` | `false` | **V4**：始终激活的共享专家（捕获共性特征） |
| `use_aux_free_balance` | `false` | **V4**：aux-free 偏置修正替代 Switch aux loss |
| `use_sqrtsoftplus` | `false` | **V4**：路由打分 √softplus 替代 softmax |

**注意力**（DeepSeek-V2 MLA 与 V4 CSA 二选一）：

| 配置项 | 默认 | 说明 |
|--------|------|------|
| `use_mla` | `false` | MLA 多头潜在注意力（低秩压缩 KV + 部分 RoPE） |
| `use_csa` | `false` | **V4** CSA 压缩稀疏注意力：块级压缩 + top-k 稀疏 + 滑窗（O(T²)→O(T·(nb+win))） |
| `csa_compress` / `csa_topk` / `csa_window` | `16` / `4` / `64` | CSA 超参 |
| `use_csa_learnable` | `true` | **V4**：可学习门控池化替代平均池化 |
| `use_hca` | `false` | **V4** HCA 重度压缩全局信号 |

**训练目标与优化器**：

| 配置项 | 默认 | 说明 |
|--------|------|------|
| `use_mtp` | `false` | **V4** 多 token 预测（额外预测 t+2，输出头与 lm_head 共享） |
| `mtp_weight` | `0.3` | MTP 损失权重 |
| `use_muon` | `false` | **V4** Muon 优化器（矩阵参数正交化，embedding/lm_head/norm 用 AdamW 保护） |
| `muon_ns_steps` | `10` | Newton-Schulz 迭代次数（系数 (2,-1.5,0.5)） |

**V4 结构设计升级**（连接方式，不增加规模；实验性，默认全关）：

| 配置项 | 默认 | 说明 |
|--------|------|------|
| `use_attn_sink` | `false` | **V4** Attention Sinks：每头一个可学习标量偏置，作为 softmax 的"垃圾桶"吸收无关注意力 |
| `use_mhc` / `hc_mult` | `false` / `4` | **V4** mHC 超连接：4 流并行残差 `X'=B·X+C·F(A·X)`，A/C sigmoid 有界、B 双重随机（Sinkhorn），子层只跑 1 次不翻倍计算 |
| `use_lightning_indexer` | `false` | **V4** 学习型块选择（简版）：idx_q/idx_k 打分选块替代 CSA raw top-k，KL 梯度桥接 |
| `num_hash_layers` | `0` | **V4** 前 N 层用 `hash(token_id)%n_experts` 确定性路由，深层才用学习路由 |

**YAML 配置系统**（替代原版 exec Python 配置）：
- 实验配置放 `config/*.yaml`，由 `config_loader.py` 加载
- 命令行覆盖：`uv run python train.py config/xxx.yaml --n_layer=4`
- 配置值带类型检查，写错类型直接报错（比如 YAML 里 `1e-3` 会被解析成字符串，会立即被拦下）

**默认 small 模型**：6 层 / 80 维 / 8000 词表 + MoE(4×2) + 共享专家 + MTP + Attention Sinks + mHC + QK-Norm + Z-Loss + turn-level EOS 数据 ≈ **2.85M 参数**（深度换宽度，规模与旧 4×96 持平），RTX 5060 上约 1 分钟一轮训练，适合快速迭代实验。当前默认模型：`out/chinese-data2-eos`（全部 6 份语料 + 全特性已验证，1500 步，val 1.8340）。

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
nanoSeek/
├── cli.py                     # 统一命令行入口（带默认预设，推荐方式）
├── model/                     # 模型核心
│   ├── __init__.py            # 重导出 GPT / GPTConfig 等
│   ├── model.py               # GPT 模型（V2/V3/V4 全部架构开关）
│   └── config_loader.py       # YAML 配置加载器
├── training/                  # 训练相关
│   ├── train.py               # 训练循环
│   └── config/                # 实验配置（YAML）
│       ├── train_chinese.yaml      # 唯一正式配置（CSA/HCA + MoE）
│       └── test.yaml               # 快速冒烟（几十步验证代码不崩）
├── inference/                 # 推理与部署
│   ├── sample.py              # Python 文本采样（任何架构都能跑）
│   ├── bench.py               # 性能基准
│   ├── runtime/               # Rust 推理框架（candle，CPU 推理 + 对话 REPL）
│   │   ├── src/               # model.rs / attention.rs / main.rs / tokenizer.rs
│   │   └── scripts/convert.py # checkpoint → safetensors + tokenizer.json
│   └── scripts/               # smoke_test.py（冒烟）/ package.py（打包）/ compare_logits.py（对拍）/ archive.py（模型归档）
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

项目配置分两种：**正式训练**用 `train_chinese.yaml`（中文对话模型，CSA/HCA + MoE）；**快速冒烟**用 `test.yaml`（几十步跑完，验证代码不崩、能训起来）。

**① 准备数据**（一次性）

```sh
uv run python data/chinese/download_dialogue.py     # 1. 从魔搭下载中文对话语料
uv run python data/chinese/train_tokenizer.py       # 2. 训 BPE 分词器 → tokenizer.json
uv run python data/chinese/prepare.py               # 3. 编码 → train.bin / val.bin
#    （默认生成「纯对话 + turn-level EOS」数据：模型学会"话说完→吐 <eos>"，
#      与默认 train_chinese.yaml 对应。旧行为用 --task-ratio 1.0 --no-insert-eos 复现）
```

**② 训练中文模型**

先快速冒烟（几十步，确认环境/代码没问题）：

```sh
uv run python training/train.py training/config/test.yaml
```

再正式训练：

```sh
uv run python training/train.py training/config/train_chinese.yaml
```

约 5-10 分钟跑完（5000 步）。**注意**：BPE 词表 8000，随机初始化 loss ≈ `ln(8000)` ≈ 9.0；这个值只和词表大小挂钩，跨数据集/词表比 loss 没有意义。

**③ 采样看效果**

```sh
uv run python inference/sample.py --out_dir=out/chinese-data2 --start="悟空" --num_samples=3 --max_new_tokens=300
```

**④ 版本化续训**（详见下方「版本化连续训练」）：`--init_from=resume` 续训、`--init_from=<路径>.pt` 基于已有模型做后训练。

**⑤ 提交实验**（每个实验一个 commit，方便回退）

```sh
git add -A && git commit -m "实验：..."
```

---

## 统一 CLI（推荐）

项目根目录提供了 `cli.py`，把散落的脚本收敛成一条命令，并内置「默认预设」。大多数日常操作不需要再记长路径：

```sh
# 训练：默认预设 = 正式中文对话配置 train_chinese.yaml
uv run python cli.py train

# 快速冒烟训练：几十步秒级验证
uv run python cli.py train --preset smoke

# 采样：默认加载 out/chinese-data2
uv run python cli.py sample --prompt "悟空"

# 准备数据（下载 → 分词器 → 编码）
uv run python cli.py data

# 查看可用的训练预设
uv run python cli.py presets

# 其他
uv run python cli.py smoke       # 架构冒烟测试
uv run python cli.py bench       # 性能基准
uv run python cli.py eval        # 对话智能度评估
uv run python cli.py convert     # 转换 → Rust 权重（默认 out/chinese-data2）
uv run python cli.py package     # 打包独立部署目录
uv run python cli.py distill     # 生成自蒸馏数据
uv run python cli.py archive     # 模型归档/索引
uv run python cli.py selftest    # 快速自检
```

**模型归档**：`cli.py archive` 会扫描 `out/` 下所有实验目录，读取 `results.csv` / `best.pt`，生成每个实验的 `manifest.json` 和汇总 `out/index.json`，之后用 `cli.py archive` 即可按 val loss / 架构特征快速横向对比。

**数据与发布溯源**：`data/chinese/prepare.py` 会额外生成 `manifest.json`（源文件哈希、切分参数、train/val 统计）；`package.py` 会在 release 目录写 `manifest.json`（来源 checkpoint、git commit、目标平台、架构配置）。

`cli.py` 统一了两种参数风格：`--key value` 和 `--key=value` 都可以用，连字符会自动转成底层脚本的下划线命名（如 `--max-iters 1000` → `--max_iters=1000`）。查看某个子命令帮助：

```sh
uv run python cli.py train --help
```

> 底层原有脚本（`training/train.py`、`inference/sample.py` 等）仍然可以直接运行，`cli.py` 只是在前面加了一层更友好的默认值与入口。

---

## 配置系统

配置文件是 YAML，键对应 `train.py` 里的全局变量：

```yaml
# training/config/train_chinese.yaml
out_dir: out/chinese-data2
dataset: chinese
n_layer: 6
n_head: 4
n_embd: 80
vocab_size: 8000
learning_rate: 0.001
```

- **命令行覆盖**：`--key=value`，值和 `train.py` 默认值类型必须一致。布尔/浮点也能覆盖，比如 `--use_rmsnorm=true --use_rope=true --use_swiglu=true` 就相当于原来的 modern 配置
- **类型检查**：YAML 里的值会和 `train.py` 全局变量比对类型，不一致立刻报错。比如 PyYAML 把 `1e-3` 解析成字符串，这种坑会被当场抓住
- **浮点写法注意**：YAML 里写浮点记得带小数点（`0.001` 而不是 `1e-3`），否则会被解析成字符串
- **配置继承**：子配置用 `extends: train_chinese.yaml`（或 `base:`）继承父配置，只需写差异项。路径相对于当前配置文件所在目录。例如：

```yaml
# training/config/train_chinese_fa.yaml
extends: train_chinese.yaml
out_dir: out/chinese-data2-fa
block_order: ffn_attn
```

---

## 架构验证记录

基于 DeepSeek-V4 技术报告（arXiv:2606.19348）逐项实现并在几 M 规模上验证：

| 技术 | 状态 | 说明 |
|------|------|------|
| 固定架构：RMSNorm + RoPE + SwiGLU | ✅ 硬编码 | 不再可开关 |
| MoE 混合专家 | ✅ 激活 | top-k 路由 + 负载均衡 |
| V4：共享专家 | ✅ **已验证有效** | val loss 5.01→4.91，已写进默认配置 |
| V4：可学习门控池化（CSA） | ✅ **已验证有效** | val loss 5.88→5.01（vs 平均池化），已写进默认配置 |
| V4：aux-free 偏置均衡 | ⚠️ 待调参 | 800 步差 0.11，需调 balance_factor |
| V4：√softplus 路由打分 | 🔬 待验证 | 开关可用 |
| V4：Muon 优化器（NS 系数 (2,-1.5,0.5)） | ⚠️ 待调 lr | 需专门 lr 缩放，不能直接用 AdamW 的 |
| V4：MTP 多 token 预测 | ⚠️ 待长训验证 | 800 步主 loss 差 0.17，数据效率收益需更久显现 |
| V2：MLA 多头潜在注意力 | ✅ 已实现 | 与 CSA 互斥，可切换 |
| V4：SwiGLU Clamp | ✅ 已实现 | 稳定性技巧，配置启用 |
| V4：Attention Sinks | ✅ **已验证有效** | A/B：单独开就打破重复坍缩，loss 还略低（val 4.21 vs 4.27），详见 [dev-notes/11](dev-notes/11-AttentionSinks打破重复.md) |
| V4：mHC 4-copy | 🔬 待验证 | **机制正确**（4 流 + Sinkhorn 双重随机 B，非之前删掉的 2 流错误版） |
| V4：Lightning Indexer（简版） | 🔬 待验证 | 学习型块选择替代 raw top-k，KL 梯度桥接 |
| V4：Hash 路由 | 🔬 待验证 | 前 N 层确定性路由，开关可用 |
| 近零参数稳定性：QK-Norm | ✅ **已验证有效，已写进默认配置** | L2 归一化 q/k + 每头可学习 scale（+24 参数）。1500 步 val 1.002→0.951（−0.052），详见 [dev-notes/25](dev-notes/25-QK-Norm与Router-Z-Loss-1500步A-B.md) |
| 近零参数稳定性：Router Z-Loss | ✅ **已验证有效，已写进默认配置** | z=logsumexp(router)² 正则（0 参数）。单独 −0.020；与 QK 合开 **−0.189 超线性**（val 1.002→0.813），详见 [dev-notes/25](dev-notes/25-QK-Norm与Router-Z-Loss-1500步A-B.md) |
| ~~预判路由~~ | ❌ 移除 | 非 V4 概念（混淆了 aux-free 偏置修正） |

**数据集**：**BPE 子词分词**（魔搭中文对话语料 ~153MB，8000 词表，ByteLevel 预分词）。数据流程：`download_dialogue.py`（魔搭下载对话）→ `train_tokenizer.py`（训 BPE）→ `prepare.py`（编码成 train.bin/val.bin）。

**默认模型**：`training/config/train_chinese.yaml`（6×80 + MTP + Sinks + CSA/HCA + MoE + mHC + QK-Norm + Z-Loss，~2.85M 参数），默认 checkpoint 为 `out/chinese-data2-eos/best.pt`（EOS 数据 + 全特性验证通过）。已验证有效的开关（共享专家、可学习门控池化、Attention Sinks、mHC、QK-Norm、Router Z-Loss）已全部写进默认配置。Attention Sinks 经 A/B 实证是打破重复坍缩的必要条件（见 dev-notes/11），mHC 经 1500 步归因收敛加速（见 dev-notes/13），QK+Z 合开超线性收敛加速（见 dev-notes/25）——均已默认开启。其余结构设计升级（Indexer/Hash）实验性默认关闭。数据采用 `prepare.py --task-ratio 0 --insert-eos`（纯对话 + turn-level EOS，模型学会"话说完→吐 `<eos>`"）。

**开发踩坑笔记（Dev Notes）**：`dev-notes/` 已按 4 大主题整理——[A 重复坍缩与训练稳定性](dev-notes/A-重复坍缩与训练稳定性.md)（核心研究线）、[B 结构消融与拓扑实验](dev-notes/B-结构消融与拓扑实验.md)、[C 数据与词表治理](dev-notes/C-数据与词表治理.md)、[D 工程与工具踩坑](dev-notes/D-工程与工具踩坑.md)。速查入口见 [dev-notes/README.md](dev-notes/README.md)；原始 25 篇逐条笔记全保留，可溯源。

## 版本化连续训练

训练按"版本"推进，每版一个 `out/` 目录，靠 YOLO 式的 `best.pt`/`last.pt` 双 checkpoint 衔接：

```sh
# v0：从零初训（会存 last.pt 每次评估 + best.pt 仅在 val 变优时存）
uv run python training/train.py training/config/train_chinese.yaml --max_iters=10000

# 续训（同一版本继续训，恢复优化器和学习率计划）
uv run python training/train.py training/config/train_chinese.yaml --init_from=resume --max_iters=15000

# 后训练（基于旧版本 best.pt 开新版本，优化器/学习率重置）
uv run python training/train.py training/config/train_chinese.yaml \
  --init_from=out/chinese-data2-eos/best.pt --out_dir=out/chinese-data2-eos-v2 --max_iters=5000
```

- `best.pt`：val loss 最优的 checkpoint，续训/部署默认用它
- `last.pt`：每次评估时的最新状态，防止训练中断丢进度
- 进度条显示当前轮次（epoch）

---

## Rust 部署（CPU 对话推理）

`runtime/` 是一个基于 candle 的 Rust 推理框架：把训练好的 PyTorch checkpoint 转成 safetensors，用 Rust 在 CPU 上跑对话，完全不依赖 Python。

**部署流程**（训练产物 → 可对话模型）：

```sh
# 1. 转换 checkpoint（默认转换当前最佳模型 out/chinese-data2/best.pt）
uv run python inference/runtime/scripts/convert.py --ckpt out/chinese-data2/best.pt --dataset chinese

# 2. 编译 Rust 运行时
cd inference/runtime && cargo build --release

# 3. 一次性生成
./target/release/nanoseek-runtime --prompt "悟空" --max-new-tokens 100

# 4. 进入对话 REPL（输入 `退出` / `exit` / Ctrl-D 结束）
./target/release/nanoseek-runtime
```

**Rust 端支持的架构**（和 `model.py` 逐位对齐，对拍误差 ~1e-6）：

| 特性 | 状态 | 说明 |
|------|------|------|
| RMSNorm / RoPE / SwiGLU | ✅ | 固定架构三件套 |
| MoE（top-k 路由 + 专家 FFN + 共享专家） | ✅ | 含 V4 共享专家 |
| SwiGLU Clamp | ✅ | V4 数值稳定性钳制 |
| CSA + HCA（压缩稀疏注意力） | ✅ | 块压缩 + top-k 稀疏选择 + 滑窗 + 全局信号，含 V4 可学习门控池化 |
| V4 Attention Sinks / mHC / Lightning Indexer / Hash 路由 | ✅ | 结构设计升级，与 Python 逐位对齐 |
| MTP | ✅ | 训练增强头，推理不需要——convert.py 跳过 MTP 权重，主干输出不受影响 |
| MLA | ❌ | Python 端有，Rust 未实现 |

⚠️ **部署注意**：MLA 仍只在 Python 端，Rust 未实现。开启 MLA 的模型无法部署。
MTP 训练的模型可正常部署（MTP 是训练时辅助预测头，推理只用主干模型的最终 logits，
convert.py 已自动跳过 MTP 权重）；其余 V4 特性（含四个结构设计升级）Rust 已全部支持。
Rust 端验证方法：用 `--print-logits` / `--dump-logits` 配合 `inference/sample.py` 的
`--dump_logits` 逐位对拍（见下方「对拍验证」）。

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
- 对拍验证：同一 prompt 下 Rust 与 Python 的 logits 应逐位对齐（最大误差 < 1e-5）。
  有 `inference/scripts/compare_logits.py` 一条命令完成（convert → build → 双端 dump → diff），
  也可手动跑：Python 端 `uv run python inference/sample.py --out_dir=out/<实验> --start="悟空" --dump_logits=/tmp/py.txt`，
  Rust 端 `./target/release/nanoseek-runtime --dump-logits /tmp/rust.txt --prompt "悟空"`，然后 diff 两个文件。

---

## 常见操作速查

| 操作 | 推荐命令（cli.py） | 等价原始命令 |
|------|---------------------|---------------|
| 冒烟测试 | `uv run python cli.py smoke` | `uv run python inference/scripts/smoke_test.py` |
| 快速自检 | `uv run python cli.py selftest` | — |
| 训练 | `uv run python cli.py train --preset smoke` / `cli.py train` | `uv run python training/train.py training/config/test.yaml` / `.../train_chinese.yaml` |
| 采样 | `uv run python cli.py sample --prompt "..."` | `uv run python inference/sample.py --out_dir=out/<实验名> --start="..."` |
| 部署到 Rust | `uv run python cli.py convert` | `uv run python inference/runtime/scripts/convert.py --ckpt out/<实验>/best.pt --dataset chinese` → `cd inference/runtime && cargo run --release -- --prompt "悟空"` |
| Rust/Python 对拍 | `uv run python cli.py compare --ckpt out/<实验>/best.pt` | `uv run python inference/scripts/compare_logits.py --ckpt out/<实验>/best.pt` |
| 模型归档/索引 | `uv run python cli.py archive` / `cli.py archive --write` | `uv run python inference/scripts/archive.py` |
| 看 checkpoint 里的配置 | `torch.load('out/xxx/best.pt', weights_only=False)['model_args']` | |
| TensorBoard 对比曲线 | `uv run tensorboard --logdir out/`（需在配置里开 `tensorboard_log: true`） | |
| 提交实验 | `git add -A && git commit -m "..."` | |
| 回退到上个实验 | `git checkout <上一个commit>` | |

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
