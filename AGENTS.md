# AGENTS.md —— 给 AI agent 的操作规程（先读这一页）

> 这个仓库是**一个人 + 多个 AI agent 断续接力**维护的。上下文会被压缩、会话会被换掉，
> 所以**不能靠记忆**。本页只讲「怎么做」；「现在是什么状态」在 `PROJECT_STATE.md`，
> 「欠了什么」在 `TECH_DEBT.md`。

---

## 1. 项目一句话

在单张 8G 的 AMD gfx1030 上训一个 **75.15M char-level 中文模型（nanoSeek-100M）**，
然后做一个 **no_grad 外部神经数据库（NDB）**，在不增加模型大小的前提下扩大有效容量
（用户明确要求：NDB **可读可写**，且**由模型自己决定**怎么读写）。

---

## 2. 开工前必读（按顺序，别跳）

| 顺序 | 文件 | 读它干什么 |
|---|---|---|
| 1 | **本页 `AGENTS.md`** | 纪律与陷阱 |
| 2 | **`PROJECT_STATE.md` 的 🚀 速查一节** | 当前状态 + 7 条常用命令，够你开工 |
| 3 | `PROJECT_STATE.md` §0.5 | 训练审查结论（噪声口径、有效 token、数据配比）|
| 4 | `TECH_DEBT.md` §2 | 已知未还的债，别重复发现 |
| 5 | `PROJECT_STATE.md` §8 | 十条运维铁律的完整版 |
| 6 | 需要时：`data/chinese/DATASET_REPORT.md` | 数据集怎么造的、哪些结论**已作废** |

**只有第 1~2 条是必须的**，其余按需。**不要**一上来通读整个仓库。

---

## 3. 环境（照抄，别自己推）

```bash
cd /home/vesita/coding/my/nanoSeek
HSA_OVERRIDE_GFX_VERSION=10.3.0 HSA_ENABLE_SDMA=0    # gfx1030 必需
.venv/bin/python                                       # 唯一 python（torch 2.9.1+rocm6.4）
```

- 8G 显存 / 15G 内存 / 12 核。训练峰值显存 **1.70G** —— 余量很大，但**探针/推理很容易吃爆**
  （见 §5.3）。
- hipBLASLt 在 gfx1030 上没有 Tensile 库，**不要**去调它。

---

## 4. 常用命令

**全部在 `PROJECT_STATE.md` 的 🚀 速查一节**（续训 / 巡检 / 配对重评 / 质量评估 /
密度体检 / 闸门 / 冒烟 七条）。这里不重复，避免两处文档分叉。

---

## 5. 测量纪律（本项目最贵的教训都在这）

### 5.1 单点 val 不可信 —— σ≈0.087，而每千步真进步只有 0.010

- `results.csv` 的 val 相邻跳动 ±0.19，**全是测量噪声**（机制：有效独立单元是
  **窗口**不是 token，窗口级 σ≈2.0 nats）。
- **要比较两个模型/两个改动，必须配对**（同一批窗口算 Δ）：
  `scripts/ckpt_paired_eval.py`。
- 只看趋势时，**≥5000 步聚合**或看 `train_loss_window.csv` 的 `window_mean`。
- ⚠️ `data/chinese/DATASET_REPORT.md` 里"val CE 标准误 1e-3 nats"的说法**是错的**
  （差 70 倍），别引用它。

### 5.2 `best.pt` 是噪声选出来的，不要用

22 个评估点里出现一次 ≥2.3σ 好运的概率是 **37%** —— 这个机制**必然**选出噪声点。
用 `last.pt` 或某个固定归档。`best.pt` 只适合"不知道用哪个"的兜底。

### 5.3 探针必须与训练**同构**

真实事故（2026-09-11）：量配对数时 chunk=50 → logits fp32 upcast 吃到 **7.88G/8G**，
而训练峰值只有 1.70G。**探针自己制造了训练里不存在的内存画像** → 险些挂掉桌面。
写任何计时/显存/前向探针前，先问：**这个形状/批量在训练里出现过吗？**

### 5.4 新测量函数先在"已知答案的输入"上跑一遍

例：`scripts/mask_density_probe.py` 的均匀采样对照必须复现
`DATASET_REPORT` 里独立测出的 **83.5% 全空窗口**。对不上 → 先修测量，别下结论。

### 5.5 结论的措辞强度必须匹配证据强度

单次、少步、单 seed 的 <1% 差异，只能写"小于 1%"，不能写"没有效果"。

---

## 6. 数据侧的三个硬事实（2026-09-11 实测，别再重新发现）

1. **有效 token 密度 39%**：`use_loss_masking=true` + `stage=full` 让 loss 只在
   「含 `<eos>`/`<cont>` 的行」上计算，全库终止符密度只有 **0.1427%**。
   每步名义 8192 token，**实际产生梯度约 3216**。
2. **名义配比 ≠ 实际训练分布**：`c4_zh`（20.25%，1.9 亿 token）**一个终止符都没有**
   → 永远不产生梯度；`wikipedia_cn`（19.67%）只有 0.0008% 密度。
   **40% 的语料对训练完全不可见。** 有效语料只有 **≈57.6M token（全库 6.14%）**。
   → 模型是个**窄域对话模型**，不是通用中文 LM。
3. **`multi_turn_dialogue` 名义 4.80%，却占 40.2% 的终止符 / 29.7% 的训练窗口**
   → 它单独决定了模型的人设（共情式心理咨询腔）。

**排查这类问题的工具**：`scripts/mask_density_probe.py`；
逐来源统计见 `PROJECT_STATE.md §0.5.2`。

---

## 7. 铁律（违反过的都在这里，完整版见 `PROJECT_STATE.md §8`）

| # | 规则 | 踩过的坑 |
|---|---|---|
| 1 | **只**用后台 `sleep N` 定时器；禁止前台 sleep、禁止 `job_output(wait=true)` | harness 会杀掉阻塞的会话 |
| 2 | 启动后台任务的命令**必须立刻返回**，绝不和 `sleep` 写在同一条里 | 整条命令被 SIGTERM，任务跟着死 |
| 3 | 训练日志写在 **`out_dir` 之外**（`out/base_v2_train.log`）| `out_dir` 内的文件会被 `_backup_old_run` 挪走，fd 跟着旧 inode → 日志永远空 |
| 4 | `pkill -f <模式>` 会匹配到**自己这条 shell** → 用括号技巧 `pkill -f "ckpt_[j]anitor.sh"`，且**不要把 pkill 和启动写在同一条命令里** | 命令自己把自己杀了（2026-09-11 又犯一次）|
| 5 | 诊断代码用 `device_type == 'cuda'`，**不要** `torch.cuda.is_available()`；取显存用 `.get(k, 0)` | CPU 模式下 `KeyError` 崩训练 |
| 6 | 巡检动作必须是**一个原子脚本** `bash scripts/watch.sh`，不许手敲长命令 | 曾出现"文档里有清理、实际定时器没跑" |
| 7 | 改完 `train.py` **必须**跑 2 步冒烟，且**显式加 `--init_from=scratch`** | 配置已是 `resume`，会去读旧 run |
| 8 | 配置键必须定义在 `config_keys` 快照**之前** | 之后定义的会被静默改回默认值 |
| 9 | 归档 checkpoint 有保留策略（默认留最新 5 个），写盘前先看 `df` | 曾写满 42GB 磁盘 |
| 10 | **重启训练 ⇒ 监控节律重置回 300s** | 问题一般发生在早期 |

---

## 8. 改配置要同步改**三处**（有测试钉着）

改 `configs/base_v2.yaml` 里的"配方"键时，必须同时改：

1. `configs/base_v2.yaml`
2. `PROJECT_STATE.md §5` 那张配方表
3. `tests/test_project_layout.py::test_base_v2_matches_documented_decisions`

`test_project_layout.py` 就是**故意**这么设计的 —— 它逼着"文档和配置一起改"。
**不要**为了让测试变绿只改测试不改文档。

---

## 9. 不要重复造轮子（先看这里，再决定写不写）

| 想做的事 | 现成的入口 |
|---|---|
| 对话质量 / 生成质量评估 | `inference/scripts/eval_dialogue.py`（rep2/3/4、空白占比、distinct-n、轮次结构）|
| 多轮对话评估 | `inference/scripts/eval_multiturn.py` |
| 采样 / 对话 | `inference/sample.py`、`inference/scripts/chat.py`（或 `cli.py sample` / `cli.py chat`）|
| 巡检 | `scripts/watch.sh` |
| 配对重评 / val 噪声 | `scripts/ckpt_paired_eval.py` |
| 有效 token 密度 | `scripts/mask_density_probe.py` |
| NDB 容量上限 | `scripts/ngram_capacity_probe.py` |
| 清理归档 ckpt | `scripts/prune_ckpts.sh`（外部）/ `training/checkpoints.py`（训练内）|

**注意**：`inference/scripts/*` 只认 `out_dir/best.pt`。要评任意 checkpoint，
先建硬链接目录：`mkdir -p out/_eval_x && ln -f <ckpt> out/_eval_x/best.pt`。

**⚠️ 评估 prompt 的格式必须和语料一致**：`eval_dialogue.py` 现在还在用
`用户：你好\n模型：`，而 v2 语料已**去标签**（用 `A：`/`B：`）。用它评 v2 模型是
**喂 OOD 输入**，`turns` 会假性全 0，而且**v1 基座在这套 prompt 上有主场优势**，
拿 `d1/d2` 比 v1/v2 会得出反向结论（`TECH_DEBT.md` P1）。比 v1/v2 请用**配对 CE**。

**⚠️ 不要另写采样/评估脚本。** 2026-09-11 的审查里犯过一次：另写了
`scripts/gen_sample.py`，而 `eval_dialogue.py` 早就覆盖了 rep/空白/多样性/轮次四类
指标，自制版只看得到"读起来像不像话"。需要定性证据时，用与语料同格式的 prompt 喂
`inference/sample.py`，客观指标交给 `eval_dialogue.py`。

---

## 10. 目录导航

```
configs/            训练配置（base_v2.yaml 是当前主线；★ 必须放在 out_dir 之外）
training/train.py   ★ 1473 行的模块级脚本 —— import 它就等于开始训练，不能单测
training/           已抽出的纯函数模块（schedules / masking / checkpoints / diag）
model/              模型与组件（gpt.py / ngram_ndb.py 是 NDB 原型）
scripts/            运维脚本（watch.sh、探针、清理）
inference/          推理/评估/采样（评估入口都在这）
tests/              单测（含 lint 门禁）；改代码后必跑
data/chinese/       语料与 tokenizer（v2 = 当前主线）
  DATASET_REPORT.md   数据集报告（⚠️ 有已作废的结论，见 §5.1）
PROJECT_STATE.md    ★ 状态 + 速查 + 铁律
TECH_DEBT.md        技术债
dev-notes/          历史实验记录（编号笔记，写新结论时接着编号）
```

---

## 11. 交付前自检

- [ ] `.venv/bin/pytest -q -m 'not slow'` 全绿
- [ ] `.venv/bin/python -m ruff check .` 全绿
- [ ] 改过 `train.py` → 跑过 2 步冒烟（带 `--init_from=scratch`）
- [ ] 后台任务没留下孤儿进程；GPU 显存已释放（`cat /sys/class/drm/card*/device/mem_info_vram_used`）
- [ ] 新结论写进了 `PROJECT_STATE.md`（状态类）或 `TECH_DEBT.md`（债类），
      并标注 **[实测] / [推断]**，以及**证据强度**（单次？配对？多 seed？）
