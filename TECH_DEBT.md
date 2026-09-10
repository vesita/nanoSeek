# 技术债清单

> 维护规则：**每条债都要能指出具体文件/行号和它的代价**（浪费时间？误导决策？掩盖 bug？），
> 并给一个"下一步动作"。没有证据的条目不要写进来 —— 那只是抱怨。
>
> 最后更新：2026-09-10（上下文压缩前后的一轮清理）

---

## 1. 本轮已偿还

### 1.1 建立了单元测试（0 → 289 条）

```bash
uv run pytest          # 289 条，跳过 slow 时 < 1 秒跑完
uv run pytest -m 'not slow'
```

覆盖面与**每条为什么值得测**见 `tests/README.md`。重点不是数量，而是：

| 测试 | 钉住的真实事故 |
|---|---|
| `test_masking.py` | mask 标记 id 从 117/119 改到 128/130 后文档没跟上；用朴素参考实现做 400 组随机对照 |
| `test_prepare_split.py` | **本项目最贵的一次事故**：val 双标准（对话 10% vs 其余 1%）→ 所有指标都是假的 |
| `test_ngram_hash.py` | 槽位对齐 off-by-L（历史上错过两次） |
| `test_utils.py` | `ns_steps=5` 会让正交化残差从 4.5e-5 炸到 0.66；混相 7 步更好 |
| `test_model_smoke.py` | 参数量被 `state_dict()` 虚高到 90.19M；`targets=None` 只返回最后一位 |
| `test_project_layout.py` | 配置放进 `out_dir` 会被归档；配置键定义在 `config_keys` 之后会被静默改回默认 |
| `test_lint.py` | 改名漏改引用（`GATES`→`GATE_NAMES`）跑 4 分钟才在打印时崩 |

### 1.2 加了 lint 门禁（零容忍，当前 0 违规）

`pyproject.toml` 的 `[tool.ruff.lint] select = ["E9", "F821", "F811", "F823"]`。

**为什么只开四条**：2026-09-10 全项目实测这四条合计只有 1 处真实违规，
所以能设零容忍而不需要 `noqa`。故意不开 `F401`（82 处未使用导入）、
`F841`（27 处未使用变量）、行宽与导入排序 —— 那些噪声会让门禁变成摆设。

已按门禁修掉：
- `local/eval_mem_dialogue.py`：`TOP_K` 先 import 后覆盖，前者失效。

### 1.3 消除了 5 处重复的数据文件名解析

`training/train.py` 里
`'train_char.bin' if char_level else 'train_byte.bin' if byte_level else 'train.bin'`
重复了 5 遍（train / val / meta / summary / 一行死代码）。加一个新模式要改 5 处，
**漏改任何一处都会让"换数据集"静默失效**。

现在收敛为 `training/schedules.py::pick_bin_names()`，可单测，并有
`test_project_layout.py::test_pick_bin_names_drives_get_batch` 用 AST 保证
`train.py` 里不再有硬编码的数据文件名。

顺带删掉的死代码：
- `train.py`：`main_bin`（赋值后从未使用，一度让人以为 `data_prefix` 没生效）、`_ds`（收敛后无调用点）
- `model/attention.py`：`k_glob`（在 HCA 路径算了两次、用零次；因为单个全局 key 的 softmax 恒为 1）
- `scripts/ngram_capacity_probe.py`：`L_max`

### 1.4 把纯函数从"模块级脚本"里抽出来

`training/train.py` 是 1466 行的模块级脚本 —— **`import` 它就会直接开训**，
里面的任何函数都无法被 pytest 覆盖。已抽出：

| 新位置 | 内容 |
|---|---|
| `training/schedules.py::lr_at` | 学习率调度（warmup + cosine/WSD），带抽取前字面实现的逐位回归测试 |
| `training/schedules.py::with_data_prefix` | 数据集文件名加前缀 |
| `training/schedules.py::pick_bin_names` | train/val/meta 三个文件名的唯一解析点 |
| `training/diag.py::mem_debug_line` | 显存调试字符串（非 CUDA 返回 None） |
| `training/diag.py::should_dump_snapshot` | 快照触发判定（取 max(allocated, reserved)） |

`tests/test_project_layout.py::test_no_test_imports_train_py` 会拦住
"测试里 import train.py" 这类把测试变成"启动训练"的写法。

### 1.5 修掉的真 bug（不是重构）

| 位置 | 问题 | 后果 |
|---|---|---|
| `model/config_loader.py` | 同时写 `extends:` 和 `base:` 时 `base` 不被 pop | `base` 作为无意义全局注入 `train.py` |
| `scripts/ngram_capacity_probe.py` | Δ 评测忘了把 `(B,T,V)` 展平成 `(N,V)` | 跑几分钟后 `gather` 抛异常 |
| 同上 | 报告段的 `for gname in GATES:` 漏改（已改名 `GATE_NAMES`） | 4 分钟评测跑完才 `NameError`，结果全丢 |
| 同上 | 旧实现的 λ 门控语义与 §12.3 不一致 | 差点把"O(N) 改写"变成"偷换语义"（λ=1 时差 4.5e-2） |
| `training/train.py` | 显存诊断的守卫写成 `torch.cuda.is_available()` | **有显卡的机器上跑 `--device=cpu` 会在 step 0 崩**（`KeyError: allocated_bytes.all.current`）。写 `configs/base_v2.yaml` 的 `ndb_debug_mem: true` 时必崩 |

最后一条是**冒烟测试**（`--max_iters=2` 真起一次训练）抓到的，不是单元测试 ——
说明"改完 train.py 就跑一次 2 步冒烟"这条纪律值得保留，已写进
`training/diag.py` 的模块文档与 `tests/test_diag.py`。

### 1.6 GPU Hang 事故的加固（2026-09-10）

探针第一次上 GPU 时把显卡跑挂：内核日志
`ring gfx_0.0.0 timeout` → `gfx_0.1.0 reset failed` → `GPU reset begin! MODE1 reset`
→ `VRAM is lost due to GPU reset`，**连带把桌面一起打死**。

加固后峰值显存 **0.93 GB**（加固前槽表常驻 537MB + 每次 λ 克隆整个 `(N,8192)` 概率矩阵）：

- 槽表查表改在 numpy 侧，GPU 上只保留 `(N,)` 小张量
- 目标概率计算从 O(N·V) 降到 O(N)（不再 clone 全量分布）
- 每 N 个 batch 打进度 + 峰值显存，**下次挂能定位到具体 batch**
- top-1 表落盘缓存，重跑不必再等建表
- 默认 `--device cpu`（模型只有 81.58M，CPU 前向约 10 分钟，零 GPU 风险）

**顺带证伪**：不是 NaN 导致的。同一 checkpoint 前向 `finite=True`、`max|logit|=13.9`；
探针两次成功运行都是 `nonfinite_logits = 0`。

### 1.7 归档检查点会写满磁盘（2026-09-11 巡检发现并修复）

**发现过程**：驻守巡检时顺手看了一眼 `df`，发现分区 270G 已用 181G、`out/` 占 74G，
而**首次 eval 才刚过（step 1000）**。顺着查到 `train.py` 逢 1000 步写
`ckpt_step_<N>.pt`，**没有任何清理**：

```
0.59GB/个 × (70000 / 1000) = 42GB     ← 光这一项
+ best.pt / last.pt 各 0.59GB
```

87GB 可用空间**勉强**够，但会被吃到只剩 ~45GB，而后续 NDB 实验（top-1 缓存 2.1GB/份、
库表 2~3GB）还要占盘。**这是一个"跑得越久越危险"的隐性故障**，不修的话
最坏情况是跑到第 60k 步时 `OSError: No space left on device` 杀死 2.7 天的训练。

**修法**：
- `training/checkpoints.py` —— 纯函数 `prune_step_checkpoints(out_dir, keep)`，
  只保留最近 `keep` 个；`training/train.py` 记完归档就调它，`keep` 取自
  `getattr(config, 'keep_step_ckpts', 5)`（**故意不用模块级全局**，避开配置快照陷阱）。
- `tests/test_checkpoints.py` —— **11 条单测**。这块代码**会删文件**，写错就是数据丢失，
  所以必须离线覆盖，不能只靠"起一次真训练看看"。
- `scripts/prune_ckpts.sh` —— 外部稀疏化（每 5000 步留一个 + 最新 2 个），
  专治**当前已在跑的旧代码进程**（改代码对它无效）。已挂进巡检命令，幂等。

**★ 关键回归点（写进测试）**：必须按**整数步号**排序，不能按文件名字典序 ——
字典序下 `ckpt_step_9000.pt > ckpt_step_10000.pt`（`'9' > '1'`），
于是"保留最近的"会**删掉真正最新的那个，留下旧的，且不报错**。
`test_prune_keeps_newest_by_numeric_order` 就是防这一条。

**通用教训**：长期训练的资源消耗要**按"最坏情况 × 总时长"算一遍**，
而不是看"现在还剩多少"。以及：**巡检时顺手看一眼 `df`，成本近乎为零。**

---

## 2. 待还的债（按「代价 ÷ 修复成本」排序）

### P0 — `training/train.py` 仍是 1466 行的模块级脚本

**代价**：无法 import、无法单测、无法局部复用。每加一个功能都要在 1466 行里找位置，
而且要记住"全局必须定义在 `config_keys` 之前"这条隐式规则（已经栽过两次）。
本轮只能把 3 个纯函数抽出来，剩下的大头没动。

**建议动作**（分步、每步都可验证）：
1. `_backup_old_run` / `_checkpoint_to_cpu` / `_save_worker` / `save_checkpoint_async` /
   `join_save_threads` → `training/checkpointing.py`（纯文件/张量操作，最好测）
2. `estimate_loss` / `print_summary` / `_plot_loss_curve` → `training/reporting.py`
3. 健康体检那一整块（`ngram_rep` / `_health_generate` / `run_health_check`）→ `training/health.py`
4. 最后把训练循环本体包进 `main()`，用 `if __name__ == '__main__': main()`

**风险**：训练循环依赖大量全局；**必须在没有长跑进行时做**，且每一步都要跑一次
`--max_iters=2` 的冒烟（`out/base_v2/smoke.log` 有先例）。

### P0 — 配置里的 `gradient_checkpointing: true` 是空开关，仍然留着

**代价**：`model/gpt.py:167` 在 `use_moe + use_aux_free_balance` 时强制关掉它，
所以这个开关**完全无效**。本项目曾用它做过 A/B（两个臂其实一模一样），结论已撤回。
配置里留着 `true` 会继续误导。

**建议动作**（三选一，都很小）：
- (a) 从 `configs/base_v2.yaml` 删掉这两行（最干净）
- (b) 保留但让 `train.py` 在启动时**打印警告**："gradient_checkpointing 被 aux-free 短路，已忽略"
- (c) 让 `load_config` 层面报错

当前 `tests/test_model_smoke.py::test_gradient_checkpointing_is_a_noop_under_aux_free_moe`
已经把这个行为记录下来，所以不会"悄悄变回去"。

### P1 — `local/` 有 57 个一次性脚本、9113 行，没有索引

**代价**：
- 找"我上次是怎么测 XX 的"要靠 `ls` 和 grep，实际经常重写一遍
- 同名的诊断脚本互相覆盖（`v4_probe.py` / `v4_hybrid_probe.py` / `probe_base.py` …）
- 新会话恢复上下文时无法判断哪个还有用

**建议动作**：
1. 加 `local/INDEX.md`：一行一个脚本 = 用途 + 关键结论 + 是否仍可用（**成本最低、收益最高**）
2. 把已有 `tests/` 覆盖其逻辑的一次性脚本标 `# 已被 tests/xxx 取代`
3. 明确归档区 `local/archive/`，把"结论已进 dev-notes、代码不再运行"的挪进去

### P1 — `training/rl/`（19 个文件 2704 行）与本项目当前目标无关

**代价**：`pytest`/lint 会扫到它，新人会以为 RL 是主线；依赖（reward engine、
GRPO updater）散落各处，改动模型接口时要一起改。

**证据**：只有 `dev_scripts/probe_v3.py`、`dev_scripts/diag_arith_reward.py` 引用它，
而 `dev_scripts/` 本身也是历史脚本。

**建议动作**：确认 RL 方向暂停后整体移到 `archive/rl/`，并在 `PROJECT_STATE.md` 记一句。

### P1 — `model/attention.py`（757 行）里 4 条互斥实现路径，只有 1 条被测

**代价**：标准注意力 / MLA / CSA+HCA / KV 记忆 用 config 开关分支。
当前训练只用 MLA，其余三条是**活代码但零测试**——改动公共部分（RoPE、mask、
输出投影）时不知道有没有弄坏它们。

**建议动作**：为每条路径各加一条"形状 + 因果性 + 梯度有限"的冒烟测试
（用 `tests/test_model_smoke.py::test_causality_*` 的模式），成本约 30 行/路径。

### P1 — `data/chinese/_backup_task17_20260910_213459/` 是遗留备份目录

**代价**：里面是旧版 `prepare.py`（452 行）+ `manifest.json`。`data/` 是数据目录，
混进代码备份会让人误以为它是数据产物。只在 `DATASET_REPORT.md:410` 提到过。

**证据**：`git status` 显示 `data/chinese/prepare.py` 是 modified，旧版在 git 历史里，
所以这个备份**冗余**。

**建议动作**：确认后删除，或移到 `archive/`。**我没有动它**（删数据目录下的文件不可逆，
留给你决定）。

### P2 — 剩余 lint 噪声：F401 82 处、F841 27 处

**代价**：未使用导入让"这个模块依赖什么"变得不可信；未使用变量里可能藏着真 bug
（本轮的 `main_bin`、`k_glob` 就是从 F841 里翻出来的）。

**建议动作**：
```bash
ruff check --select F401 --fix          # 82 处，自动修，需人工 review diff
ruff check --select F841 --output-format concise   # 27 处，逐个看，可能又翻出 bug
```
建议**按目录分批**（`model/` 和 `training/` 先做，`local/` 最后），别一次改 82 个文件。

### P2 — `state_dict()` 参数量虚高这个坑只在文档和测试里防着

**代价**：文档写"真实 81.58M、state_dict 求和 90.19M"，但代码层面没有任何提示。
下一个统计参数量的人还会踩。

**建议动作**：给 `GPT` 加一个 `count_params(verbose=False)` 类方法，
或者在 `get_num_params` 的 docstring 里加一行"**不要用 `sum(state_dict().values())`**"。
（`tests/test_model_smoke.py::test_state_dict_sum_inflates_due_to_weight_tying`
已经做了精确对账，至少不会再退化。）

### P2 — `out/` 下多份运行产物没有索引

**代价**：`out/ndb_run`（旧基线）、`out/base_v2`（未启动）、`out/ngram_full`、
`out/bench`、`out/mem_store`… 哪份是哪次实验、基座 step 多少，只能靠 `config.yaml` 反推。
本轮就发生过"探针用了 v1 语料训的 checkpoint 去评 v2 数据"这种口径不一致
（`Δ` 因此是保守估计）。

**建议动作**：加 `out/INDEX.md`：一行一个运行 = 目录 + 配置 + 数据 + step + 一句话结论。
（注意：`out/` 在 `.gitignore` 里，索引要么提交、要么放在 `dev-notes/`。）

### P3 — 无 CI / 无 pre-commit

**代价**：289 条测试和 lint 门禁目前只能靠人记得跑。

**建议动作**：加一个 `scripts/check.sh`（`ruff check && pytest -q`），
再考虑 `.pre-commit-config.yaml`。本机是单机开发，CI 不是必需，
但"一条命令跑完所有检查"应该有。

---

## 3. 已确认**不是**债的（别再重复讨论）

| 事项 | 结论 | 证据 |
|---|---|---|
| `state_dict()` 求和 ≠ 真实参数量 | 是 PyTorch 的正常行为（共享张量重复列举 + buffer），不是 bug | `test_state_dict_sum_inflates_due_to_weight_tying` 做了精确对账 |
| `targets=None` 时 logits 只有 1 个位置 | 是**刻意的推理优化**（`gpt.py:207`），省 T-1 次大矩阵乘 | `test_forward_without_targets_returns_only_last_position` |
| `model.eval()` 下 `router_bias` 仍被改 | 是 aux-free 的既有设计（V4），整个 19781 步基线都带着它。要改应做单独 A/B | `test_aux_free_router_bias_mutates_even_in_eval` |
| `load_state_dict` 后两个优化器共享 buffer | 是 PyTorch 的行为（dtype 匹配时不拷贝）。真实续训路径无害（源自磁盘） | `test_load_state_dict_aliases_source_tensors` |
| `data/external/` 下几万个文件 | 第三方语料仓库，已从 lint/测试扫描里排除 | `pyproject.toml` 的 `extend-exclude` |
| `ns_steps=7, aggressive=4` 比 10 步更好 | 实测：残差 1.6e-6 vs 3.4e-5（良态）、4.3e-2 vs 6.1e-2（病态），且省 30% 计算 | `test_ns_mixed_4_aggressive_3_classic_...` |
| `muon_split` / `use_mhc` 这些开关是不是空转 | 已有"它确实改变了行为"的测试 | `test_ns_split_differs_from_whole_matrix` 等 |

---

## 4. 现在有哪几道闸

新增债务最容易从这几个口子进来，现在都有人守着：

| 口子 | 闸 |
|---|---|
| 改了数据切分逻辑，忘了重新生成数据 | `test_shipped_v2_manifest_matches_this_logic` 把产物与代码绑起来 |
| 加了配置键，定义位置不对 | `test_every_config_key_is_overridable`（AST 静态分析 `train.py`） |
| 配置放进 `out_dir` | `test_config_file_lives_outside_its_own_out_dir` |
| 改名/重构漏改引用 | `test_lint.py`（F821/F811） |
| 写了"两臂其实一样"的空测试 | `tests/README.md` 的坑表 + 各处的"对照必须能区分"断言 |
| 测试变成"启动一次训练" | `test_no_test_imports_train_py` |
| 配置与文档分叉 | `test_base_v2_matches_documented_decisions` 逐项对照 PROJECT_STATE §5 |
| 文档里的数字变成谎言 | `test_documented_cosine_vs_wsd_table` 直接复现 §5 的表 |
