# 技术债清单

> 维护规则：**每条债都要能指出具体文件/行号和它的代价**（浪费时间？误导决策？掩盖 bug？），
> 并给一个"下一步动作"。没有证据的条目不要写进来 —— 那只是抱怨。
>
> 最后更新：2026-09-10（上下文压缩前后的一轮清理）

---

## 1. 本轮已偿还

### 1.1 建立了单元测试（0 → 数百条）

```bash
.venv/bin/python -m pytest -q -m 'not slow'   # 全绿；跳过 slow 时 < 1 秒跑完
```
⚠️ 用 `.venv/bin/python`，**不要** `uv run`（会另解析一套环境，见 `PROJECT_STATE §11.1`）。
⚠️ 条数不写死（会过期）。

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

**修法**（⚠ 这一节记的是**第一版**做法，已被 §1.8 取代，保留以见演进）：
- `training/checkpoints.py` —— 纯函数 `prune_step_checkpoints(out_dir, keep)`，
  只保留最近 `keep` 个；`training/train.py` 记完归档就调它。
  ⚠ 第一版写的 `getattr(config, 'keep_step_ckpts', 5)` 其实是个**死旋钮**
  （该全局从未定义在 `config_keys` 之前 ⇒ 永远是 5），已在 §1.8 修正。
  ⚠ "只留最近 N 个"这个策略本身也有副作用，也已在 §1.8 换成稀疏保留。
- `tests/test_checkpoints.py` —— 单测。这块代码**会删文件**，写错就是数据丢失，
  所以必须离线覆盖，不能只靠"起一次真训练看看"。
- `scripts/prune_ckpts.sh` —— 外部稀疏化，专治**当时已在跑的旧代码进程**。
  ⚠ 它依赖巡检看守调用，而看守会被会话重启杀掉（§1.9）⇒ 已不再可靠，见 §1.8。

**★ 关键回归点（写进测试）**：必须按**整数步号**排序，不能按文件名字典序 ——
字典序下 `ckpt_step_9000.pt > ckpt_step_10000.pt`（`'9' > '1'`），
于是"保留最近的"会**删掉真正最新的那个，留下旧的，且不报错**。
`test_prune_keeps_newest_by_numeric_order` 就是防这一条。

**通用教训**：长期训练的资源消耗要**按"最坏情况 × 总时长"算一遍**，
而不是看"现在还剩多少"。以及：**巡检时顺手看一眼 `df`，成本近乎为零。**

### 1.8 保留/清理策略从外部搬进训练内部（2026-09-11 晚）

**症状 A —— `results.csv` 续训被清空**：`train.py` 无条件用 `'w'` 打开它，
暂停在 step 22000 后重启，967 字节（22 行评估记录）→ **0 字节**，无任何提示。
**症状 B —— 两个清理策略互相打架**：训练内"只留最新 5 个" vs 外部
`prune_ckpts.sh`"每 5000 步留一个"。前者会在 step 25000 之后把
`ckpt_step_5000/10000/15000` 这些**阶段回溯点**当旧文件删掉。
**症状 C —— 外部清理靠不住**：它由巡检看守调用，而看守会在会话重启时被
SIGKILL（见 §1.9）⇒ 清理实际没跑，而文档以为在跑。

**修法（唯一实现 + 内置）**：
- `training/checkpoints.py` 新增 `steps_to_keep` 作为**唯一决策函数**
  （每 `sparse_every` 步留一个 + 最新 `newest_keep` 个；两条规则都关时**直接抛错**，
  因为那等于"删光全部归档"，绝不该是任何人的意图）。
  `prune_step_checkpoints` 退化为 `sparse_every=0` 的薄包装，旧测试全绿。
- `train.py` 每次存档后调用 `prune_step_checkpoints_sparse`，并把**保留了什么步号**
  打进 tqdm 日志（看不见的策略等于没有策略）。策略异常被 try/except 吞掉 ——
  **清理绝不允许有能力搞崩训练**。
- `scripts/prune_ckpts.sh` 退化成 `python -m training.checkpoints` 的薄包装，
  shell 与 Python 不再各有一份实现。
- `training/run_logs.py` 新增续写策略：`resuming` + 文件非空 ⇒ **追加且不重写表头**；
  0 字节文件（上次启动被 SIGKILL 的残留）⇒ 当作新建。`results.csv` / `ndb.csv` 都改用它。
- 顺带修掉一个**死旋钮**：旧代码写 `getattr(config, 'keep_step_ckpts', 5)`，
  但 `keep_step_ckpts` **从未**被定义为 `config_keys` 快照之前的全局
  ⇒ `load_config` 不会覆盖它、`config` 里也没这个键 ⇒ 该旋钮**永远是 5**，
  在 yaml 里设了也没用。新键 `ckpt_sparse_every` / `ckpt_newest_keep`
  定义在快照之前，是真的可覆盖。

**测试**：`tests/test_checkpoints.py`（20 条，含"step 25000 不得删掉 5000/10000/15000"的
回归）、`tests/test_run_logs.py`（9 条）。
端到端也验过：连跑两次（scratch → resume），`results.csv` 从 3 行变 5 行、旧行保留、表头唯一。

### 1.9 长跑被会话重启 SIGKILL —— 改用 systemd 用户单元（2026-09-11 晚）

**症状**：23:18 用 `setsid nohup ... & disown` 启动训练，23:22 训练**凭空消失**。
日志无 traceback、无 OOM，内核无 OOM 记录，系统内存还剩 12G 可用。
`journalctl` 里找到真凶：
```
dsh-subprocess-14240-<hash>.scope: Killed unit cgroup '...' with SIGKILL on client request.
```
**根因**：harness 每次工具调用建一个 systemd scope；**会话重启会 SIGKILL 整条 cgroup**。
`setsid` 只脱离会话/tty，**脱离不了 cgroup** ⇒ 照死。看守进程同样死。

**修法**：改用 `systemd-run --user --unit=<name> --collect ...`，
它建**独立的用户单元**（`/user.slice/.../app.slice/<name>.service`），不在 dsh 的 scope 里。
已实测：重启训练 + 看守后，两者 cgroup 与当前 dsh scope 完全不同。
命令固化在 `PROJECT_STATE §0.4`，纪律固化在 `AGENTS.md` 铁律 0 / `PROJECT_STATE §8` 铁律 0。

**通用教训**：**"后台"不等于"持久"**。判断一个后台任务能否活过宿主的生命周期，
要看它落在哪个 **cgroup/unit**，而不是看有没有 `nohup`/`setsid`。

### 1.10 `out/` 老实验清理（只删派生物，留证据）

`out/` 曾有 120+ 个实验目录、78.6 GB。实测体积构成：`.pt` 74.06 GB（94.2%）、
`.npz` 4.50 GB（5.7%）、**其余全部（.log/.json/.csv/.png）只有 35 MB（0.04%）**。
⇒ 整目录删会连结论一起丢；只删 `.pt`/`.npz` 能释放 98% 空间、丢 0 条结论。

已清理 **72.51 GB**（`out/` 78.6G → 6.1G，磁盘可用 64G → 112G），
保留 `base_v2`（当前 run）与 `nanoseek_100m`（v1 基座，§0.5.6 配对比较要用）的权重，
并生成 `out/CLEANUP_MANIFEST.md` 记录每个目录删了什么。
工具：`scripts/cleanup_out.py`（默认 dry-run，`--apply` 才删）。

### 1.11 文档里残留的 `setsid nohup` 启动命令 + 已退役的"守夜人"仍在被推荐（2026-09-11 晚）

**问题**：§0.4 已改成 `systemd-run --user`（铁律 0），但**另外两处没跟着改**：

1. `PROJECT_STATE §5` 的**两阶段命令块**（阶段一启动、65000 暂停看守、阶段二退火）
   仍写着 `setsid nohup ... & disown`。这是**最危险的一处** ——
   阶段二的对话退火命令会被照着抄，而按铁律 0，那样起的训练**会话一重启就被 SIGKILL**。
2. `PROJECT_STATE §0.1` 把 `scripts/ckpt_janitor.sh` 称作「★ 守夜人」并推荐启动，
   而它的启动说明同样是 `setsid nohup`；更根本的是**它的职责已被铁律 11 取消**
   （保留策略搬进了训练内部，每次归档落盘就地稀疏化），
   留着一个"会被误当成第二道防线"的进程本身就是负资产。

同批修掉的过期内容：`§0.4` 里重复粘贴了两遍的「前提」段落；
`§5` 里写死的旧 PID（`297961`，每次续训都会变）；`§0.1` 里"当前这个运行进程是旧代码"、
"`keep_step_ckpts` 默认 5"（那是已修的死旋钮）；「恢复上下文三件事」里的
`pgrep -f "training/train.py configs/base_v2"`（**这个模式字面量会匹配到执行它的 shell 自己**，
正是铁律 4 的坑）与写死的测试条数。

**已修**：
- `§5` 两阶段命令全部改为 `systemd-run --user`（单元名 `nanoseek-base-v2` /
  `nanoseek-pause-65000` / `nanoseek-base-v2-p2`），并补上阶段二启动前的**顺序自检**
  （阶段一单元必须 `inactive`，否则两个训练抢同一张卡与同一个 `out_dir`）。
- `ckpt_janitor.sh` 改为**拒绝运行的退役桩**（打印退役理由 + 替代命令，`exit 64`），
  而不是只加注释 —— 注释挡不住复制粘贴，退出码挡得住。
- `§0.1` 写明退役记录与两条理由；`§0.4` 去重；"三件事"改用
  `systemctl --user is-active`（既不会自我匹配，也不依赖会变的 PID）。

**教训（写进 `AGENTS.md` 判据）**：改掉一条铁律后，必须**全文 grep 那条旧写法**，
而不是只改你当时看到的那一处 —— 一处旧命令留在文档里，比整篇没写更危险，
因为它带着"这是本项目认可的做法"的权威。

---

### 1.12 归档清理报的「释放 XMB」**恒为 0**（2026-09-12 巡检发现并修复）

**症状**：巡检日志出现 `删除 1 个，释放 0MB`。看着像"文件是空的"或"只是硬链接"，
实际那个 `ckpt_step_21000.pt` 是 **0.59 GiB**。

**根因**：`training/checkpoints.py` 的 `main()` 先调 `_remove()` 删文件，
**删完再**对同一批文件名做 `os.path.getsize()` → 每个都 `OSError` → 被
`except OSError: pass` 吞掉 → `freed` 永远是 0。

```python
removed = _remove(out_dir, doomed)          # 文件在这里已经没了
for name in removed:
    try: freed += os.path.getsize(...)      # ← 必然抛 OSError，恒加 0
    except OSError: pass
```

**为什么这么久没被发现**：这是个**"测量函数自己坏了"**的 bug —— 它的输出是 0，
而 0 恰好等于"没有可删对象"时的正确输出，所以**两种情况长得一模一样**。
只有拿**已知答案的输入**当对照才会暴露（`AGENTS.md` §5.4 的同一条纪律）。

**已修**：大小在**删除之前**采集，`dry-run` 也报告"将会释放"多少。
新增 3 条测试，其中一条是**对照**（无可删对象时必须仍报 0，不能把"算不出"包装成"释放很多"），
另两条断言报告的 MB 数与磁盘上真实减少的字节数**相等**。

**过程中顺带验证了纪律有效性**：我第一版测试的期望值写错了（把 `newest_keep=1` 下
只删 1 个写成了删 2 个），**是测试先失败、暴露出我的期望错了，而不是代码错了** ——
这正是"先拿已知答案的输入跑一遍"该有的效果。修正期望后全绿。

---

### 1.13 对话自然度评估的两个入口对 v2 语料误配（2026-09-13 评审时发现并修复）

**症状**（三个叠在一起，都让"对话自然度"测不准）：

1. `eval_multiturn.py` 的 prompt **和**轮次判定都写死 `用户：/模型：`（v1 约定）。
   评 v2 基座（语料是 `A：/B：`）时是**喂 OOD 输入**；而且模型自己开的 `A：` 轮次
   **不计入**「自开轮次率」→ 该指标恒 0%，把"会开轮次"误报成"不会"。
2. `sample_py.py::_truncate_at_turn`（`stop_on_turn` 的唯一实现）同样只认两套 v1 标签，
   v2 模型自己开 `\nA：/\nB：` 时**检测不到** → 截断不触发：`chat.py` 表现为"收不住"，
   评估里回复一路顶到 `max_new_tokens`。
3. `eval_multiturn` 的「EOS率」用 `eos_pos != -1` 统计，而它传的是
   `stop_on_eos=False` —— 那条路径下 `<eos>` **根本不置** `eos_pos`，
   于是统计到的其实是**轮次截断**。step 20000 那份报「EOS率 100%」就是这么来的。

**已修**：

* `sample_py.TURN_MARKERS = ("\n用户：", "\n模型：", "\nA：", "\nB：", "\nUser:", "\nModel:")`，
  `_truncate_at_turn` 与逐步 tail 检测统一用它（`A：/B：` 必须带换行前缀，避免英文 `A:` 误判）。
* `eval_multiturn` 增加 `--style`（默认 `ab`，与 `eval_dialogue.py` 同一约定），
  prompt 由 `build_scenarios(style)` 生成；停止原因改用 `stop_kind_ref`，
  汇总拆成 **收尾率 / 自开轮次率 / 收不住率**，不再用文本里有没有标签去猜。

**对照（已知答案的输入，`AGENTS.md` §5.4）**：v2 自开 `A：`、v1 自开 `模型：` 都必须截断；
行内 `Option A: ...` 与**无换行**的 `这是A：标记` 都必须不截断 —— 四条全过。

**为什么当初漏了**：9-11 的 P1 只修了 `eval_dialogue.py` 这**一个**入口
（见 §"已偿还"那条），而 `sample_py` 的轮次标签、`eval_multiturn` 的 prompt 都还在用
v1 约定 —— **同一类缺陷有几个入口，就得逐个改**，改一个不等于改一类。

---

### 1.14 训练跑满 `max_iters` 之后**还会多跑一个优化器步**（2026-09-14 用户实测发现并修复）

**症状**：`--max_iters=3` 打印「训练完成：**4** 步（达 max_iters 3）」，tqdm 走到 `4it`（>100%）。
**根因**：终止判据 `if iter_num > max_iters: break` 写在循环**末尾**、且在 `iter_num += 1`
**之后**，于是流程是 `[eval@k] → [优化器步 k] → k+=1 → (k>max_iters 才 break)`。
跑满之后那一步：① 不评估、不落盘（**白算**，每个 run 固定浪费 1 步）；
② 但它**改了内存里的权重**，而 `last.pt` 是循环顶部 `iter_num == max_iters` 时存的
⇒ **盘上权重与内存权重差一步**；`resume` 时那一步会被重做。
**修法**：判据上移到**评估之后、优化器步之前**，算符改 **`>=`**（不是 `>`）。
修后语义：优化器步**恰好** `max_iters` 次（`iter_num` 0..`max_iters-1`），
最后一次评估/落盘仍在 `iter_num == max_iters` ⇒ **`results.csv` 口径与 ckpt 编号都不变**。
**证据（已知答案的输入）**：`--max_iters=2` 冒烟修前打印 3 步、修后打印 **2 步**；
`results.csv` 两版都是 step 0/1/2（只有优化器步数变了）。
**闸门**：`tests/test_training_loop.py` —— 用 **AST 读源码**（`train.py` 是模块级脚本，
不能 import），断言 `termination < optimizer < iter_num+=1` 且算符必须是 `GtE`；
自带**手写旧形状/新形状片段**的已知答案对照（判据不是恒真的）。

**为什么现在才发现**：这个 off-by-one 只在"打印的步数"和"盘上权重"上露出，
loss/val 曲线完全正常（1/14000 步的差异测不出来）⇒ 单测与曲线都不会报警，
只有人读日志时数着"怎么是 4 步"才会发现。

### 1.15 NDB 从 9 个模块收敛到 2 个（2026-09-15）

**背景**：`model/` 下曾同时躺着 9 个 NDB 相关模块，其中 7 个是已被实测淘汰的路线。
它们会让下一个人（和 AI）读错方向 —— 实测代价就是：我按文档把 `ngram_ndb.py` 判成
"已被取代"，于是说错了"当前 NDB 是灌注、写门控是死的"，**而真正可学写的那个就在旁边**。

**保留的两个**（`model/` 下与 NDB 有关的**只有**这两个）：

| 模块 | 是什么 |
|---|---|
| `model/ngram_ndb.py` | **可读可写的 NDB，读写策略都由模型自己学**。`w_t=σ(W_w·h)` 写门控 + `g_t=σ(W_r·[h;槽统计量;w_t])` 读门控 + `softmax(level_weight)` 多级混合；表是 no_grad 的 token 计数，不进 `state_dict`。**`w_t` 进 `read_gate` 输入 ⇒ `∂L/∂W_w ≠ 0`**（实测 grad=0.0134，`tests/test_ngram_ndb.py:292` 钉着） |
| `model/memory_cross_attn.py` | **神经元级长程读接口**（RETRO-lite）。读侧实测最强（共训 Δ=−0.0738）；写侧 `write_online` 是**规则式种子写**（人给的惊讶分位），不是模型决定 |

**删除的 7 个**（连同 `GPTConfig` 的 6 个旧字段、`Block` 的挂载点、`train.py` 的 GC/导出例程、
`gpt.py` 的参数分组子句一起清掉）：
`neural_db.py`（旧 PK-NDB，values 参与梯度 ⇒ 撑爆 8G 卡）、`residual_neural_db.py`（Δ→−0.0002）、
`product_key_memory.py`、`neuron_db.py`（v5）、`neuron_db_mix.py`、`external_memory.py`、
`interface_network.py`。删除历史见提交信息与 `git log --diff-filter=D`。

**验收**（全绿）：pytest / ruff；`--init_from=scratch` **和** 真 warm start
（`out/base_v3_dlg/last.pt`）各 2 步冒烟均 `exit 0`、`参数量 75.15M`；
确认 `out/base_v2/last.pt`、`out/base_v3_persona/` 未被触碰（冒烟走 `/tmp`）。

⚠ **`local/` 下约 10 个一次性脚本会 ImportError**（它们 import 了被删模块）。
`local/` 是 gitignore 的本机杂物间，不是交付物；需要那些脚本时用
`git log --diff-filter=D` 取回对应模块即可。

---

## 2. 待还的债（按「代价 ÷ 修复成本」排序）

### P0 — 一手实验记录只放在 `out/`（gitignore）里 ⇒ 已经真丢过一站（2026-09-15）

**位置**：`out/ndb_run/STATE.md`（以及一切 `out/**/STATE.md` 形态的 handoff 文件）。
**代价（已兑现，不是假想）**：NDB「共训站」的**唯一完整记录**（471 行）只写在
`out/ndb_run/STATE.md`，而 **`.gitignore:57` 忽略 `out/`** ⇒ **它从未进过 git**。
配套制品（基座 `out/base_probe/best.pt` step 12000、库 `out/mem_store/store5mA.pt`、
`out/ndb_run/` 的 ckpt）**已全部消失**（多个 `out/` 目录 mtime 停在 2026-09-11 23:31，
**原因无记录**）。结果是：全项目最硬的 NDB 正面结果（**共训 Δ=−0.0738**，冻结基线 −0.0348 的 2.1×）
在主文档里**一个数字都没有**，并且 4 天后被另一份文档写成了「已被否决」。
**已做**：原件逐字归档 → `analysis/NDB_cotrain_STATE_2026-09-10.md`；
复核写进 `dev-notes/80-NDB共训站丢失事故与方向纠偏.md`。
**待还**：`out/` 下**其余** handoff / state / 证据文件按同一标准清点一遍；
并考虑给 `training/train.py` 加一条"run 结束时把 STATE 摘要落到 `analysis/` 或日志目录"的约定。
**纪律（已进 `AGENTS.md §1`）**：**只把结论写在 gitignore 的目录里 = 没写。**

### P1 — NDB 的「写」与 `ndb_att_sim`（鲁棒性修法）都从没在训练里跑过（2026-09-15）

**位置**：`model/memory_cross_attn.py`（`write_online` 已有，未接线）、
`training/train.py`（`ndb_att_sim` / `ndb_retr_noise` 键已在、`:763-768` 已消费）。
**代价**：
1. **「写」是用户的硬要求**（`PROJECT_STATE §1` 第 3 条、§6.5），但至今 `train.py` 里
   NDB **写入调用数是 0**；历史上的库全是离线 `--mode mean` 一次建成的。
2. **检索鲁棒性悬崖未修**：实测 **10% 检索出错就废掉 65% 收益，25% 出错时记忆净有害**
   （`analysis/NDB_cotrain_STATE_2026-09-10.md` §2.2）。探针（P6）已验证
   **`ndb_att_sim` λ=2** 能把它翻正（25% 错误：+0.24 vs λ=0 的 −0.36），
   而**专门为它准备的 pilot（`out/pilot_sim/`，`ndb_att_sim: 2.0`）跑了 6 分钟就死了**
   （`ndb.csv`/`results.csv` 都是 0 字节）。
**动作**：把 `write_online` 接进训练循环（配置键**必须定义在 `config_keys`(:313) 之前**，铁律 8；
写只在训练 micro-batch 内、`estimate_loss`/体检期间关闭，防 val 泄漏）；
然后跑**有写 / 无写** A-B，并**同时报 Δ 与鲁棒性曲线**（只报 Δ 会掩盖"写污染了库"）。
**为什么没当场做**：用户要求**启动训练前先报方案等批准**，且这需要先重建库（旧库已删）。

### P1 — 单流 `<resp>` 格式在**采样/评估入口**没有一等支持（2026-09-14）

**位置**：`inference/scripts/sample_py.py`、`inference/scripts/eval_dialogue.py`、
`inference/scripts/eval_multiturn.py`。
**代价**：人格层（`v3_persona`，单流 `<resp>` 格式）训出来的模型，
现成入口**给不出干净的评估证据**，三处叠加：

1. **机制符被吃掉**（本项目栽过三次的那个坑，§7 铁律 14）：`sample_py.py` 命中 `<eos>` 时
   `gen = gen[:new_start+i]` **把 `<eos>` 从返回的 ids 里删掉**（"EOS 及其后不输出"），
   且所有 `tok.decode(...)` 都用默认 `skip_special_tokens=True`
   ⇒ 采样产物**不能直接喂 `training.dialogue_stream.parse_log`**（是一个没闭合的 `<resp>`），
   打印出来的样本也看不见 `<resp>`/`<eos>`。
2. `eval_multiturn.TURN_MARKERS`（§1.13 刚补的）**不含 `<resp>`** ⇒ 单流模型自己开的轮次
   检测不到，截断不触发、自开轮次率失真。
3. `eval_dialogue.py` / `eval_multiturn.py` 的 `--style` **只有 `ab` / `user-model`**，
   没有单流这一种。
**临时绕过**：`scripts/single_stream_e2e.py` 用 `generate_ids` 拿 token 级结果、
把采样器**故意删掉**的那个 `<eos>` 补回去、再用 `skip_special_tokens=False` 解码，
然后喂 `parse_log` 做往返 + loss 区间检查。
**动作**：给 `sample_py.py` 加 `--keep-special`（保留结尾 `<eos>` + 全量解码，默认关，
**不改既有评估产物**）；给两个 eval 入口加第三种 `--style`（单流）。
**为什么不当场改**：改 `sample_py.py` 的默认 decode 会**动既有 8 个模型的评估产物**，
属于"口径变更"，要有对照臂才动（同 §1.13 的教训）。

---

### P1 — 分段训练的"验收尺子"是**两条数据管线**，只报一把会得出相反结论（2026-09-14）

**位置**：`scripts/per_source_ce_probe.py` 的默认口径（`val_char_v2.bin`）vs
分段训练自己那段的 val（`val_char_v3_dlg.bin`）。
**代价**：B 段（14000 步，warm start 自 v2@61000）在这两把尺子上**符号相反** ——
v2 val real **3.0095 → 3.7458**（+0.736，25 源全变差），
v3_dlg val real **2.7415 → 2.1875**（−0.554，8/8 非空源变好）；
连同一个源都相反（`dailychat`：+1.09 / −0.49）。
两边的**污染率都是 0**（0/369、0/153，32-gram 逐字）⇒ 既不是"见过"也不是"背过"，
而是**模型向 v3 管线那套分布迁移**。任何只引用其中一把的报告都是选择性汇报。
**动作**：验收协议已写进 `AGENTS.md §5.12`（两把尺子 + 污染率 + 生成侧证据三者齐报）。
**证据**：`analysis/B_stage_review.md`、`analysis/per_source_ce_{after_B,B_ownval}.txt`、
`analysis/val_train_contamination_{v2val,v3dlgval}.md`。

### P1 — `AGENTS.md` 里"v2 见过 v3 val 的 99%"与实测不符（待澄清口径）

**位置**：`AGENTS.md §1`（原话用来论证"只能用 v2 val 验收"）。
**代价**：2026-09-14 用 32-gram 逐字匹配实测：`val_char_v3_dlg` 在 `train_char_v2` 里
命中 **0/153**。若那句话被当成实测依据，会推出"v3 val 完全不可用"的过强结论。
**动作**：已在 `AGENTS.md §1/§9` 标注为"待更正或注明口径"。若原意是"同源/同分布"，
请补上口径定义；否则删掉。
**证据**：`analysis/val_train_contamination_v3dlgval.md`（含已知答案对照）。

### P2 — `eval_dialogue.py` 的 `turns` 指标**恒为 0**（含历史 8 个模型）

**位置**：`inference/scripts/eval_dialogue.py::dialogue_turn_structure`。
**代价**：`out/_nat_{20000..61000}`、`nanoseek_100m`、`base_v2`、`_eval_last22000`
共 8 个存档全是 `turns=0.0` + `style=none` ⇒ 它是**空指标**，
但报告里和 d1/d2/repN 并列，容易被读成"轮次结构退化"（B 段当场就可能被误读）。
**动作**：要么修到能真的区分（先 dump 被判定为"有轮次结构"的样本，§5.9），
要么在输出里明确标"本指标在所有历史模型上恒 0，暂不可用"。
**证据**：`analysis/eval_dialogue_B.txt`（B 与 v2 都是 0.0）+ 上列 8 个存档的 samples JSON。

### P0 — `DATASET_REPORT.md` 里"val CE 的标准误 1e-3 nats"是错的（差 70 倍）

**位置**：`data/chinese/DATASET_REPORT.md` §"结论摘要"第 2 条 / §5。
**代价**：它按 **token 独立**估 SE（61 万有效 token → 1e-3），但 val loss 的有效独立
单元是**窗口**不是 token —— 实测窗口级 CE 的 σ≈2.0 nats，`eval_iters=200`（800 窗口）
的真实 **σ_eval ≈ 0.087**。这条错误结论会直接诱导人做"跑 N 步比单点 val"的判断，
而 NDB 的 Δ 只有 0.02~0.07 —— **信号会被噪声淹掉 2~4 倍**。
**动作**：改掉那句话，指向 `PROJECT_STATE.md §0.5.3`。
**证据**：`scripts/ckpt_paired_eval.py`（800 窗口分块 sd 实测 0.0865）+ 两条独立推算。

### P0 — 名义数据配比 ≠ 实际训练分布：40% 的语料对训练完全不可见

**位置**：`configs/base_v2.yaml` 的 `use_loss_masking: true` + `stage: full`，
配合 `data/chinese/manifest_v2.json` 的来源配比。
**代价**：`build_assistant_mask` 是行级判据，而 `c4_zh`（20.25%，1.9 亿 token）
**一个 `<eos>/<cont>` 都没有** → 它永远不产生梯度；加上 `wikipedia_cn`（19.67%），
**40% 的语料对训练完全不可见**。模型实际上只见过约 57.6M 有效 token
（全库的 6.14%），却要在上面跑 3.9 遍。
这不是效率问题（拒绝采样已经自动跳过了这些源），是**能力覆盖问题**：
数据卡上写着 40% 的百科/网页，模型一天都没学过。
**动作**：二选一，取决于目标 ——
(a) 通用中文 LM：`use_loss_masking: false`（有效 token 6.14% → 100%，**16.3×**，
    但**不要**用 `stage: pretrain`，它会把 train 切成不存在的 `pretrain.bin`）；
(b) 对话模型：改名义配比，把 c4_zh/wikipedia 的权重让给对话源。
**证据**：全库逐来源终止符精确计数（`c4_zh` = 0）+ 拒绝采样后的实际来源分布。

### ✅ 已偿还（2026-09-11 晚）—— `eval_dialogue.py` 的提示词格式

**原症状**：`DIALOGUE_PROMPTS` 写死「用户：/模型：」，而 v2 语料早已去标签改成 `A：`/`B：`
（dev-notes/61）。后果全是静默的：`turns` **三个模型全是 0.0**（看着像"碎片拼贴"，
其实只是提示词不匹配），而且 **v1 基座在旧标签上有主场优势**，
`d1/d2` 那一列拿去比 v1/v2 **会得出反向结论**。

**修法**：
- `PROMPT_STYLES = {'ab': [...], 'user-model': [...]}`，`--style` 显式选择，
  **默认 `ab`**（v2 主线）；评 v1 及更早显式 `--style=user-model`。
- `dialogue_turn_structure()` 改成**同时识别两套约定**，并返回 `style_detected`：
  这样既能兼容新旧模型，也能发现**格式漂移**（用 `A：` 提示却生成 `用户：`）。
  顺带修掉一个旧逻辑错误：旧实现 `user+model >= 2 and has_model` 会把
  "`B：` 刷屏"这种**单侧复读**误判成"有轮次结构"；现在要求两侧各至少出现一次。
- 命令行会打印 `样式=X(给了 Y)`，不一致时直接打警告。

**测试**：`tests/test_eval_dialogue_prompts.py`（10 条），含"默认样式必须是 ab"、
"两套样式不得串味"、"单侧复读不算结构"、"格式漂移可见"等回归。

**仍成立的告诫**：真要比较 v1/v2，**不要**用本脚本的 d1/d2 —— 用**配对 CE**
（`scripts/ckpt_paired_eval.py`），CE 只吃 token，不受标签格式影响。
**证据**：`out/eval_dialogue.log`（2026-09-11）。

### P2 — `inference/scripts/*` 只认 `out_dir/best.pt`，评不了任意 checkpoint

**位置**：`inference/scripts/sample_py.py::build_model_from_checkpoint` 写死 `best.pt`。
**代价**：想评 `ckpt_step_21000.pt` 这种中间归档，只能先
`ln -f <ckpt> out/_eval_x/best.pt` 造目录 —— 而 `best.pt` 本身是
**噪声选出来的**（`PROJECT_STATE §0.5.4`），所以"评估工具默认评 best.pt"
这件事本身就在**推荐用噪声点**。
**动作**：加 `--ckpt <path>` / `--ckpt-name` 参数，默认仍 best.pt 但允许覆盖。


### P2 — 训练摘要框写「续训自动从 best.pt 恢复」，与代码相反（会误导重启决策）

**位置**：`training/train.py` 的 `print_summary()` 里
`"last.pt（最新）· 续训自动从 best.pt 恢复"`。
**实际**：`train.py:536-538` 是 **`last.pt` 优先**，只有 `last.pt` 不存在才回退 `best.pt`：
```python
ckpt_path = last_path if os.path.exists(last_path) else best_path
```
**代价**：`best.pt` 是**噪声选出来的**（§0.5.4：step 20000 的 1.6432 是 2.31σ 好运，
而 `last.pt` 是 22000）。如果有人信了这句摘要，会以为"续训会退到 20000"，
从而做出错误的重启安排（或反过来，以为能用 best.pt 挑便宜）。
**动作**：把摘要框那半句改成 `last.pt 优先，缺失才回退 best.pt`。


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

**代价**：测试和 lint 门禁目前只能靠人记得跑。（条数不写死在这里 —— 写死的数字会过期，
`AGENTS.md §11` 给的是命令。）

**建议动作**：加一个 `scripts/check.sh`（`ruff check && pytest -q -m 'not slow'`），
再考虑 `.pre-commit-config.yaml`。本机是单机开发，CI 不是必需，
但"一条命令跑完所有检查"应该有。

### P3 — `results.csv` 的 `time` 列会随每次续训归零

**位置**：`training/train.py` 的 `train_start`（`time.time()`）。
**现象**（[实测] 2026-09-12）：`results.csv` 里 `22000` 那行 `time=75120.0`，
续训后 `23000` 那行 `time=3480.0`。`time` 只是**本进程内**的累计秒数。
**代价**：低。它只被当作进度参考，不影响任何决策（MFU 是单独一列）。
但有人若想"用最后一行的 time 减去第一行"算总训练时长，会得到**严重偏小**的数。
**已做的缓解**：在 `train.py` 那一行加注释写明"续训会归零、只能用相邻差值"。
**建议动作**：续训时读 `results.csv` 最后一行的 `time` 作为偏移量累加。
需要 `training/run_logs.py` 加一个 `last_csv_row()`（现有 `count_csv_rows` 是同一类工具），
改完按铁律 7 跑冒烟。

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
