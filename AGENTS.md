# AGENTS.md —— 给 AI agent 的操作规程（先读这一页）

> 这个仓库是**一个人 + 多个 AI agent 断续接力**维护的。上下文会被压缩、会话会被换掉，
> 所以**不能靠记忆**。本页只讲「怎么做」与「怎么查」；**本项目没有"状态文档"** ——
> 状态与数字写下来就会过期，需要时**现场读**：`out/<run>/results.csv`、`out/<run>/ndb.csv`、
> 训练日志、`git log`、`systemctl --user status`。「欠了什么」在 `TECH_DEBT.md`，
> 历史结论与当时的数字在 `dev-notes/`（dated snapshot，不承担"保持最新"的义务）。

---

## 1. 项目一句话

在单张 8G 的 AMD gfx1030 上训一个 **75M char-level 中文模型（nanoSeek）**，
外加一个 no_grad 外部神经数据库（NDB，可读可写、由模型自己决定读写）。

**最近一段：意图跟随密集监督段**（`configs/base_v3_intent.yaml`，已跑完并验收，
审查见 `dev-notes/86`：intent_probe 0/28 → 17/28（语义 ≈9~10），代价 = 文言源 −0.6 级；
下一轮修法也写在 86 §5）。
依据链：`analysis/next_phase_options.md` + `analysis/mask_stage_review.md` §10
（B 段 8 倍算力 = 对照臂 5/28 ⇒ **预算不是瓶颈，缺的是密集监督**；掩码无增益已定案）。
intent 线验收基线：v3_lang 起点 **4.5521**；intent_probe 对照 know2 0 / 掩码臂 4 / 对照臂与 B 段 5 /
intent 17 / intent2 20（关键词口径）。

分段接力已跑完：`base_v2`（61000）→ `v3_dlg`（B 段）→ `v3_persona`（站1）→ `v3_know2`（站2）
→ 掩码实验两臂（空结果）。**链路末端 = `out/base_v3_intent2/last.pt`**（intent 线三轮中最优：v3_lang **4.5521** +
intent_probe 20/28；第三轮 intent3 一次改太多变量而退化，已按判据**回退**，
教训见 `dev-notes/86` §7——每轮只动一个变量；守卫指标要带 avg_len 上下界防"变短"钻空子），
新实验从这里 warm start。
★ 验收任何新段都按 §5.12：**两把尺子**（本段自己的 val + 通用尺子 `v3_lang`，起点基线见
`analysis/know2_stage_review.md`）+ **intent_probe 28 条**（对照数字：know2 0/28、掩码臂 4/28、
对照臂与 B 段 5/28）+ 污染率 + 读生成原文。

**已定案、别再翻案**（证据在 analysis/ 与 dev-notes/）：
- 站 1 人格段配方（小语料 ~86k token、高 lr、无 replay、~85 epoch）会造成灾难性遗忘；
  再跑人格/小语料段必须 **低 lr + 少 epoch + 通用 replay**。
- NDB：行为上可读可写、覆盖率在涨，但收益是记忆性的，**Δ 上界 ≤0.0013 nats，不迁移到 val**。
- 吞吐已到顶，**不要**再调 batch/compile/dtype（§5.7）。
- 语料里"助手显式开新话题"<0.4% ⇒ 缺数据不是缺标注；但反问/递话头 18~38% 供给充足。
- 人格的具体设定（性格/语气/措辞）**不许写进仓库**；语料与规格都在 `~/datasets/persona/`。
  含生成正文的报告只写仓库外（`~/datasets/persona/reports/`）。

---

## 2. 开工前必读

| 顺序 | 文件 | 用途 |
|---|---|---|
| 1 | 本页 | 纪律与陷阱（harness 自动注入，必须保持最新且自足）|
| 2 | 本页 §4 常用命令 + §1 主线 | 够开工 |
| 3 | `TECH_DEBT.md` | 已知未还的债，别重复发现 |
| 4 | `dev-notes/`（按需）| 历史结论与当时的数字（快照，不保最新）|
| 5 | 需要时 `data/chinese/DATASET_REPORT.md` | ⚠ 内含已作废结论，引用前先核对 §5.1 |

**只有 1~2 条是必须的**，不要一上来通读整个仓库。

---

## 3. 环境（照抄，别自己推）

```bash
cd /home/vesita/coding/my/nanoSeek
HSA_OVERRIDE_GFX_VERSION=10.3.0 HSA_ENABLE_SDMA=0    # gfx1030 必需
.venv/bin/python                                       # 唯一 python（torch 2.9.1+rocm6.4）
```

- 8G 显存 / 15G 内存 / 12 核。训练峰值显存 ~1.7G；**探针/推理很容易吃爆**（§5.3）。
- ⚠ **唯一的 python 是 `.venv/bin/python`**，不要 `uv run` / 裸 `python`
  （uv 会另解析一套环境，"测试全绿"证明的不是本项目环境）。
- 训练跑在 **systemd 用户单元**（铁律 0）。当前是否有单元在跑**一律现场读**：
  ```bash
  systemctl --user list-units 'nanoseek-*' --no-pager
  ```
  ★ `--collect` 起的单元跑完自己消失，`inactive` / "could not be found" 都是正常的；
  成败看日志里的 `最终 best_val_loss` 与异常关键字。
  ⚠ `systemctl --user` 单元不随会话重启而死；`setsid nohup` 起的会死（铁律 0）。

---

## 4. 常用命令（不要从别处抄，dev-notes 里的会过期）

```bash
# ① 启动/续训当前段（★ 必须 systemd 单元；换站时同步改 ①b 单元名、日志名、
#    scripts/watch.sh 的默认 OUT_DIR/LOG —— 不同步 ⇒ ckpt 不被 prune，磁盘写满）
systemd-run --user --unit=<单元名> --collect \
  --property=WorkingDirectory=/home/vesita/coding/my/nanoSeek \
  --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=HSA_ENABLE_SDMA=0 \
  /bin/bash -c '.venv/bin/python -u training/train.py configs/<配置>.yaml > out/<日志>.log 2>&1'

# ② 巡检只认这一条（定时器每 30 分钟也跑一次，追加进 out/watch_heartbeat.log）
bash scripts/watch.sh
systemctl --user list-timers nanoseek-watch.timer --no-pager

# ③ 配对重评（单点 val 噪声 σ≈0.087，只信配对）
.venv/bin/python scripts/ckpt_paired_eval.py --ckpts <a.pt> <b.pt> --batches 800

# ④ 逐源语言能力（两把尺子工具；--data/--offsets/--manifest/--train-bin 指到 v3_* 换 val）
.venv/bin/python scripts/per_source_ce_probe.py \
  --data data/chinese/val_char_v3_lang.bin --offsets data/chinese/val_char_v3_lang.off \
  --manifest data/chinese/manifest_v3_lang.json --train-bin data/chinese/train_char_v3_lang.bin \
  --ckpts <起点.pt> <终点.pt>

# ⑤ 意图跟随 28 条（先 --selftest 验判据可用；报告逐条落原文）
HSA_OVERRIDE_GFX_VERSION=10.3.0 HSA_ENABLE_SDMA=0 \
  .venv/bin/python scripts/intent_probe.py --out-dir out/_eval_<名> \
  --out analysis/intent_probe_<名>.md --json analysis/intent_probe_<名>.json

# ⑥ 闸门：改完代码必须全绿
.venv/bin/python -m pytest -q -m 'not slow' && .venv/bin/python -m ruff check .

# ⑦ 改完 train.py 的 2 步冒烟（显式 --init_from=scratch 且带 config）
.venv/bin/python training/train.py configs/base_v2.yaml \
  --out_dir=out/_smoke --init_from=scratch --device=cpu --compile=false \
  --batch_size=1 --block_size=256 --gradient_accumulation_steps=1 --max_iters=2 \
  --eval_interval=1 --eval_train_split=false

# ⑧ 当前状态一律现场读
tail -c 1500 out/<日志>.log; tail -5 out/<run>/results.csv; tail -5 out/<run>/ndb.csv
git log --oneline -10
```

⚠ 训练日志写在 **`out_dir` 之外**（铁律 3）。看 tqdm 日志先 `tr '\r' '\n'`。
★★ **换 run 必须同步改 `scripts/watch.sh` 的默认 `OUT_DIR`/`LOG`** ——
不同步的后果是**新目录 ckpt 不被 prune、磁盘写满**，且异常扫描指向旧 run。

---

## 5. 测量纪律（本项目最贵的教训都在这）

### 5.1 单点 val 不可信 —— 只信配对
`results.csv` 的 val 相邻跳动 ±0.19 全是测量噪声（每千步真进步 ~0.010）。
比较两个模型必须配对（`ckpt_paired_eval.py`）或看 ≥5000 步聚合。
⚠️ `DATASET_REPORT.md` 里"val CE 标准误 1e-3 nats"的说法是错的，别引用。

### 5.2 `best.pt` 是噪声选出来的
22 个评估点里出现一次 ≥2.3σ 好运的概率 37%，best.pt 必然选出噪声点。
用 `last.pt` 或固定归档。

### 5.3 探针必须与训练同构
写任何计时/显存/前向探针前先问：这个形状/批量在训练里出现过吗？
（实测事故：chunk=50 的配对探针 fp32 upcast 吃到 7.88G/8G，而训练峰值 1.7G。）

### 5.4 新测量函数先过"已知答案的输入"
每个测量函数配一个正确答案已知的对照输入；对照不过先修测量，别下结论。
★ 对照要针对**最可能搞错的那一步**，并用物理判据兜底（CE≈9.01=log 词表 = 瞎猜）。

### 5.5 结论措辞强度必须匹配证据强度
单次、少步、单 seed 的 <1% 差异只能写"小于 1%"，不能写"没有效果"。

### 5.6 val 口径断裂 —— `results.csv` 不能跨口径比较
`use_loss_masking` / `data_prefix` / `mask_mode` 任一变化 = 换尺子（采样分布与 loss 分母都变）。
每段开头重新立基线，只做段内配对比较；跨段一律走 `per_source_ce_probe`（口径与 masking 无关）。
两臂 `results.csv` 的 val **不可互比**（分母不同），比较走 probe。

### 5.7 吞吐已到顶 —— 别再调
已测过且无杠杆：dtype、compile、gradient_checkpointing（空开关）、ns_steps、微批扫描。
用户明确"速度够了"。

### 5.8 rep3 是长度/结构敏感的
短对话回复上 rep3>0.3 是重复坍缩好判据；长文档/结构化文本（CoT、c4 网页）上合法文本也能到 0.3~0.8。
跨类型比较必须长度匹配；跨源比较用 n-gram 覆盖率 + 模板注入对照自证。

### 5.9 ★ 关键词判据改动前，先把被翻转的样本抽出来读
任何"含 X 就标 Y / 超过 T 就丢"的规则：改动前 dump 20~40 条命中/翻转样本读原文。
命中率小不等于影响小，命中率大也不等于判对了。（rep3 阈值、CoT 过滤、意图挖掘都栽过。）

### 5.10 ★ 任何采样/掩码/口径改动，必须量"有效 token / 可达 token 占比"
且配一个能区分好坏口径的已知答案对照。历史上连续三次"语料无声不产生梯度"事故：
loss masking 让 88% 语料白读；pack_align=True 让 67~73% token 永远采不到 ⇒
**训练必须 `use_doc_packing: true` + `pack_align: false`**。

### 5.11 ★ 对比数字前，先核对源集合、分母、口径是不是同一个
写"A 降到 B"前先答三问：①A 和 B 在同一批源/分母上算的吗？②这个数能复现吗（给命令）？
③Δ 能归因给谁（规则 X 只解释 26% 就不能写"因为 X"）？
配套：让复核者复核"数字的来源"，不只是复核数值。

### 5.12 ★ 分段验收必须同时报两把尺子
- 本段自己的 val（域内拟合）+ 一把与本段语料无关的通用尺子 `v3_lang`（能力距离）。
  两者符号可以相反，只报一把会得出相反结论。
- 汇总会平均掉灾难：**至少读完一个单源**（烟感器选模型本来擅长且 val 块多的现代中文源：
  `wikipedia_cn` / `c4_zh`；四大名著与文言源本来就弱 ~2 nats，测抖动会过敏，降为次要读数）。
- 必须配污染率（`val_train_contamination_probe.py`，**必须流式扫**，建全量索引会 OOM）
  与生成侧证据（`eval_dialogue` / `eval_multiturn` / `intent_probe`，同批 prompt 比指标 + 读样本）。
- 必须写清"代价"：域内变好伴随通用 CE 上升，不写就是选择性汇报。
- 上游打坏的下游要花步数还 ⇒ 每段独立验收不是形式主义。

### 5.13 分段验收的判据要**启动前写死**在配置注释里，事后不许挑有利的一条宣布胜利
（掩码实验：5 条判据 3 过 1 部分 3 不过 ⇒ "本段不算成功"，正是判据写死才敢这么说。）

---

## 6. 数据侧的硬事实（细节与出处见 dev-notes/85、data/chinese/DATASET_REPORT.md）

- `v3_dlg` 是**真多轮**；终止符**逐轮覆盖**且必须**在 bin 上量**
  （`stages/v3_*/*.txt` 是标注前文本，grep 终止符得 0）。
- 助手**反问/递话头**供给充足（18~38%）；**显式开新话题** <0.4%（缺数据，刷标注长不出来）；
  `<topic>`（id 141）注入零成本（写进 .txt 即单个 id），但"注得进去"≠"有东西可注"。
- 身份污染（自称 OpenAI/ChatGPT）真但小（最大源 3.4% 字符），不足以解释能力失败。
- 有效 token / 可达 token 的口径事故史见 §5.10；排查工具 `scripts/mask_density_probe.py`。
- **意图跟随密集监督语料**由 `data/chinese/build_intent_stage.py` 生成
  （挖掘 598 + 模板合成 177 + 通用 replay；**14 条探针原句留出不进训练**）；
  阶段目录 `data/chinese/stages/v3_intent/`，重建后必须跑终止符验收。

---

## 7. 铁律（编号只增不改；每条都有踩坑现场）

| # | 规则 | 坑 |
|---|---|---|
| **0** | 长跑用 `systemd-run --user --unit=...`，绝不用 `setsid nohup` | 会话重启 SIGKILL 整条 dsh cgroup，`setsid` 逃不出 cgroup；现场=进程消失+journalctl 一行 `Killed unit cgroup` |
| 1 | 等待分两种：**有明确终点**（构建/测试/一次性扫描）正常等到结束（`job_output(wait=true)` 或看日志结束标记）；**无终点长跑看护**才用后台 `sleep N` 定时唤醒 | 把多天训练的结论过度推广到所有任务，"等 6 分钟"变"等 6 分钟才发现早结束了" |
| 2 | 启动命令必须立刻返回，绝不和 `sleep` 写在同一条里 | 整条命令被 SIGTERM，刚启动的作业跟着死 |
| 3 | 训练日志写在 `out_dir` 之外 | `out_dir` 会被 `_backup_old_run` 挪走，fd 跟旧 inode → 日志永远空 |
| 4 | `pkill -f`/`pgrep -f` 会匹配自己这条 shell → 用括号技巧 `pkill -f "foo_[b]ar"`；别把模式字面量写进同一条命令 | 自己杀过自己两次 |
| 5 | 诊断代码用 `device_type == 'cuda'` 守卫，取值一律 `.get(k, 0)`；诊断逻辑抽成纯函数（`training/diag.py`）| CPU 模式 `KeyError` 崩训练 |
| 6 | 巡检必须是一个原子脚本 `bash scripts/watch.sh`，不许手敲长命令 | "文档里有清理、定时器实际没跑" |
| 7 | 改完 `train.py` 必须跑 2 步冒烟：显式 `--init_from=scratch` 且**带配置文件** | 配置是 resume 会读旧 run；不带 config 则改过的键走默认，测不到你改的路径 |
| 8 | 配置键必须定义在 `config_keys` 快照**之前** | 之后定义的被静默改回默认值 |
| 9 | ckpt 保留策略内置在训练里（`training/checkpoints.py`），写盘前看 `df` | 写满过 42GB 磁盘 |
| 10 | 重启训练 ⇒ 监控节律重置回 300s | 问题多发在早期 |
| 11 | 保留/清理策略不许只活在外部进程里（看守随会话重启一起死）| `results.csv` 被截成 0 字节 |
| **12** | warm start（`init_from=<路径>.pt` ≠ `resume`）必须配**独立 `out_dir`**，否则 `_backup_old_run` 把起点 ckpt 全挪进 `old/`（先加载后归档，不当场崩，最阴险）| 三条断言钉着：`test_v3_stage_config_safety` / `test_all_config_out_dirs_are_pairwise_distinct` / `V3_STAGE_STEPS` |
| **13** | "定期要发生"的运维动作必须落在 systemd 用户 `*.timer`；`sleep` 链只用来让自己醒来（醒来先读 `out/watch_heartbeat.log` 补盲区）| 会话 `sleep` 链是单点故障；通知可能因 API 限额迟几小时，期间"以为在监控" |
| **14** | 别用 `cmd \| tail` 串 `&&`（退出码被 tail 换成 0）→ 前加 `set -o pipefail`；测量脚本要当代码审（`decode` 默认吞机制符，栽过三次）| 跑红的测试被当通过提交过 |

---

## 8. 改配置要同步改的地方（有测试钉着）

- `configs/base_v2.yaml` 的"配方"键 ⇄ `tests/test_project_layout.py::test_base_v2_matches_documented_decisions`。
- v3 分段：`configs/base_v3_*.yaml`（`extends: base_v2.yaml`，只覆盖本段配方）
  ⇄ `V3_STAGE_STEPS`（步数登记）。新增阶段配置必须登记，否则测试拦下。
- 开 `use_loss_masking` 只许两种组合（测试按 `mask_mode` 分支拦）：
  `resp_span` + `v3_persona`（单流 `<resp>`），`eos_line` + `v3_dlg`（答案段掩码实验）。
  其余语料前缀开 masking 会被拦（`v3_lang` 无终止符 ⇒ 零梯度）。

---

## 9. 不要重复造轮子

| 想做的事 | 现成入口 |
|---|---|
| 对话质量评估 | `inference/scripts/eval_dialogue.py`（必带 `--style`，默认 `ab`）|
| 多轮对话（收尾率/自开率/收不住率）| `inference/scripts/eval_multiturn.py`（必带 `--style ab`）|
| 意图跟随 28 条 | `scripts/intent_probe.py`（先 `--selftest`；子进程采样自带 gfx 环境变量硬拦；去提示词回显；见其文件头两条事故注记）|
| 逐源能力 real/shuffled/unigram | `scripts/per_source_ce_probe.py`（`--dump-windows` 落窗口起点供污染审计；`--control-random` 是已知答案对照，下结论前先看这行）|
| val 污染率 | `scripts/val_train_contamination_probe.py`（**流式扫**，别建全量索引，OOM 过两次）|
| 采样 | `inference/scripts/sample_py.py`（`--out_dir/--prompt/--temperature/--seed`）|
| 单流 `<resp>` 语料（读/写/loss 区间/parse_log）| `training/dialogue_stream.py`；bin 验收 `scripts/resp_bin_probe.py`；掩码 `training/masking.py::build_resp_span_mask`（与 `dialogue_stream.loss_token_spans` 必须逐位一致）|
| 分句/滑窗/上下文管理（全项目统一）| `training/segmentation.py`（canonical）；`data/chinese/split_sentences.py` 是薄壳转发 |
| NDB 载具 | `model/ngram_ndb.py`（外挂，不进 GPTConfig；梯度来自读路径；判"写门控在学"看 read 侧；接线测试 `tests/test_ndb_wiring.py`）|
| 配对重评 | `scripts/ckpt_paired_eval.py` |
| 有效 token 密度 | `scripts/mask_density_probe.py` |
| `<eos>` 先验 | `scripts/eos_prior_probe.py`（判"会不会真截断"看**概率**不看 rank）|
| 清 ckpt / 清 out/ | `scripts/prune_ckpts.sh`（薄包装）/ `scripts/cleanup_out.py`（默认 dry-run）|
| 语料清洗 | `data/chinese/clean_corpus.py`（默认 dry-run；别自己拍阈值）；独立验收 `verify_clean_corpus.py --selftest` |
| 外部数据导入 | `data/chinese/import_external.py`（六种格式 + selftest）|
| 分阶段切语料 + 建 bin | `data/chinese/build_stages.py`（`--extra` 默认 `[]`，新源必须显式给，否则静默漏掉；`--build` 后自动跑终止符位置验收；改文本层逻辑后必须重建 bin）|
| **意图跟随语料** | `data/chinese/build_intent_stage.py`（挖掘+合成+replay 一体；`--dump N` 抽样供 §5.9 判读；`--selftest`；`--stage` 产出 `stages/v3_intent/`）|
| 样本打包 | `training/packing.py` + `--use_doc_packing`（必配 `pack_align=false`，§5.10）|

注意：
- `inference/scripts/*` 只认 `out_dir/best.pt`。评任意 checkpoint 建硬链目录：
  `mkdir -p out/_eval_x && ln -f <ckpt> out/_eval_x/best.pt`。
- **不要另写采样/评估脚本**——现成入口覆盖的指标远多于自制版。
- 改了"文本层"逻辑（annotate/分句/清洗）后，**旧 bin 不会自己变对**，必须重建并看终止符验收；
  重建后核对 token 数变了没有（逐位相同=新源没进去）。

---

## 10. 目录导航

```
configs/            训练配置（base_v2 = 基座；base_v3_* = 分段配方，★ 放 out_dir 之外）
training/           纯函数模块（schedules/masking/checkpoints/run_logs/diag/packing/segmentation）
training/train.py   模块级脚本 —— import 即训练，不能单测；改后必跑冒烟（铁律 7）
model/              模型与组件（NDB 载具 ngram_ndb.py）
scripts/            运维与探针（watch.sh、per_source_ce_probe、intent_probe、ckpt_paired_eval…）
inference/          推理/评估/采样（评估入口都在这）
tests/              单测（含 layout/接线门禁）；改代码后必跑
data/chinese/       语料与 tokenizer
  build_intent_stage.py   意图跟随语料生成（挖掘+合成+replay）
  clean_v3/               治理后语料（清洗产物）
  stages/v3_*/            阶段目录（v3_lang / v3_know / v3_dlg / v3_persona / v3_intent）
  new_sources/            导入/生成的转换产物（intent_mined.txt、intent_synth.txt 在这）
/home/vesita/datasets/NLP/   原始语料（在仓库外，2.4GB）
TECH_DEBT.md        技术债
dev-notes/          历史实验记录（编号快照；写新结论接着编号）
analysis/           各段效果审查（B/know2/mask/next_phase_options…）
out/                run 产物与训练日志（.gitignore；旧日志在 out/logs_archive/）
```

---

## 11. 交付前自检

- [ ] `.venv/bin/python -m pytest -q -m 'not slow'` 全绿；`.venv/bin/python -m ruff check .` 全绿
- [ ] 改过 `train.py` → 2 步冒烟（`--init_from=scratch` + 带 config）；
      动过步数/终止/评估节律 → 另跑 `pytest tests/test_training_loop.py`
- [ ] 后台任务没留孤儿进程；长跑是 systemd 单元（铁律 0）
- [ ] 新结论写进 `dev-notes/NN` 或 `TECH_DEBT.md`，标 **[实测]/[推断]** + 证据强度
      （单次/配对/多 seed）。一手证据放 git 跟踪路径（`out/` 不进 git）
- [ ] 改掉铁律/换掉命令 → 全文 grep 旧写法（含 .md/.sh，排除 .venv），每处改掉或标作废；
      退役脚本删掉或改成拒绝运行的桩（只加注释挡不住复制粘贴）
- [ ] **本页更新了吗？** 判据："下一个只读本页的人，会不会踩同一个坑/走同一条死路？"
      会 → 写进本页（纪律进 §5/§7，路由进 §1/§9）。
      **状态与数字不写进任何文档**（写下来就会过期），现场读（§4 ⑧）。

**维护本页的约束**：预算 65536 字节。**再往里加之前，先合并或删减已有条目**
（写作纪律：只写现状、不写沿革与负描述、一条信息只说一遍、写判据不写考古）。
