# AGENTS.md —— 给 AI agent 的操作规程（先读这一页）

> 这个仓库是**一个人 + 多个 AI agent 断续接力**维护的。上下文会被压缩、会话会被换掉，
> 所以**不能靠记忆**。本页只讲「怎么做」；「现在是什么状态」在 `PROJECT_STATE.md`，
> 「欠了什么」在 `TECH_DEBT.md`。

---

## 1. 项目一句话

在单张 8G 的 AMD gfx1030 上训一个 **75.15M char-level 中文模型（nanoSeek-100M）**，
然后做一个 **no_grad 外部神经数据库（NDB）**，在不增加模型大小的前提下扩大有效容量
（用户明确要求：NDB **可读可写**，且**由模型自己决定**怎么读写）。

**当前主线（2026-09-14 起，用户拍板：陪聊人格；工程侧见 `PROJECT_STATE §0.5.14`）**：
 0. **★ 人格层 `v3_persona` 是当前站** —— 配置 `configs/base_v3_persona.yaml`，
    单元 `nanoseek-persona-smoke`（**跑完自己消失**，`inactive` 不代表训练"死了"），
    接 `out/base_v3_dlg/last.pt`，**单流 `<resp>` 格式 + `mask_mode: resp_span`**。
    ★★ 语料是 **86k token 的冒烟量 / 300 步 ≈ 28 epoch** ⇒ 它只证明**链路**，
    **别**拿它的 loss 或样本当人格质量结论。
    ★★ **人格的具体设定（性格/语气/措辞）不许写进仓库**（用户：「这些句话除了在上下文中，
    其他地方不要留下任何记录」）—— 语料与规格都在 `~/datasets/persona/`（仓库外），
    仓库里只放格式/掩码/配置/验收工具。**不要**把对话正文或人设描述粘进文档、提交信息、测试。
    ★ 验收工具 `scripts/resp_bin_probe.py`（查 `<resp>`/`<eos>` 配平、**有效 token 占比**、
    两条 loss 口径逐位一致、编解码往返）；实测 2,313/2,313 配平、**64.87% 有效**。
    ⏭ **Step B（未做）**：把**在线写** NDB 接进 `train.py`（现在**调用数是 0**，
    `out/mem_store/` 空；现成的 `MemoryCrossAttention` 是**预构建库只读** = 用户否决的灌注路线），
    再做「有写 / 无写」A-B 对照。
1. **B 段对话专修已跑完（2026-09-14 13:28）** —— 单元 `nanoseek-v3-dlg`（`Result=success`，
   14000/14000 步，12:59:55，`NRestarts=0`），配置 `configs/base_v3_dlg.yaml`，
   warm start 自 `out/base_v2/last.pt`（step **61000**）。产物在 `out/base_v3_dlg/`
   （`last.pt` = 14000，另有 5000/10000/13000 归档）。
   ★ **A 段（`base_v3_know.yaml`）已被用户拍板跳过** —— 理由见 `PROJECT_STATE §0.5.10`；
   配置保留，将来补知识段可再跑。
   ★★ **效果审查已落档：`analysis/B_stage_review.md`**（含结论与两把尺子的分歧）。
2. **验收要看两把尺子，别只看一把**（2026-09-14 实测，详见 §5.12）：
   - **stage 自己的 val**（B 段是 `val_char_v3_dlg.bin`）：real **2.7415 → 2.1875**（−0.554）、
     上下文净利用 −1.998 → **−2.432**，8/8 非空源同向；污染率 **0/153**（不是背下来）。
   - **v2 的 val**（`analysis/per_source_ce_before_B.txt` 那套）：real **3.0095 → 3.7458**
     （**+0.736**），25 源全变差；污染率 **0/369**（也不是"见过"）。
   ⇒ 两把尺子**符号相反**，量的是"自己的分布"与"离旧分布多远"两件事。
   ★ 本节曾写"v2 见过 v3 val 的 99%"——**逐字 32-gram 口径实测是 0/153**，
   该说法要么是别的口径、要么需更正，**别**再拿它当"v3 val 已被背下"的依据。
3. **提速不再是核心** —— 用户明确"目前速度够了"。吞吐已实测到顶（`PROJECT_STATE §0.6`），
   **不要**再去调 batch/compile/dtype。
4. 基座就绪后再回到 NDB。**下一轮训练怎么调（混通用数据 / 降 lr / 补 A 段）尚无定论，
   是一个训练 seed、无对照臂 ⇒ 不许把代价归因给单一原因**（见审查报告 §6.6）。
5. **★ B 段练出的是"对话的形状"，不是"对话的内容"**（2026-09-14 实测，`analysis/B_ood_prompts.txt`）：
   OOD/身份类提示词全线失败 —— `你叫什么名字？` 答成 `你叫乔丹的，也叫马里兰卡`；
   `1+1等于几？` 把问题抄回来；`介绍一下北京` 串成南京；`用 Python 写快排` 只吐语料碎片；
   **连"失恋/压力"这种域内强项也只是模板级**（`你还有老师，也有老师关注你，安稳发展哦`）。
   ⇒ 别拿"CE 变好了"推断"会答问题了"；身份/自我认知在单轮 QA 拼盘里**没有监督信号**，
   而这正是被跳过的 A 段/新知识本该补的东西。

---

## 2. 开工前必读（按顺序，别跳）

| 顺序 | 文件 | 读它干什么 |
|---|---|---|
| 1 | **本页 `AGENTS.md`** | 纪律与陷阱（**本页由 harness 自动注入**，所以它必须永远是最新的）|
| 2 | **`PROJECT_STATE.md` 的 🚀 速查一节** | 当前状态 + 常用命令，够你开工 |
| 3 | `PROJECT_STATE.md` §0.5 + §0.6 | 训练审查结论（噪声口径、有效 token、数据配比）+ 吞吐已测过的旋钮 |
| 4 | `TECH_DEBT.md` §2 | 已知未还的债，别重复发现 |
| 5 | `PROJECT_STATE.md` §8 | 运维铁律的完整版（比 §7 那张表更详细；**编号只增不改**，因为有交叉引用）|
| 6 | 需要时：`data/chinese/DATASET_REPORT.md` | 数据集怎么造的、哪些结论**已作废** |

**只有第 1~2 条是必须的**，其余按需。**不要**一上来通读整个仓库。

> ⚠️ **harness 自动注入的只有两个文件**：
> 1. `~/.dsh/AGENTS.md`（**用户全局**，跨项目偏好，加载在最前）
> 2. **本页**（项目根 `AGENTS.md`）
>
> 项目里**没有** `CLAUDE.md` / `AGENTS.local.md` / 嵌套 `AGENTS.md`，
> 所以**不会有第二份互相矛盾的指令**（若将来要加，注意 harness 只对"内容一致"的同级文件去重 ——
> 内容不同的 `CLAUDE.md` 会作为**第二份指令**一起注入）。
>
> 因此：**"下一个人必须知道"的东西要写进本页**，只写进 `PROJECT_STATE.md` 的话，
> 得等他自己去翻。反之，状态/数字变化频繁的内容放 `PROJECT_STATE.md`，本页只放**不轻易变的纪律与路由**。

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
- ⚠️ **唯一的 python 是 `.venv/bin/python`，不要用 `uv run`**（也**不要**裸 `python`）。
  文档里曾散落着 `uv run pytest` 这类命令（2026-09-12 清掉）—— `uv run` 会按
  `pyproject.toml` 另行解析/同步一套环境，可能跑在**不同的 torch/依赖**上，
  于是"测试全绿"证明的**不是本项目的环境**。命令统一写成：
  `.venv/bin/python -m pytest -q -m 'not slow'` / `.venv/bin/python -m ruff check .`
- **训练跑在 systemd 用户单元里**（铁律 0），最新一站是 **人格层 `v3_persona`（Step A 冒烟）**，看它一眼用：
  ```bash
  systemctl --user is-active  nanoseek-persona-smoke.service
  systemctl --user status      nanoseek-persona-smoke.service --no-pager | head -14
  ```
  单元名：`nanoseek-persona-smoke`（**当前**：人格层，跑 `configs/base_v3_persona.yaml`。
  ★ 它用 `--collect` 起的 ⇒ **跑完单元自己消失**，`is-active` 返回 `inactive` 且
  `systemctl status` 报 "could not be found" **都是正常的**，要看日志判断是成功还是失败）。
  ⇒ 要接着训是**新的一站**，必须新起单元名（并同步改 `scripts/watch.sh:20-21`）。
  ✅ **上一站 `nanoseek-v3-dlg`（B 段对话专修）2026-09-14 13:28 正常结束**
  （`Result=success`、14000/14000 步）；⚠ **旧单元 `nanoseek-base-v2` 已停**
  （2026-09-13 12:48 stop，终态 step **61776**，它的 `last.pt` = step 61000）；
  `nanoseek-pause-65000`（65000 步看守）**从未触发、已作废**，
  **别**再照抄它们的命令去判断"训练是不是死了"。
  ⚠ `systemctl --user` 的单元**不随会话重启而死**，但**随手用 `setsid nohup` 起的会死**。

---

## 4. 常用命令

**全部在 `PROJECT_STATE.md` 的 🚀 速查一节**（续训 / 看单元状态 / 巡检 / 配对重评 /
质量评估 / 密度体检 / 闸门 / 冒烟）。这里不重复，避免两处文档分叉。

**四条最常用的**（其余去速查节抄）：
```bash
bash scripts/watch.sh                                  # 巡检（唯一认可的入口，铁律 6）
systemctl --user is-active nanoseek-persona-smoke.service   # 训练还活着吗
tail -c 1500 out/base_v3_persona_train.log             # 最新进度
tail -40 out/watch_heartbeat.log                       # ★ 我不在时，定时器替我记的巡检心跳
```
⚠ 训练日志在 **`out/base_v3_persona_train.log`**（`out_dir` **之外**，铁律 3）。
⚠ 上面的单元名/日志名**跟着当前站走**：当前站是人格层 `v3_persona`（§1 第 0 条）。
换站时这四条里的后三条、以及下面那条 `watch.sh` 的默认值**都**要一起改。

★★ **巡检已由 systemd 定时器兜底：`nanoseek-watch.timer`（每 30 分钟跑一次 `watch.sh`，
输出追加进 `out/watch_heartbeat.log`）**。2026-09-14 实测的教训：
AI 会话的 `sleep N` 链**是单点故障** —— 半夜 bash 定时器到点结束后，harness 的完成通知
**迟了 6 小时**才送到我手里（会话被挂起/休眠时通知不推进），于是那 6 小时里**没有任何巡检**，
而我（和读文档的人）会以为在监控。训练没受影响（systemd 单元独立于会话，
`NRestarts=0`、进度条时长与墙钟逐秒吻合），但"以为在监控"本身就是本项目最忌讳的失败模式
（同铁律 6/11 的根因）。
⇒ 纪律：**任何"定期要发生"的动作，必须是 systemd 用户单元（`*.timer`），
`sleep` 链只用来让我自己醒来后看一眼**；醒来先读 `out/watch_heartbeat.log` 补盲区，
再跑一次 `watch.sh` 做即时确认。定时器本身用
`systemctl --user list-timers nanoseek-watch.timer` 验证。
⚠★ **换 run 必须同步改 `scripts/watch.sh:20-21` 的默认 `OUT_DIR`/`LOG`** ——
它俩是写死的默认值。不同步的后果不是"少看日志"：**新目录的 ckpt 不会被 prune，磁盘会被写满**，
而且日志/异常关键字扫的是旧 run（2026-09-14 起默认已指向 `out/base_v3_persona`）。

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

### 5.6 ★ val 口径在 **step 22000 断裂** —— `results.csv` 不能跨这条线比较

`use_loss_masking` 于 step 22000 由 `true` 改成 `false`，于是**采样分布和 loss 分母同时变了**：

```
step ≤ 22000   拒绝采样到「含 <eos>/<cont> 的行」上算 loss   val ≈ 1.6 ~ 1.9
step > 22000   全 token 均匀采样、全部算 loss                 val ≈ 4.0 起步，然后下降
```

**这不是训练崩了**，是换了一把尺子。同一个 step-22000 模型在两把尺子下分别是
**1.82**（对话行）与 **3.98**（全量语料）—— 困惑度 6.2 vs 53.5。

⇒ 纪律：
- **禁止**拿 step>22000 的 val 与 ≤22000 的历史比，也**禁止**把两段画在同一条曲线上下结论
- 阶段一（22000→65000，masking off）与阶段二（65000→70000，masking on 对话退火）
  **两段之间也不可比**
- 每个阶段开头**先重新立基线**，之后只做**同阶段内**的配对比较
- `results.csv` 里的断点已用 `out/results_p0_step22000.csv` 单独存档（阶段 0 的 22 个点）

### 5.7 吞吐已实测到顶 —— 不要再花时间调

用户 2026-09-11 明确"提速不是核心，目前速度够了"。**已测过的旋钮别再测**：
`dtype=float16`（更慢）、`compile=false`（慢 5.4%）、`gradient_checkpointing`
（**空开关**，`gpt.py:167` 短路，从未生效）、`ns_steps=5`（灾难）、INT8/FP8（平台不支持）。
微批也扫过：**bs=4 最优，bs=2 只快 9.7%**，`gpu_busy=99%` 但只有峰值 ~5% 算力
⇒ **没有空闲可填**，微调无用。完整结论与证据见 `PROJECT_STATE §0.6`。

### 5.8 ★ `rep3` 是**长度/结构敏感**的 —— 阈值不能跨文本类型搬

同一个 `rep3`，在**短对话回复**（30~80 字）上是"重复坍缩"的好判据（阈值 0.3，干净源误伤 0.1%）；
在**长文档 / 结构化文本**上完全不是：

| 文本 | rep3 |
|---|---|
| 干净散文（四大名著），50 字 | 0.024 |
| 同一段干净文本，1600 字 | 0.131 |
| 合法性 CoT（deepseek，`### 第一步`/`$$`/编号） | **0.3~0.8** |
| c4_zh 网页（同长度下是名著的 2.7 倍） | 0.27 |

**2026-09-13 实测教训**：清洗器对全语料套 `rep3>0.3` 会砍掉 **45%**，抽检发现被砍的 CoT
**全是合法推理**。⇒ 用 rep3 前先问"这段文本的长度与结构像对话回复吗"；跨类型比较必须
**长度匹配**（都用同一长度重新切）。同类坑在 v3 出现过三次（跨源排序、c4 判定、CoT 过滤）。

**跨源比较更要用 n-gram 覆盖率**（"高频 K-gram 覆盖了多少字符"），它对长度不敏感，
且能用**模板注入对照**自证：往干净文本里注入已知比例的模板，覆盖率必须与注入比例逐点相等。

### 5.9 ★ 关键词判据在"接线/改阈值"之前，必须先把**会被翻转的样本**抽出来看

**文档说要这么做，不等于这么做是对的。** 实测事故（2026-09-13）：
`prepare.py` 里有个**无调用点**的死常量 `CONTINUE_QUESTION`（含「吧」「呢」），
文档（`dev-notes/61`）写着"含疑问标记或递回短语（？/？/**吧**/**呢**/你觉得…）→ `<cont>`"。
看起来是个"接线就完事"的漏做项 —— **真接线就错了**。先量了 1,466,294 条回复：

| 接线方式 | 被翻转成 `<cont>` | 抽检看到什么 |
|---|---:|---|
| 整条回复以 吧/呢 结尾 | 0.65% | 全是语气助词（`我还没去拿呢`）与祈使/建议（`我们明天做个计划吧`） |
| 正文出现过 吧/呢 | 3.02% | 第一条就是 **《摔跤吧！爸爸》**（电影名）、`…垃圾分类吧！从分类垃圾开始…` |

⇒ 常量表达的**意图本身是错的**，已删除，并补了反向对照测试钉住。

**规则**：任何形如"含 X 就标 Y / 超过 T 就丢"的判据，改动前**先 dump 被它翻转/命中的样本**
（抽 20~40 条原文读一遍）。命中率小不等于影响小，**命中率大也不等于判对了** ——
只有读原文才能区分。
（同类：§5.8 的 `rep3` 阈值。（对同一批数据连续踩了三次，所以单独立一条。））

### 5.10 ★ 任何"采样/掩码/口径"改动，都要量**有多少 token 真的产生了梯度**

**我们修完一个"语料无声不产生梯度"的 bug，紧接着又自己引入了一个同类 bug。**
（`use_loss_masking` 让 88% 语料白读 → 修好 → 打包的块对齐模式又让 **67~73%** 的 token 永远采不到。）

实测（2026-09-13 打包对抗审计，主 AI 独立复算过）：`pack_align=True` 把窗口**锚在块尾**，
位置 p 可达 ⟺ p 落在某块边界前 T 个 token 内 ⇒ 比 T 长的块，**前 L−T 个 token 永远进不了任何窗口**：

| bin | 不可达 train token |
|---|---:|
| `v3_dlg` | **67.39%** |
| `v3_know` | **72.91%** |
| `v3_lang` | **67.05%** |
| `v2` | **70.67%** |

⇒ **训练必须 `--use_doc_packing=True --pack_align=False`**（全域随机窗口 + 同一张块对角掩码，覆盖 ≈100%）。

**规则**：新增/修改任何**采样策略、loss 掩码、窗口对齐、数据过滤**之后，**必须**给出一个
"**有效 token / 可达 token 占比**"的数字，并配一个**已知答案对照**（判据要能区分好坏两种口径，
否则是恒真的空测试）。这条与 §6 的第 1、2 条是同一个教训的第三次复现，所以单独立一条。

### 5.11 ★ 对比两个数字前，先核对**源集合、分母、口径**是不是同一个

**2026-09-13 对抗复核抓到一次：我把两个不同源集合的数字并排写成了一条"改进曲线"。**

| 我写的 | 真相 |
|---|---|
| 「单轮占比 **90.5% → 7.9%**」 | 90.5% 是 **v2 整库**（25 源）；7.9% 是 **v3_dlg 的 8 个源**。而那 8 个源**在未治理的 raw 状态下本来就是 5.646%**，治理后反而升到 7.914%（整源删了 `multi_turn_dialogue`）⇒ 真正的变化是**重新分桶**，不是清洗的功劳 |
| 「块 **1,856,966 → 1,553,429**（−16.4%）」 | **找不到任何文件子集给出这两个数**。可复现的是 raw_all **1,895,503** → clean_v3 **1,648,376**（=三阶段 manifest `blocks` 之和，逐位相等）。而且精确 blake2b 去重**只解释 80,318 个重复块**，最多占声称降幅的 26% ⇒ 那个"−16.4%"**不能归因于去重** |
| 「12-gram 覆盖 **16.4% → 8.8%**」 | 只有 **union/字符覆盖**口径能对上（实为 17.81% → 8.28%）；"窗口起点"口径只有 7.55% → 3.00% |

**规则**：写"A 降到 B"之前，先答三问 ——
1. **A 和 B 是在同一批源 / 同一分母上算的吗？**（换源集合 = 换了尺子，见 §5.6）
2. **这个数能复现吗？** 给我一条命令、一个口径定义。**记一个自己没能复现的数字，等于埋雷。**
3. **Δ 归因给谁？** 若规则 X 只能解释 Δ 的 26%，就不能写"这是因为 X"。

配套纪律：**让复核者去复核数字，而要复核"数字的来源"** —— 这次复核最大的价值不是算得不一样，
是它**找不到那两个数的出处**。

---

### 5.12 ★ 分段训练的验收必须**同时报两把尺子** —— 只报一把会得出相反结论

**2026-09-14 B 段实测（`analysis/B_stage_review.md`）**：同一个 14000 步的模型，

| 尺子 | v2 基座 61000 | B 段 14000 | 读法 |
|---|---:|---:|---|
| `val_char_v2.bin`（v2 管线，25 源） | real 3.0095 | real **3.7458**（**+0.74 变差**） | "离旧分布远了" |
| `val_char_v3_dlg.bin`（B 段自己的，9 源） | real 2.7415 | real **2.1875**（**−0.55 变好**） | "自己的分布学好了" |

连**同一个源**都给出相反符号（`dailychat`：v2 val **+1.09** / v3 val **−0.49**）
⇒ 两个 bin 来自两条数据管线（v2 的旧 `prepare.py` vs v3 清洗重建），**不是同一把尺子**。

⇒ 纪律：
- **禁止**只拿"旧 run 的 val"给分段训练打分，也**禁止**只拿"stage 自己的 val"宣布成功 ——
  前者量的是**管线距离**，后者量的是**域内拟合**，两者符号可以相反。
- **必须**同时报两个方向，并各自配**污染率**（见下）与**生成侧证据**（`eval_dialogue` /
  `eval_multiturn --style=ab`，同一批 prompt 下比指标 + 读样本）。
- **必须在结论里写清"代价"**：域内变好往往伴随旧分布 CE 上升，不写出来就是选择性汇报。

**配套工具（都是为这件事写的）**：
- `scripts/per_source_ce_probe.py --dump-windows <json>` —— 把**实际用到的窗口起点**落盘。
  ★ 不要靠"照抄采样逻辑"来复现窗口：该函数的 rng 还被 shuffled 的 permutation 消耗，
  抄错一个消耗就会拿到另一批窗口，污染率立刻不可比。
- `scripts/val_train_contamination_probe.py` —— 用那批窗口查"是不是 train 的近重复"。
  ★★ **必须流式扫、不要建全量索引**：9.4 亿 token 的 bin 建索引要 ≈7.5GB，
  实测被 OOM 杀掉两次（子代理连着失败也栽在这里）。
  ★ 必带三组对照：train 原样片段**必须命中**、同段打乱**必须不命中**、
  v2 自己的 train 查 v2 自己的 val **应≈0**。
- B 段实测污染率：**v2 val 0/369、v3_dlg val 0/153**（32-gram 逐字）⇒ 上面那两个相反的数字
  **都不是"见过"造成的**。⚠ 覆盖面只有 48%/60%（含 `<eos>` 的窗口查不了），且只测逐字、
  **近重复未测**。

---

## 6. 数据侧的三个硬事实（2026-09-11 实测，别再重新发现）

> **★ 2026-09-11 晚状态变更**：下面 1/2 两条描述的是 `use_loss_masking: true` 时的情形，
> 而该开关**已在 step 22000 改为 false**（全量语料预训练，见 `PROJECT_STATE §5`）。
> 结论仍然有效、且正是**改它的理由**：它们说明了旧口径下 88% 的语料为什么白读。
> 现在每步有效 token 是 **8192（100%）**，语料覆盖 **937.8M（16.3×）**。

1. **有效 token 密度 39%**（旧口径）：`use_loss_masking=true` + `stage=full` 让 loss 只在
   「含 `<eos>`/`<cont>` 的行」上计算，全库终止符密度只有 **0.1427%**。
   每步名义 8192 token，**实际产生梯度约 3216**。
2. **名义配比 ≠ 实际训练分布**：`c4_zh`（20.25%，1.9 亿 token）**一个终止符都没有**
   → 永远不产生梯度；`wikipedia_cn`（19.67%）只有 0.0008% 密度。
   **40% 的语料对训练完全不可见。** 有效语料只有 **≈57.6M token（全库 6.14%）**。
   → 旧口径下模型是个**窄域对话模型**，不是通用中文 LM。
3. **`multi_turn_dialogue` 名义 4.80%，却占 40.2% 的终止符 / 29.7% 的训练窗口**
   → 它单独决定了模型的人设（共情式心理咨询腔）。
4. **★ 2026-09-13 追加两条（都在 v3 治理里处理了）**：
   - **语料里的"多轮"是离散拼接**：72.6% 的对话 block **恰好 1 轮**（按回复字符算 **≈90.5% 是单轮 QA**）；
     相邻两轮 8-gram 承接性 **Δ 中位数 = 0.0000**（"打乱轮序"对照）⇒ **没有可测的轮间依赖**。
     ⚠ **90.5% 是 v2「整库」口径**；治理后 `v3_dlg` **那 8 个源**的数字是 **7.9%**，而它们在未治理的
     raw 状态下本来就已是 **5.646%** ⇒ **这两个数不能并排读成一条下降曲线**（2026-09-13 对抗复核纠正）。
     唯一全员多轮的 `multi_turn_dialogue`（占 ≥2 轮 block 的 **50.1%**）模板覆盖 **96.4%**，
     已整源清洗掉 —— 2026-09-13 实测：**100,000 块 → 剩 1 块 / 600 字节**（文件仍在，
     仍登记在 `v3_dlg` 里，但已不构成任何训练信号）。⇒ "模型学不会对话逻辑"首先是个**数据问题**。
   - **`<eos>` 曾插在回复首行之后**（`prepare.py::annotate_replies`）：deepseek（27.3% 字符）
     100% 回复首行是 `<think>`，终止符落在回复 **0.33%** 处 ⇒ 在教"`<think>` 之后就停"。
     已在 v3 修为"贴在整条回复末尾"，并补了 5 条测试（此前**零覆盖**）。
     ★ **2026-09-13 实测：这个"教学"在权重里的残留很弱** —— 61000 的权重在 `<think>\n`
     之后给 `<eos>` 的概率只有 **1.32e-4**（随机位置 7.0e-5，只高 1.9×；真正该收尾处 6.0e-1，
     低 4500×）。所以"`<think>` 就停"**不会在采样里兑现**，**别**拿它当"必须重训一段来冲刷"
     的理由。注意它的 **rank 是 28**（vs 随机 844）—— rank 与概率会给出相反读法，
     判"会不会真截断"要看**概率**。量它的工具：`scripts/eos_prior_probe.py`。

**★ 量化"不聪明"的那个数**：同一个 step-22000 模型，在对话行上 loss=1.82（困惑度 6.2），
在**全量语料**上 loss=**3.98**（困惑度 **53.5**）—— 8.6 倍差距。
这就是"背得出语料里的心理鸡汤、却答不出'介绍一下北京'"的硬数字。

**排查这类问题的工具**：`scripts/mask_density_probe.py`；
逐来源统计见 `PROJECT_STATE.md §0.5.2`。

---

## 7. 铁律（违反过的都在这里，完整版见 `PROJECT_STATE.md §8`）

| # | 规则 | 踩过的坑 |
|---|---|---|
| **0** | **★ 长跑用 `systemd-run --user --unit=... ` 启动，绝不用 `setsid nohup`** | **会话重启会 SIGKILL 整条 dsh cgroup，`setsid` 逃不出 cgroup** —— 2026-09-11 实测杀掉过一次训练（无 traceback、无 OOM，只有 journalctl 一行 `Killed unit cgroup`）|
| 1 | **只**用后台 `sleep N` 定时器；禁止前台 sleep、禁止 `job_output(wait=true)` | harness 会杀掉阻塞的会话 |
| 2 | 启动后台任务的命令**必须立刻返回**，绝不和 `sleep` 写在同一条里 | 整条命令被 SIGTERM，任务跟着死 |
| 3 | 训练日志写在 **`out_dir` 之外**（`out/base_v2_train.log`）| `out_dir` 内的文件会被 `_backup_old_run` 挪走，fd 跟着旧 inode → 日志永远空 |
| 4 | `pkill -f <模式>` 会匹配到**自己这条 shell** → 用括号技巧 `pkill -f "ckpt_[j]anitor.sh"`，且**不要把 pkill 和启动写在同一条命令里**。⚠ `pgrep -f` 取受害者 PID 时同理：**别把模式字面量写进同一条命令**（会匹配到自己）| 命令自己把自己杀了（2026-09-11 犯过**两次**：一次 `pkill`、一次 `pgrep -f` 循环）|
| 5 | 诊断代码用 `device_type == 'cuda'`，**不要** `torch.cuda.is_available()`；取显存用 `.get(k, 0)` | CPU 模式下 `KeyError` 崩训练 |
| 6 | 巡检动作必须是**一个原子脚本** `bash scripts/watch.sh`，不许手敲长命令 | 曾出现"文档里有清理、实际定时器没跑" |
| 7 | 改完 `train.py` **必须**跑 2 步冒烟，且**显式加 `--init_from=scratch`**；冒烟**必须带配置文件** `configs/base_v2.yaml` | 配置已是 `resume`，会去读旧 run；不带 config 则改过的键走默认值，**测不到你刚改的路径** |
| 8 | 配置键必须定义在 `config_keys` 快照**之前** | 之后定义的会被静默改回默认值（`keep_step_ckpts` 就是这么变成"永远是 5"的死旋钮）|
| 9 | 归档 checkpoint 的保留策略**内置在训练里**（`training/checkpoints.py`，每 5000 步留一个 + 最新 2 个），写盘前先看 `df` | 曾写满 42GB 磁盘；也曾在 step 25000 后静默删掉早期回溯点 |
| 10 | **重启训练 ⇒ 监控节律重置回 300s** | 问题一般发生在早期 |
| 11 | **保留/清理策略不许只活在外部进程里** —— 外部看守会随会话重启一起死（见铁律 0）| `results.csv` 续训被截成 0 字节；归档清理依赖看守 |
| **12** | **★ warm start（`init_from=<路径>.pt`）必须配独立 `out_dir`** —— 判据是 `init_from != 'resume'`，而 `<路径>.pt` **不是** `resume` ⇒ `_backup_old_run` 照样触发 | 2026-09-13 拟三阶段方案时命令块里写了 `--init_from=out/base_v2/last.pt` 却**漏了 `--out_dir`**；照抄 + `configs/base_v2.yaml`（`out_dir: out/base_v2`）会把基座 61000 步的**全部 ckpt（含 `last.pt` 自己）静默挪进 `old/`**。⚠ 模型**先加载、后归档**，所以**不会当场崩** —— 这正是它阴险的地方。已建 `configs/base_v3_{know,dlg}.yaml` 并加两条断言（`test_v3_stage_config_safety` / `test_all_config_out_dirs_are_pairwise_distinct`）。与铁律 3、9 **同一个根因**（`out_dir` 会被整体归档）|
| **13** | **★「定期要发生」的运维动作必须落在 systemd 用户 `*.timer` 里，不许只活在 AI 会话的 `sleep` 链里** | 2026-09-14：`sleep 1800` 巡检链 02:05 到点后，harness 的**完成通知迟了 6 小时**才送达（会话挂起时通知不推进）⇒ 6 小时内**零巡检**，而所有人都以为在监控。训练没受影响（`NRestarts=0`），但"以为在监控"是铁律 6/11 同族失败模式。已建 `nanoseek-watch.timer` → `out/watch_heartbeat.log`；`sleep` 链只用来让自己醒来。详见本页 §4 |
| **14** | **★ 别用 `cmd \| tail` 串 `&&`** —— 管道会把退出码换成 `tail` 的 **0**，**失败被静默吞掉** | 2026-09-14：`pytest ... \| tail -3 && git commit` 把**跑红的测试**当成通过提交了（`42ebe2f` 就是坏状态）。修法：命令前加 **`set -o pipefail`**（或分开写、取 `$?`）。★ 同族的还有"**验证脚本自己用错 `decode` 默认值**"（`Tokenizer.decode` 默认 `skip_special_tokens=True`，会吞掉 `<eos>`）——**测量脚本要当代码审**，我在这上面连栽三次 |

---

## 8. 改配置要同步改**三处**（有测试钉着）

改 `configs/base_v2.yaml` 里的"配方"键时，必须同时改：

1. `configs/base_v2.yaml`
2. `PROJECT_STATE.md §5` 那张配方表
3. `tests/test_project_layout.py::test_base_v2_matches_documented_decisions`

`test_project_layout.py` 就是**故意**这么设计的 —— 它逼着"文档和配置一起改"。
**不要**为了让测试变绿只改测试不改文档。

**★ 分段配方是另一套（2026-09-13 新增，2026-09-14 加入人格层）**：`configs/base_v3_*.yaml`
用 `extends: base_v2.yaml` 继承架构与优化器，只覆盖"这一段训练的配方"。
它们的步数在 `tests/test_project_layout.py::V3_STAGE_STEPS` 里又写了一遍，
改步数必须同时改**两个文件**（`PROJECT_STATE` 的方案表 §0.5.10 / §0.5.14 + 那个 dict）。
★★ 人格层是**唯一**开着 `use_loss_masking` 的阶段（`mask_mode: resp_span`）——
`test_v3_stage_config_safety` 第 (4) 条**按 `mask_mode` 分支**，不是按文件名白名单。

---

## 9. 不要重复造轮子（先看这里，再决定写不写）

| 想做的事 | 现成的入口 |
|---|---|
| 对话质量 / 生成质量评估 | `inference/scripts/eval_dialogue.py`（rep2/3/4、空白占比、distinct-n、轮次结构；**必带 `--style`**，见下）|
| 多轮对话评估 | `inference/scripts/eval_multiturn.py`（**也必带 `--style`**：收尾率/自开轮次率/收不住率；2026-09-13 前它写死 v1 标签，评 v2 会喂 OOD，见 `TECH_DEBT` §1.13）|
| 采样 / 对话 | `inference/sample.py`、`inference/scripts/chat.py`（或 `cli.py sample` / `cli.py chat`）|
| 巡检 | `scripts/watch.sh` |
| **v3 分段训练配方（当前主线 = 人格层 `v3_persona`）** | ★ **人格层（当前）**：`configs/base_v3_persona.yaml`（`extends: base_v2.yaml`），`init_from: out/base_v3_dlg/last.pt`，`data_prefix: v3_persona`，**`use_loss_masking: true` + `mask_mode: resp_span`**（单流格式必需，见下一条）。★ **B 段**：`configs/base_v3_dlg.yaml`，`init_from: out/base_v2/last.pt`（step 61000）—— **A 段 `configs/base_v3_know.yaml` 已被用户 2026-09-13 拍板跳过**，配方保留备用。启动就是 `train.py configs/base_v3_*.yaml`，**不要再堆一长串命令行参数**（方案文档里那串参数已经全部写进配置）。★★ 铁律 **12**：warm start 的 `init_from=<路径>.pt` **不是** `resume`，照样触发 `_backup_old_run` ⇒ **每段必须有自己的 `out_dir`**。三条断言钉着：`test_project_layout.py::test_v3_stage_config_safety`（第 (4) 条按 `mask_mode` 分支）/ `::test_all_config_out_dirs_are_pairwise_distinct` / `V3_STAGE_STEPS` |
| **单流 `<resp>` 语料的 loss 掩码 / 产物验收** | ★ 掩码：`training/masking.py::build_resp_span_mask`（`<resp>` 之后 → 对应 `<eos>` 含，**不含 `<resp>`**）；权威定义是 `training/dialogue_stream.py::loss_token_spans`，**两者必须对所有输入逐位一致**（相邻 `<resp>` 这种退化输入曾让它们分叉 —— 那次是**实现**错，见 `PROJECT_STATE §0.5.14`）。`train.py` 用 `--mask_mode=resp_span` 选它（未知值**断言退出**，不静默退回）。★ bin 验收：`scripts/resp_bin_probe.py` —— `build_stages.py` 的 `check_terminators` **看不见 `<resp>`**，本脚本补上：`<resp>`/`<eos>` 配平、孤儿计数、**有效 token 占比**、两条口径逐位一致、编解码往返（★ `decode` 必须 `skip_special_tokens=False`）。`--selftest` 带已知答案对照 |
| 配对重评 / val 噪声 | `scripts/ckpt_paired_eval.py` |
| **逐来源的语言能力（"会不会认字"）** | `scripts/per_source_ce_probe.py` —— 按 manifest 的 `val_blocks` 把 val 切回**来源**，报 `real / shuffled / unigram` 三级对照。`real − shuffled` = 真的在读上下文的净度量（`shuffled` 保住相邻对、毁掉长上下文）。★ **口径与 `use_loss_masking` 无关**，所以 **step 22000 那条断裂线在它的表里不存在**，可以跨全程比较。★ **可以换 val**：`--data/--offsets/--manifest/--train-bin` 指到 `v3_*` 就能评 stage 自己的 val（§5.12 要求两把尺子都报）。★★ 曾经写在"源对齐对照"里的三个源名是 **v2 manifest 专属**，换 manifest 会 `KeyError`（2026-09-14 修，见 `align_control_names()` + 4 条测试）。★ `--dump-windows <json>` 把**实际用到的窗口起点**落盘，供污染率审计复用。★ 加 `--control-random` 会再评一个**随机初始化**的模型当已知答案对照（实测 real−shuffled = **+0.001**，B 终态是 **−1.65**，随机权重落在 log(8192)=9.01 ≈ 瞎猜）——**下结论前先看这一行**。⚠ 此前那句"v3 val 已被 v2 见过 99%"**在逐字 32-gram 口径下实测为 0/153**，别再引用它当理由 |
| **val 是不是训练集的近重复（污染率）** | `scripts/val_train_contamination_probe.py` —— 拿 `per_source_ce_probe --dump-windows` 落盘的那批窗口，做 32-gram 定长哈希**流式**扫训练 bin。★★ **必须流式、别建全量索引**：9.4 亿 token 的 bin 建索引要 ≈7.5GB，实测被 OOM 杀过两次。★ 自带三组对照（train 原样片段必须命中 / 同段打乱必须不命中 / v2 自己的 train 查 v2 自己的 val 应≈0）。★ 2026-09-14 实测：B 段 v2 val **0/369**、v3_dlg val **0/153** |
| **某一段训练到底训成什么样（效果审查报告）** | `analysis/B_stage_review.md`（B 段：两把尺子 + 污染率 + 生成侧指标 + OOD 提示词 + 结论与债）；原始输出在 `analysis/per_source_ce_{after_B,B_ownval,B_controlcheck}.txt`、`analysis/eval_{dialogue,multiturn}_B.txt`、`analysis/B_ood_prompts.txt`、`analysis/val_train_contamination_{v2val,v3dlgval}.md`。★ 采样入口是现成的 `inference/scripts/sample_py.py`（`--out_dir/--prompt/--temperature/--seed`），**不要另写采样脚本** |
| **★ 分句 / 流式输入 / 上下文管理 / 自然文本入口（全项目统一）** | ★★ `training/segmentation.py` —— **canonical，别再自己写分句或滑窗**。① `split_line()` / `split_text()`：分句；**机制符原子**（不切进 `<...>`）、且 `<eos>` 这类后缀**不单独成句**（否则滑窗会把 `<eos>` 单独弹掉 = 坏数据）。② `StreamSegmenter`：**流式**（用户输入 / 长文本 / 分块到达）；**跨块的 `<eos>` 也粘得住**（`hold_last=True`，逐字喂也对）；`pending` 是压着的尾巴，`flush()` 收尾。③ `ContextWindow`：**上下文管理** —— 超预算从头部**整句**弹出（用户 2026-09-14 定的规则）；`budget >= NO_LIMIT` 时**不做任何测量**（否则解析整份语料 O(n²)，实测 60s+ 超时 → 0.01s）。④ `prepare_natural_text()` / `process_file()` / `python -m training.segmentation --file X`：**自然文本 → 训练可用**（规整 → 一句一行 → 空行仍是块分隔）。★ `data/chinese/split_sentences.py` 只是它的**薄壳转发**（`prepare.py` 一行没改）；测试见 `tests/test_segmentation.py`（含与旧实现逐字对拍 + 负向对照） |
| **对话流（单流 + `<resp>` + loss 区间）** | `training/dialogue_stream.py` —— `DialogueStream`（`append/commit/prompt/render/rename/loss_token_spans`）、`iter_training_samples`、**`parse_log`**（把日志读回来）、**`read_corpus`/`write_corpus`**（语料统一读写，别再手写 `split('\n\n')`）。★ **上下文管理已委托**给 `segmentation.ContextWindow`，**别在 `DialogueStream` 里再加一套滑窗**。★ 模型自己轮次的标记是 **`<resp>`**（单 token，id 140），不是 `自己：`。★ **换话题标记 `<topic>`**（单 token，id 141）插在**开启新话题那一段的开头**（`append/commit(..., new_topic=True)`，或剧本三元组 `(speaker, text, True)`）；放在**模型自己**那段时它**落在 loss 区间内**，所以模型能学会**主动换话题** |
| 有效 token 密度 | `scripts/mask_density_probe.py` |
| **量"`<eos>` 先验"（数据 bug 在权重里的残留）** | `scripts/eos_prior_probe.py` —— 在 `<think>\n` 之后 / 真·收尾处 / 换轮边界 / 随机中段四组位置上，量 P(`<eos>`) 与它的 **rank**。★ 自带**已知答案对照**（真·收尾组 rank 应为 0，实测通过），`--show` 会 dump 位置前文供人工核对（§5.9）。★★ **rank 和概率会给出相反读法**（实测：`<think>` 后 rank 28 但 P 仅 1.3e-4）—— 判"会不会真的截断"必须看**概率**，rank 只能说明"学到了" |
| NDB 容量上限 | `scripts/ngram_capacity_probe.py` |
| 清理归档 ckpt | `scripts/prune_ckpts.sh`（薄包装）/ `training/checkpoints.py`（**训练内自动**，唯一策略实现）|
| 清理 `out/` 老实验（只删 .pt/.npz，留证据）| `scripts/cleanup_out.py`（默认 dry-run，`--apply` 才删；会先写 `out/CLEANUP_MANIFEST.md`）|
| 指标 CSV 的续写策略 | `training/run_logs.py`（续训**追加**不截断）|
| 语料清洗（去重 / 套话 / 垃圾）| `data/chinese/clean_corpus.py`（**默认 dry-run**，`--apply` 才写；口径与阈值见 `PROJECT_STATE §0.5.9` —— ★ 别自己拍阈值，每条规则都要有对照）|
| 清洗结果的**独立验收** | `data/chinese/verify_clean_corpus.py`（自己重算，**不复用清洗器代码**；含 4 条已知答案对照，`--selftest`）|
| 外部数据 → 项目格式 | `data/chinese/import_external.py`（qa / messages / alpaca / sharegpt / belle / wildchat 六种格式 + selftest）|
| 分阶段切语料 + 构建 bin | `data/chinese/build_stages.py`（lang / know / dlg；`--emit-offsets` 产出打包用的边界表；`--only <阶段>` 只重建/只检查一个）。★★ **`--extra` 默认是 `[]`**：只写在 `new_sources/` 里的源（如 `escov_zh.txt`）**必须显式 `--extra data/chinese/new_sources`**，否则会被静默漏掉（现已改成**字面源名匹配不到就 exit 1**）。★ **`--build` 之后会自动跑"终止符位置验收"**（块内最后一个 `<eos>/<cont>` 必须落在块尾）：2026-09-13 实测 `v3_know` 的 bin 是 `annotate_replies` 修好**之前**构建的，**32.6%** 的块终止符落在回复中段。**改了 `annotate_replies` 之类的文本层逻辑，一定要重建 bin 并看这条验收** —— 旧 bin 不会自己变对。★ **重建后要核对 token 数变了没有** —— 数量逐位相同就说明新源没进去 |
| 样本打包（防跨样本污染）| `training/packing.py` + `--use_doc_packing`（默认关；实测 **28.4% 的随机窗口跨样本**，打包后 A→B 污染 `max|Δ|=0`；⚠ **会改变 val 口径**）。★★ **必须配 `--pack_align=False`**：块对齐模式把窗口锚在块尾，**67~73% 的 token 永远采不到**（§5.10）。I1（注意力）经对抗审计成立（反向对照报 1040 个泄漏，判据非恒真）；标签侧曾有右端漏网，见 §0.5.12 |

**注意**：`inference/scripts/*` 只认 `out_dir/best.pt`。要评任意 checkpoint，
先建硬链接目录：`mkdir -p out/_eval_x && ln -f <ckpt> out/_eval_x/best.pt`。

**⚠️ 评估 prompt 的格式必须和模型训练语料一致** —— 这是 `TECH_DEBT` P1，**已于 2026-09-11 修**：
`eval_dialogue.py` 现在有 `--style`，**默认 `ab`**（= v2 语料的 `A：`/`B：`）；
评 v1 及更早的基座要显式 `--style=user-model`。
选错的后果是喂 OOD 输入：`turns` 假性归零，且**见过那套标签的模型获得虚假主场优势**，
足以把 v1/v2 的结论评反。脚本现在还会报 `样式=` 并在检测到**格式漂移**
（用 `A：` 提示却生成 `用户：`）时打警告。
**但真正要比较 v1/v2 时，仍请用配对 CE**（`scripts/ckpt_paired_eval.py`）——
CE 只吃 token，不受标签格式影响，是更硬的证据。

**⚠️ 不要另写采样/评估脚本。** 2026-09-11 的审查里犯过一次：另写了
`scripts/gen_sample.py`，而 `eval_dialogue.py` 早就覆盖了 rep/空白/多样性/轮次四类
指标，自制版只看得到"读起来像不像话"。需要定性证据时，用与语料同格式的 prompt 喂
`inference/sample.py`，客观指标交给 `eval_dialogue.py`。

---

## 10. 目录导航

```
configs/            训练配置（base_v2.yaml = 基座；base_v3_persona.yaml = **当前站**；base_v3_{know,dlg}.yaml = 前两站；★ 必须放在 out_dir 之外）
training/train.py   ★ 1473 行的模块级脚本 —— import 它就等于开始训练，不能单测
training/           已抽出的纯函数模块（schedules / masking / checkpoints / run_logs / diag）
model/              模型与组件（gpt.py / ngram_ndb.py 是 NDB 原型）
scripts/            运维脚本（watch.sh、探针、清理、cleanup_out.py）
inference/          推理/评估/采样（评估入口都在这）
tests/              单测（含 lint 门禁）；改代码后必跑
data/chinese/       语料与 tokenizer（★ **原始语料已移出仓库**，见下）
  DATASET_REPORT.md   数据集报告（⚠️ 有已作废的结论，见 §5.1）
/home/vesita/datasets/NLP/   ★ 原始语料（2026-09-13 用户要求搬出仓库；29 文件 / 2.4GB）
data/chinese/raw_all/        软链聚合目录（清洗的**输入**）
data/chinese/clean_v3/       治理后的语料（清洗**产物**，进 bin 的输入）+ CLEANING_REPORT.md
data/chinese/stages/v3_*/    阶段软链目录（v3_lang / v3_know / v3_dlg / v3_persona）
data/chinese/new_sources/    新导入数据的转换产物（qa_knowledge / sharegpt / belle / wildchat）
PROJECT_STATE.md    ★ 状态 + 速查 + 铁律
TECH_DEBT.md        技术债
dev-notes/          历史实验记录（编号笔记，写新结论时接着编号）
```

---

## 11. 交付前自检

- [ ] `.venv/bin/python -m pytest -q -m 'not slow'` 全绿
- [ ] `.venv/bin/python -m ruff check .` 全绿
- [ ] 改过 `train.py` → 跑过 2 步冒烟（带 `--init_from=scratch` **和 `configs/base_v2.yaml`**）
- [ ] 后台任务没留下孤儿进程；长跑用 `systemd-run` 起的（铁律 0）
- [ ] 新结论写进了 `PROJECT_STATE.md`（状态类）或 `TECH_DEBT.md`（债类），
      并标注 **[实测] / [推断]**，以及**证据强度**（单次？配对？多 seed？）
- [ ] **改掉了一条铁律 / 换掉了一条命令 → 全文 grep 那个旧写法**
      （`grep -rn "setsid nohup" --include='*.md' --include='*.sh' .`，排除 `.venv`），
      把**每一处**都改掉或标成已作废。
      2026-09-11 实测教训：铁律 0 把启动方式换成 `systemd-run` 后，
      `PROJECT_STATE §0.4` 改了，但 **§5 的两阶段命令块没改** ——
      而 §5 那一段正是阶段二对话退火要照抄的。**残留的旧命令比不写更危险**，
      因为它带着"这是本项目认可的做法"的权威。同批还清掉了写死的旧 PID、
      重复粘贴的段落、"当前进程是旧代码"这类会立刻过期的描述（见 `TECH_DEBT §1.11`）。
      顺带：**别在文档里留会自我匹配的 `pgrep -f "<长模式>"`**（铁律 4）。
      退役的脚本要么删掉，要么改成**拒绝运行的桩**（打印替代命令 + 非零退出码）——
      只加注释挡不住复制粘贴。
- [ ] **★ 本页（`AGENTS.md`）更新了吗？** —— 它是**项目侧唯一被 harness 自动注入**的文档
      （另一个是全局 `~/.dsh/AGENTS.md`），所以判据不是"这次改了什么"，而是：
      **"下一个接手的人，如果只读本页，会不会踩同一个坑 / 走同一条死路？"**
      会 → 就把那条写进本页（纪律进 §5/§7，路由进 §1/§2/§9）。
      **跨项目**的偏好才进 `~/.dsh/AGENTS.md`，本项目的别往那放。

**维护本页的两条约束**：
- **预算 65536 字节**（harness 的 `maxBytes`），超了会**从宽泛的文件开始省略**。
  本页约 16~17 KB、**占预算不到三成**，余量充足 —— 但别把它写成第二本 `PROJECT_STATE.md`
  （那本已经 70+ KB，超预算十倍，正因为它不自动注入才放得下）。
- **只放不轻易变的**：纪律、路由、陷阱。**状态与数字**（step 数、val、磁盘）放
  `PROJECT_STATE.md` —— 写在本页会立刻过期，而过期的指令比没有指令更危险。
