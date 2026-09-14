> ⚠ **冻结快照，不再维护**（2026-09-15 用户决定移除 `PROJECT_STATE.md`：它与 `AGENTS.md`
> / `dev-notes` 大量重复，同一个事实写两处就必然有一处先过期 ⇒ **它本身是漂移来源**）。
> **不要照抄本文的数字与状态**（step / val / 磁盘 / 配置键都会过期）：
> - 纪律与路由 → `AGENTS.md`（唯一被 harness 自动注入的页，必须自足）
> - 常用命令 → `AGENTS.md §4` · 欠债 → `TECH_DEBT.md`
> - 当前状态 → 从 `out/<run>/results.csv`、`out/<run>/ndb.csv`、训练日志与 `git log` 读
> 本文只作为**历史证据**保留（当时实测的数字与当时的推理）。

# nanoSeek 项目状态（上下文压缩后的唯一恢复入口）

> **上下文被压缩后，第一件事就是读这个文件。** 不要凭记忆重启训练、不要凭记忆改配置。
> 最后更新：2026-09-15
>
> **本页只放「当前状态 + 怎么操作 + 铁律」。** 历史实验叙述（2026-09-11~14 的一批审查与方案）
> 已按字节搬到 **`dev-notes/81-PROJECT_STATE历史归档.md`** —— 原处只留标题 + 指针，
> 所以 `§0.5.10` 这类交叉引用仍然解析得到。**归档 ≠ 作废**，引用那些结论时请标明出处。
>
> 配套文档：`TECH_DEBT.md`（技术债与闸门）、`AGENTS.md`（**给 AI agent 的操作规程，先读它**）、
> `dev-notes/`（按编号的实验记录）、`tests/README.md`（测试布局）。

## 🚀 速查（最常用的东西就在这一节，别往下翻）

**当前状态**：🧩 **主线是「人格层」（`v3_persona`）—— Step A 冒烟已跑完，Step B（NDB 在线写）未开始**。
单元 `nanoseek-persona-smoke`（`--collect` ⇒ **跑完自己消失，`inactive` 是正常的**）：
`Result=success`、300/300 步、19.8 min、异常 0；`results.csv` train/val **3.5563/3.4986 → 0.1233/0.1278**
（★ 86k 语料 × 300 步 ≈ 28 epoch ⇒ **背下来了**，只证明链路，**别当效果**）。
端到端验收（采样 → `parse_log` 回读）四条全过：收尾 **6/6**、往返一致、`<resp>` 不进 loss、无相邻同说话人。
配置 `configs/base_v3_persona.yaml`，`init_from: out/base_v3_dlg/last.pt`，
**`mask_mode: resp_span`**（单流格式必需）。日志 `out/base_v3_persona_train.log`，产物 `out/base_v3_persona/`。
详细口径与"不许把人设写进仓库"的约束见 **§0.5.14**（先读它的开头那条自我约束）。
★★ **NDB 侧 2026-09-15 有一次重要的方向复核：整个「NDB 共训」站（`out/ndb_run/`，Δ=−0.0738）
此前不在主文档里，已补进 §6.0** —— 谈 NDB 之前先读它；`out/ndb_run/STATE.md`（471 行）是原始记录。
★ 同批修掉一个 off-by-one：跑满 `max_iters` 后旧代码**还会多跑一个优化器步**
（`--max_iters=3` 打印 4 步）—— 已修，见 §0.5.14 末尾。
✅ **B 段对话专修已跑完**（2026-09-14 13:28）—— 单元 `nanoseek-v3-dlg`，
`Result=success`、**14000/14000** 步、12:59:55、`NRestarts=0`；配置 `configs/base_v3_dlg.yaml`，
`init_from: out/base_v2/last.pt`（= step **61000**）。产物 `out/base_v3_dlg/`
（`last.pt` = 14000；归档 5000/10000/13000/14000）。
⚠ 该单元**现在 inactive 是正常的**；⚠ 它是**修 off-by-one 之前**跑的，优化器步实际 14001。
**A 段（`base_v3_know.yaml`）已被用户拍板跳过**（见 §0.5.10）。
⚪ 基座 `nanoseek-base-v2` **已停**（2026-09-13 12:48 stop，终态 step **61776**，其 `last.pt` = 61000）；
`nanoseek-pause-65000` 看守**从未触发、已作废** —— 别照抄它们的命令判断"训练死了没"。
**B 段效果审查**（完整版 `analysis/B_stage_review.md`）：
- **自己那段的 val**（`val_char_v3_dlg.bin`）：real **2.7415 → 2.1875**（−0.554）、
  上下文净利用 −1.998 → **−2.432**，8/8 非空源同向；污染率 **0/153**。
- **v2 的 val**（`val_char_v2.bin`）：real **3.0095 → 3.7458**（**+0.736**），25 源全变差；污染率 **0/369**。
- 生成侧同向变好：空白 1.09%→**0%**、distinct-2 0.83→**0.98**、自开轮次 0%→**17%**。
⇒ 两条尺子**符号相反**（同一个源都相反）⇒ 读作"**向 v3 管线那套分布迁移，并付出 v2 分布的代价**"，
**不要**只引用其中一条（纪律见 `AGENTS.md §5.12`）。
**关键数字**：单点 val 噪声 **σ≈0.087** · 有效 token 密度 **100%**（Stage1 起）·
B 段**起跑前基线**（`per_source_ce_probe` @61000）：real **3.0095** / real−shuffled **2.050**（`analysis/per_source_ce_before_B.txt`）。
⚠ **val 口径已于 step 22000 断裂**；B 段验收**两把尺子都要报**（见 §0.5.10 末尾 + `AGENTS.md §5.12`）。

```bash
cd /home/vesita/coding/my/nanoSeek

# ① 启动 / 续训**当前这一站**（★ 必须用 systemd 单元，不能用 setsid nohup —— 见铁律 0）
#    当前站 = 人格层 v3_persona（§0.5.14）；换站时**必须**同时改这里、①b、
#    和 scripts/watch.sh:20-21 的默认 OUT_DIR/LOG（不同步 ⇒ 新目录 ckpt 不被 prune）
systemd-run --user --unit=nanoseek-persona-smoke --collect \
  --property=WorkingDirectory=/home/vesita/coding/my/nanoSeek \
  --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=HSA_ENABLE_SDMA=0 \
  /bin/bash -c '.venv/bin/python -u training/train.py configs/base_v3_persona.yaml > out/base_v3_persona_train.log 2>&1'
#    后台等它跑完（有明确终点 ⇒ 正常等到结束，不用 sleep 链）：
#    systemd-run ... --wait ...   ← 或 systemctl --user is-active nanoseek-persona-smoke

# ①b 看训练单元的存活与日志
systemctl --user status nanoseek-persona-smoke.service --no-pager | head -14

# ② 巡检：只认这一条，别手敲长命令（铁律 11）★ 默认已指向 out/base_v3_persona
bash scripts/watch.sh

# ②b 补盲区：systemd 定时器 nanoseek-watch.timer 每 30 分钟自动跑一次 watch.sh，
#     输出追加到这里。AI 会话被挂起/通知迟到时，这是唯一可靠的记录（§8 铁律 14）
tail -40 out/watch_heartbeat.log
systemctl --user list-timers nanoseek-watch.timer --no-pager   # 它还在跑吗

# ③ 要看 val 趋势 → 配对重评（单点 val 不可信，见 §0.5.3）
.venv/bin/python scripts/ckpt_paired_eval.py --ckpts out/base_v2/ckpt_step_22000.pt --batches 800

# ④ 对话质量评估（客观指标，现成的，别自己另写采样器）
#    ★ --style 必须和模型训练语料一致，否则喂 OOD、turns 假性归零（TECH_DEBT 已偿还那条）
#      默认 ab = v2 主线（A：/B：）；评 v1 及更早的基座用 --style=user-model
.venv/bin/python inference/scripts/eval_dialogue.py --dirs out/base_v2 --style ab
#    ⚠ 它只认 out_dir/best.pt；要评任意归档先建硬链接目录（见 §9 说明）

# ⑤ 有效 token 密度体检
.venv/bin/python scripts/mask_density_probe.py --n 3000

# ⑥ 闸门：改完代码必须全绿
.venv/bin/pytest -q -m 'not slow' && .venv/bin/python -m ruff check .

# ⑦ 改完 train.py 的 2 步冒烟（★ 必须显式 --init_from=scratch，且**带 config**，
#   否则 use_loss_masking 等键走默认值，测不到你刚改的那条路径 —— 见 §8 铁律 8）
.venv/bin/python training/train.py configs/base_v2.yaml \
  --out_dir=out/_smoke --init_from=scratch --device=cpu \
  --compile=false --batch_size=2 --gradient_accumulation_steps=1 --max_iters=2 \
  --eval_interval=1 --eval_iters=1 --eval_train_split=false
```

**四个最容易犯的错**：

| 别做 | 该做 | 为什么 |
|---|---|---|
| 用 `setsid nohup` 启长跑 | `systemd-run --user --unit=...` | **会话重启会 SIGKILL 整条 dsh cgroup，setsid 逃不掉**（2026-09-11 实测丢过一次训练）|
| 拿 `results.csv` 的单点 val 判断好坏 | 配对重评，或 ≥5000 步聚合 | σ≈0.087，而每千步真进步只有 0.010 |
| 用 `best.pt` 当"最好的模型" | 用 `last.pt` 或固定归档 | `best.pt` 是噪声选出来的（§0.5.4）|
| 自己写采样/评估脚本 | 用 `inference/scripts/eval_dialogue.py` | 已覆盖 4 类已知失败模式 |
| 把监控写成手敲的长命令 | `bash scripts/watch.sh` | 曾出现"文档里有清理、实际定时器没跑"|

---

## 0. 一句话现状

🟸 **主线基座训练已按用户要求「暂停」在 step 22000 / 70000（2026-09-11 20:21）**，
存档完整、可续训（见 §0.4）。今天的训练告一段落。
NDB 方向经**三次定位**，**当前载体 = ③ RETRO-lite（no_grad 长程 chunk KV + cross-attention）**：
① 后缀键+梯度值（§78）Δ≈−0.0046 **已放弃**；② token 级后缀 n-gram（TDB，`model/ngram_ndb.py`）
同预算 Δ−0.0077 —— ⚠ **「用户已否决 TDB」是 2026-09-14 的误归因，见 §6 的 ★ 更正（2026-09-15）**；
③ **RETRO-lite Δ=−0.0348（256 窗）＝ 同预算 4.5× TDB，三个伪影检查全过**（`dev-notes/79`，见 §6）。
★ ② 与 ③ 是**互补不是替代**（`dev-notes/79 §7` 原话：TDB 管局部续写、长程 NDB 管跨窗依赖）。
★★ **但上面这三个数都不是 NDB 的最好成绩** —— **共训跑过 `out/ndb_run/`，Δ=−0.0738（step 19000，
是冻结基线 −0.0348 的 2.1×），然后被用户停在 19781 去跑 pilot，pilot 没跑完，这一站就从主文档里
消失了** ⇒ **见 §6.0，那是全项目最硬的 NDB 结果，也是"交接失败"的现场**。
★★ **NDB 已收敛到单一载具**：`model/ngram_ndb.py` —— **读与写都由模型门控决定**，
表在训练中**在线累积**（不需要离线建库、没有逐段重建、没有基座漂移问题）。
`ndb_*` 全键在 `train.py` 顶部，`ndb_slots = 0` 即关闭，**默认由 `configs/base_v2.yaml` 开启**。
⚠ 上面 ② ③ 与 §6.0 的 Δ 数字都来自**已退役的载具**（离线建库 + 只读 cross-attn），
留作历史证据；**本轮方案的 Δ 尚无实测**，第 1 段的 `ndb.csv` 就是第一份数据（`dev-notes/82`）。
工程侧有 **300 条单元测试 + lint 门禁**（§11）。

### 0.4 ▶️ 启动 / 续训（复制即用）

> ⚠️ **2026-09-11 血的教训：绝对不要用 `setsid nohup ... &` 启长跑。**
> harness 每次工具调用都建一个 systemd scope（`dsh-subprocess-<pid>-<hash>.scope`），
> **会话重启时会 `SIGKILL` 整条 cgroup**，而 `setsid` **逃不出 cgroup**。
> 实测：23:18 用 `setsid nohup` 启动训练，23:22 会话重启，训练连同看守一起被杀
> （日志无 traceback、无 OOM，只有 `journalctl` 里一行
> `Killed unit cgroup ... with SIGKILL on client request`）。
> ✔ 正确做法是 `systemd-run --user`：它建**独立的用户单元**，不在 dsh 的 scope 里。

```bash
cd /home/vesita/coding/my/nanoSeek

# 启动 / 续训 B 段（★ 用 systemd 用户单元，脱离 dsh cgroup）
systemd-run --user --unit=nanoseek-v3-dlg --collect \
  --property=WorkingDirectory=/home/vesita/coding/my/nanoSeek \
  --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=HSA_ENABLE_SDMA=0 \
  /bin/bash -c '.venv/bin/python -u training/train.py configs/base_v3_dlg.yaml > out/base_v3_dlg_train.log 2>&1'

# 查看存活 / 停止 / 重看日志
systemctl --user status nanoseek-v3-dlg.service --no-pager | head -14
systemctl --user stop   nanoseek-v3-dlg.service
bash scripts/watch.sh
```

**前提（B 段）**：`configs/base_v3_dlg.yaml` 里 `init_from: out/base_v2/last.pt`（**warm start，不是 resume**）
与 `out_dir: out/base_v3_dlg` —— **两者都已写死在配置里**，所以命令永远就是上面那一条。
⚠ `init_from=<路径>.pt` **不是** `resume`：它只搬权重，**优化器 / LR / step 全部重置**（铁律 12），
并且照样触发 `_backup_old_run(out_dir)` ⇒ 每段必须有独立 `out_dir`。
（旧的 `configs/base_v2.yaml` 是 `init_from: resume` + `out_dir: out/base_v2`，会走 `out/base_v2/last.pt`
自动恢复模型/优化器/LR/RNG 且跳过载入那一步的评估与存档；**基座已停，别再用它续训**。）

⚠️ **重申：用 `systemd-run` 而不是 `setsid nohup`**（理由见本节开头的实测）。
若因故只能用 `setsid nohup`（例如在没有 systemd 的环境），必须知道**会话重启会杀掉它**，
并在重启后立刻检查 `systemctl --user is-active` 等价的存活状态。

⚠️ **想从零开一个新 run 时**，必须显式覆盖：`--init_from=scratch --out_dir=out/<新目录>`。
否则 `init_from=resume` 会去读旧 run 的 `last.pt`。
（同理，§8 铁律 8 的 2 步冒烟命令要加 `--init_from=scratch`，否则会尝试 resume 空目录。）

**暂停点存档清单**（`out/base_v2/`，每份 0.59 GiB）：

| 文件 | step | val | 备注 |
|---|---|---|---|
| `last.pt` | **22000** | 1.9107 | ★ **续训用这个** |
| `ckpt_step_22000.pt` | 22000 | 1.9107 | 同步归档 |
| `ckpt_step_21000.pt` | 21000 | 1.7924 | |
| `ckpt_step_20000.pt` | 20000 | **1.6432** | = `best.pt` |
| `ckpt_step_15000.pt` | 15000 | 2.0631 | 稀疏保留点 |
| `ckpt_step_10000.pt` | 10000 | 2.0809 | 稀疏保留点 |
| `ckpt_step_5000.pt` | 5000 | 2.1694 | 稀疏保留点 |
| `best.pt` | 20000 | 1.6432 | ⚠️ 见下 |

⚠️ **`best.pt` 是"噪声选出来的"，不要当作"最好的模型"**：
它的 val 1.6432 明显低于左右邻居（19000: 1.9203、21000: 1.7924、22000: 1.9107），
而单点 eval 噪声 **σ≈0.087**（2026-09-11 实测，见 §0.5）→ 这个 −0.15~0.27 的优势里
**主要是抽样运气**。做 NDB 对照实验请用 `last.pt`（22000）或固定某个稀疏归档，**不要用 `best.pt`**。

**暂停时的状态**：`lr` 仍在 WSD 稳定段（3e-4，衰减要到 step 56000）→ 续训无 LR 断层。

### 0.5 🔍 训练审查（2026-09-11 晚，step 22000 / 70000）

对 21 小时、22000 步的 run 做了一次完整的量化审查。**结论：训练本身干净，但有两个
结构性发现，都跟"算力花在哪"有关。** 全部数字都是实测，脚本见 §0.5.4。

#### 0.5.1 ⭐ 有效 token 密度只有 39%，有效语料只有全库的 6.14%

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`（只搬了位置，内容字节未改）。本页只保留**当前状态与规程**。

#### 0.5.2 ⭐⭐ 名义数据配比 ≠ 实际训练分布（`c4_zh` 一个终止符都没有）

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`（只搬了位置，内容字节未改）。本页只保留**当前状态与规程**。

#### 0.5.3 eval 噪声：机制已定位，σ≈0.087

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`（只搬了位置，内容字节未改）。本页只保留**当前状态与规程**。

#### 0.5.4 收敛与其它观察

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`（只搬了位置，内容字节未改）。本页只保留**当前状态与规程**。

#### 0.5.6 过拟合与基线对照（配对，同一批 3200 窗口）

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`（只搬了位置，内容字节未改）。本页只保留**当前状态与规程**。

#### 0.5.5 ⭐⭐⭐ 训练质量：loss 健康，但模型是个「心理咨询机器人」，且没有世界知识

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`（只搬了位置，内容字节未改）。本页只保留**当前状态与规程**。

#### 0.5.7 🗣 对话自然度基线（阶段一，2026-09-13 @ step 61000 / 训练中）

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`（只搬了位置，内容字节未改）。本页只保留**当前状态与规程**。

#### 0.5.8 ⭐ 归因：对话自然度差，**主因是数据集**，不是模型太小（2026-09-13）

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`（只搬了位置，内容字节未改）。本页只保留**当前状态与规程**。

#### 0.5.9 🧹 v3 语料治理 + 三阶段切分（2026-09-13；训练停在 step 61776）

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`（只搬了位置，内容字节未改）。本页只保留**当前状态与规程**。

### 0.5.10 📋 训练方案（2026-09-13 拟定；用户决定"等翻译完成后再训"）

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`（只搬了位置，内容字节未改）。本页只保留**当前状态与规程**。

#### ★ 2026-09-13 实测：A 段的**第一条**理由（"冲刷 `<eos>` 先验"）**被推翻了**

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`（只搬了位置，内容字节未改）。本页只保留**当前状态与规程**。

### 0.5.11 🩺 对话语料健康度体检（2026-09-13，对 v3_dlg 全量）

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`（只搬了位置，内容字节未改）。本页只保留**当前状态与规程**。

### 0.5.12 🐛 打包对抗审计（2026-09-13）：一个 P0 覆盖漏洞 + 一个标签漏网

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`（只搬了位置，内容字节未改）。本页只保留**当前状态与规程**。

### 0.5.13 🌐 escov 机翻对话的验收与取舍（2026-09-13）

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`（只搬了位置，内容字节未改）。本页只保留**当前状态与规程**。

### 0.5.14 🧩 人格层（`v3_persona`）：单流 `<resp>` 格式 + `mask_mode=resp_span`（2026-09-14）

★★ **本节的自我约束（先读这条再往下）**：人格的**具体设定**（性格、语气、要避免什么、
要传递什么）**故意不写进仓库** —— 用户明确要求「这些句话除了在上下文中，其他地方不要留下
任何记录」。语料本体与规格都在仓库外（`~/datasets/persona/`，规格 `_spec.md`）；
仓库里只放**工程侧**的东西：格式、掩码口径、配置、验收工具。
⇒ **不要**把对话正文、人设描述、示例回复粘进本文档 / 提交信息 / 测试 / 注释。
本节提到人格时只用中性说法（"人格层"、"人设绑定"）。

**为什么会有这一段**：用户 2026-09-14 把主线从"通用对话专修"改成"**陪聊人格**"，
理由是 NDB 本来就不基于上下文，人格比知识更贴合那条路线（原话大意：陪聊更符合我们用
NDB 这种不基于上下文的预设）。B 段（`v3_dlg`）练出的是**对话的形状**不是内容
（`analysis/B_ood_prompts.txt` 实测 OOD/身份类全线失败）⇒ 人格层要补的正是
"**我是谁 / 我怎么说话**"这一层监督信号。

#### 0 ⏭ 2026-09-15 长跑计划：`persona(1) → 通识 → persona(2)`，**NDB 默认启用**

**用户 2026-09-15 拍板**：进入长程训练；顺序 `人格 → 通识 → 人格`；**NDB 从现在起是训练的
默认组件**（不再是要记得加的可选项）。理由（用户原话大意）：最后一段决定最终人设，
所以中间插一段通识补能力、末尾再用人格把声线收回来。

**两个数字先钉死**（都是实测算出来的，别凭感觉改）：

| 语料 | train token | 步/epoch @ 8192 | 本计划取的步数 | = epoch |
|---|---:|---:|---:|---:|
| `v3_persona` | 809,598 | 98.8 | 900（两轮各） | **9.1** |
| `v3_know` | 487,755,407 | 59,540 | **18,000** | **0.302** |

⇒ 通识段**不是"把知识背完"**，是"吃进 17%~30% 的新 QA + 一次 LR 重启"。
（1 epoch = 59,540 步 ≈ 56h 跑不起；3k 步 = 0.05 epoch 已被 §0.5.10 论证为"最站不住的档位"。）

**四段一览**（每段**独立 WSD、各自退火到 `min_lr`** ⇒ 最后一段定最终人设）：

| 顺序 | 配置 | `init_from` | `out_dir` | 步数 | 语料 | masking | 时长估算 |
|---|---|---|---:|---:|---|---|---|
| **λ 短臂 ×2** | `configs/pilot_persona_lam{0,2}.yaml` | `out/base_v3_dlg/last.pt` | `out/pilot_persona_lam{0,2}` | 400 ×2 | `v3_persona` | `resp_span` | ~1.2h |
| **1 人格** | `base_v3_persona.yaml` | `out/base_v3_dlg/last.pt` | `out/base_v3_persona` | **900** | `v3_persona` | `resp_span` | ~1.3h |
| **2 通识** | `base_v3_know2.yaml` | `out/base_v3_persona/last.pt` | `out/base_v3_know2` | **18,000** | `v3_know` | 关 | **~22~25h** |
| **3 人格** | `base_v3_persona2.yaml` | `out/base_v3_know2/last.pt` | `out/base_v3_persona2` | **900** | `v3_persona` | `resp_span` | ~1.3h |

合计 **≈27h**（NDB 开着；老 run 实测 5.0 s/步）。

**为什么先跑 λ 两臂（400 步 ×2，只差 `ndb_att_sim` 一个键）**：
`ndb_att_sim`（把检索相似度 z 分数加进注意力 logit）只在**免训练探针**上验过
（step 19000；25% 检索错误时 λ=2 是 **+0.24**、λ=0 是 **−0.36**），**从没进过训练**。
用 <1.5h 把它问清楚，比在 27h 里带着一个未验证的旋钮强。
判据：看 `ndb.csv` 的**配对** Δ（四条件同批 val），**不看**单点 val（σ≈0.087，§5.1）。

**NDB 的取用方式（本轮）**：**只开读**。库由 `local/build_chunk_store.py --mode mean` 离线建成，
**每段按该段起点的基座重建**（键 = chunk 均值，基座漂移了检索空间就对不上）。
写侧 `MemoryCrossAttention.write_online` 是**规则式**的（人给的惊讶分位），
不是"模型自己决定写"⇒ 本轮不开（见 `TECH_DEBT §2` P1；"模型自己学写"的实现在
`model/ngram_ndb.py`，见 `AGENTS.md §1` 的门面表）。
库文件名规范：`out/mem_store/store_<建库所用ckpt目录名>_gen10m.pt`。

★ **"NDB 默认启用"已变成断言**：`tests/test_project_layout.py::test_v3_stage_configs_enable_ndb`
要求每个 `base_v3_*` 阶段 `ndb_store` 非空（两份历史配方在 `NDB_EXEMPT_STAGES` 里显式豁免并写理由）。

#### 一、用户拍板的决定（照抄，别再自己发明）

| 决定 | 内容 | 备注 |
|---|---|---|
| 格式 | **单流**（方案"乙"）：整个对话是一条流，模型轮的标记是 **`<resp>`** | 不是多流、不是 `用户：/模型：` |
| 模型自己的标记 | **`<resp>`**（单 token，**id 140**） | 取代早期的 `自己：`/`<你该说话了>`；它由 harness 喂，**模型不该生成它** |
| 换话题标记 | **`<topic>`**（单 token，**id 141**），插在**开启新话题那一段的开头** | 放在**模型自己**那段时它落在 loss 区间内 ⇒ 模型能学会**主动**换话题（commit `a95cd0d`） |
| 对方编号 | 未绑定身份时用 **`对象A` / `对象B` / `对象C`** | 用户提议、已用在生成的语料里 |
| 身份绑定 | `对象A：我是xx。` ⇒ 后续改用 **`xx：`** | 允许绑定后改名（`DialogueStream.rename`）|
| `<cont>` | **已退役**（token 保留，id 130，绝不删——删了会动其它 id） | 用户："直接合并 cont 吧，保留 eos" |
| 语料边界（a+b） | 人设描述**不进仓库**；语料**放仓库外的 `~/datasets/`** | 仓库里只留软链（`data/chinese/new_sources/` 本身已 gitignore） |

#### 二、loss 口径：**必须** `mask_mode: resp_span`（这是本层唯一的新机制）

单流格式里每一行都自带终止符，所以：

- 旧规则（`mask_mode='eos_line'`：token 所在**行**内含 `<eos>/<cont>` ⇒ **整行**算 loss）
  不会误伤对方行（它们没有 `<eos>`），但会把 **`<resp>` 自己**算进 loss ——
  每轮只多 1 个 token，方向却是错的（在教模型顺手输出 `<resp>`）。
- 关掉 masking 更糟：全部 token 等权。
- ⇒ 新规则 `resp_span`：**`<resp>` 之后 → 对应 `<eos>`（含）** 算 loss，**不含 `<resp>`**；
  `<topic>` 落在区间内照常算 loss。实现 `training/masking.py::build_resp_span_mask`
  （纯 cummax/cummin），权威定义是 `training/dialogue_stream.py::loss_token_spans`。

★ **写这一版时抓到一个真 bug（值得记，因为它差点被"参考实现"掩盖）**：第一版向量化实现用
"**含自身**的 `cummax` 得 `prev_resp`" + 断言 `prev_resp < t` 来表达"cue 自身不算 loss"。
它在 **相邻两个 `<resp>`**（`<resp><resp>…<eos>`）上给 `[F,F,T]`，而**权威顺序扫描**给
`[F,T,T]`（第一个 `<resp>` 的扫描把第二个当**正文**吞进区间）。实测 `[140,140,128]` 分歧。
⇒ 真正的判据是「**严格早于** t 存在 `<resp>`」，改用右移一格的严格版。
真实流里 `append/commit` 不会产出相邻 cue，所以这不是行为 bug，但
"向量化实现 ≡ 权威 span 定义"必须**对所有输入**成立。
**教训**：我最初的测试参考实现也写成了一次扫描的状态机，它**同样**在未闭合 `<resp>` 上出错
⇒ 是**参考错了，不是实现对了**；消融前先怀疑参考，并且要用**两条独立写法**的参考互钉
（`tests/test_masking.py::test_resp_span_two_references_agree_exhaustively` 穷举 L≤7 的
全部短行，再把实现穷举到 L≤6 的 15625 行 —— 400 行随机对照**没抽到**相邻 cue 那个组合）。

#### 三、产物与验收（`[实测]`，2026-09-14）

构建（语料是 800 段单流对话 / 107,850 字符）：

```bash
.venv/bin/python data/chinese/build_stages.py --src data/chinese/clean_v3 \
    --extra data/chinese/new_sources --only v3_persona --apply --build
```

| 项 | 实测值 |
|---|---|
| `train_char_v3_persona.bin` | **85,967** token（792 段；`.off` 793 个边界）|
| `val_char_v3_persona.bin` | **850** token（8 段，`--val-all` 口径与 train 同源）|
| 终止符位置验收 | `rel_p50=0.9815`、`rel<0.5` 占比 **0.00%** ✅ |
| `<resp>` / `<eos>` 计数 | **2,313 / 2,313**（配平）、孤儿 **0 / 0** |
| `<topic>` 计数 | **0**（本层语料还没用它；留给 persona 的 dialogue 层）|
| ★ **有效 token 占比** | **55,767 / 85,967 = 64.87%**（`resp_span` 口径；AGENTS §5.10 要求）|
| 两条 loss 口径 | 向量化 **逐位等于** 权威顺序扫描 ✅ |
| 编解码往返 | 3/3 块逐字相同 ✅（`skip_special_tokens=False`！）|

验收工具：**`scripts/resp_bin_probe.py`**（`--selftest` 带已知答案对照、`--json` 落盘）。
它补的正是 `build_stages.py` 的 `check_terminators` **看不见**的那一半 ——
后者只查"块内最后一个终止符落在块尾"，**完全不看 `<resp>`**。

#### 四、配置与测试例外

`configs/base_v3_persona.yaml`（`extends: base_v2.yaml`）：
`out_dir: out/base_v3_persona`、`init_from: out/base_v3_dlg/last.pt`（B 段终态 step 14000）、
`data_prefix: v3_persona`、`use_doc_packing: true` + `pack_align: false`、
**`use_loss_masking: true` + `mask_mode: resp_span`**、`eval_interval: 50`（段短，否则
`best.pt` 永不落盘）。

★ 它与其余 v3 阶段的一条硬不变量**相反**：别处要求 `use_loss_masking is False`，
本层必须为 `True`。`tests/test_project_layout.py::test_v3_stage_config_safety` 的
第 (4) 条因此改成**按 `mask_mode` 分支**：
`mask_mode == 'resp_span'` ⇒ 必须开 masking，且 `data_prefix` 必须是 `v3_persona`
（别的阶段写 `resp_span` 会被拦下——那说明配错了语料，后果是 mask 全 False ⇒ loss NaN）。
例外**绑在格式上、不绑在文件名上**，所以将来新增单流语料阶段也能复用这条判据。

#### 五、Step A（冒烟）与 Step B（NDB 在线写）

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`。
> ★ **结论仍在用**：Step A 冒烟 ✅ 跑通（`Result=success`、300 步、19.8 min、异常 0；
> val 3.50 → 0.128 但那是 **96k 语料 × 28 epoch ＝ 背下来了**，只证明链路）。
> ★ **Step B 已被 2026-09-15 的计划取代** —— 现在的四段安排见本节上面的「第 0 小节」。

### 0.6 🚀 吞吐扫描：**已到顶，别再花时间**（2026-09-11 深夜实测）

用户 2026-09-11 明确："提速不是核心的，核心是提升模型智能程度，目前速度够了"。
下面是**结论**，不是待办 —— 想再动吞吐前先读完这一节。

**一、主假设被推翻：微批越小越快，不是越大越快**

原假设是"瓶颈在 kernel 粒度，所以微批越大越快"。`scripts/throughput_sweep.sh` 实测**单调相反**：

| 臂 | 微批 | 步时 | 折算 8192 tok/步 | 每微批 tok/s |
|---|---|---|---|---|
| **A bs4/ga8（现状）** | 4 | **3.86 s** | **2124 tok/s** | **8480** |
| B bs8/ga4 | 8 | 4.51 s | 1815 tok/s | 7500 |
| C bs16/ga2 | 16 | 4.71 s | 1738 tok/s | 6940 |
| E bs8/ga8（2× token/步）| 8 | 8.39 s | — (1952 tok/s) | — |

`scripts/throughput_focus.sh` 再往下探（ABAB 交替，吸收机上游负载漂移）：
**bs=2/ga=16 = 3.54 s/步，比 bs=4 快 9.7%**（基线自身漂移仅 **0.7%**，所以这个 9.7% 是真的）。
⇒ 微批趋势在 bs<4 **仍在延续**，但幅度只有 ~10%，`bs=1` 未测。这一项**没采纳**（收益小）

**二、`gpu_busy_percent = 99%` —— 最容易被读错的一个数**

GPU **不是空闲**，是被大量低效 kernel 占满。按 6ND 估算：8192 tok/步 × 3 × 2 × 75.15M
÷ 3.41 s ≈ **1.08 TFLOPS**，而 gfx1030 的 fp32 峰值约 20 TFLOPS ⇒ **约 5%**。
"99% 占用 + 5% 峰值算力" 恰恰说明**没有任何空闲可填**，所以
「瓶颈是 kernel 启动间隙」的假设与数据不符（若真是启动间隙，微批*变大*该更快，实测更慢）。
**推论：吞吐要提升只能靠"改成更大/更高效的运算"（=改模型或换精度），微调旋钮无用。**

**三、这些旋钮**早就测过了**，不要再测（出处 §2.3）**

| 旋钮 | 已有结论 | 状态 |
|---|---|---|
| `dtype=float16` | 端到端 **+0.5%（更慢）** | ❌ 已否决 |
| `compile=false` | 慢 **5.4%** | ❌ 保留 compile |
| `gradient_checkpointing` | **被 `gpt.py:167` 短路，从未生效** | ❌ **空测试** |
| `ns_steps=5` | −11% 但正交化残差 0.66，灾难 | ❌ 已否决 |
| INT8 / FP8 | 本平台无 gfx1030 内核 | ❌ 不可用 |
| bs=2/ga16 | 快 9.7%（见上） | ⏸ 收益小，未采纳 |

- `gradient_checkpointing` 之所以是空测试：`gpt.py:167` 有
  `if use_ckpt and use_moe and use_aux_free_balance: use_ckpt = False`
  —— 基座正是 MoE + aux-free，所以 `configs/base_v2.yaml` 里那个 `true` **从来没生效过**。
  ⇒ 想测它必须先改那行短路，否则测出来必然是"无差异"（**测量的函数没有区分能力**）。
- ⚠ **教训（我自己犯的）**：曾准备连开 gc / fp16 / compile=false 三个臂，
  查文档后发现**三个都已测过**，其中 gc 还是空测试。**开跑前先查 §2.3 和本表。**

**四、测量纪律：扫描里踩到的三个坑**（已修进 `scripts/throughput_sweep.sh`）
1. **显存读数必须在进程退出后等稳定**：不 wait 就采样会打印出
   `2231→4059→4721→6121` 的"阶梯"，看起来像显存泄漏，其实只是**读数时刻的假象**
   （全部进程停掉后回到 2292 MiB，与起始 2294 一致 ⇒ 无泄漏）。现已加 `settle()`。
2. **必须记录 `gpu_busy_percent`**：机上有用户自己的游戏在占 GPU（实测 27~53%），
   绝对值完全不可信，只有**同批次内的比值**可用。
3. **臂必须交替，不能串行**：负载会漂移，串行时漂移与臂效应混在一起无法分离。
   ABAB 设计下基线自身漂移被实测为 0.7%，才敢下 9.7% 的结论。

### 恢复上下文后先做这三件事
```bash
cd /home/vesita/coding/my/nanoSeek
tail -40 analysis/B_stage_review.md                    # ① 最新一站的结论与"待办/债"（B 段已跑完，先看这个）
tail -c 1500 out/base_v3_dlg_train.log                 # ①′ 上一站训练是怎么结束的（日志在 out_dir **之外**！铁律 3）
systemctl --user is-active nanoseek-v3-dlg.service     # ①″ 应输出 inactive = **正常结束**，不是"训练死了"
cat PROJECT_STATE.md TECH_DEBT.md                      # ② 恢复记忆
.venv/bin/python -m pytest -q -m 'not slow'            # ③ 全绿（跳过 slow 时 <1 秒）
```
⚠ **下一站训练尚未启动**（A 段已跳过、B 段已跑完）：要起新站必须
①新配置 + ②新 `out_dir`（铁律 12）+ ③新单元名 + ④同步改 `scripts/watch.sh:16-17` 与
`nanoseek-watch.timer`/`nanoseek-watch.service` 里的路径。**别直接复用旧单元名。**

### 0.1 🕐 看护节律（当前任务：低上下文定期巡检）

#### ★ 驻守授权（人类不在场时的行为准则）
> **为什么驻守**：无人监控。所以出问题时的目标是 —— **尽可能修复，然后维持监控继续运行**。
> 这期间有**自由裁定权**（自己决定怎么修、要不要重启、要不要调参），不需要等人确认。
>
> **唯一必须记住的一条**：**训练一旦重启，监控节律必须重新从 5 分钟（300s）开始**。
> 理由：出问题**一般都是早期发生的**（编译尖峰、显存爬升、配置错误、数据路径），
> 重启后的前几分钟是信息密度最高的窗口，用长间隔会漏掉。

#### 巡检动作
**唯一动作**：每轮唤醒只做「查状态 → 挂下一个后台定时器」，**不读源码**（避免上下文膨胀）。

```bash
# 每轮唤醒的固定动作就这一条（不读源码、不跑测试）
bash scripts/watch.sh
```

**为什么巡检里带归档清理**：`train.py` 逢 1000 步写一个 `ckpt_step_<N>.pt`（≈0.6GB），
70000 步 ⇒ 70 个 ⇒ **42GB**。保留策略**已经内置在训练内部**
（`training/checkpoints.py` 的 `steps_to_keep`：每 5000 步留一个 + 最新 2 个），
在**每次归档落盘时**就地执行（`train.py` 存档点），2026-09-11 23:23 起跑的进程已带这份代码。
`watch.sh` 里再跑一遍 `python -m training.checkpoints` 是**幂等巡检 + 把结果打进巡检日志**，
不是唯一防线（铁律 11）。脚本幂等、只删归档、绝不碰 `best.pt`/`last.pt`。

**★ 巡检动作必须是一个脚本，不能是每次现敲的长命令。**
2026-09-11 事故：我把 prune 写进了**文档里**的巡检命令，但实际挂出去的定时器只有
`ls | wc -l`（只数个数）→ **看起来在清理，实际没清**。
现在巡检 = `bash scripts/watch.sh`（一条不可分割、可审计的脚本），定时器只负责
`sleep N && bash scripts/watch.sh`，没机会漏步骤。

**★ 守夜人 `scripts/ckpt_janitor.sh` —— 已于 2026-09-11 晚退役，不要再启动它。**
退役理由两条，都是实测出来的：

1. 保留策略**已经搬进训练内部**（铁律 11），每次归档落盘就地稀疏化。
   外部看守不再是防线，而是一个**会被人误当成防线的重复进程**。
2. 它自己的启动说明写的是 `setsid nohup` —— 那恰是**铁律 0 明令禁止**的形态
   （会话重启 SIGKILL 整条 cgroup；2026-09-11 实测因此丢过一次训练）。
   文档里留一句"照抄就会踩坑"的命令，**比没有这句更危险**。

文件保留只为历史追溯。将来若真需要外部看守，用 `systemd-run --user --unit=...` 起。

**节律（以 300s = 5 分钟起步，逐次翻倍，5 小时 = 18000s 封顶）**：
```
300 → 600 → 1200 → 2400 → 4800 → 9600 → 18000 → 18000 → …
```
👉 **当前档位：18000s（5 小时封顶，之后固定不变）**
⚠ **每次 harness loop 结束前，必须确认后台定时器仍在 running**（`job_list` 查一眼），否则会永久睡死。
⚠ **训练重启 ⇒ 节律重置回 300s**，从头再爬一遍阶梯（见上「驻守授权」）。

**基准健康值**（超出这个范围才需要深挖，否则一律只看不读）：
| 指标 | 正常 | 含义 |
|---|---|---|
| s/it | 3.35 ~ 3.40 | >4.5 或漂移 = 有问题 |
| `显存` | 1.7G | peak 1.70 / rsv 1.78，台阶式上涨才可疑 |
| `oom` / `retry` | 0 / 0 | 非 0 且持续上涨 = 尖峰 |
| `dynamo unique_graphs` | 22 且稳定 | 上涨 = 重编译 |
| `异常关键字` | 0 | 任何非 0 立刻查 traceback |
| **磁盘可用** | **> 40G** | 每个 ckpt 0.6GB、70 个 = 42GB；低于 40G 立刻加密集巡检 |

**已确认的正常里程碑**（不用重复核查）：
- **step 1000 首次 eval + 存档 ✅**：`train/loss 3.3983` · `val/loss 3.3923`（**val 略低于 train，无过拟合，数据路径正确**）
  · `lr 3.0e-4`（WSD 稳定段，与调度一致）· `time 3407s/1000 步 = 3.41 s/步`（含 eval 开销）
  · 产出 `best.pt` / `last.pt` / `ckpt_step_1000.pt`，各 ≈ 0.59GB

### 0.2 ⚠️ 指标口径：`results.csv` 的 train/loss 列**不能**用来判断过拟合

2026-09-11 巡检看到 step 8000 出现 `train 1.8050 / val 2.1769`，一度以为是过拟合。
**逐项核对后判定是假警报**，但暴露了一个必须记住的口径问题：

| 指标 | 口径 | 可信度 |
|---|---|---|
| `val/loss` | `estimate_loss` 200 batch × token 加权，**同一套代码、同一 eval 模式** | ✅ **判断泛化的唯一依据** |
| `train/loss`（results.csv） | **train split 的 200 batch eval**（`eval_train_split: true`，EMA 分支是死代码） | ⚠️ **跳动 ±0.2~0.35，不可用于趋势** |
| `train_loss_window.csv` 的 `window_mean` | 每 20 步的训练侧窗口均值；**按 1000 步聚合后 SE ≈ 0.014** | ✅ 可用（要做聚合，别看点值） |

**判定依据（可复算）**：把 `train_loss_window.csv` 按 1000 步聚合：

```
区间         窗口均值μ    8000-9000: μ=2.1496  (min 1.947 max 2.336)
7000-8000   2.1623       ← 训练侧**已平台**
6000-7000   2.1722
```

**★ 结论已用 step 9000 验证（2026-09-11 08:04）**：val 从 2.1769 掉回 **1.9696**，
train 列从 1.8050 回到 2.0436 —— 两者**互相靠近到 ~2.0**，证明 step 8000 那对数字
**两边都是噪声抽样**，不是过拟合。但这也**修正了我上面过强的说法**：

```
★ 实测 eval 噪声：σ ≈ 0.10（不是我以为的 0.03）
   由 val 相邻 1000 步差分的 sd(0.14)/√2 估出
   而真实进步速率只有 ~0.05/1000 步
   ⇒ **单点 eval 的噪声比我们要测的信号还大**
```

**推论（对未来 NDB A/B 至关重要，比原来的说法更强）**：
- NDB 的 Δ 只有 0.02~0.07 量级，**任何"跑 N 步比单点 val"的做法都测不出来**，
  噪声 σ≈0.10 会把信号整个淹掉。
- 必须用**配对评估**：在**同一个 batch** 上同时算 `CE_off` 与 `CE_ndb` 再取差
  （batch 级噪声直接对消）。`scripts/ndb_online_ab.py` 已经是这个设计——
  现在有了量化的理由，别改成"分别跑两个 arm 比 val"。
- 或把 `eval_iters` 提上去（噪声 ∝ 1/√n，要 σ 从 0.10 降到 0.01 需要 ×100 的 batch 数）。
- 判断训练趋势用「1000 步聚合的 window_mean」（SE≈0.014），**不要**看单点。

### 0.3 📉 收敛趋势记录（2026-09-11 15:44，step 18000 / 70000）

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`（只搬了位置，内容字节未改）。本页只保留**当前状态与规程**。

## 1. 项目目标

1. 把 nanoSeek-100M（char-level 中文）训好；
2. 在此之上做一个 **no_grad 外部神经数据库（NDB）**，在不增加模型大小的前提下扩大有效容量；
3. 用户明确要求：**NDB 必须同时支持「读」和「写」**。

---

## 2. 关键事实速查（全部为实测，不要凭直觉推翻）

### 2.1 模型规模（曾被算错）
- **真参数量 = 81.58M**（关 MTP 后 **75.15M**）
- `state_dict()` 求和会**虚高到 90.19M** —— `wte` / `lm_head` / `mtp_head` 是**同一个张量**被列了 3 遍
- 分解：FFN/MoE **62.88M (77.1%)** · Attention MLA 8.06M (9.9%) · MTP 6.44M (7.9%) · embed+head 4.19M (5.1%)

### 2.2 步时归因（baseline 4.965 s/step）
| 阶段 | s/step | 占比 |
|---|---|---|
| **bwd** | 3.295 | **66.4%** |
| **muon** | 1.077 | **21.7%** |
| fwd | 0.559 | 11.3% |
| **data** | 0.024 | **0.5%** |

- `bwd/fwd ≈ 5.2×`（正常 2×）。**关 MTP / 关 mHC / 关 MoE 都改不动它** → 是这套栈的固有属性，**已停止追查**。
- **取数只占 0.5%** → "主机喂不饱 GPU" 的假设**已证伪**。
- `--no_moe` 单关会炸到 12.95 s/step（与 mHC 的交互 bug），**不是编译污染，已复核**；但那是我们永不使用的配置。

### 2.3 速度杠杆（哪些行、哪些不行）
| 项 | 结果 | 状态 |
|---|---|---|
| `batch_size 8→4, grad_accum 4→8` | **−9.7%**，峰值显存 2.91→1.70G | ✅ 采用 |
| `use_mtp=false` | **−10.3%** 步时，省 7.9% 参数 | ✅ 采用 |
| `use_mhc=false` | **−5.4%**（已证明 hc_mult=2 逐位无效，见 §4）| ✅ 采用 |
| 混相 NS 7 步（4 激进 + 3 经典）| **−7.7%**，且正交化残差**更好** | ✅ 已实现 |
| `ns_steps=5`（纯经典砍半）| −11% 但残差 0.66（灾难）| ❌ **否决** |
| `dtype=float16` | 端到端 **+0.5%（更慢）** | ❌ **否决** |
| INT8 / FP8 | 本平台**不支持**（hipBLASLt 无 gfx1030 内核 / addmm 未实现）| ❌ 不可用 |
| `gradient_checkpointing` | 被 `gpt.py:167` 短路（aux-free MoE 不兼容），**从未生效** | ❌ 空测试 |
| `compile=false` | 慢 5.4% | ❌ 保留 compile |

**合并实测：4.965 → 3.371 s/step（−32.1%），峰值显存 1.70G（−42%）。**

### 2.4 三个"假指标"（都在代码里修好了）
- `mfu`（`gpt.py:396`）：ROCm 读不到 `clock_rate` → 回退 A100 的 312 TFLOPS
- tqdm 的 `吞吐`（`train.py`）：用**单步** `dt`，实测抖到 2×（看 `s/it`，别信它）
- `state_dict()` 求和当参数量（见 §2.1）

### 2.5 学习曲线（纯 CE，固定 256 val 窗口，配对设计）
```
step    纯CE      每千步增量
12000   1.3060
13000   1.2647    −0.0413
14000   1.2544    −0.0103
15000   1.2483    −0.0061
16000   1.2445    −0.0038
17000   1.2406    −0.0039
18000   1.2392    −0.0014
19000   1.2369    −0.0023
```
- 斜率 **−0.00434 ± 0.00061 /千步（t = −7.08）** → **统计上仍在下降，不是平台**
- 但速率 4000 步内掉 ~5 倍；按任何外推，**剩余 50000 步只值 0.01 nats 量级**
- **"平台化"的判断我错了两次**，根因是用了混合口径（见下）

### 2.6 混合 loss 必须先拆（被这条误导过 3 次）
`loss = CE + Σ MoE aux + mtp_weight × MTP`
```
step 19000：纯 CE 1.2369 (74.7%) · MoE aux 0.0000 (0.0%) · MTP×0.3 0.4191 (25.3%) · 合计 1.6560
```
- **MoE aux 恒为 0.0000** → "去掉负载均衡辅助损失"是无效提议
- `ndb.csv` 的 `base_off` = CE + 0.3×MTP，噪声 σ=0.093，**测不出纯 CE 的趋势**（t=−1.59）

---

## 3. 数据集：v2 已重建并验收通过

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`（只搬了位置，内容字节未改）。本页只保留**当前状态与规程**。

### 3.1 结论

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`（只搬了位置，内容字节未改）。本页只保留**当前状态与规程**。

### 3.2 旧 bug（已修）

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`（只搬了位置，内容字节未改）。本页只保留**当前状态与规程**。

### 3.3 附带发现

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`（只搬了位置，内容字节未改）。本页只保留**当前状态与规程**。

### 3.4 产物

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`（只搬了位置，内容字节未改）。本页只保留**当前状态与规程**。

### 3.5 语料混合（建议，未动手）

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`（只搬了位置，内容字节未改）。本页只保留**当前状态与规程**。

## 4. 架构审计：mHC 是逐位无效的（已证明，已关闭）

`model/gpt.py` 的 `use_aux_free_balance=True` 使 `use_mhc` 的多流机制完全空转：

```
每个 block 输出处 max|stream0 − stream1| = 0.000e+00   （12/12 层，精确零）
raw_B ≡ 0（12 层全是精确 0.0）    raw_A = [0.3487, 0.3487]（分量恒等）
```

**自锁链**：两流初始相同 → B（双重随机混合）的梯度**恰好为 0** → B 永远均匀 →
输出仍两流相同 → 回到起点。而且 `raw_A` 就算动了也**被下游 LayerNorm 吃掉**（LN 对正缩放不变）。

**净效果**：`x ← x + c·F(ln(x))`，c≈0.634 —— 只是一个标量。**关掉它省 5.4% 步时 + 2× 残差内存，功能不变。**

其余审计结论：
- **MoE 在干活**：专家权重两两余弦 ≈0.000，输出相对差 0.77~1.22（相同=0，随机=1.41）→ **保留**
- **SwiGLU clamp**：0.034% 激活 >10，最大 49.6 → **开 `swiglu_clamp=10`**
- 无死层（每层动量 1.6e-4~7.3e-4）
- `router_bias` 前两层为 0，第三层起长到 `[2.50, 3.56, 1.10, 1.06]` —— 路由器本能想偏斜，被均衡器硬掰回

---

## 5. 新基座配置：**已启动，暂停在 step 22000 / 70000**，2026-09-11 转入全量语料

**文件：`configs/base_v2.yaml`**（放在 `out_dir` **之外** —— 见 §7 的坑）

| 键 | 旧 | 新 | 依据 |
|---|---|---|---|
| `data_prefix` | — | `v2` | 验收通过的新数据集 |
| `out_dir` | `out/ndb_run` | `out/base_v2` | — |
| `init_from` | resume | **resume** | 暂停/续训的安全默认：写 `scratch` 的话，误用 §0.4 那条命令会触发 `_backup_old_run`（`train.py:916` 的 `init_from != 'resume'`），把所有存档**静默挪进 `out/base_v2/old/`**（且 `old/` 已存在时先 `rmtree`）→ `last.pt` 不在原位，续训直接失败，再犯一次还会删掉上一代存档。开新 run 必须显式 `--init_from=scratch --out_dir=out/<新目录>` |
| `batch_size` / `gradient_accumulation_steps` | 8 / 4 | **4 / 8** | −9.7% 步时，显存 −30% |
| `use_mhc` | true | **false** | 已证明逐位无效 |
| `use_mtp` | true | **false** | −10.3% 步时 + 省 7.9% 参数 |
| `swiglu_clamp` | 0.0 | **10.0** | 实测有异常值（最大 49.6）|
| `muon_ns_steps` / `muon_ns_aggressive` | 10 / — | **7 / 4** | 残差 8.79e-06 vs 4.53e-05，配对 t=−10.5 |
| `lr_decay_iters` | 30000 | **70000** | 修掉"退火在 30000 结束、之后 40000 步平在 1e-4" |
| `use_loss_masking` | true | **false**（2026-09-11 改）| 全量语料预训练。true 时行级 masking 让**只有 6.14% 的语料产生梯度**（`c4_zh` 1.89 亿字符**零**终止符 → 恒不可见），有效语料仅 57.6M token，且 `multi_turn_dialogue` 独占 40.2% 的终止符 → 训出的是窄域复读机。改 false：每步有效 token **3216→8192（2.55×）**、覆盖 **57.6M→937.8M（16.3×）**。⚠ val 口径随之改变，**历史 val 不可比** |
| `ndb_store` / `ndb_heldout` | store5mA | **`""`** | 这一轮先做纯基座 |

- **冒烟测试已通过**：参数量 75.15M、step0 loss 9.0689（≈ln8192=9.011）、**3.38 s/it**、显存 1.7G
- **数据开关已验证**：`--data_prefix=v2` → 读 `train_char_v2.bin`；`--data_prefix=''` → 读旧文件
- ETA：70000 × 3.371 s = **65.5 小时 = 2.73 天**；0.61 epoch

### ★ 2026-09-11 路线：**全量语料预训练 + 末尾对话退火**（用户已确认）

动机：step 22000 的审查发现模型是**窄域复读机**（会背语料里的心理咨询原句，但无世界知识），
根因是 88% 的语料从不产生梯度（见 §0.5）。加速已到顶（§0.6），所以转向**数据**。

```
阶段一  step 22000 → 65000   use_loss_masking=false（configs/base_v2.yaml 里的值）
        全量 937.8M 语料，每步 8192 有效 token（是原来的 2.55×）
        45.5 h。LR 走 WSD：56000 起开始线性退火 3e-4 → 1e-4
阶段二  step 65000 → 70000   use_loss_masking=true（**命令行显式覆盖**）
        切回对话口径做 5000 步退火，把对话能力找回来，同时避免再次人设坍塌
        （现在是 22000 步纯对话 → 人设独占；5000 步不会重演）
```

⚠️ **命令已全部改用 `systemd-run --user`**（铁律 0：`setsid nohup` 会被会话重启
SIGKILL，2026-09-11 实测踩过）。下面这条是**当前真正在跑的阶段一命令**。

```bash
cd /home/vesita/coding/my/nanoSeek

# 阶段一：启动 / 续训（config 已含 use_loss_masking=false）
# ★ 单元名 = nanoseek-base-v2；日志仍在 out_dir 之外（铁律 3）
systemd-run --user --unit=nanoseek-base-v2 --collect \
  --property=WorkingDirectory=/home/vesita/coding/my/nanoSeek \
  --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=HSA_ENABLE_SDMA=0 \
  /bin/bash -c '.venv/bin/python -u training/train.py configs/base_v2.yaml > out/base_v2_train.log 2>&1'

# 阶段一：在 65000 优雅暂停（等 ckpt 落盘再发信号；train.py 没有信号处理器）
# 同样必须用 systemd 单元，否则会话重启会连同看守一起被杀
systemd-run --user --unit=nanoseek-pause-65000 --collect \
  --property=WorkingDirectory=/home/vesita/coding/my/nanoSeek \
  /bin/bash -c 'bash scripts/pause_after_eval.sh 65000 > out/pause_65000.log 2>&1'

# 阶段二：切回对话退火，跑到 70000
# 前提：阶段一的单元已 inactive（--collect 会自动回收单元名），
#       且 out/base_v2/last.pt 是 65000 步那一份
systemd-run --user --unit=nanoseek-base-v2-p2 --collect \
  --property=WorkingDirectory=/home/vesita/coding/my/nanoSeek \
  --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=HSA_ENABLE_SDMA=0 \
  /bin/bash -c '.venv/bin/python -u training/train.py configs/base_v2.yaml --use_loss_masking=true > out/base_v2_train_p2.log 2>&1'
```

**阶段二启动前后的自检**（顺序不能反）：
`systemctl --user is-active nanoseek-base-v2.service` 必须是 `inactive`（否则两个训练
抢同一张卡、抢同一个 `out_dir`）；`ls -la out/base_v2/last.pt` 的 mtime 应贴近 65000 步。

✅ **阶段二的开关已于 2026-09-11 提前冒烟验证**（[实测]，CPU 2 步，`exit=0`）：
```
.venv/bin/python -u training/train.py configs/base_v2.yaml \
  --out_dir=out/_smoke_p2 --init_from=scratch --device=cpu --compile=false \
  --batch_size=2 --gradient_accumulation_steps=1 \
  --max_iters=2 --eval_interval=1 --eval_iters=1 --eval_train_split=false \
  --use_loss_masking=true
→ 日志第 13 行：Overriding: use_loss_masking = True
```
这一步**必须**做，因为 `use_loss_masking` 若定义在 `load_config` 之后，
命令行会"看起来生效"（打印 Overriding）而实际被静默改回默认值
（铁律 8 的死旋钮坑；`keep_step_ckpts` 就是这么坏的）。
**注意这只是"开关生效 + 代码路径不崩"的验证，不是质量验证** ——
阶段二的真实效果要到 65000 步之后才能测。

⚠ **val 口径断裂**：阶段一与阶段二、以及它们与 step≤22000 的历史日志，
val 数值**都不可直接比较**（采样分布和 loss 分母都变了）。
阶段一开始后第一件事是**重新立基线**，之后只做**同阶段内**的配对比较。

### ✅ 已决定：LR 调度用 **WSD**（2026-09-10）
`training/train.py` 两种都已实现（`schedule = 'cosine' | 'wsd'`，`stable_frac: 0.8`）。
```
      step    cosine(decay=70000)   WSD(stable_frac=0.8)
     10000          2.903e-04             3.000e-04
     50000          1.378e-04             3.000e-04
     60000          1.099e-04             2.429e-04
     69999          1.000e-04             1.000e-04
```
**为什么选 WSD**（实测见 `scripts/run_sched_compare.sh` 与 `out/_sched_driver.log`）：
1. 本项目有明确的中断史（长跑在 19781 被停、pilot 被 kill 两次）。WSD 下稳定段随便停、
   最后再退火；cosine 下每次中断都停在退火中途。
2. **中断续训的 LR 轨迹已实测逐点相同**（`wsd_whole` vs `wsd_seg1+seg2`，
   15 个公共步点最大 LR 差 **0.00e+00**）。这条是选 WSD 的核心理由，已验证。
3. MiniCPM4 用 7T 稳定 + 1.3T 退火，DeepSeek-V3 亦然。

⚠ **未测**：两者的**训练质量**差异。150 步的对照测不出这个（预算太小），
本决策只基于"可中断性"，不基于"哪个 loss 更低"。
**当前 `configs/base_v2.yaml` 里写的是 `wsd`。**

### ★ 启动命令（注意日志路径）

启动命令见 §0.4 与上面的两阶段块（**统一用 `systemd-run --user`**）。
这里只强调日志路径：

**日志必须放在 `out_dir` 之外**（这里是 `out/base_v2_train.log`，不是 `out/base_v2/train.log`）。
原因见 §8 铁律 3：shell 在进程启动前就创建了 `out/base_v2/train.log`，而
`_backup_old_run` 会把 `out_dir` 里的**所有**文件挪进 `old/` —— 进程的 fd 跟着被挪走，
`out/base_v2/train.log` 永远是空的（人会被这个假象骗很久）。

**存活检查**（不要再记 PID —— PID 每次续训都会变）：
```bash
systemctl --user is-active nanoseek-base-v2.service
systemctl --user status  nanoseek-base-v2.service --no-pager | head -14
bash scripts/watch.sh
```

### 深度：**保持 12 层，不要加深**
`data/paper/2601.20994v1.pdf`（*The Depth Delusion*，W=512 的 U 型曲线）：
```
深度   参数    loss     阶段
 2     58M    3.945    D ≪ Dcrit
 8     77M    3.543    D < Dcrit
16    102M    3.435    D ≈ Dcrit     ← 最优
24    127M    3.468    D > Dcrit     （多 25% 参数，loss 反而高 0.033）
```
`D_crit ≈ 2.43·ln(W)` → **W=512 时 D_crit ≈ 15.2。我们的 12 层已经很接近最优。**
大规模同向：1B（24L/1792 vs 80L/1024）deep 差 0.16；7B（32L/4096 vs 64L/2816）deep 差 0.12。
⚠ 该文只测 dense（全文未提 MoE），我们是 MoE → **方向可信、幅度存疑**。
⚠ 与 MobileLLM/SmolLM2（30 层 × 576 宽）冲突，**冲突未解释**。

---

## 6. NDB：当前方向 = RETRO-lite（长程 chunk KV + cross-attention）

> **三站沿革**（详见 `dev-notes/79-长程神经元记忆-RETRO方向决策与计划.md`）：
> ① 后缀键+梯度值（本文 §6.1–§6.4，`dev-notes/78`）Δ−0.0046 → **已放弃**；
> ② token 级后缀 n-gram（TDB，`model/ngram_ndb.py`）同预算 Δ−0.0077 → ③ 在**同预算**上 4.5× 于它；
> ③ **RETRO-lite（`model/memory_cross_attn.py`）Δ=−0.0348（256 窗）＝ 同预算 4.5× TDB** ← **当前载体**。
> ⚠ `training/train.py:729` 挂的 `MemoryCrossAttention`（`ndb_store` + `ndb_layer=-6` + `ndb_chunk=64`
> + `ndb_exclude_radius=512`）**就是 ③ 这个架构**，不是旧遗留。⚠ 但**最近两次训练（v2 61000 步 /
> B 段 14000 步）都没接 NDB**（`configs/base_v2.yaml:80` 是 `ndb_store: ""`，`base_v3_dlg.yaml` 无 `ndb_*` 键）。
>
> ### ★ 更正（2026-09-15）：没有「用户否决 TDB / v7」这回事
> 本页 §0 与本节上面曾写「② TDB … **用户已否决**（「与 harness 记忆工具无本质区别」）」。
> **该归因错误。** 全仓库唯一出处是 `1e0c121`（2026-09-14 21:50），它把**两句不同时间、不同对象**
> 的话粘成了一句（`git log -S'用户已否决'` 只命中这一个提交）：
> - 「与 harness 记忆工具无本质区别」出自 **`dev-notes/78 §14.1`（2026-09-09）**，说的是**当时**的
>   v4/v5「后缀键+梯度值」范式（原文："§13 的「写 token 计数」据此作废"）⇒ 结论是**改神经元级**；
>   它**没有**评价后来才出现的 RETRO-lite，也**没有**说 TDB 本身无意义。
> - 用户 **2026-09-10** 的原话在 **`dev-notes/79 §1`**：「**如果不能超越 tdb，那么 ndb 其实就没什么
>   意义。直接朝着可能超越 tdb 的方向前进，不然就放弃这个方向。**」⇒ TDB 是**标尺**，不是被否决项。
> - 且 `dev-notes/79 §7` 自己写着：「**不是替代，是互补**：TDB 管局部续写，长程 NDB 管跨窗依赖。」
>   ⇒「§6.5 的 token 级 v7 标注作废」这句**与它引用的源文档直接矛盾**。
>
> **更正后什么是真的**：② 与 ③ **并列**（互补）。③ 是当前载体的真正理由是**用户 2026-09-09 定的
> 「神经元级」方向**（value 是残差流空间的向量，不是 token 分布）—— v7 存 token 计数，落在那条线外。
> **不是**"用户否决了它"。★ 而 ② 的缺口是**数字缺失**：v7 的**在线 A/B 从没跑过**（见 §12.5、
> §10 第 3 条），它的 −0.0723 是**离线且跨预算**（268M 槽 ≈2.1GB vs ③ 的 10M token 预算）
> ⇒ 依 §5.11 **不能**与 −0.0348 并排读。
> ★★ **下一个人：不要说「v7 被否决」，要说「v7 的在线数字从未跑过」。**
> ★ 2026-09-15 实际进展 + **同日更正**：`model/memory_cross_attn.py` 补了 `write_online`
> （环形库），但它是**规则式**的（人给的惊讶分位 `quantile`，即"灌注"的在线版）。
> 同提交里那个 `write_gate` **无梯度路径 = 死参数**，**已删除**。
> ⇒ **结论改了：两条线没有"合流"。** 「模型自己决定读写」的实现在 **`model/ngram_ndb.py`（v7）**，
> 那里 `w_t` 进 `read_gate` 输入、`∂L/∂W_w ≠ 0`（实测 grad=0.0134，`tests/test_ngram_ndb.py:292`）。
> 要把"模型自己决定写"搬到神经元级载体上，**移植的是那套接线，不是这个空壳**。

### 6.0 ★★ 被丢掉的一站：**NDB 共训 `out/ndb_run/`** —— 全项目最硬的 NDB 正面结果
**来源：`out/ndb_run/STATE.md`（471 行，2026-09-10 维护；2026-09-14 21:31 被加了「已被本页取代」的头）。**
★ **2026-09-15 已把原件逐字归档进仓库：`analysis/NDB_cotrain_STATE_2026-09-10.md`**
（`out/` 在 `.gitignore` 里 ⇒ **原件从来没进过 git，这就是它会被丢掉的原因**）。
★★ **完整复核（时间线、全部找回的数字、三处纠偏的取证、现状盘点、固化纪律）见
`dev-notes/80-NDB共训站丢失事故与方向纠偏.md`** —— 本节只是摘要。
★★ **谈 NDB 方向之前先读它。** 本页此前（含 §0 那三行）**完全没有这一站** —— 最强的证据被漏了。

| 项 | 值 |
|---|---|
| 基座 | `out/base_probe/best.pt`（step **12000**，81.58M）← **制品已删（目录空）** |
| 库 | `out/mem_store/store5mA.pt`（78,124 chunks, `--mode mean`）← **已删（目录空）** |
| 挂载 | 层 −6 · top_k 4 · chunk 64 · exclude_radius 512 · dropout 0.1 · 接口 lr 3e-4（独立 AdamW） |
| 步数 | 12000 → **19781（用户手动停）**；`out/ndb_run/` 现**只剩日志/CSV，ckpt 全无** |

**共训 Δ（`out/ndb_run/ndb.csv`，四个条件同一批 val batch = 配对）**

| step | base_off | mem | **Δ** | Δ_rand | Δ_held | Δ_held/Δ |
|---|---|---|---|---|---|---|
| 13000 | 2.0043 | 1.9630 | −0.0413 | +0.0316 | −0.0391 | 0.95 |
| 16000 | 1.6986 | 1.6373 | −0.0613 | +0.1124 | −0.0520 | 0.85 |
| 19000 | 1.8060 | 1.7322 | **−0.0738** | +0.1793 | −0.0608 | 0.82 |

① **Δ 单调变负、还没饱和**（每千步增量 −0.0086→−0.0027）；
② **Δ_held/Δ = 0.82~0.95**（判据 >0.8）⇒ 学到的是**可迁移读策略**，不是把 5M 库背下来；
③ **是冻结基座基线 −0.0348 的 2.1×** —— 本页与 §0 一直只引 −0.0348，**低估了自己一倍**；
④ **Δ_rand 强正且增长（+0.03→+0.18）** ⇒ 随机检索**主动有害**，即模型确实在读内容；
   代价是它**选择信任记忆，而不是「不确定就不注入」**。

**★ 同一份 STATE.md 记录的头号问题（比 Δ 更重要）——检索鲁棒性悬崖**

| top-k 被随机替换 | 0% | 10% | 25% | 50% | 100% |
|---|---|---|---|---|---|
| Δ | −0.0611 | −0.0212 | **+0.0222** | +0.0701 | +0.1509 |
| 占 Δ(0) | 1.00 | **0.35** | **−0.36** | −1.15 | −2.47 |

⇒ **10% 检索出错就废掉 65% 收益；25% 出错时记忆净有害。**
免训练探针（`local/eval_ndb_probe.py`，改推理期读取结构、不训练）定位了根因与修法：
检索用 sim 选出条目后**把 sim 扔了**，注意力只用学出来的 wq/wk 重打分
（P3：k=4 时四个槽的污染代价几乎相同 +0.118~+0.121 ⇒ **模型完全不认排名**；
P5：`corr(sim_top, 增益) = +0.003`）。
**修法已在代码里、且在探针里验过**：`ndb_att_sim`（λ，把 sim 的 z-score 加进注意力 logit）——
λ=2 时用 **11% 的干净增益换来「检索出错时记忆仍然有益」**（25% 错误：λ=2 是 **+0.24**，λ=0 是 **−0.36**）。
⚠ **这个修法只在探针里验过，从没在训练里跑过。**

**★ 由这一站推翻的三个旧说法**
1. **「库越大越好」**：5M vs 10M 的 Δ 是 −0.0612 vs −0.0570（10M 反而略差）⇒ **容量不是瓶颈**。
   STATE.md 的结论：瓶颈在**读取分辨率**（chunk 均值 + top-4 太粗），**不在库大小**。
   ⇒ **别再把「换更大的库」当解法**（这条直接推翻了本节上面那个"更大的库"思路）。
2. **「MemoryCrossAttention 是只读、没接进 train.py」**：**它接进去了、而且跑过 8000 步。**
   `train.py:203-213` 的 `ndb_store / ndb_layer / ndb_chunk / ndb_top_k / ndb_exclude_radius /
   ndb_dropout / ndb_retr_noise / ndb_att_sim` 全在，`ndb_store` 非空即启用，且全在
   `config_keys`（:313）之前。缺的**只有「写」**（库是离线 `--mode mean` 一次建成的）。
3. **「NDB 的当前水平是 −0.0348」**：那是**冻结基座**的数；**共训是 −0.0738**。

**★ 这一站为什么会消失 —— 交接失败的直接证据（2026-09-15 复核）**
用户把长跑停在 19781，是为了**先跑 2000 步 pilot 回答设计问题、胜者再回灌长跑**：
`out/pilot_sim/` = 已知最好配方 **+ `ndb_att_sim: 2.0`**（正是上文那个鲁棒性修法）；
`out/pilot_sim_noise/` = 再加检索噪声 0.15。
**但两个 pilot 一个都没跑完**：`pilot_sim` 日志停在 step 19069（6 分钟，
`ndb.csv` 与 `results.csv` **都是 0 字节**），`pilot_sim_noise` **一行没跑**（只有 config.yaml）。
⇒ **NDB 这条线是「实验做了一半被丢下」，不是「得出结论后被否定」。**
26 分钟后（2026-09-14 21:31）STATE.md 被加上「已被 `PROJECT_STATE.md` 取代、
本文件不再维护方向结论」的头，再过 19 分钟 `1e0c121` 写下「② 用户已否决 / ③ 当前 = RETRO-lite」，
而**这一站连一行都没进主文档**。
⇒ ★★ **纪律：结论只能来自"跑完并有产物"的实验。「跑了一半 + 若干天后被别的文档宣告作废"，
既不是否定也不是结论 —— 它只是丢了。**

### 6.1 完整实验史在 `dev-notes/78-残差神经数据库P0P2落地与数学优化路径.md`（1065 行）

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`（只搬了位置，内容字节未改）。本页只保留**当前状态与规程**。

### 6.2 ★ 为什么"写残差"注定失败 —— 定理级（§11.4 / §12.2）

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`（只搬了位置，内容字节未改）。本页只保留**当前状态与规程**。

### 6.3 ★ 什么才有效 —— **全量探针实测（2026-09-10，已推翻 §12.3）**

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`（只搬了位置，内容字节未改）。本页只保留**当前状态与规程**。

### 6.4 已确认的负结果（不要重犯）

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`（只搬了位置，内容字节未改）。本页只保留**当前状态与规程**。

### 6.5 用户判定的 NDB 形态：**必须支持读和写**

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`（只搬了位置，内容字节未改）。本页只保留**当前状态与规程**。

### 6.6 ⚠ 一个必须正面处理的矛盾

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`（只搬了位置，内容字节未改）。本页只保留**当前状态与规程**。

## 7. 任务状态

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`（只搬了位置，内容字节未改）。本页只保留**当前状态与规程**。

### ✅ 任务 1（已完成 2026-09-10）：全量 n-gram 容量探针

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`（只搬了位置，内容字节未改）。本页只保留**当前状态与规程**。

### ⏭ 下一步（NDB 方向，按优先级）

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`（只搬了位置，内容字节未改）。本页只保留**当前状态与规程**。

## 8. 运维铁律（违反过，别再犯）

> 编号只增不改 —— 文档里散落着"见 §8 铁律 N"的交叉引用，插队重排会把它们全部指错。
> 所以新规律一律给**新号**（0、11、12…），不给已有编号让位。

0. **★ 长跑必须用 `systemd-run --user` 启动，绝不用 `setsid nohup`。**（2026-09-11 实测踩到）
   harness 每次都把工具调用放进一个 systemd scope
   （`/user.slice/.../dsh-subprocess-<pid>-<hash>.scope`），**会话重启时 SIGKILL 整条 cgroup**；
   `setsid` 只脱离会话/tty，**脱离不了 cgroup**，所以照样被杀。
   现场特征：日志**无 traceback、无 OOM**，最后一行 tqdm 停在某一秒，
   `journalctl --since` 里能看到 `Killed unit cgroup ... with SIGKILL on client request`。
   → 正确姿势见 §0.4；看守进程（`pause_after_eval.sh` 等）**同样要**用 `systemd-run` 起。

1. **绝不用前台 `sleep` / `job_output(wait=true)` 等待**。只用后台 `sleep N` 定时唤醒，
   每次查完状态**立刻挂下一个**。节奏 300→600→1200→2400→4800→9600→18000→18000…
2. **日志管道里不准放过滤器**。`grep`/`sed` 必须等 `\n`，而 tqdm 只写 `\r` → 输出全卡在缓冲区。
   正确姿势：`python -u ... >> log 2>&1`（单文件、双定向、`2>&1`）。tqdm 默认写 **stderr**。
3. **`out_dir` 里的 `config.yaml` 会被归档走**：`train.py` 在 `init_from != 'resume'` 时
   调 `_backup_old_run(out_dir)`，把目录里**所有**文件挪进 `old/`。
   → **config 必须放在 `out_dir` 之外**（已改成 `configs/base_v2.yaml`）。
4. **改 `train.py` 时注意 `config_keys` 快照在 `load_config` 之前**。
   定义在它之后的全局变量会被**静默改回默认值**（`data_prefix`、`gradient_checkpointing` 都栽在这）。
   → 现在有测试自动拦：`tests/test_project_layout.py::test_every_config_key_is_overridable`。
5. **`train.py` 有 OOM 现场落盘**：前 3 次 OOM 会写 `oom_dump_<step>.txt`
   （完整 traceback + allocated/reserved/peak + batch/架构快照）。快照触发条件已改为
   `max(allocated, reserved) > 阈值`。
6. **★ 启动后台作业时，绝对不要在**同一条** shell 命令里再 `sleep`。**
   harness 的超时会 SIGTERM 整条命令，**连带杀掉刚在同一条命令里启动的后台作业**
   （无论它是 `setsid nohup` 还是 `systemd-run` 的启动动作——后者若被拦腰 SIGTERM，
   单元可能根本没起来）。
   （2026-09-10 实际发生：一次 11 分钟的 Δ 扫描在第 60 秒被静默杀掉，
   日志停在半路、内核无任何报错，排查花了十几分钟。）
   正确姿势：**第一条命令只负责启动并立刻返回**，监控**另起**一条后台 `sleep` 命令。
7. **★ 诊断代码不能有能力搞崩训练。** 记忆/显存诊断必须用**实际训练设备**
   （`device_type`）判断，**不能用 `torch.cuda.is_available()`** ——
   在有显卡的机器上跑 `--device=cpu` 时后者仍为 True，会走进 CUDA 分配器统计并
   `KeyError` 崩在 step 0。已加固：`training/diag.py` + `tests/test_diag.py`。
8. **★ 改完 `train.py` 必须跑一次 2 步冒烟**（成本 20 秒，`--device=cpu` 即可）：
   ```bash
   .venv/bin/python -u training/train.py configs/base_v2.yaml \
     --out_dir=out/_smoke --init_from=scratch --device=cpu --compile=false \
     --batch_size=2 --gradient_accumulation_steps=1 \
     --max_iters=2 --eval_interval=1 --eval_iters=1 --eval_train_split=false
   ```
   ⚠ `--init_from=scratch` **必须加**：`configs/base_v2.yaml` 现已是 `init_from: resume`
   （为了让重启命令永远只有一条），不加就会去 `out/_smoke/last.pt` 找续训点然后报错。
   上面第 7 条那个崩溃**单元测试抓不到**，就是冒烟测试抓到的。
   跑完记得 `rm -rf out/_smoke`（会写 ~1.2GB 的 ckpt）。
9. **★ 训练日志必须放在 `out_dir` 之外**（用 `out/base_v2_train.log`，
   不要用 `out/base_v2/train.log`）。
   shell 的重定向 `>> out/base_v2/train.log` 在进程启动**之前**就创建了文件，
   而 `_backup_old_run` 会把 `out_dir` 里的**所有**文件挪进 `old/` ——
   进程的 fd 跟着 inode 被挪走，于是 `out/base_v2/train.log` 永远是**空的**。
   看护的人会以为训练挂了，实际它在正常跑、日志在 `out/base_v2/old/train.log`。
   （与第 3 条"配置不能放 out_dir 内"是同一个根因：**out_dir 会被整体归档**。）
10. **★ 离线预热是 NDB 在线实验的前提。** 小预算在线实验里表几乎是空的
    （300 步 × 1024 token = 30 万次观测 vs 6700 万槽位 → 覆盖率 0.5%），
    必须先用 `NgramNDB.observe_tokens()` 在大量语料上把表填到有覆盖率，
    再在训练中增量写。没有这一步，测出来的 Δ 是噪声。
11. **★ 巡检动作必须是一个原子脚本**（`bash scripts/watch.sh`），不许手敲长命令。
    起因：曾**口头**把 `prune_ckpts.sh` 写进文档的巡检命令，但实际挂出去的定时器里
    只有 `ls | wc -l`（数个数），**没有真的清理** —— 人（和压缩后的自己）照文档以为在清理，
    实际没清。定时器只负责 `sleep N && bash scripts/watch.sh`，没机会漏掉任何一步。
12. **★ 保留/清理策略必须内置在训练里，不能只靠外部进程。**（2026-09-11 改）
    旧设计有两套清理：训练内"只留最新 N 个" + 外部 `prune_ckpts.sh`"每 5000 步留一个"。
    两个后果都是静默的：① 外部那套依赖看守活着，而看守随会话重启一起被杀（见铁律 0）；
    ② 训练内那套会在 step 25000 之后把 `ckpt_step_5000/10000/15000`
    这些**阶段回溯点**当"旧文件"删掉。
    → 现在**唯一实现**在 `training/checkpoints.py`（`steps_to_keep`，有 20 条单测），
    `train.py` 每次存档后自己调用，`scripts/prune_ckpts.sh` 退化成调用同一个实现的薄包装。
    同理 `results.csv` 续训时改为**追加**（`training/run_logs.py`），
    不再每次重启就把指标历史截成 0 字节。
13. **★ warm start（`init_from=<路径>.pt`）必须配独立的 `out_dir` —— 它**不是** `resume`。**
    （2026-09-13 拟三阶段方案时差点踩到：命令块里写了 `--init_from=out/base_v2/last.pt`，
    **漏了 `--out_dir`**。）
    判据在 `train.py:1022`：`if master_process and init_from != 'resume': _backup_old_run(out_dir)`。
    warm start 走的是 `init_from.endswith('.pt')` 分支 ⇒ **`!= 'resume'` 成立** ⇒ 归档照样触发。
    事故场景：照抄命令 + `configs/base_v2.yaml`（`out_dir: out/base_v2`）
    ⇒ 基座 61000 步的**全部归档 ckpt（含 `last.pt` 自己）**被静默挪进 `out/base_v2/old/`，
    新旧两个 run 的产物混在同一目录。
    ⚠ 模型是**先加载、后归档**的，所以训练**不会当场崩** —— 这正是它阴险的地方。
    → 已为三阶段各建独立配方（`configs/base_v3_know.yaml` / `base_v3_dlg.yaml`），
    并用两条断言钉住：`test_project_layout.py::test_v3_stage_config_safety`
    与 `::test_all_config_out_dirs_are_pairwise_distinct`。详见 §0.5.10。
    与第 3、9 条**同一个根因**（`out_dir` 会被整体归档），只是触发条件不同。

14. **★ 「定期要发生」的运维动作必须落在 systemd 用户单元（`*.timer`）里，不许只活在 AI 会话的 `sleep` 链里。**
    （2026-09-14 实测踩到）2026-09-13 23:35 我挂的 `sleep 1800` 巡检定时器于 02:05 到点结束，
    但 harness 的**完成通知迟了 6 小时**才送达（会话被挂起/休眠时通知不推进）⇒
    02:05~08:06 这 6 小时里**一次巡检都没发生**，而我和读文档的人都以为在监控。
    ⚠ 训练本身**没受影响**（`systemd` 单元独立于会话：`NRestarts=0`、
    进度条累计时长 7:37:49 与墙钟 00:28:29→08:06:29 逐秒吻合），
    但"以为在监控"正是铁律 6/11 同一族失败模式。
    → 已建 `nanoseek-watch.timer`（每 30 分钟跑一次 `watch.sh`，输出追加进
    **`out/watch_heartbeat.log`**；单元文件在 `~/.config/systemd/user/`）。
    纪律：**醒来先读 `out/watch_heartbeat.log` 补盲区，再跑 `watch.sh` 做即时确认**；
    `sleep` 链降级为"让我自己醒来"的提示器。验证：
    `systemctl --user list-timers nanoseek-watch.timer`。

---

## 9. 相关 skill（已固化本轮全部教训）

| skill | 内容 |
|---|---|
| `ml-experiment-attribution`（354 行）| 空测试、假指标、同构探针、微基准陷阱、比值归因、数据侧对比三坑、**架构审计（对称初始化是稳定鞍点）** |
| `longrun-train-monitor`（460 行）| 后台定时器铁律、日志清洗、**OOM 现场落盘**、快照触发修正、续训架构键覆盖陷阱、参数量统计 |

---

## 10. 下一步（按优先级）

1. 🔄 **等主线基座训完**（`configs/base_v2.yaml`，WSD，70000 步，ETA 2.73 天）。
   看护按 §8 铁律 1 的后台定时器节奏；进度看 `out/base_v2_train.log` 与
   `out/base_v2/train_loss_window.csv`。
2. **用 v2 基座重跑容量探针**（`scripts/run_delta_sweep.sh`）——
   §6.3 的 Δ=−0.0723 是 **v1 基座评 v2 数据**，必须重测。这是 NDB 方向下一个硬数字。
3. **跑 NDB v7 在线 A/B**（`scripts/ndb_online_ab.py`）—— 回答
   "离线 −0.0723 在线能不能兑现"。脚本已写好并冒烟通过（§12.4）。
   ⚠ 冒烟用的是退化配置（M=4M 槽装 20M token），**要看结论必须用
   `--slots 268435456 --prewarm_m 900 --steps 300`**（约 30-45 分钟）。
4. **软检索**（top-K 计数分布取代 hard top-1）—— §6.3 结论 3 说 23.27% 是
   hard top-1 的本征上限；模块已支持 `top_k>1`，只差一次对照。
5. 上面三条都清楚了，再把 NDB **正式接进 `train.py`**（不是现在的独立脚本）。

---

## 11. 工程侧（2026-09-10 本轮新增，与算法无关但很值）

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`（只搬了位置，内容字节未改）。本页只保留**当前状态与规程**。

### 11.1 单元测试：0 → 数百条

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`（只搬了位置，内容字节未改）。本页只保留**当前状态与规程**。

### 11.2 lint 门禁：只开"一定是 bug"的四条

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`（只搬了位置，内容字节未改）。本页只保留**当前状态与规程**。

### 11.3 结构优化（都是行为保持的）

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`（只搬了位置，内容字节未改）。本页只保留**当前状态与规程**。

### 11.4 技术债清单：`TECH_DEBT.md`

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`（只搬了位置，内容字节未改）。本页只保留**当前状态与规程**。

## 12. NDB v7 原型：**可读可写、no_grad、模型自己决定**（2026-09-10 新增）

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`（只搬了位置，内容字节未改）。本页只保留**当前状态与规程**。

### 12.1 在哪

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`（只搬了位置，内容字节未改）。本页只保留**当前状态与规程**。

### 12.2 与 v5/v6 的根本区别（三处，都针对已被证伪的旧范式）

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`（只搬了位置，内容字节未改）。本页只保留**当前状态与规程**。

### 12.3 「模型自己决定读写」落在哪

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`（只搬了位置，内容字节未改）。本页只保留**当前状态与规程**。

### 12.4 已做的验证

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`（只搬了位置，内容字节未改）。本页只保留**当前状态与规程**。

### 12.5 已知未做（下一步，见 §10）

> 📦 **本节正文已归档** → `dev-notes/81-PROJECT_STATE历史归档.md`（只搬了位置，内容字节未改）。本页只保留**当前状态与规程**。
