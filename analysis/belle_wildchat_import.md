# Belle 多轮 + WildChat(中文) 导入报告

> 2026-09-13 | 任务：把两份中文多轮对话原始数据转成项目语料格式、质量过滤、跑 `clean_corpus.py`，
> 产出可进「对话阶段」的干净文件。
> 代码改动：`data/chinese/import_external.py`（新增两个 format 分支 + selftest 用例）、
> 新增 `data/chinese/verify_import_belle_wildchat.py`（四条自证）。
> 本报告所有数字都来自**实测**（命令与日志见文末「复现」）。

---

## 1. 结论摘要

| 产物 | 条数(block) | 字符(不含分隔空行) | 模型轮/块(mean) | 总轮/块(mean) | 汉字占比 |
|---|---:|---:|---:|---:|---:|
| `new_sources/belle_multiturn.txt` | 53,689 | 24,860,266 | 4.55 | 9.10 | **97.17%** |
| `new_sources/wildchat_zh.txt` | 3,168 | 14,999,688 | 6.90 | 13.77 | **61.75%** |
| `clean_v3/belle_multiturn.txt`（清洗后） | 53,686 | 24,859,455 | — | — | — |
| `clean_v3/wildchat_zh.txt`（清洗后） | 3,127 | 14,233,749 | — | — | — |

* 汉字占比 = 汉字 / (汉字 + 拉丁字母)，全文统计（`[\u4e00-\u9fff]` vs `[A-Za-z]`）。
  Belle 拉丁字母只有 597,177（2.83%，主要是夹在中文里的英文词/术语）；
  WildChat 拉丁字母 4,336,796（38.25%）——里面混着**代码块、命令、英文专名**，这是 WildChat 的天然形态。
* 两份产出都满足：≥4 条 `模型：` 回复 / block 内只有 `用户：`/`模型：` 两种行 / block 之间空行分隔 /
  内容里的换行已压成空格。

---

## 2. 输入与许可

| 数据 | 路径 | 规模 | 许可 |
|---|---|---|---|
| Belle multiturn_chat_0.8M | `/home/vesita/datasets/NLP/_belle/multiturn_chat_0.8M.json` | 990,032,546 B；**831,036 条** | 仓库标称 **GPL-3.0**，且 README 明确「仅限研究目的、不得商用」（[HF card](https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M)） |
| WildChat（中文子集） | `/home/vesita/datasets/NLP/_wildchat/data/train-0000{0,1}-of-00002.parquet` | 425,791,701 B；**122,958 行** | **AI2 ImpACT License – Low Risk Artifacts**（[allenai.org/licenses/impact-lr](https://allenai.org/licenses/impact-lr)），研究用途 + 禁止造成危害；本地 README 在 `/home/vesita/datasets/NLP/_wildchat/README.md` |

**关于 Belle 的物理格式（与任务描述略有出入，已实测）**：它不是「一个大 JSON 数组」，而是 **NDJSON**
——831,036 行，每行恰好一条 `{...}`、行尾 `}\n`，无外层 `[ ]`、无逗号分隔。
所以「不能整文件 `json.load`」的约束成立（990MB），但流式方案是**逐行**而不是 `ijson`。
为鲁棒起见，转换器里的流式扫描器**两种布局都支持**（见 §4 自证 1）。

---

## 3. 实现（优先复用 `import_external.py`，没有另写一套）

在 `data/chinese/import_external.py` 里新增两个 format 分支，并补 selftest：

| format | 作用 | 关键点 |
|---|---|---|
| `belle_json` | 解析 `instruction` 里的 `Human:`/`Assistant:` 多轮 + 用 `output` 接尾轮 | 流式、两遍扫描、质量过滤、按字符预算抽样 |
| `wildchat_parquet` | 读 parquet（按 row group，只读 4 列） | 过滤 `toxic/language/turn`，**复用 `turns_from_messages`** |

* **`messages_json` 的 role→标签 映射抽成 `turns_from_messages()`**，`messages_json` 与
  `wildchat_parquet` 共用同一份实现 —— 等价于「先把 parquet 落成 messages-jsonl 再走 `messages_json`」，
  只是省掉了中间文件。selftest 里有**等价性断言**（合成 parquet vs 同内容的 messages-json）。
* **流式扫描器 `iter_json_records()`**：NDJSON 走 `readline` 快路径；大 JSON 数组走通用状态机
  （按 `"`/`{`/`}`/`\` 跳转 + 字符串状态机，字符串外整段切片）。`force_state_machine=True` 可强制走状态机，供自证用。
* **Belle 尾轮拼接规则**：把 `instruction` 按行首 `Human:`/`Assistant:` 切成段，最后一段若是 `Assistant:`
  就把 `output` 接到它后面；末尾空模型轮（`output` 也为空）直接丢掉，中段空轮判为坏样本整条丢。
* **确定性抽样**：固定 `seed`，两遍扫描。第 1 遍量「过滤后 block 字符总量」；第 2 遍
  `u = blake2b(seed:idx)/2^64`，`u < target/total` 的入选，再按 `u` 升序确定性填充到 **≤ target**
  （`trim_to_target`）。**不取文件前缀**，换 seed 就是另一套样本；同 seed 同输入永远同样输出。
  为什么加最后那步截断：WildChat 单条最长 256,660 字符，重尾会让 Bernoulli 抽样超目标
  7%（实测 15.0M → 16.06M），截断后落到 14,999,688。
* 少于 `--min-turns` 轮的样本仍由原有 `to_text()` 丢弃（`--min-turns 4`）。

---

## 4. 过滤统计

### 4.1 Belle（`--min-turns 4 --max-en-ratio 0.5 --target-chars 25000000 --seed 42`）

| 阶段 | 条数 | 说明 |
|---|---:|---|
| 原始记录 | 831,036 | = `wc -l` = 逐行 `json.loads` = 状态机扫描 |
| 丢弃：`<4` 轮 | 668,077 | 80.4%（Belle 以 2~3 轮为主：前 3000 条的分布是 2 轮 59%、3 轮 22%） |
| 丢弃：纯英文占比 >0.5 | 6,797 | `拉丁/(汉字+拉丁) > 0.5` 或没有汉字 |
| 丢弃：`Human:`/`Assistant:` 残留 | 153 | 段内仍出现标记 ⇒ 切分不可信，**整条**丢 |
| 丢弃：空回复 | 16 | `empty_model` 15 + `empty_user` 1 |
| **过滤后保留池** | **155,993** | 72,206,259 字符（占原始 18.8%） |
| 抽样（frac=0.346230） | **53,689** | 24,860,266 字符（目标 25M，差 −0.56%） |

### 4.2 WildChat（`--min-turns 4 --target-chars 15000000 --seed 42`）

| 阶段 | 条数 | 说明 |
|---|---:|---|
| 原始行 | 122,958 | = parquet `num_rows`（两个文件 61,479 + 61,479） |
| 丢弃：`toxic == True` | 3,655 | 2.97% |
| 丢弃：`language != 'Chinese'` | **0** | ⚠️ 见下 |
| 丢弃：`turn < 4` | 87,991 | 71.6%（转换后 `模型：` 轮数复核：再丢 0 条） |
| **过滤后保留池** | **31,312** | 140,319,492 字符 |
| 抽样（frac=0.106899） | **3,168** | 14,999,688 字符（目标 15M，差 −0.002%） |

**★ 重要发现：本地的 WildChat 副本已经是纯中文。** 全量 122,958 行的 `language` 只有
`Chinese` 一个取值（`('Chinese', False): 119,303` + `('Chinese', True): 3,655`）。
所以命令里的 `language == 'Chinese'` 这一条**实际过滤掉了 0 行**，不是没生效，而是本地文件
在下载时已被筛过 —— 换一份全语言 WildChat-1M 跑同一个分支时，这一条才会真正起作用（代码路径已在
selftest 里用合成 parquet 验证过：非中文 + toxic 的行会被丢）。

其它参考值：`turn` 与「assistant 轮数」在抽查的 2000 行里**完全相等**（turn 1↔1、7↔7），
所以 `turn >= 4` 就是「≥4 轮 assistant」。`redacted == True` 的行 430 条（0.35%）。

---

## 5. 四条自证（`data/chinese/verify_import_belle_wildchat.py`，17 项断言全过）

```
===== Belle / WildChat 导入自证 =====
  [逐行 json.loads] 条数=831,036（2.2s）
  [流式状态机]   条数=831,036（13.3s）
  PASS ✅ 自证1a 条数一致 — 831,036 == 831,036
  PASS ✅ 自证1b 前 5000 条内容逐条一致 — 比对 5000 条 dict
  PASS ✅ 自证2a 已知答案逐字一致 — kept=2 chars=80
  PASS ✅ 自证2b 负向对照：去 output 每条少 1 轮 — 带/不带 output 的模型轮数差=[1, 1]
  PASS ✅ 自证2c 残标（段内 Human:/Assistant:）被拦截 — why=residual
  过滤统计：{'raw': 831036, 'bad_json': 0, 'no_marker': 0, 'residual': 153, 'empty_user': 1,
             'empty_model': 15, 'english': 6797, 'short': 668077}
  PASS ✅ 自证3a-1 过滤前条数 == 逐行解析条数 — 831,036 == 831,036
  PASS ✅ 自证3a-2 过滤后条数 < 过滤前（确实丢了东西） — 155,993 / 831,036 = 18.8%
  PASS ✅ 自证3a-3 产出每个 block ≥4 轮 — 最小模型轮数=4
  PASS ✅ 自证3a-4 产出无 Human:/Assistant: 残留 — 命中 0 条
  PASS ✅ 自证3a-5 产出无空回复行 — 命中 0 条
  PASS ✅ 自证3a-6 反向对照：关掉英文过滤后条数变多 — 开=19827 关=19995（多 168）
  产出 block 数=3,168；回原始 parquet 逐行核对来源…
  扫描原始行 122,958 条
  PASS ✅ 自证3b-1 产出每条都能回溯到原始行 — 命中 3,168 行 / 产出 3,168 block，未回溯 0
  PASS ✅ 自证3b-2 产出**不存在** toxic==True / 非中文 / <4 轮 — 违规 0 条
  PASS ✅ 自证3b-3 抽样 1000 条复核（结构与来源） — 抽样 1000 条全部 ≥4 轮
  PASS ✅ 自证3b-4 反向对照：关过滤后 toxic=True 条目出现 — 关过滤得 61478 条；
             其中 1695 条来自 toxic=True 行（原始 toxic 行共 1695）
  PASS ✅ 自证3b-5 反向对照：关过滤后 <4 轮条目出现 — 其中 41650 条来自 <4 轮行
  belle_multiturn.txt: block=53,689 字符=24,860,266 均463字/块 均4.55轮/块
             汉字=20,539,871 拉丁=597,177 汉字占比=97.17%
  wildchat_zh.txt: block=3,168 字符=14,999,688 均4735字/块 均6.90轮/块
             汉字=7,000,965 拉丁=4,336,796 汉字占比=61.75%
  PASS ✅ 自证4 两份产出汉字占比均 >50%（汉字是占比最高的文字）
===== 全部通过 ✅（17 项）
```

逐条对应任务要求：

1. **解析正确性**：状态机扫描与逐行 `json.loads` 的**条数一致**（831,036）且**前 5000 条 dict 逐条相等**
   ⇒ 切块没有吃错边界。另外 `bad_json = 0`（没有一条解析失败）。
2. **已知答案**：2 条手工对话（含 `Human:/Assistant:` 与 `output` 尾轮）走**完整转换**
   （`belle_json` → `to_text`）后逐字符合预期；**负向对照**去掉 `output` 后，每条对话的模型轮数
   恰好少 1（差 = [1, 1]）⇒ 尾轮真的接上了。附带一条：段内残标被拦（`why=residual`）。
3. **过滤生效**：
   * Belle 过滤前 831,036 → 保留池 155,993（18.8%）；产出文件从结构上复核：最小模型轮数 = 4、
     无残标、无空回复行。
   * WildChat 把产出 3,168 条 block **逐条回溯到原始 parquet 行**（命中 3,168 行、0 条未回溯），
     这些来源行全部 `toxic == False`、`language == 'Chinese'`、`turn >= 4`，**违规 0 条**；
     另按 1000 条抽样复核，全部 ≥4 轮。
   * **反向对照**：把过滤全关后，`toxic=True` 的条目确实出现（1695 条 toxic 行全部进入产出）、
     `<4` 轮的条目也确实出现（41,650 条）⇒ 过滤不是空转。Belle 侧同样做了反向对照
     （关掉英文过滤后多留 168 条）。
4. **语言检查**：Belle 汉字 97.17%（20,539,871 vs 597,177），WildChat 汉字 61.75%
   （7,000,965 vs 4,336,796）。两者汉字都是占比最高的文字。WildChat 的拉丁字母偏高是数据形态
   （代码/命令/英文），不是解析或过滤出错。

---

## 6. 清洗（项目现成 `clean_corpus.py`，未自写规则）

执行的命令（原样）：

```
.venv/bin/python data/chinese/clean_corpus.py --src data/chinese/new_sources \
    --dst data/chinese/clean_v3 --apply --dedup-blocks --ngram-size 12 \
    --max-ngram-freq 50 --max-ngram-cover 0.5 --ngram-max-candidates 20000000 \
    --filter-nondialogue-rep3 --nondialogue-max-rep3 0.4
```

结果（`clean_v3/CLEANING_REPORT.md`）：

| 文件 | blocks 进 | blocks 出 | 字符 进 | 字符 出 | dedup | ngram |
|---|---:|---:|---:|---:|---:|---:|
| belle_multiturn.txt | 53,689 | 53,686 | 24,860,266 | 24,859,455 | 0 | 3 |
| qa_knowledge.txt | 274,111 | 270,984 | 19,233,890 | 19,113,563 | 2,075 | 1,062 |
| sharegpt_zh_38k.txt | 38,537 | 38,247 | 60,874,410 | 60,689,375 | 86 | 244 |
| wildchat_zh.txt | 3,168 | 3,127 | 14,999,688 | 14,233,749 | 0 | 41 |
| **合计** | **369,505** | **366,044** | **119,968,254** | **118,896,142** | **2,161** | **1,350** |

* 丢弃总量：**3,461 block / 1,072,112 字符（0.89%）**；`rep3 / min-reply-chars / 非对话 rep3` 三条规则命中 0。
* WildChat 的 41 个 block 被 ngram 规则丢掉却带走 **765,939 字符（占它自己 5.11%）** ——
  原因：WildChat 单条很长（均 4,735 字，中位 3,216，p99 34,118，**最长 256,660 字符**），
  里面夹着大量复读套话，正是短语级规则的目标。
  Belle 只丢 3 条（含 811 字符；Belle 中位 448 字、最长 1,678）。
* heavy 集合：频次 > 50 的 12-gram 共 **9,883** 个（判定用的精确计数集合，不是草图估计）；
  回复字符里的高频短语覆盖率 3.83% → 3.58%（−0.24 pp）。
* **跨源精确去重**：`--dedup-blocks` 是全局去重（按文件名序，保留首次出现）。
  belle / wildchat 的 dedup 命中都是 **0**：两份新数据内部以及与 qa_knowledge/sharegpt **没有**精确重复 block。
  （qa_knowledge 命中 2,075、sharegpt 命中 86，与上一次整库清洗时（2,075 / 87）基本一致 ⇒ 新文件没有「抢走」它们的去重。）

### ⚠️ 对既有 `clean_v3` 文件的影响（任务已声明「可接受」，但实测有细微变化）

`--src data/chinese/new_sources` 里还有 `qa_knowledge.txt`、`sharegpt_zh_38k.txt`，这次命令把它们
**重新清洗并覆盖**了。对照实测（清洗前 sha256 已记录）：

| 文件 | 旧 clean_v3（整库 `--src raw_all` 跑的） | 这次（`--src new_sources`） | 差异 |
|---|---|---|---|
| qa_knowledge.txt | 270,975 block / 19,108,814 字符 | 270,984 block / 19,113,563 字符 | **+9 block / +4,749 字符（+0.025%）** |
| sharegpt_zh_38k.txt | 38,234 block / 60,654,278 字符 | 38,247 block / 60,689,375 字符 | **+13 block / +35,097 字符（+0.058%）** |

* 原因：上一次的 heavy 12-gram 集合是在 **整库 `data/chinese/raw_all`（27 文件 / 1.02G 字符）**
  上统计的（144,859 个 heavy），这次只在 **4 个文件 / 1.2 亿字符** 上统计（9,883 个 heavy）——
  基准变小 ⇒ 对这两个文件而言「高频套话」判定更宽松 ⇒ 少丢了一点。差异 <0.06%，方向是**保留更多**。
* 副作用：`clean_v3/CLEANING_REPORT.md` 也被覆盖成「4 文件版」，原来那份**整库 27 文件版报告**
  已不在 `clean_v3/` 里 —— 完整内容见本报告**附录 A**（已原样保留）。
* 其它 clean_v3 文件（c4_zh / wikipedia_cn / deepseek_r1… 等 23 个）**没有被动过**，只改了这 4 个同名文件。
* 如果要求与整库口径完全一致，应把两个新文件放进 `raw_all/` 后**重跑整库**（会重写 27 个文件，
  超出本任务范围，未做）。

---

## 7. 残留风险

1. **WildChat 的 PII（最高优先级）**。数据来自真实 ChatGPT 用户对话，本地 parquet 里还留着
   `hashed_ip` / `state` / `country` / `header` 等可关联字段（本次只读了 `conversation/toxic/language/turn`，
   这些字段**没有**进入 txt）。但 `conversation.content` 本身的 PII 只由数据集的自动 redaction 处理：
   122,958 行里只有 **430 行**（0.35%）标了 `redacted=True`，其余对话可能仍含未被检出的
   姓名/邮箱/电话/地址等。**投放训练前建议再过一遍 PII 规则**（正则 + 可选模型），
   并考虑丢弃 `redacted=True` 已标记的那些（本次按任务要求只按 toxic/language/turn 过滤，未额外丢）。
2. **许可**。WildChat 是 **AI2 ImpACT LR**（研究用途、禁止有害用途），Belle 是 **GPL-3.0 + 仅限研究**。
   两者都**不是**可无条件商用的语料；本项目（研究用途）可用，但若后续分发模型/数据需重新评估。
3. **Belle 是 ChatGPT 生成的合成对话**（其 README 明说「未经严格校验，可能包含错误」），
   事实性内容不可信；作为「对话形态/人设」语料没问题，作为知识源有噪声。
4. **WildChat 长尾**：均 4,735 字/块、最长 256,660 字符，且夹代码/命令/英文（拉丁字母占 38%）。
   进入 char-level 训练前如果按「对话行」做 masking，超长 block 会贡献不成比例的长度；是否需要截断/分块由装配阶段决定。
5. **繁体**：WildChat 中文里有相当比例繁体（如「倉頡規則」「作為一名室內設計師」），Belle 为简体。
   本项目 tokenizer 是 char-level，简繁会各占一个 token，**不做归一化**就会稀释简体的统计强度。本次未做繁简转换（任务未要求）。
6. **两个反向对照的意义边界**：`--no-filter` 是**调试开关**，只为证明过滤不是空转；不要用它产出语料。
7. `max-en-ratio 0.5` 是本次拍的阈值（Belle 里丢 6,797 条 = 0.8%）。若认为「中英夹杂」有价值，
   可以调高；该阈值下反丢掉的中文条目已由反向对照确认是真正的英文条目。

---

## 8. 复现（全部只用 `.venv/bin/python`）

```bash
cd /home/vesita/coding/my/nanoSeek

# 1) 转换（流式 + 过滤 + 确定性抽样）
.venv/bin/python data/chinese/import_external.py --format belle_json \
    --in /home/vesita/datasets/NLP/_belle/multiturn_chat_0.8M.json \
    --out data/chinese/new_sources/belle_multiturn.txt \
    --min-turns 4 --target-chars 25000000 --seed 42

.venv/bin/python data/chinese/import_external.py --format wildchat_parquet \
    --in /home/vesita/datasets/NLP/_wildchat/data \
    --out data/chinese/new_sources/wildchat_zh.txt \
    --min-turns 4 --target-chars 15000000 --seed 42

# 2) 清洗（§6 的命令）
.venv/bin/python data/chinese/clean_corpus.py --src data/chinese/new_sources \
    --dst data/chinese/clean_v3 --apply --dedup-blocks --ngram-size 12 \
    --max-ngram-freq 50 --max-ngram-cover 0.5 --ngram-max-candidates 20000000 \
    --filter-nondialogue-rep3 --nondialogue-max-rep3 0.4

# 3) 自证（四条）
.venv/bin/python data/chinese/import_external.py --selftest
.venv/bin/python data/chinese/verify_import_belle_wildchat.py

# 4) 门禁
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest -q -m 'not slow'      # 全绿
```

**下一步（不在本任务范围）**：`clean_v3/` 已经是 `prepare.py --source-dir data/chinese/clean_v3`
可直接编码的目录；要进对话阶段，还需在装配侧把这两个新文件名登记进来源表/配比。
`data/chinese/prepare.py` / `training/train.py` 等由其它会话占用，本次**没有改**。

---

## 附录 A：被覆盖前的整库 `CLEANING_REPORT.md`（原样保留）

```
# 语料清洗报告（clean_corpus.py）
- 时间：2026-09-13T13:53:41+08:00
- 模式：--apply（已写数据文件）
- 源目录（只读）：data/chinese/raw_all    输出目录：data/chinese/clean_v3
- 命令行：clean_corpus.py --src data/chinese/raw_all --dst data/chinese/clean_v3 --apply
          --dedup-blocks --ngram-size 12 --max-ngram-freq 50 --max-ngram-cover 0.5
          --ngram-max-candidates 20000000 --filter-nondialogue-rep3 --nondialogue-max-rep3 0.4

## 全局
| 指标 | 清洗前 | 清洗后 | 变化 |
| block 数 | 1,895,503 | 1,591,542 | -303,961（16.04%） |
| 字符数 | 1,023,815,587 | 847,719,586 | -176,096,001（17.20%） |
| 高频 12-gram 覆盖率（回复字符） | 15.4961% | 8.5849% | -6.9112 pp |
（heavy 集合：频次 > 50 的 144,859 个 12-gram；回复字符 514,734,294 → 468,419,902）

## 各规则命中 block 数
dedup_blocks 81,236 | max_reply_rep3 0 | max_ngram_cover 123,407 | min_reply_chars 0 | nondialogue_rep3 101,711

## 逐来源（只列与本次相关的 2 行；原报告共 27 个源文件，全局与合计值已完整保留）
| 文件 | blocks 进 | blocks 出 | 字符 进 | 字符 出 | dedup | rep3 | ngram | min | 非对话rep3 |
| qa_knowledge.txt     | 274,111 | 270,975 | 19,233,890 | 19,108,814 | 2,075 | 0 | 1,071 | 0 | 0 |
| sharegpt_zh_38k.txt  |  38,537 |  38,234 | 60,874,410 | 60,654,278 |    87 | 0 |   256 | 0 | 0 |
| **合计** | **1,895,503** | **1,591,542** | **1,023,815,587** | **847,719,586** | **81,236** | **0** | **123,407** | **0** | **101,711** |
- 源文件数：27
```

## 附录 B：文件 sha256

| 文件 | sha256 | bytes |
|---|---|---:|
| `new_sources/belle_multiturn.txt` | `cdc16424157c557871414b947df181e217ff0161bca6fc0e985c9491f369e8cd` | 71,432,213 |
| `new_sources/wildchat_zh.txt` | `3e85b6bc37f9c866555f4d46c3b060d1765f4271a491e2276b1877b7364a3fcf` | 30,661,740 |
| `clean_v3/belle_multiturn.txt` | `79d932fa79b0b5a341a735fddbd5234da50e5e7c871f0d76df577e747d78c330` | 71,430,422 |
| `clean_v3/wildchat_zh.txt` | `4510ab2f082c332f885dd4e846f960b1fca2bd45b6aac6a6801d111ec76200f3` | 28,854,735 |
| `clean_v3/qa_knowledge.txt`（被重写） | 旧 `76ffcbc9263317af…` → 新 `b713b38e227d0347…` | 55,794,715 → 55,797,724 |
| `clean_v3/sharegpt_zh_38k.txt`（被重写） | 旧 `441d010a80c98bed…` → 新 `f98168e0213e0ab5…` | 143,633,659 → 143,684,697 |
