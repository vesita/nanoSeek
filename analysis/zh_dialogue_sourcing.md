# 中文多轮对话语料侦察报告（zh_dialogue_sourcing）

> 任务：找出**真正多轮、连贯的中文对话**语料候选，并给出可执行的引入方案。
> 日期：2026-09-13。执行人：子 agent（只读侦察 + 小规模抽样下载）。
> 所有数字标注 **[实测] / [推断]**，并给证据强度（抽样量 / 单次或多次）。
> 测量脚本：`temp/zh_dialogue_probe/{measure.py, measure_downloaded.py, dss_sample.py, make_table.py}`；
> 汇总数据：`temp/zh_dialogue_probe/evidence_all.json`；原始抽样文件在 `temp/zh_dialogue_probe/dl{,2}/`。

---

## 0. 结论速览

1. **[实测] 本地语料里"真多轮"最好的三个是 `kdconv`（0.94M 字符，99.4% ≥2 轮、74.9% ≥4 轮，
   承接 ΔLCS≥4 **+4.6pp**，12-gram 覆盖仅 12.8%）、`multi_turn`（45.5M 字符，100% ≥2 轮、
   100% ≥4 轮、mean 5.48，**但 12-gram 字符覆盖 85.9%**）、`glm`（5.05M 字符，72.7% ≥2 轮、
   12-gram 覆盖 47.5%）。**论"干净 + 有承接" `kdconv` 最好，论"量" `multi_turn` 最大，
   但 `multi_turn` 是全部来源里最模板化的那一个。**
2. **[实测] 承接性对照（真实轮序 vs 角色保持打乱）在本地来源上几乎全部为 0**：
   8-gram Δ>0 占比全部 ≤+0.4pp；只有 `kdconv` 用更敏感的最长公共子串（LCS≥4）能看到
   **+4.6pp**。也就是说：**本地中文对话语料里"上一轮的内容被下一轮接住"这件事，
   在字面上几乎测不到**——与"模型缺多轮推进"的判断一致。
3. **[实测] 候选下载源一测就分开了**：`shareAI/ShareGPT-Chinese-English-90k` 的
   `unknow_zh_38k.jsonl`（真实下载 150.7MB）LCS≥4 承接 **+15.1pp**、8-gram **+6.4pp**；
   `WildChat-Chinese` **+14.5pp / +9.2pp**；`BelleGroup/multiturn_chat_0.8M` **+8.3pp / +3.1pp**。
   即：**"有没有承接"这件事在可下载源上比在本地语料上强一个数量级**。
4. **[实测] 最推荐三件套**（详见 §4）：
   - `shareAI/ShareGPT-Chinese-English-90k :: sharegpt_jsonl/unknow_zh_38k.jsonl`
     （**150.7MB，Apache-2.0，单文件，已在本次实测下载并跑完指标**）
   - `BelleGroup/multiturn_chat_0.8M`（**990MB，GPL-3.0，100% ≥2 轮、质量最干净**；
     文件是 JSONL，可以**随机区间抽样 ~100MB** 而不必下全量）
   - `benchang1110/WildChat-Chinese`（**435.8MB，真实用户多轮，承接最强**；
     但 8.25% 带 `toxic` 标记、许可未标注，需要过滤）
5. **[实测] 明确"否掉"的三个**：`shareAI/shareAI-Llama3-DPO-zh-en-emoji`（ModelScope，
   已下载 7.9MB 全量文件验证）是**单轮 DPO**；`svjack/GLM-Open-Dialogue-Chinese-Dataset`（123MB）
   是**单轮续写模板**且本地 `glm_dialogue.txt` 已来自它；`lorinma/Slim-LCCC-zh` **401 gated，
   无 token 下不了**。
6. **[实测] `thu-coai/lccc` 本身没有数据**（HF 仓库只有 3 个文件：README + `lccc.py` 加载脚本），
   脚本里的 `_URLS` 指向 `silver/lccc`。要下 LCCC 请直接下 `silver/lccc`。

---

## 1. 测量口径（口径不写清楚，数字就没有意义）

### 1.1 抽样

- 所有 txt/parquet 一律**随机字节偏移 / 随机行组**，**不取文件前缀**（AGENTS §5.1、skill §8.1）。
- 质量指标（rep3、高频 12-gram）按**字符预算**从全库助手回复里随机抽（预算 1.2M 字符），
  不是"取前 N 条"。
- 每源 ~1500 段对话（小源全取，如 `muice` 1500、`ada_zh` 472、`identity` 12）。
- 大体积 HF 源（Belle 990MB、MOSS 8.7GB、WildChat 435MB）用
  **datasets-server `/rows` + 随机 offset** 抽样，不下载全量。

### 1.2 承接性：为什么必须带"角色保持打乱"对照

"相邻轮 8-gram 重叠中位数"本身会骗人：真实对话里相邻对大多是 `用户→模型`，
打乱整条轮序后 `用户→模型` 会被换成 `模型→模型` 之类，**角色模式的变化会混进差值**。
所以主对照是**角色保持打乱**（`role_preserving_shuffle`）：只把同角色的文本在各自位置上重排，
`u2a / a2u` 的定义完全不变，变的只有"这两个相邻轮是否真的来自同一段对话"。

同时注意一个**结构性事实**：单轮对话在角色保持打乱下是**恒等变换**（只有一个 user、一个 assistant
文本），Δ 恒为 0。所以表里另给一列 **"真多轮子集"**（≥2 个 assistant 轮的对话）——只有这一列
对"多轮推进"有意义。

### 1.3 已知答案对照（每次运行都跑，`ok=True`）

```
carry_same_doc_median            = 0.8   （同一话题的 u/a 对，应高）
carry_unrelated_median           = 0.0   （四句互不相关，应 0）
carry_cross_topic_swap_median    = 0.0   （uA→aB→uB→aA 全跨话题，应 0）★ 这一条才是
                                           "打乱后应该发生什么"的对照
role_preserve_roles_ok           = True  （打乱后角色序列逐位不变）
ok                               = True
```

### 1.4 8-gram 的已知盲区（本次实测踩到，必须说明）

中文对话轮普遍很短（LCCC 助手回复中位数 **9 字符**、GLM 15、Belle 90）。4~6 字的实体/短语
（"理查德·柯蒂斯"）永远凑不满 8-gram，所以 **8-gram 会把"有承接"误判成 0**——本地全部来源的
8-gram Δ 都≈0，而同一批数据用 4-gram / LCS≥4 能测出信号。
⇒ 报告以 8-gram 为主口径（任务要求），同时给 **4-gram 与 LCS≥4** 做敏感性复核；
**不要把"8-gram≈0"读成"无承接"**，只能读成"8-gram 这个尺子在这些数据上没有分辨力"。

---

## 2. 逐源测量结果

列含义：Δ8gram>0 = 真实轮序与角色保持打乱的"含重叠对占比"之差（全体对话 / 仅真多轮子集）；
ΔLCS≥4 = 相邻轮最长公共子串 ≥4 的比例之差（仅真多轮子集）；hf12 = 高频 12-gram
（df≥5）的回复覆盖率 / 字符覆盖率。全部 **[实测]**，单次抽样（n≈1500 段/源）。

### 2.1 本地 `data/chinese/`（现已被搬成 `data/chinese/raw_all/` 的软链，内容不变）

| 源 | 抽样对话数 | 抽样字符 | 轮数 mean/med | ≥2 / ≥4 / ≥8 轮 | Δ8gram>0 全体/真多轮 | ΔLCS≥4 | rep3 均值(>0.3占比) | hf12 回复/字符 | 回复中位长 |
|---|---|---|---|---|---|---|---|---|---|
| lccc | 1500 | 0.06M | 1.57 / 1 | 43.9% / 2.7% / 0.0% | +0.0% / +0.1% | +0.9% | 0.008 (0.6%) | 0.0% / 0.0% | 9 |
| multi_turn | 1500 | 0.62M | 5.48 / 5 | 100.0% / 100.0% / 0.0% | +0.0% / +0.0% | +0.1% | 0.029 (1.4%) | 99.1% / **85.9%** | 63 |
| glm | 1500 | 0.15M | 2.88 / 2 | 72.7% / 31.0% / 2.1% | +0.0% / +0.0% | +1.8% | 0.000 (0.0%) | 37.8% / **47.5%** | 15 |
| kdconv | 1500 | 0.51M | 4.44 / 5 | **99.4% / 74.9%** / 0.0% | +0.2% / +0.2% | **+4.6%** | 0.025 (0.3%) | 25.0% / 12.8% | 40 |
| muice | 1500 | 0.07M | 1.14 / 1 | 10.6% / 0.9% / 0.0% | +0.4% / +0.9% | +2.2% | 0.014 (0.8%) | 0.6% / 0.7% | 20 |
| dailychat | 622 | 0.03M | 1.00 / 1 | 0.0% / 0.0% / 0.0% | — (单轮) | +0.0% | 0.006 (0.0%) | 2.4% / 3.1% | 34 |
| zhihu_kol | 1500 | 0.30M | 1.00 / 1 | 0.0% / 0.0% / 0.0% | — (单轮) | +0.0% | 0.041 (0.9%) | 1.6% / 0.5% | 117 |
| coig_other | 1500 | 0.56M | 0.99 / 1 | 0.0% / 0.0% / 0.0% | — (单轮) | +0.0% | 0.066 (2.4%) | 3.8% / 0.2% | 296 |
| coig_cqia | 1500 | 0.38M | 0.92 / 1 | 0.0% / 0.0% / 0.0% | — (单轮) | +0.0% | 0.044 (1.4%) | 0.0% / 0.0% | 58 |
| coig_wiki | 1500 | 0.59M | 0.97 / 1 | 0.0% / 0.0% / 0.0% | — (单轮) | +0.0% | 0.077 (1.2%) | 0.5% / 0.1% | 266 |
| coig_logic | 1500 | 0.37M | 0.83 / 1 | 0.0% / 0.0% / 0.0% | — (单轮) | +0.0% | 0.072 (4.6%) | 9.2% / 1.7% | 62 |
| coig_math | 765 | 0.24M | 0.93 / 1 | 0.0% / 0.0% / 0.0% | — (单轮) | +0.0% | 0.095 (9.8%) | 6.2% / 1.8% | 80 |
| coig_code | 1384 | 0.57M | 0.89 / 1 | 0.0% / 0.0% / 0.0% | — (单轮) | +0.0% | 0.102 (8.2%) | 7.7% / 0.8% | 157 |
| code_alpaca | 1500 | 0.33M | 1.00 / 1 | 0.0% / 0.0% / 0.0% | — (单轮) | +0.0% | 0.065 (0.5%) | 99.3% / 77.5% | 120 |
| gsm8k_cot | 1500 | 0.85M | 1.00 / 1 | 0.0% / 0.0% / 0.0% | — (单轮) | +0.0% | 0.306 (50.1%) | 100.0% / 19.4% | 279 |
| deepseek_r1 | 1500 | 3.56M | 1.00 / 1 | 0.0% / 0.0% / 0.0% | — (单轮) | +0.0% | 0.266 (35.4%) | 94.4% / 1.5% | 1930 |
| qwen3_235b | 1500 | 1.79M | 1.00 / 1 | 0.0% / 0.0% / 0.0% | — (单轮) | +0.0% | 0.221 (26.8%) | 37.9% / 2.6% | 898 |
| identity | 12 | 0.00M | 1.00 / 1 | 0.0% / 0.0% / 0.0% | — (单轮) | +0.0% | 0.004 (0.0%) | 0.0% / 0.0% | 34 |

**"轮数 mean" < 1 的源**（coig_*、`identity`）是解析副作用：这些文件的"用户/模型"成对不总完整，
少量样本只有 `模型：` 行被计入 0 个 assistant——不影响它们"单轮为主"的结论。

**每源全文字符数 [实测，逐文件流式计数]**（不是抽样外推）：

| 源 | 文件 MB | 全文字符 | `用户：` 行数 |
|---|---|---|---|
| multi_turn_dialogue.txt | 133.0 | 45.50M | 550,295 |
| deepseek_r1_distill_dialogue.txt | 601.3 | 258.03M | 137,094 |
| qwen3_235b_distill_dialogue.txt | 279.6 | 136.27M | 110,134 |
| coig_wiki_dialogue.txt | 112.7 | 40.86M | 30,946 |
| zhihu_kol_dialogue.txt | 76.9 | 27.56M | 136,811 |
| code_alpaca_dialogue.txt | 25.3 | 25.15M | 14,909 |
| lccc_dialogue.txt | 22.7 | 8.06M | 236,441 |
| coig_other_dialogue.txt | 14.1 | 5.50M | 9,856 |
| glm_dialogue.txt | 13.6 | 5.05M | 119,523 |
| coig_cqia_dialogue.txt | 12.1 | 4.57M | 5,634 |
| gsm8k_cot_dialogue.txt | 4.8 | 4.22M | 7,473 |
| kdconv_dialogue.txt | 2.6 | 0.94M | 10,991 |
| 其余（coig_code/logic/math、muice、dailychat、identity） | 4.4 | 3.05M | 22,474 |
| **合计（对话类，不含 c4/wikipedia/四大名著/诗词）** | | **≈564M** | |

> ⚠️ `lccc_dialogue.txt` 的 8.06M 字符 = **LCCC-base 全量的约 3.4%**：
> 它自己的下载脚本 `data/chinese/download_daily_chat.py` 写着"读文件前缀即可拿到所需条数
> （目标 ~15 万条 ≈ 前 2-3%），HTTP 提前断连"。本次专门做了**前缀 vs 随机区间对照**：
> 前缀 1500 段 vs 未压缩镜像随机 3×20MB 区间的 1500 段，轮结构 43.7% vs 45.0% ≥2 轮、
> ΔLCS≥4 +0.8pp vs +0.9pp——**本次没测出前缀偏差**（注意：这只能说明"没测出"，不等于"一定没有"）。

### 2.2 本地 HF / ModelScope 缓存候选

| 源 | 抽样对话数 | 抽样字符 | 轮数 mean/med | ≥2 / ≥4 / ≥8 轮 | Δ8gram>0 全体/真多轮 | ΔLCS≥4 | rep3 均值(>0.3占比) | hf12 回复/字符 | 回复中位长 |
|---|---|---|---|---|---|---|---|---|---|
| cjy180k（HF 本地 3/32 shard） | 1500 | 0.11M | 1.00 / 1 | 0.0% | — (单轮) | +0.0% | 0.013 (0.2%) | 1.2% / 1.5% | 34 |
| oasst1_zh（HF 本地全量 40MB） | 1575 | 0.64M | 1.32 / 1 | 40.3% / 0.0% / 0.0% | +2.2% / +2.8% | +3.6% | 0.084 (6.4%) | 47.1% / 41.7% | 106 |
| qiao（ModelScope，274k QA） | 1500 | 0.10M | 1.00 / 1 | 0.0% | — (单轮) | +0.0% | 0.006 (0.1%) | 0.0% / 0.0% | 36 |
| ada_dst（**英文**多轮 1300 段） | 1300 | 3.19M | 14.14 / 13 | 100.0% / 100.0% / 99.1% | **+7.2% / +7.2%** | **+10.5%** | 0.045 (0.0%) | 74.3% / 30.7% | 76 |
| ada_zh（473 条单轮） | 472 | 0.02M | 1.00 / 1 | 0.0% | — (单轮) | +0.0% | 0.003 (0.0%) | 0.0% / 0.0% | 18 |
| ada_alpaca（英文单轮 3672） | 3672 | 0.51M | 1.00 / 1 | 0.0% | — (单轮) | +0.0% | 0.034 (0.1%) | 30.3% / 8.0% | 64 |

结论：本地 HF/ModelScope 缓存里**没有一个中文多轮源**。
- `cjy180k` **确认是语音对话**（parquet 里 `instruct_wav`/`prediction_wav` 是音频字节），文本只有单轮；
  32 个 shard 只下了 3 个，且**读它很慢**（1500 行要 122 秒，因为要跳过音频列）。**建议删掉或不再续下**。
- `oasst1_zh` 是 OASST1 全量里筛出的 **zh 子集，抽样里只有 ~2.3%**（1907/82335 行），
  1500 段里 60% 还是单轮、没有任何 ≥4 轮的——**不是可用量级**。
- `ada_dst` 的英文多轮结构确实很好（mean 14.1 轮、承接 +7.2pp），但**是英文**，对中文模型只能做英文对话能力，不解决中文多轮缺口；`ada_zh` 只有 473 条且被拍平成单轮。

### 2.3 本次抽样下载/在线抽样的**可下载候选**
（`unknown_zh_38k`、LCCC 是本地下载后实测；Belle/MOSS/WildChat 走 datasets-server 随机行抽样）

| 源 | 抽样对话数 | 抽样字符 | 轮数 mean/med | ≥2 / ≥4 / ≥8 轮 | Δ8gram>0 全体/真多轮 | ΔLCS≥4 | rep3 均值(>0.3占比) | hf12 回复/字符 | 回复中位长 |
|---|---|---|---|---|---|---|---|---|---|
| **sharegpt_unknow_zh_38k（真实下载）** | 1500 | 2.30M | 4.13 / 2 | 59.5% / 31.9% / 13.5% | **+6.4% / +6.7%** | **+15.1%** | 0.145 (14.1%) | 18.0% / 2.7% | 274 |
| lccc_base（60MB gz 前缀解压，随机行） | 1500 | 0.07M | 1.57 / 1 | 43.7% / 2.3% / 0.0% | +0.1% / +0.0% | +0.8% | 0.008 (0.9%) | 0.0% / 0.0% | 9 |
| lccc_base（未压缩镜像**随机区间**） | 1500 | 0.07M | 1.59 / 1 | 45.0% / 3.0% / 0.1% | +0.0% / +0.1% | +0.9% | 0.009 (0.6%) | 0.0% / 0.0% | 9 |
| **BelleGroup/multiturn_chat_0.8M** | 600 | 0.23M | 2.77 / 2 | **100.0% / 21.8%** / 0.3% | +3.1% / +2.7% | **+8.3%** | **0.033 (0.2%)** | **4.6% / 0.6%** | 90 |
| YeungNLP/moss-003-sft-data | 600 | 3.45M | **5.91 / 6** | **100.0% / 96.7% / 13.7%** | **+18.5% / +18.9%** | +8.9% | **0.342 (59.0%)** | 63.6% / **17.7%** | 712 |
| **benchang1110/WildChat-Chinese** | 800 | 1.67M | 2.47 / 1 | 46.5% / 21.2% / 5.0% | **+9.2% / +11.2%** | **+14.5%** | 0.173 (17.7%) | 23.1% / 4.1% | 430 |

**读法**：
- **结构（轮数）**：MOSS > unknow_zh_38k > Belle > WildChat > LCCC。
- **承接性**：MOSS（+18.9pp）> WildChat（+11.2pp）≈ unknow_zh_38k（+6.7pp）> Belle（+2.7pp）> LCCC（≈0）。
- **质量（套话/复读）**：Belle（hf12 字符覆盖 0.6%）最好；unknow_zh_38k（2.7%）、WildChat（4.1%）
  中等；MOSS（17.7% 字符覆盖 + rep3 均值 0.34 + **59% 的回复 rep3>0.3**）最差。
- 因此 **MOSS 的"承接"有一部分是"复读"**：它长（回复中位 712 字符）、模板句反复出现，
  Δ8gram 高不完全是"真承接"。**不推荐**。

---

## 3. 可下载候选清单（重点）

抽样一律**随机区间/随机行**；`<200MB` 的已实测抽样，>200MB 的只给命令和体积，等拍板。

| # | 名称 | 路径（HF / ModelScope） | 规模 | 轮次结构（实测） | 质量（实测） | 许可 | 下载体积 | 建议 |
|---|---|---|---|---|---|---|---|---|
| 1 | ShareGPT-Chinese-English-90k | HF `shareAI/ShareGPT-Chinese-English-90k`；MS `AI-ModelScope/ShareGPT-Chinese-English-90k` | 90k 双语；中文两文件 `unknow_zh_38k`(38,557 段) + `common_zh_70k`(70k 段) | **59.5% ≥2 轮 / 31.9% ≥4 轮 / 13.5% ≥8 轮，mean 4.13** | rep3 0.145（14.1% >0.3）；hf12 覆盖 2.7% | **Apache-2.0** | `unknow_zh_38k.jsonl` **150.7MB**；`common_zh_70k.jsonl` 482.6MB | ★★★ 首选 |
| 2 | BelleGroup multiturn_chat_0.8M | HF `BelleGroup/multiturn_chat_0.8M` | 831,036 段，990MB，JSONL | **100% ≥2 轮，21.8% ≥4 轮，mean 2.77** | rep3 0.033（0.2% >0.3）；hf12 0.6% —— **最干净** | **GPL-3.0** | 990MB（可随机区间抽样 100MB） | ★★★ 质量最佳（但是英译中） |
| 3 | WildChat-Chinese（WildChat-1M 中文子集） | HF `benchang1110/WildChat-Chinese`（上游 `allenai/WildChat-1M`，ODC-BY） | 122,958 段，435.8MB（2 parquet） | 46.5% ≥2 / 21.2% ≥4 / 5.0% ≥8，mean 2.47 | rep3 0.173（17.7%）；hf12 4.1%；**toxic 8.25%** | 派生集**未标许可**（上游 ODC-BY，需署名） | 435.8MB | ★★☆ 承接最强但需过滤 |
| 4 | LCCC-base | HF `silver/lccc`（`thu-coai/lccc` 的脚本指向它；未压缩镜像 `hysi-lab/lccc_large`） | 6,820,506 段（train），933MB 文本，gz 下载 369.9MB | 43.9% ≥2 轮 / 2.7% ≥4 轮；**回复中位 9 字符** | rep3 0.008；hf12 **0%**（几乎无跨回复套话） | HF 标 **MIT**（上游 CDial-GPT 历史上是"仅供研究"，[推断] 有歧义） | 369.9MB（gz）/ 912MB（未压缩） | ★★☆ 自然闲聊，但短、承接弱 |
| 5 | sharegpt_gpt4（zh 子集） | HF `shibing624/sharegpt_gpt4`；MS `AI-ModelScope/sharegpt_gpt4` | 103,415 段（3 文件） | `sharegpt_zh_38K_format.jsonl` 与 #1 的 `unknow_zh_38k` **同源（首轮哈希重合 4983/4984 = 100%）** | 同 #1 | **CC-BY-4.0** | `sharegpt_zh_38K_format.jsonl` 153.8MB | ★☆ **与 #1 重复，别下两遍** |
| 6 | moss-003-sft-data | HF `YeungNLP/moss-003-sft-data` | 670,948 段，8.7GB | **mean 5.91 / 96.7% ≥4 轮**（结构最好） | **rep3 0.342、59% 回复 >0.3、hf12 17.7%**（模板化最重） | **未标注** | 8,718MB | ✗ 超 2GB + 质量差 |
| 7 | GLM-Open-Dialogue-Chinese-Dataset | HF `svjack/GLM-Open-Dialogue-Chinese-Dataset`（另有 v1/v2） | train csv 123.0MB + valid 11.4MB | schema 是 `source_text`（含"上下文：…答案："模板）+`target_text`，**单轮续写** | —— | 未标注 | 134.4MB | ✗ 本地 `glm_dialogue.txt` 已来自它 |
| 8 | shareAI-Llama3-DPO-zh-en-emoji | MS `shareAI/shareAI-Llama3-DPO-zh-en-emoji` | 合并文件 7.9MB + 分片，总计 ~16MB | **单轮 DPO**（`{question, answer_zh, ...}`） | —— | **Apache-2.0** | 7.9MB（已实测下载） | ✗ 单轮 |
| 9 | Slim-LCCC-zh | HF `lorinma/Slim-LCCC-zh` | `LCCC_sharegpt_10K.jsonl` 2.7MB | 未能抽样 | —— | 未标注 | 2.7MB | ✗ **gated（HTTP 401）**，无 token 下不了 |

### 3.1 你要我特别核实的四个

| 数据集 | 结论 | 证据 |
|---|---|---|
| `thu-coai/lccc` | **仓库里没有数据**，只有 `README.md` + `lccc.py`（加载脚本），脚本 `_URLS` 指向 `silver/lccc`。SDK/`datasets` 加载会去 `https://huggingface.co/datasets/silver/lccc/resolve/main/lccc_base_train.jsonl.gz` 拉数据 | **[实测]** HF API tree 只有 3 个 0MB 文件；读了 `lccc.py` 源码 |
| `lorinma/Slim-LCCC-zh` | **gated auto**：未登录直接 401 `Access to dataset ... is restricted`。要下必须先登录 HF 并接受条款 | **[实测]** `curl -L` 返回 401 + 明确文案 |
| `svjack/GLM-Open-Dialogue-Chinese-Dataset` | 能下（range 请求 206），`train_all_df.csv` 123.0MB，列 `source_text,target_text,cate`；**是单轮续写模板**（`根据上下文，得到后续的对话…[SEP]…答案：`），不是多轮；本地 `glm_dialogue.txt` 已用它（只见 `cate=='gen'`） | **[实测]** head 抽样 + 读了 `download_glm_chat.py` |
| MS `shareAI/shareAI-Llama3-DPO-zh-en-emoji` | 能下。文件清单：`merged_dpo_zh_emoji.jsonl` 7.9MB、`merged_dpo_zh_emoji_for_firefly.jsonl` 9.5MB、`0_dpo_loji/1_dpo_zhihu_1/1_dpo_zhihu_2/2_dpo_ruozhiba` 等；**打开 merged 文件确认是单轮** `{question, answer_zh}`（emoji 风格） | **[实测]** SDK 列文件 + 下载 7.9MB 全量并解析 |

### 3.2 下载命令（照抄）

**A. HF 单文件（`hf` CLI 已实测可用，版本 1.29）**
```bash
cd /home/vesita/coding/my/nanoSeek
# 首选：150MB，Apache-2.0
.venv/bin/hf download shareAI/ShareGPT-Chinese-English-90k \
    sharegpt_jsonl/unknow_zh_38k.jsonl --repo-type dataset --local-dir data/external/sharegpt90k
# 追加：482MB（同源更大子集）
.venv/bin/hf download shareAI/ShareGPT-Chinese-English-90k \
    sharegpt_jsonl/common_zh_70k.jsonl --repo-type dataset --local-dir data/external/sharegpt90k
# LCCC-base（369.9MB gz）
.venv/bin/hf download silver/lccc lccc_base_train.jsonl.gz --repo-type dataset --local-dir data/external/lccc
```
> 未登录也可下（会提示 `unauthenticated requests`）；`lorinma/*` 这类 gated 仓库必须先 `hf auth login`。

**B. ModelScope（本项目 `temp/download.py` 的写法 / 直链）**
```python
# 方式一：SDK 流式（download_dialogue.py 里的写法）
from modelscope.msdatasets import MsDataset
ds = MsDataset.load('AI-ModelScope/ShareGPT-Chinese-English-90k', split='train', use_streaming=True)
for ex in ds:   # 取到需要的条数就 break，只传实际读到的字节
    ...
```
```bash
# 方式二：直链（本次对 shareAI / qiaojiedongfeng / adafny123 都实测成功）
base=https://modelscope.cn/api/v1/datasets
curl -sL -o unknow_zh_38k.jsonl \
  "$base/AI-ModelScope/ShareGPT-Chinese-English-90k/repo?Source=SDK&Revision=master&FilePath=sharegpt_jsonl%2Funknow_zh_38k.jsonl"
```

**C. >200MB 单文件：随机区间抽样（JSONL 专用，本次在 `hysi-lab/lccc_large` 与 Belle 上实测可用）**
```bash
SZ=990032546; CHUNK=10000000
for i in 1 2 3 4 5; do
  OFF=$((RANDOM * 32768 % (SZ - CHUNK)))
  curl -sL -H "Range: bytes=$OFF-$((OFF+CHUNK-1))" \
    "https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M/resolve/main/multiturn_chat_0.8M.json" \
    >> belle_sample.jsonl
done
# 每个区间丢弃首行残片（第一个 \n 之前的内容）
```
> ⚠️ **只对 JSONL/逐行 JSON 有效**。`moss-003-sft-data.jsonl`、`BelleGroup` 的 `.json`（实测是 JSONL）、
> `hysi-lab/lccc_large` 的 `lccc_base_train.jsonl` 都适用；`c4` 那种单行巨 JSON 不适用。

**D. 快速核验（不下载）**
```bash
# 行数/字节
curl -s "https://datasets-server.huggingface.co/size?dataset=BelleGroup%2Fmultiturn_chat_0.8M"
# 随机行
curl -s "https://datasets-server.huggingface.co/rows?dataset=BelleGroup%2Fmultiturn_chat_0.8M&config=default&split=train&offset=123456&length=100"
```

---

## 4. 推荐方案：补 5~20M token 的中文多轮对话

### 4.1 我的判断（先说量级）

- 需要的是**"多轮对话段数"**，不是无脑堆 token。按本次实测的每段字符数换算：
  `unknow_zh_38k` ≈ 1487 字符/段、`Belle` ≈ 380、`WildChat-zh` ≈ 2093。
  所以 **5M token ≈ 2.4k~13k 段；20M token ≈ 9.6k~53k 段**（取决于用哪个源、按字符还是按 CJK 字符算）。
  换成"多轮对话段数"这个更有意义的单位：**要补的不是 5~20M token，而是 ~1 万~5 万段真多轮对话**。
- 现有训练预算（阶段一，step 22000→65000）共约 `43000 × 8192 ≈ 352M` 个采样 token。
  语料若从 937.8M 涨到 ~957M，新增 20M 只占 **2.1%**，模型在剩余步数里大约只见到
  **~7.5M** 个多轮 token。
- ⇒ **建议按上限补（~15~20M token），并在阶段二（对话退火）对多轮样本做 3~5× 上采样**，
  而不是把语料堆到几百 M 去"稀释"它。多轮推进是**行为目标**，不是通用知识，靠比例是推不动的。
- **验收口径**：用 `scripts/ckpt_paired_eval.py` 在**留出的多轮窗口**上做配对 CE（同批、同位置），
  再用 `inference/scripts/eval_multiturn.py --style=ab` 看收尾率/自开轮次率。
  单点 val 在这个量级（Δ<0.05 nats）上分辨不出来（AGENTS §5.1）。

### 4.2 推荐组合（按优先级）

| 优先级 | 源 | 取多少 | 下载量 | 预期轮次结构 | 说明 |
|---|---|---|---|---|---|
| **P0** | `shareAI/ShareGPT-Chinese-English-90k :: unknow_zh_38k.jsonl` | **全量 38.5k 段**（≈57M 字符，其中 CJK 64.6% ≈ 37M 字符） | **150.7MB** | 59.5% ≥2 轮、31.9% ≥4 轮、mean 4.13 | 单文件、Apache-2.0、已实测下载+跑指标；承接 ΔLCS≥4 **+15.1pp**。若只要 5~20M token，随机取前 30~60% 的段即可 |
| **P1** | `BelleGroup/multiturn_chat_0.8M` | **随机区间抽 60~100MB**（≈15~25 万段，≈6~10M 字符）或全量 990MB | 60~100MB（抽样） | 100% ≥2 轮、21.8% ≥4 轮、mean 2.77 | 质量最干净（rep3 0.2%、hf12 0.6%），补"多轮感"；**但是英译中，有翻译腔风险**，建议只占混合的 30~40% |
| **P2（可选）** | `benchang1110/WildChat-Chinese` | 过滤后全量（`toxic==False`） | 435.8MB | 46.5% ≥2 轮、21.2% ≥4 轮 | **真实人类 ↔ ChatGPT 多轮**，承接 Δ8gram +11.2pp 最强；需去 PII/去毒，且上游 ODC-BY 要署名 |
| **P3（可选）** | `silver/lccc :: lccc_base_train.jsonl.gz` | 随机区间抽 30~60MB | 30~60MB | 43.9% ≥2 轮，极短轮 | 只用来补"自然人类口语"，**对多轮推进贡献≈0**；且必须**去掉分词空格** |

**预期混合结果**（P0 60% + P1 40%，按 token）：
`≥2 轮 ≈ 75~80%`、`≥4 轮 ≈ 25~28%`、`每段 assistant 轮数 mean ≈ 3.3~3.6`，
且套话覆盖（hf12 字符）从 `multi_turn` 的 **85.9%** 降到 **~2%** 量级。

### 4.3 必须做的清洗（否则会把老毛病带回来）

1. `unknow_zh_38k` / `sharegpt_*` 带 **GPT-4 腔**：markdown（`**`、`###`、列表）、emoji、
   "作为一个AI"、拒绝模板。项目此前被 `glm`/`multi_turn` 的套话坑过一次，建议按
   **hf12 高频 12-gram 黑名单 + emoji 比例 + markdown 标记**过滤（脚本已有 `clean_corpus.py` 可扩）。
2. `BelleGroup` 是**英译中**：用"的/了/吗"密度、`你我他` 直译腔、专有名词音译等启发式抽检；
   建议抽样人读 50 段再决定比例。
3. `LCCC` 是 **jieba 分词后空格连接**（`你 去 那儿 竟然 不喊 我`）——**必须剥掉所有空白**，
   否则模型会学成"每个字之间加空格"。
4. `WildChat` 有 `toxic`(8.25%)/`redacted`/`hashed_ip`/`country` 字段：按 `toxic==False` 过滤、
   去重、去掉 URL/邮箱/电话；`state/country/hashed_ip` 不要进语料。
5. 所有源都要**跨源去重**：`shibing624/sharegpt_gpt4` 与 `shareAI/...` 的 zh 子集实测
   **100% 同源**（中段随机 20MB 抽样 4984 段，首轮 user 哈希命中 4983）。

### 4.4 不建议做的

- **不要把 `moss-003-sft-data`（8.7GB）当多轮主力**：结构最好（mean 5.91 轮）但 rep3 均值 0.342、
  59% 回复 rep3>0.3、12-gram 覆盖 17.7%，且许可未标注 —— 会直接强化"又长又套"的毛病。
- **不要再下 `CJY/Chinese-Dialogue-180k`**：语音对话、文本单轮、读一个 shard 要 90+ 秒。
- **不要再下 `adafny123/visual_noval_atri`**：英文多轮（1300 段）+ 中文 473 条拍平单轮，中文量级可忽略。

---

## 5. 操作风险（本次踩到的，交接必读）

1. **ModelScope hub 缓存会被并行会话清理**：13:04 还在的
   `qiaojiedongfeng/qiaojiedongfeng/train.jsonl`（58.9MB）和 `adafny123/fine-tune_dst.json`（7MB）
   到 13:12 就没了。⇒ 任何依赖 `~/.cache/modelscope/hub/datasets/downloads/` 的探测/清洗脚本
   **必须先把要用的文件拷到自己目录**（本次已把 4 个文件拷进 `temp/zh_dialogue_probe/dl2/`）。
2. **`data/chinese/*.txt` 在本次任务中途被并行会话搬到了 `data/chinese/raw_all/`（软链到
   `/home/vesita/datasets/NLP/`）**（就是 v3 清洗那条线在做）。本报告所有 `data/chinese/` 数字
   都对应 `raw_all/` 里的同一批文件（探测脚本已加 `raw_all/` 回退）。**谁要引用这些文件，
   请用 `data/chinese/raw_all/` 或符号链接解析后的真实路径**。
3. **不要用 `~/.cache/huggingface/hub` 里那几个"空壳"**：`thu-coai/lccc`、`lorinma/Slim-LCCC-zh`、
   `svjack/GLM-Open-Dialogue-Chinese-Dataset` 的本地目录只有 README/refs，**没有数据**。
4. 抽样下载一共只用了：`unknow_zh_38k.jsonl` 150.7MB（全量）+ LCCC 未压缩镜像 3×20MB 随机区间
   + LCCC gz 前 60MB + 小文件若干，均 <200MB/文件。**更大的（990MB Belle、435.8MB WildChat、
   369.9MB LCCC gz）没有下**，等你/用户拍板。

---

## 6. 复现

```bash
cd /home/vesita/coding/my/nanoSeek
# 全量本地测量（25 源，~5 分钟；每测完一源落盘 metrics.json）
.venv/bin/python temp/zh_dialogue_probe/measure.py --all --n 1500 \
    --out temp/zh_dialogue_probe/metrics.json
# 已下载抽样文件（sharegpt 38k / LCCC 前缀 / LCCC 随机区间）
.venv/bin/python temp/zh_dialogue_probe/measure_downloaded.py all
# datasets-server 随机行抽样（Belle / MOSS / WildChat-zh，不下载）
.venv/bin/python temp/zh_dialogue_probe/dss_sample.py belle 600
.venv/bin/python temp/zh_dialogue_probe/dss_sample.py moss 600
.venv/bin/python temp/zh_dialogue_probe/dss_sample.py wildchat_zh 800
# 渲染表格
.venv/bin/python temp/zh_dialogue_probe/make_table.py temp/zh_dialogue_probe/evidence_all.json
```

**证据强度总述**：全部为 **[实测] 单次抽样**（每源 600~1500 段/3672 条），抽样方式为随机偏移/
随机行组/随机行号；对照函数在每次运行都过 `ok=True` 的已知答案检查。**没有做多种子重复**，
所以 <1pp 的差异不要当结论；本报告里用到的差异都在 **+2.7pp ~ +18.9pp** 量级。

**与背景材料不一致的两处（如实记录）**：
1. 背景说 `multi_turn_dialogue.txt` 的 `rep3>0.3` 回复占 **4.6%**，本次随机偏移抽样测得 **1.4%**。
   差异很可能来自抽样方式（本次按字符预算随机抽助手回复），**不影响结论方向**（两者都低）——
   真正指出模板问题的是新增的 **hf12 字符覆盖 85.9%**，`rep3` 这个指标对"跨回复套话"不敏感。
2. 背景说本地"真正的多轮对话只有 `multi_turn_dialogue.txt`"，本次实测 **`kdconv`（99.4% ≥2 轮）
   和 `glm`（72.7% ≥2 轮）也是真多轮**，只是体量小（0.94M / 5.05M 字符，合计不到 `multi_turn`
   的 13%），在按 token 计的占比里确实可以忽略——两种说法不矛盾。

