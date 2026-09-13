# 语料清洗报告（clean_corpus.py）

- 时间：2026-09-13T14:04:20+08:00
- 模式：**--apply（已写数据文件）**
- 源目录（只读）：`data/chinese/new_sources`
- 输出目录：`data/chinese/clean_v3`
- 命令行：`data/chinese/clean_corpus.py --src data/chinese/new_sources --dst data/chinese/clean_v3 --apply --dedup-blocks --ngram-size 12 --max-ngram-freq 50 --max-ngram-cover 0.5 --ngram-max-candidates 20000000 --filter-nondialogue-rep3 --nondialogue-max-rep3 0.4`
- 启用规则：`--dedup-blocks`（block 级精确去重，全局）; `--max-ngram-freq 50 --ngram-size 12 --max-ngram-cover 0.5`（短语级 K-gram）; `--filter-nondialogue-rep3`（非对话块 rep3 阈值 0.4，显式 G）

## 全局

| 指标 | 清洗前 | 清洗后 | 变化 |
|---|---|---|---|
| block 数 | 369,505 | 366,044 | -3,461（0.94%） |
| 字符数 | 119,968,254 | 118,896,142 | -1,072,112（0.89%） |
| 高频 12-gram 覆盖率（回复字符） | 3.8275% | 3.5826% | -0.2449 pp |

（heavy 集合：频次 > 50 的 9,883 个 12-gram；覆盖口径 = 回复里被至少一个 heavy K-gram 覆盖的字符占比。回复字符 91,933,863 → 91,568,719）

## 各规则命中 block 数

| 规则 | 命中 |
|---|---|
| dedup_blocks | 2,161 |
| max_reply_rep3 | 0 |
| max_ngram_cover | 1,350 |
| min_reply_chars | 0 |
| nondialogue_rep3 | 0 |

> 一条 block 可能同时命中多条规则，所以"命中数之和"≥ 被丢的 block 数。

## 逐来源

| 文件 | blocks 进 | blocks 出 | 字符 进 | 字符 出 | dedup | rep3 | ngram | min | 非对话rep3 |
|---|---|---|---|---|---|---|---|---|---|
| belle_multiturn.txt | 53,689 | 53,686 | 24,860,266 | 24,859,455 | 0 | 0 | 3 | 0 | 0 |
| qa_knowledge.txt | 274,111 | 270,984 | 19,233,890 | 19,113,563 | 2,075 | 0 | 1,062 | 0 | 0 |
| sharegpt_zh_38k.txt | 38,537 | 38,247 | 60,874,410 | 60,689,375 | 86 | 0 | 244 | 0 | 0 |
| wildchat_zh.txt | 3,168 | 3,127 | 14,999,688 | 14,233,749 | 0 | 0 | 41 | 0 | 0 |
| **合计** | **369,505** | **366,044** | **119,968,254** | **118,896,142** | **2,161** | **0** | **1,350** | **0** | **0** |

（默认：不含 `模型：` 的非对话 block 只受 `--dedup-blocks` 约束、**不受** rep3 / min-reply-chars / K-gram 覆盖率约束；要连非对话块一起按 rep3 过滤，显式加 `--filter-nondialogue-rep3`。）

- 源文件数：4
