# 中文语料数据集重建报告（char-level v2）

- 日期：2026-09-10
- 仓库：`/home/vesita/coding/my/nanoSeek`
- 执行环境：`.venv/bin/python`（Python 3.12.14 / tokenizers 0.23.1 / numpy 2.5.2 / torch 2.9.1+rocm6.4），**全程 CPU，未使用 GPU**
- 本报告所有陈述按三类标注：
  - **[实测]**：本次跑出来的数字/日志，命令可复现
  - **[推断]**：由实测数字推出来的结论，未直接观测
  - **[建议]**：只给建议，本次没有执行任何混合比例调整或来源删减

---

## 0. 结论摘要

1. **[实测]** 修前 val 确实是"双标准"切分：`DIALOGUE_FILES`（实际只命中 4 个已存在文件）按 10% 抽 val，其它 21 个来源按 1% 抽 val。后果是 val 里对话严重超配。
2. **[实测]** 修后（统一 1%）有效 token 占比 train/val 从 **4.64 倍**收敛到 **1.02 倍**；终止符密度从 **3.15 倍**收敛到 **1.06 倍**；全空窗口占比从相差 30 个百分点收敛到 0.3 个百分点。
3. **[实测]** bigram 独立判据：修前 val 与 train 后半段 CE 差 **−0.2462 nats**，修后 **−0.0275 nats**（收缩 8.9 倍，已低于任务书目标 0.15）；k∈{0.01,0.1,1.0} 不改变结论。
4. **[实测+推断]** 还发现一个**任务书没提到的旧 train 缺口**：旧 `train_char.bin` 比"当前语料按切分逻辑应产出"少 **94,773 个 block / 约 1.0 亿字符**，且缺口全在 train 侧。**[推断]** 该缺口可由 `--source-ratio c4_zh=0.5`（只降 train）**精确复现**（block 总数、首个 `<eos>` 位置都对上）。旧构建的**字面命令行没有落盘**，不可恢复。
5. **[实测]** 顺带修掉 `encode_to_bin` 一个会导致 OOM 的真 bug（详见 §7）。
6. **[实测]** 原 `train_char.bin` / `val_char.bin` **一个字节都没动**（sha256 与开工前备份一致，见 §10）。

产物：

| 文件 | 大小 | token 数 |
|---|---:|---:|
| `data/chinese/train_char_v2.bin` | 1,875,546,180 B（1788.7 MiB） | 937,773,090 |
| `data/chinese/val_char_v2.bin` | 18,835,890 B（18.0 MiB） | 9,417,945 |
| `data/chinese/meta_char_v2.pkl` | 85 B | — |
| `data/chinese/manifest_v2.json` | 13,517 B | — |

---

## 1. 问题与修复（`data/chinese/prepare.py`）

### 1.1 val 双标准（本次核心 bug）

修前逻辑（简化）：

```
if fn in DIALOGUE_FILES:  # 硬编码 5 个名字
    10% 进 val
elif args.val_all:
    1% 进 val
```

**[实测]** `DIALOGUE_FILES` 里的 `agent_dialogue.txt` 在 `data/chinese/` 中不存在，所以真正吃到 10% 的只有 4 个文件：`dailychat_dialogue.txt`、`muice_dialogue.txt`、`multi_turn_dialogue.txt`、`zhihu_kol_dialogue.txt`。其余 21 个来源（含 573 MB 的 `deepseek_r1_distill_dialogue.txt`、466 MB 的 `c4_zh.txt`、456 MB 的 `wikipedia_cn.txt`、267 MB 的 `qwen3_235b_distill_dialogue.txt`）全部只按 1% 进 val。

**修复**（保持向后兼容）：
- 新增模块级函数 `split_one_source(fn, blocks, args, src_ratio)`，把切分逻辑从 `main()` 里抽出来，审计脚本可复用。
- 分支顺序改为 **`args.val_all` 优先**：传了 `--val-all` 时，**所有来源统一用 `--val-ratio`**，`DIALOGUE_FILES` 不再有第二套标准。
- **不传 `--val-all` 时行为与旧版逐字节一致**（已用单测验证：对 `multi_turn_dialogue.txt` / `c4_zh.txt` / `wikipedia_cn.txt`，新旧切分结果完全相同）。
- 去掉 val 侧 `max(1, ...)`：block 数 < 100 的小源不再被整段抽进 val（例如 `identity_dialogue.txt` 12 个 block，1% 取整后 val=0，全部留 train）。

### 1.2 新增开关（不静默改语义）

- `--out-prefix PREFIX`：产物加后缀。`--out-prefix v2` → `train_char_v2.bin` / `val_char_v2.bin` / `meta_char_v2.pkl` / `manifest_v2.json`。默认空 = 旧文件名（不覆盖既有数据集）。
- `--seed N`：可选，固定 `annotate_replies` 的随机种子，使重建可复现；不传保持旧行为（不设种子）。
- `write_manifest` 现在记录**完整命令行 `argv`** + `val_all` / `val_ratio` / `source_ratio` / `char_level` / `seed` / `out_prefix`，并输出逐来源 `source_breakdown`。

### 1.3 文件头注释

已在 `prepare.py` 头部写明：改了什么、为什么、修复前后语义差异、`--source-ratio` 仍只作用于 train 的注意事项。

---

## 2. 旧 train 的不可考降采样（重要发现）

### 2.1 [实测] 缺口

| 量 | 数值 |
|---|---:|
| 当前 `data/chinese/` 全部 `.txt` 的 block 总数（按 `\n\n` 切分并 strip） | 1,582,855 |
| 旧 `manifest.json` 的 `train_samples + val_samples` | 1,450,526 + 37,556 = 1,488,082 |
| 缺口 | **94,773 个 block** |
| 旧 `train_char.bin` token 数 | 835,846,272 |
| 同语料、不做降采样重建后的 train 字符数 | 944,122,571（≈ 1.0 亿字符缺口） |

**[实测]** 缺口 100% 在 train 侧：`val_samples` 恰好等于"双标准 val"的预测值（我按每个文件 `int(N × ratio)` 预测 37,553，旧 manifest 37,556，差 3 属取整误差）。同目录备份 `_backup_barv_20260908_222540`（22:21 生成，args 无 `val_all`）也呈现同样的 ~95,730 block 缺口。

### 2.2 [推断] 缺口来源 = `c4_zh` train 侧 ×0.5

- 旧词表/旧文件里 `zhuangxialie.txt` 已不存在，且旧 `manifest.json` 的 `source_files` 里也没有它——所以它不可能是缺口来源。
- 用 `--source-ratio c4_zh=0.5` 复现切分，预测 `train_samples=1,450,529`、`val_samples=37,553`，**合计 1,488,082，与旧 manifest 完全一致**（不降采样则预测 1,545,302，相差 94,776）。
- 用首个 `<eos>` 位置独立验证：旧 `train_char.bin` 中首个 id=128 出现在 **96,427,430**。train 按文件名顺序拼接，`c4_zh.txt` 第一、`classical_poetry.txt` 第二、`code_alpaca_dialogue.txt` 第三；只有对话行才会产生 `<eos>`。若 c4 全量应在 ~191.3M 处，若 c4 砍半应在 ~96.19M 处。实测 96.43M → **c4 被砍半**。
- 旧 val 里的 c4 字符数（1,842,739）与新 val 完全相同 → 降采样**只作用于 train**，与 `--source-ratio` 的现有语义一致。

**结论**：旧构建的命令行**没有**任何落盘记录（shell history 为空、`manifest.json` 未记录 `source_ratio`、目录里没有构建日志），字面命令**不可考/不可复现**；但其**效果**是 `c4_zh` train 侧约 0.5 的降采样。这属于 **[推断]**：block 总数与首个 `<eos>` 两个独立证据都吻合，但我没有直接看到那条命令。

---

## 3. 来源占比表

字符数为**标注前**的 raw 字符（block 拼接、不含 `\n\n` 分隔符）。占比按各自 split 内部归一。

### 3.1 A. 修前来源表（双标准 val + `c4_zh` train ×0.5）

train 832,333,518 字符 / val 15,935,661 字符

| 来源 | 大小MB | blocks | train字符 | train占比 | val字符 | val占比 |
|---|---:|---:|---:|---:|---:|---:|
| deepseek_r1_distill_dialogue.txt | 573.4 | 109,999 | 255,231,274 | 30.66% | 2,577,288 | 16.17% |
| wikipedia_cn.txt | 456.3 | 253,054 | 183,711,218 | 22.07% | 1,799,660 | 11.29% |
| qwen3_235b_distill_dialogue.txt | 266.7 | 109,999 | 134,611,323 | 16.17% | 1,439,002 | 9.03% |
| c4_zh.txt | 466.2 | 191,459 | 94,081,765 | 11.30% | 1,842,739 | 11.56% |
| multi_turn_dialogue.txt | 126.9 | 100,000 | 40,774,164 | 4.90% | 4,530,529 | 28.43% |
| coig_wiki_dialogue.txt | 107.5 | 194,534 | 40,058,407 | 4.81% | 406,930 | 2.55% |
| zhihu_kol_dialogue.txt | 73.4 | 136,934 | 24,559,562 | 2.95% | 2,723,518 | 17.09% |
| code_alpaca_dialogue.txt | 24.2 | 148,512 | 24,316,134 | 2.92% | 251,949 | 1.58% |
| lccc_dialogue.txt | 21.6 | 150,000 | 7,683,873 | 0.92% | 75,584 | 0.47% |
| coig_other_dialogue.txt | 13.4 | 30,348 | 5,388,909 | 0.65% | 43,670 | 0.27% |
| glm_dialogue.txt | 13.0 | 42,530 | 4,916,689 | 0.59% | 49,784 | 0.31% |
| coig_cqia_dialogue.txt | 11.6 | 47,191 | 4,424,412 | 0.53% | 45,319 | 0.28% |
| gsm8k_cot_dialogue.txt | 4.6 | 7,477 | 4,166,488 | 0.50% | 40,846 | 0.26% |
| classical_poetry.txt | 5.8 | 24,556 | 2,106,588 | 0.25% | 21,982 | 0.14% |
| coig_code_dialogue.txt | 2.1 | 4,907 | 1,001,554 | 0.12% | 5,672 | 0.04% |
| kdconv_dialogue.txt | 2.5 | 2,438 | 926,030 | 0.11% | 9,092 | 0.06% |
| 水浒传.txt | 2.5 | 6,824 | 832,032 | 0.10% | 11,662 | 0.07% |
| 红楼梦.txt | 2.4 | 2,654 | 831,685 | 0.10% | 7,214 | 0.05% |
| coig_logic_dialogue.txt | 1.8 | 6,246 | 730,775 | 0.09% | 4,774 | 0.03% |
| 西游记.txt | 1.9 | 3,311 | 650,724 | 0.08% | 7,239 | 0.05% |
| 三国演义.txt | 1.7 | 1,566 | 587,328 | 0.07% | 7,263 | 0.05% |
| coig_math_dialogue.txt | 1.1 | 3,750 | 493,459 | 0.06% | 5,306 | 0.03% |
| muice_dialogue.txt | 0.6 | 3,642 | 200,923 | 0.02% | 23,160 | 0.15% |
| dailychat_dialogue.txt | 0.1 | 912 | 47,682 | 0.01% | 5,434 | 0.03% |
| identity_dialogue.txt | 0.0 | 12 | 520 | 0.00% | 45 | 0.00% |

**畸高/畸低 [实测]**：
- `multi_turn_dialogue.txt`：train 4.90% / val **28.43%**（放大 5.8 倍）
- `zhihu_kol_dialogue.txt`：train 2.95% / val **17.09%**（放大 5.8 倍）
- `muice_dialogue.txt`：train 0.02% / val 0.15%（放大 7.5 倍）
- 4 个命中 `DIALOGUE_FILES` 的文件合计：train **7.88%** / val **45.70%**（放大 5.8 倍）
- 反向畸低：`deepseek`（30.66% → 16.17%）、`wikipedia`（22.07% → 11.29%）、`qwen3`（16.17% → 9.03%）、`c4_zh`（11.30% → 11.56%）、`coig_wiki`（4.81% → 2.55%）

### 3.2 B. 修后 v2 来源表（所有来源统一 val 1%）

train 933,967,068 字符 / val 9,379,225 字符

| 来源 | 大小MB | blocks | train字符 | train占比 | val字符 | val占比 |
|---|---:|---:|---:|---:|---:|---:|
| deepseek_r1_distill_dialogue.txt | 573.4 | 109,999 | 255,231,274 | 27.33% | 2,577,288 | 27.48% |
| c4_zh.txt | 466.2 | 191,459 | 189,158,879 | 20.25% | 1,842,739 | 19.65% |
| wikipedia_cn.txt | 456.3 | 253,054 | 183,711,218 | 19.67% | 1,799,660 | 19.19% |
| qwen3_235b_distill_dialogue.txt | 266.7 | 109,999 | 134,611,323 | 14.41% | 1,439,002 | 15.34% |
| multi_turn_dialogue.txt | 126.9 | 100,000 | 44,852,730 | 4.80% | 451,963 | 4.82% |
| coig_wiki_dialogue.txt | 107.5 | 194,534 | 40,058,407 | 4.29% | 406,930 | 4.34% |
| zhihu_kol_dialogue.txt | 73.4 | 136,934 | 27,011,301 | 2.89% | 271,779 | 2.90% |
| code_alpaca_dialogue.txt | 24.2 | 148,512 | 24,316,134 | 2.60% | 251,949 | 2.69% |
| lccc_dialogue.txt | 21.6 | 150,000 | 7,683,873 | 0.82% | 75,584 | 0.81% |
| coig_other_dialogue.txt | 13.4 | 30,348 | 5,388,909 | 0.58% | 43,670 | 0.47% |
| glm_dialogue.txt | 13.0 | 42,530 | 4,916,689 | 0.53% | 49,784 | 0.53% |
| coig_cqia_dialogue.txt | 11.6 | 47,191 | 4,424,412 | 0.47% | 45,319 | 0.48% |
| gsm8k_cot_dialogue.txt | 4.6 | 7,477 | 4,166,488 | 0.45% | 40,846 | 0.44% |
| classical_poetry.txt | 5.8 | 24,556 | 2,106,588 | 0.23% | 21,982 | 0.23% |
| coig_code_dialogue.txt | 2.1 | 4,907 | 1,001,554 | 0.11% | 5,672 | 0.06% |
| kdconv_dialogue.txt | 2.5 | 2,438 | 926,030 | 0.10% | 9,092 | 0.10% |
| 水浒传.txt | 2.5 | 6,824 | 832,032 | 0.09% | 11,662 | 0.12% |
| 红楼梦.txt | 2.4 | 2,654 | 831,685 | 0.09% | 7,214 | 0.08% |
| coig_logic_dialogue.txt | 1.8 | 6,246 | 730,775 | 0.08% | 4,774 | 0.05% |
| 西游记.txt | 1.9 | 3,311 | 650,724 | 0.07% | 7,239 | 0.08% |
| 三国演义.txt | 1.7 | 1,566 | 587,328 | 0.06% | 7,263 | 0.08% |
| coig_math_dialogue.txt | 1.1 | 3,750 | 493,459 | 0.05% | 5,306 | 0.06% |
| muice_dialogue.txt | 0.6 | 3,642 | 222,104 | 0.02% | 1,979 | 0.02% |
| dailychat_dialogue.txt | 0.1 | 912 | 52,587 | 0.01% | 529 | 0.01% |
| identity_dialogue.txt | 0.0 | 12 | 565 | 0.00% | 0 | 0.00% |

### 3.3 C. 并排：旧 val / 新 val / 新 train（按新 train 占比降序）

| 来源 | 旧val占比 | 新val占比 | 新train占比 |
|---|---:|---:|---:|
| deepseek_r1_distill_dialogue.txt | 16.17% | 27.48% | 27.33% |
| c4_zh.txt | 11.56% | 19.65% | 20.25% |
| wikipedia_cn.txt | 11.29% | 19.19% | 19.67% |
| qwen3_235b_distill_dialogue.txt | 9.03% | 15.34% | 14.41% |
| multi_turn_dialogue.txt | **28.43%** | 4.82% | 4.80% |
| coig_wiki_dialogue.txt | 2.55% | 4.34% | 4.29% |
| zhihu_kol_dialogue.txt | **17.09%** | 2.90% | 2.89% |
| code_alpaca_dialogue.txt | 1.58% | 2.69% | 2.60% |
| lccc_dialogue.txt | 0.47% | 0.81% | 0.82% |
| coig_other_dialogue.txt | 0.27% | 0.47% | 0.58% |
| glm_dialogue.txt | 0.31% | 0.53% | 0.53% |
| coig_cqia_dialogue.txt | 0.28% | 0.48% | 0.47% |
| gsm8k_cot_dialogue.txt | 0.26% | 0.44% | 0.45% |
| classical_poetry.txt | 0.14% | 0.23% | 0.23% |
| coig_code_dialogue.txt | 0.04% | 0.06% | 0.11% |
| kdconv_dialogue.txt | 0.06% | 0.10% | 0.10% |
| 水浒传.txt | 0.07% | 0.12% | 0.09% |
| 红楼梦.txt | 0.05% | 0.08% | 0.09% |
| coig_logic_dialogue.txt | 0.03% | 0.05% | 0.08% |
| 西游记.txt | 0.05% | 0.08% | 0.07% |
| 三国演义.txt | 0.05% | 0.08% | 0.06% |
| coig_math_dialogue.txt | 0.03% | 0.06% | 0.05% |
| muice_dialogue.txt | 0.15% | 0.02% | 0.02% |
| dailychat_dialogue.txt | 0.03% | 0.01% | 0.01% |
| identity_dialogue.txt | 0.00% | 0.00% | 0.00% |

**[实测] 双标准具体在哪几个源上失真（新 val vs 新 train 几乎相等，旧 val vs 新 train 才是失真的）**：
- `multi_turn_dialogue.txt`：28.43% → 4.82%（新 val），新 train 4.80%
- `zhihu_kol_dialogue.txt`：17.09% → 2.90%（新 val），新 train 2.89%
- `muice_dialogue.txt`：0.15% → 0.02%
- `deepseek`：16.17% → 27.48%（新 val），新 train 27.33%
- `wikipedia`：11.29% → 19.19%（新 val），新 train 19.67%
- `c4_zh`：11.56% → 19.65%（新 val），新 train 20.25%
- `qwen3`：9.03% → 15.34%（新 val），新 train 14.41%

---

## 4. 修前 vs 修后对照表（硬性方法论）

抽样一律**全库随机**（规则 1）：`rng = np.random.default_rng(20260910)`，`ix = rng.integers(0, len(d)-256-2, 3000)`，窗口 256。有效 token 用仓库的 `training.masking.build_assistant_mask(y, [128,130], None)` 计算（`<eos>=128`、`<cont>=130`，见 `char_tokenizer.json`）。

### 4.1 三项中间量（规则 2）

| 指标 | 修前 train | 修前 val | 修前 val/train | 修后 train | 修后 val | 修后 val/train |
|---|---:|---:|---:|---:|---:|---:|
| 有效 token 占比 | 6.0203% | 27.9530% | **4.64×** | 6.3560% | 6.4883% | **1.02×** |
| 全空窗口占比 | 82.9667% | 53.0667% | 0.64× | 83.5333% | 83.2333% | 1.00× |
| 终止符(<eos>+<cont>)密度 | 0.1480% | 0.4669% | **3.15×** | 0.1322% | 0.1401% | **1.06×** |
| <eos> 密度 | 0.1143% | 0.2954% | 2.58× | 0.0958% | 0.1042% | 1.09× |
| <cont> 密度 | 0.0337% | 0.1715% | 5.09× | 0.0363% | 0.0359% | 0.99× |

**[实测]** 修前数字与任务书给的证据同量级（任务书：train 6.29% / val 27.39%，终止符 3.23×；本次：6.02% / 27.95%，3.15×，差异来自随机窗口）。

### 4.2 bigram 交叉验证（规则 3，不受 masking 影响）

在 **train 全量**上建 bigram 计数表（token id 直接建，`(8192,8192) float64`），add-k=0.1，用**同一张表**给 train 后半段和 val 打分。

| 判据 | 修前 | 修后 |
|---|---:|---:|
| CE(train 后半段) | 4.5080 | 4.4622 |
| CE(val) | 4.2617 | 4.4347 |
| **gap = val − train后半段** | **−0.2462 nats** | **−0.0275 nats** |
| gap 绝对值 | 0.2462 | 0.0275（收缩 8.9×，< 0.15 ✅） |

稳健性补充（`verify_bigram_extra.py`）：

| 判据（k=0.1） | 修前 | 修后 |
|---|---:|---:|
| val − train后半段 | −0.2228 | +0.0116 |
| val − train全域随机 | −0.1518 | +0.0439 |
| 来源顺序噪声底：train后半 − train前半 | +0.1379 | +0.0694 |
| k=0.01 / 0.1 / 1.0 下 val − 后半 | −0.218 / −0.223 / −0.229 | +0.016 / +0.012 / +0.009 |

**结论 [实测]**：修后 val 与 train 的 bigram CE 差已收缩到 ≤0.05 nats 量级，远低于 0.15 目标，且对 add-k 不敏感。

### 4.3 [实测] 与任务书给出的 old bigram 绝对值不一致（如实记录，未调参对齐）

- 任务书：train CE 4.9359、val CE 5.7967（**val 更难 +0.86**）
- 本次实现：train后半 4.5080、val 4.2617（**val 更易 −0.2462**）

我的实现细节：表建在 **train 全量**；窗口为该区间内**全库随机** 3000×256；add-k=0.1。
我做了两个对照定位差异，均**未**复现任务书符号：
- add-k 从 0.01 到 100：gap 在 −0.218 ~ −0.253 之间，符号不变；
- 只用 **train 前半段**建表（消除自包含）：后半 4.9733 / val 5.0030，gap **+0.0297**（绝对值远小于 +0.86）；量级接近任务书的 train 4.94，但 val 5.80 仍对不上。

**[推断]** 任务书那组数字大概率用了更小的建表样本/更强的平滑/不同的"后半段"窗口定义（具体未记录，无法复现）。**我没有为了让数字对齐而修改计算**。本报告的核心判据是"同一实现下修前→修后 gap 的收缩"，该收缩在 k、表范围、窗口种子的多种组合下都稳健。

### 4.4 规则 4：测量函数先过已知答案

`verify_dataset.py --selftest`：

```
[selftest] mask.valid: got=0.187500 exp=0.187500 OK
[selftest] mask.empty: got=0.500000 exp=0.500000 OK
[selftest] mask.term:  got=0.062500 exp=0.062500 OK
[selftest] 同数据自比 mask 统计最大差 = 0.0 (应为 0)
[selftest] bigram CE 规则交替=0.1517 vs 乱序=5.5091 OK
[selftest] 空计数表 CE=9.010913 期望=log(8192)=9.010913 OK
[selftest] ALL OK
```

另外 bigram 的"同一份数据自己跟自己比"两次都测得差值 **0.00e+00**（修前、修后各一次）。

---

## 5. val 比例的选择及理由

**选择：`--val-ratio 0.01`（1%），对所有 25 个来源统一。**

理由：
1. **与实跑意图一致**：旧 `manifest.json` 记录 `val_all=true, val_ratio=0.01`，说明 0.01 就是原定比例，只是被双标准破坏。
2. **统计上绰绰有余**：1% ≈ **9,417,945 token** val。按修后有效 token 占比 6.49% 算约 61 万个有效 token；即便只按 1/10 保守估计，val CE 的标准误也在 1e-3 nats 量级，足够稳定跟踪训练。
3. **训练数据损失最小**：val 只占 1%，train 拿 99%；相比 2% 可多保留 ~940 万 token 训练数据。
4. **来源覆盖**：1% 下除 `identity_dialogue.txt`（12 block，取整为 0）外，所有来源在 val 中都有 >0 样本；主要来源每个都有上千个 val block，来源构成已经足够稳（见 §3.3，新 val 与新 train 逐源占比几乎相等）。

**[实测] 代价**：`identity_dialogue.txt` 在新 val 中 0 条（train 仍有 565 字符）。这是"统一比例 + 取消 `max(1)`"的直接后果；如需它在 val 出现，应显式提高比例或给它单独处理，**本次未擅自特殊化**。

### val token 数变小是预期行为

旧 val 16,081,949 token → 新 val 9,417,945 token。**[推断]** 原因是旧 val 用 10% 抽对话源、1% 抽其它源，对话源被放大，val 总量被撑大；统一 1% 后 val 就是"全库的 1%"。这不是 bug。

---

## 6. train 变大是预期行为

新 `train_char_v2.bin` 937,773,090 token（1788.7 MiB）比旧 835,846,272（1594.3 MiB）多约 1.02 亿 token，两个原因：
1. **[实测] 旧 train 被 `c4_zh ×0.5` 砍掉约 1.0 亿字符**（§2），v2 恢复 c4 全量；
2. **[实测]** 统一 val 1% 后，对话源多留在 train（如 `multi_turn` train 从 40.77M → 44.85M 字符）。

**[建议]** 如果目标是"和旧 run 逐 token 可比"，v2 做不到也不应该做；v2 是一个**新数据集**，需要从头训（这一点已与父代理确认，旧基线可以放弃）。

---

## 7. 顺带修复的真 bug：`encode_to_bin` 的 OOM

**[实测]** 修前 `encode_to_bin` 只在"两个 `<eos>`/`<cont>` 之间的整段"处理完之后才 flush。train 按文件名拼接，第一个来源 `c4_zh.txt` 有 ~1.89 亿字符且**不含任何特殊符**，于是整段被塞进 `buf`（Python int 列表）：

- 进程 RSS **7.1 GB**
- 系统可用内存只剩 **469 MB**
- swap 已用 15.39 GB / 空闲 451 MB，系统濒临 OOM

已在 `prepare.py` 中改为在 **1M 字符分块内 flush**，并用已知输入做了**逐字节等价自检**（`ref tokens=64500 new tokens=64500 identical=True`）——输出与旧实现完全一致，只是不再堆积内存。这条已写进 `prepare.py` 文件头注释。

---

## 8. [建议] 语料混合是否需要进一步治理

> 以下全是建议，本次**没有**做任何降采样/删减/加权。

**[实测] v2 train 里占比 >15% 的源：**

| 来源 | 新 train 占比 | 类型 |
|---|---:|---|
| `deepseek_r1_distill_dialogue.txt` | 27.33% | 模型蒸馏对话（573 MB） |
| `c4_zh.txt` | 20.25% | 网页/通用文本（466 MB） |
| `wikipedia_cn.txt` | 19.67% | 百科文本（456 MB） |

紧随其后：`qwen3_235b_distill_dialogue.txt` 14.41%（模型蒸馏对话）。**这 4 个源合计 81.7%。**

按大类聚合 **[实测]**：
- 模型蒸馏对话（deepseek + qwen3）：**41.74%**
- 网页/百科/poetry/books 等非对话文本（c4 + wikipedia + coig_wiki + classical_poetry + 四大名著）：**44.75%**
- 人类自然对话（multi_turn + zhihu + lccc + glm + kdconv + dailychat + muice + identity + 各种 coig 对话 + code_alpaca + gsm8k）：约 **13.51%**

**dev-notes/47 描述的"单一指令格式源占比过半导致碎片拼贴"风险是否还在？**
- **[实测]** 当时的元凶 `zhuangxialie.txt` 已不在 `data/chinese/`，也不在旧 manifest 的 `source_files` 里 → 该类风险当前**不存在**。
- 但现在换成了另一种集中：**蒸馏数据 41.7%**。它不是"指令模板拼贴"，但存在学习 teacher 的措辞/格式（如 `A：/B：`、`<eos>/<cont>` 节奏）而非自然中文的风险；同时**自然人类对话只有约 13.5%**，如果目标仍是聊天模型，这是一个值得注意的偏低项。
- **[实测]** `c4_zh`(20.25%) + `wikipedia_cn`(19.67%) 合计 ~40% 的"书面/百科"文本，对生成风格是"书面化"压力。

**建议（择一或组合，均需单变量 A/B）：**
1. 给任一单源设 15%~20% 上限（例如 `--source-ratio deepseek_r1=0.6`），但注意 `source-ratio` 只作用于 train，会让 val 该源占比偏高——若要严格一致，应同时把比例作用到 val（当前代码不支持，需要新增开关）。
2. 若目标是对话模型，考虑对自然对话源整体上采样（或对蒸馏/百科整体降采样），把"自然对话 ≥ 20%"作为一条可检验的目标。
3. 无论怎么调，都请用**逐源 val 占比表 + 三项中间量**复核，并用 `manifest_v2.json` 的 `source_breakdown` 留痕。
4. 不要直接删源：`identity_dialogue.txt`(565 字符)、`dailychat`(52,587) 这类极小源基本不可见，删/留都无所谓，但它们会污染"来源占比"表的可读性。

---

## 9. 复现方法（命令行 + 种子）

所有命令在仓库根目录 `/home/vesita/coding/my/nanoSeek` 执行，使用 `.venv/bin/python`。

**重建 v2（本次实际命令，已写进 `manifest_v2.json` 的 `argv`）：**

```bash
.venv/bin/python data/chinese/prepare.py \
  --char-level --val-all --val-ratio 0.01 \
  --out-prefix v2 --seed 20260910
```

- `--seed 20260910` 固定 `annotate_replies` 的随机种子；切分另有每文件固定 seed（`1337 + sum(ord(c)) + 7` 等），所以整个重建是确定性的。两次运行产出的 `train_char_v2.bin` / `val_char_v2.bin` 字节数完全一致。
- 未使用 `--with-books`（未触发任何下载；四大名著 `.txt` 已在本地，会作为普通 `.txt` 被读入）。
- 未使用任何 `--source-ratio`（`manifest_v2.json` 里 `source_ratio: []`）。

**修前来源审计（冻结旧逻辑）：**

```bash
PYTHONPATH=data/chinese .venv/bin/python data/chinese/audit_split.py \
  --mode prefix --val-ratio 0.01 --out data/chinese/audit_prefix.json
# §2 的 c4_zh=0.5 反推：由 mix_tables.py 里 audit_split.split_prefix(..., src_ratio=('c4_zh',0.5)) 精确重算
.venv/bin/python data/chinese/mix_tables.py   # 输出三张表 -> data/chinese/mix_tables.txt
```

**修前/修后验证（全库随机 3000×256，seed=20260910）：**

```bash
.venv/bin/python data/chinese/verify_dataset.py --selftest
.venv/bin/python data/chinese/verify_dataset.py \
  --train data/chinese/train_char.bin  --val data/chinese/val_char.bin \
  --label old_dual_standard --out data/chinese/verify_old.json
.venv/bin/python data/chinese/verify_dataset.py \
  --train data/chinese/train_char_v2.bin --val data/chinese/val_char_v2.bin \
  --label new_v2_unified --out data/chinese/verify_new.json
# k 敏感性 + 顺序噪声底
.venv/bin/python data/chinese/verify_bigram_extra.py --train data/chinese/train_char.bin    --val data/chinese/val_char.bin    --label old_dual --out data/chinese/verify_bigram_extra_old.json
.venv/bin/python data/chinese/verify_bigram_extra.py --train data/chinese/train_char_v2.bin --val data/chinese/val_char_v2.bin --label new_v2   --out data/chinese/verify_bigram_extra_new.json
```

---

## 10. 产物清单与完整性

### 新数据集（v2）

| 文件 | 大小(B) | sha256 |
|---|---:|---|
| `data/chinese/train_char_v2.bin` | 1,875,546,180 | `6a21ca5ec60c386a4fe9811d65292647f336fc081378a2a21abd9fae102ff2b2` |
| `data/chinese/val_char_v2.bin` | 18,835,890 | `a724ee1be896ce33418fadf9bc1b6a4a88782d0b20848c42f89e4bb764ebff05` |
| `data/chinese/meta_char_v2.pkl` | 85 | `a5e16acc691e0af348d520b8d006feb644820705d84ac6dba20f0daede380dd4` |
| `data/chinese/manifest_v2.json` | 13,517 | `bfbd5a41dae03dac22f47110343be2644c5b2aeb3586878439bc31a5f85d8aa3` |

`meta_char_v2.pkl` 内容：`vocab_size=8192, char_level=True, tokenizer_path='char_tokenizer.json'`。
`manifest_v2.json` 含：完整 `argv`、`source_ratio: []`、`val_all: true`、`val_ratio: 0.01`、`seed: 20260910`、25 条 `source_breakdown`（逐源 train/val 字符数与占比）、源文件 sha256、产物 sha256。

### 原数据集完整性（**[实测] 未被触碰**）

| 文件 | 大小(B) | 与开工前备份 sha256 对比 |
|---|---:|---|
| `data/chinese/train_char.bin` | 1,671,692,544 | **一致**（`7c780c28364d6b55...`） |
| `data/chinese/val_char.bin` | 32,163,898 | **一致**（`2d9d349b3ff2f9e2...`） |

开工前备份（带时间戳）：`data/chinese/_backup_task17_20260910_213459/{prepare.py,manifest.json}`。

### 新增/修改的脚本

| 文件 | 作用 |
|---|---|
| `data/chinese/prepare.py` | **修改**：统一 val、`--out-prefix`、`--seed`、`argv` 落盘、OOM 修复 |
| `data/chinese/audit_split.py` | 冻结修前切分 + 调用修后 `split_one_source` 的来源审计 |
| `data/chinese/verify_dataset.py` | 三项中间量 + bigram 交叉验证 + `--selftest`（规则 1/2/3/4） |
| `data/chinese/verify_bigram_extra.py` | k 敏感性 + 来源顺序噪声底 + 顺序无关对照 |
| `data/chinese/mix_tables.py` | 生成 §3 三张来源占比表 |
| `data/chinese/DATASET_REPORT.md` | 本报告 |

---

## 11. 明确声明：未验证 / 不确定项

1. **[推断]** 旧构建的**字面命令行**无法从仓库任何地方恢复（shell history 空、manifest 未记录、无构建日志）。`c4_zh ×0.5` 是对**效果**的精确反推（block 总数与首个 `<eos>` 两个独立证据吻合），不是直接观测到的命令。
2. **[未验证]** 任务书给出的旧 bigram 绝对值与符号（4.9359 / 5.7967，val 难 +0.86）未被我的实现复现；我用 k 敏感性、前半段建表、全域对照做了排查，仍不能完全解释，原因未定。**未做参数对齐**。
3. **[未验证]** 本次**未跑任何神经模型**（GPU 被训练占用），因此"val loss 是否仍低于 train loss""修后模型指标是否更可信"均未在本报告验证；本报告只覆盖数据分布层。
4. **[未验证]** `c4_zh` 恢复全量后对训练的实际影响（质量/收敛）没有做实验；v2 相对旧 run 不可直接续训比较。
5. **[实测]** `build_assistant_mask` 的 `reply_ids` 用了 `[128,130]`（当前 `char_tokenizer.json` 的实际 id）。`training/masking.py` 的 docstring 里写的是旧词表的 117/119，**本次未修改该文件**，仅在此提示文档过期。
6. **[实测]** `verify_dataset.py` 的 bigram 表建在 `train` 全量上，会用"后半段自己"给自己打分；这正是任务书要求的方法（同一张表），但也意味着绝对 CE 偏低。我用"前半段建表"和"顺序噪声底"做了交叉参照。
7. **[实测]** 全空窗口占比 ~83% 是**既有事实**（masking 只对含终止符的回举行算 loss），不是本次引入；本次没有改动 masking 或训练逻辑。

---

## 12. 对训练方的一句话

**[实测]** v2 的 val 现在是"全库 1% 的随机样本"，train 与 val 的逐源占比、终止符密度、有效 token 占比、bigram CE 都已对齐。请把 v2 当作**新数据集从头训**；旧的 `train_char.bin`/`val_char.bin` 字节未变，仍可继续当前实验。
