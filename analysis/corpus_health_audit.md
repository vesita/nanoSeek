# v3 语料治理 —— 对抗性独立复核报告

- **复核者**：独立子代理（换输入通道：自写解析/统计代码，从原始文本重新计算）
- **日期**：2026-09-13
- **方法声明**：
  - **没有**读 `dev-notes/83-dev-notes/83最终快照.md` / `analysis/*.md` 的结论表；
  - **没有**复用 `data/chinese/clean_corpus.py` / `verify_clean_corpus.py` 的代码；
  - 所有代码写在 `/tmp/audit/`（`dlg_lib.py` / `a1_singleturn.py` / `a2_continuity.py` /
    `a3_control.py` / `coverage.c` / `a4_dedup.py` / `a4diag.py` / `a5a6_bin.py` /
    `a1b2_v2.py`），唯一产物写入 `analysis/corpus_health_audit.md`；
  - Python 一律 `.venv/bin/python`（未用 `uv run`，未用裸 `python`）；未用 GPU。
- **证据等级**：[实测] = 我亲手跑出的原始输出；[推断] = 由实测外推、未经直接测量。
  「单次」= 一次确定性计算（无随机种子敏感性）；「配对」= 同一输入上的对照比较；
  「多组」= 多个已知答案输入上的对照。

---

## 0. 数据版本钉住（重要：语料在我复核期间被别的会话修改过）

我测量期间，语料**不是冻结的**：另一个会话 15:13 重写了 `wildchat_zh.txt`（PII scrub），
15:20 重建了 `train_char_v3_dlg.bin` / `.off` / `manifest_v3_dlg.json`。
下表是各文件在我测量时的哈希/时间；断言 1/2 我先在 14:04 版上测，随后在 15:13 版上复测，
结论数字变化 < 0.01pp（见 §1/§2 两版并列）。

| 文件 | mtime | size | sha256(前 16 hex) | 备注 |
|---|---|---|---|---|
| clean_v3/belle_multiturn.txt | 2026-09-13 14:03:25 | 71430422 | 79d932fa79b0b5a3 | 全程未变 |
| clean_v3/dailychat_dialogue.txt | 13:50:19 | 109653 | 5945ec43832e6832 | 未变 |
| clean_v3/glm_dialogue.txt | 13:51:01 | 13577151 | be5eabee5a675649 | 未变 |
| clean_v3/kdconv_dialogue.txt | 13:51:02 | 2605151 | a719bc80a97a0881 | 未变 |
| clean_v3/lccc_dialogue.txt | 13:51:07 | 22659836 | df387657fba15822 | 未变 |
| clean_v3/multi_turn_dialogue.txt | 13:51:44 | 600 | 3676268b2533bda5 | 未变（已被整源清空到 1 block） |
| clean_v3/sharegpt_zh_38k.txt | 14:04:16 | 143684697 | f98168e0213e0ab5 | 未变 |
| clean_v3/wildchat_zh.txt (旧) | 14:04:20 | 28854735 | **4510ab2f082c332f** | 断言 1/2 首测用这版 |
| clean_v3/wildchat_zh.txt (新) | **15:13:29** | 28844849 | **461b9d50e7911b09** | 复测用这版 |
| train_char_v3_lang.bin | 14:11:34 | 560835952 | 2fb750534c89e1b8 | 未变 |
| train_char_v3_know.bin | 14:44:44 | 975510814 | 858956d26916d577 | 未变 |
| train_char_v3_dlg.bin (旧) | 14:22 | 227469458 | 8724f34a6bbf0014 | a5a6 首测用这版 |
| train_char_v3_dlg.bin (新) | **15:20:50** | 227485006 | **8aed9449f98d493a** | 复测用这版 |

**口径提示**：任务描述里「clean_v3 28 个 `*.txt`、约 787M 字符」与我实测不符：
我数到 **29 个 `*.txt`**，块内字符合计 **886,845,832**（三个 manifest 的
`train_chars_raw+val_chars_raw` 合计约 897M）。见 §「我无法验证的部分」。

---

## 判定汇总

| # | 断言 | 我的数字 | 判定 |
|---|---|---|---|
| 1 | v3_dlg 单轮（回复字符）占比 7.9%（v2 对照 90.5%） | **7.914%**（line，去标签）；复测 7.914% | **未能推翻**（v2 对照口径不成立，见 §1） |
| 2 | 88.5% 字符来自「有承接」的 4 个源 | 4 源字符占比 **88.759%**；4 源 Δ 全 >0 | **未能推翻**（但「恰好 4 个」在 Δ>0 规则下不唯一） |
| 3 | 高频 12-gram 覆盖率 16.4%→8.8% | union 口径 **17.81%→8.28%**；window 口径 7.55%→3.00% | **未能推翻（方向与量级）**；绝对值依口径 |
| 4 | 原始 1,856,966 块 → 治理后 1,553,429 块（blake2b 精确去重） | clean=**1,648,377**（=manifest 恰好相等）；raw_all=1,895,504，精确重复仅 **80,318** | **已推翻**（数字与 dedup 归因都对不上） |
| 5 | v3_lang bin 里 eos/cont=0；know/dlg 有终止符 | lang eos=0 cont=0；know eos=494,174 cont=174,405；dlg eos=622,358 cont=160,732 | **未能推翻** |
| 6 | know/dlg train bin 中 rel<0.5 的块占比都 <5% | know **0.000000**（0/668,049）；dlg **0.000031**（9/287,607） | **未能推翻** |

---

## 断言 1 —— v3_dlg 单轮占比 7.9%

### 我的实现与复现命令
口径：块 = 文本按空行 `\n\n` 切分；回复 = 以 `模型：` 开头的**行**；块轮数 = 该块内这种行的条数；
单轮占比 = 「回复数==1」的块里的回复字符数 / 全部回复字符数。两种口径都算：
`line`（回复=单行）/`turn`（回复=到下一个 `用户：`/`模型：` 前的多行正文）；
标签字符是否计入也各算一遍。

```bash
.venv/bin/python /tmp/audit/a1_singleturn.py        # 首测（14:04 版 wildchat）
.venv/bin/python /tmp/audit/a1_singleturn.py        # 15:13 版 wildchat 复测，见 §0
```

### 对照（已知答案）
构造 3 个块：1 轮 `AAAA`(4 字) / 2 轮 `BB`+`CCCCCC` / 3 轮 `D`+`EE`+`FFF`。
- 期望（去标签）：single=4，total=18 → **22.2222%**
- 期望（含标签）：single=7，total=36 → **19.4444%**

实测输出原文：

```
EXPECTED excl label: single=4 total=18 -> 22.2222%
EXPECTED incl label: single=7 total=36 -> 19.4444%
  total_reply_chars(excl_label) = 18
  singleturn_reply_chars(excl_label) = 4
  share_excl_label = 0.2222222222222222
  total_reply_chars(incl_label) = 36
  singleturn_reply_chars(incl_label) = 7
  share_incl_label = 0.19444444444444445
  share_excl_label pct = 22.2222%
  share_incl_label pct = 19.4444%
```

| 对照 | 期望 | 实测 | 判定 |
|---|---|---|---|
| 去标签单轮占比 | 22.2222% | 22.2222% | 通过 |
| 含标签单轮占比 | 19.4444% | 19.4444% | 通过 |

### 全量结果（v3_dlg 8 个源）
```
（首测，wildchat 14:04 版）
  belle_multiturn.txt      blocks=  53686 single_blk=     0 share_line=  0.00%
  dailychat_dialogue.txt   blocks=    622 single_blk=   622 share_line=100.00%
  glm_dialogue.txt         blocks=  42486 single_blk= 12214 share_line=  9.86%
  kdconv_dialogue.txt      blocks=   2438 single_blk=     9 share_line=  0.23%
  lccc_dialogue.txt        blocks= 149903 single_blk= 83398 share_line= 36.89%
  multi_turn_dialogue.txt  blocks=      1 single_blk=     0 share_line=  0.00%
  sharegpt_zh_38k.txt      blocks=  38247 single_blk= 14969 share_line= 10.94%
  wildchat_zh.txt          blocks=   3127 single_blk=     0 share_line=  0.00%
AGGREGATE v3_dlg:
  line-mode excl label: single/total = 6749259/85293090 = 7.913%
  line-mode incl label: single/total = 7082895/87665991 = 8.079%
  turn-mode excl label:                          = 7.913%
（复测，wildchat 15:13 版）
  line-mode excl label: single/total = 6749259/85287636 = 7.914%
  line-mode incl label:                          = 8.080%
```

**结论**：声称 7.9%，实测 **7.913%/7.914%**。差异 0.014pp，且 line/turn 两口径一致。
**判定：未能推翻。**

### 口径分歧 / 发现（v2 对照 90.5% 不成立）
1. **line 与 turn 完全等价**（7.913% vs 7.913%）：v3_dlg 的回复没有多行正文，
   所以「回复=首行」这个歧义对 v3_dlg 无影响。含/不含 `模型：` 标签也只差 0.17pp（8.08% vs 7.91%）。
2. **v2 对照 90.5% 不是同一口径的对照**。我用同一函数从原始源算 v2（turn 口径）：
   - 8 个 v3_dlg 源在 raw 下的单轮占比 = **5.646%**（首测 `a1b2_v2.py` 输出原文）
   - 全部 29 个 raw 源（turn 口径）= **79.170%**
   - 去掉 belle/wildchat/multi_turn 后 ≈ **89.85%**（= 433,547,821 / 482,524,030），接近 90.5%
   - 去掉上述三个 + kdconv 后 ≈ **89.97%**
   即：**90.5% 是"被大量单轮 QA 源（deepseek/qwen/qa_knowledge/zhihu）主导的整库"的数**，
   而那些源在 v3 被分流到了 `v3_know`，**不在 v3_dlg**。
   更关键：v3_dlg 的这 8 个源在 raw 下就已经是 **5.65%** 单轮，
   治理后变成 **7.91%** —— **是升高，不是从 90.5% 降到 7.9%**（原因是 multi_turn_dialogue
   100,000 个多轮块被整源删除，单轮占比被动上升）。
   ⇒ 「7.9%（对照 v2 90.5%）」这个并排对比**会误导**：它把「源集合换了」写成了「治理效果」。
3. 复现命令：`.venv/bin/python /tmp/audit/a1b2_v2.py`（见 §「无法验证」——我无法复现整库 90.5% 的精确源集合）。

---

## 断言 2 —— 88.5% 字符来自「有轮间承接」的源

### 我的实现与复现命令
- 块内按 `用户：`/`模型：` 边界切成「轮」（角色, 正文）；块需 ≥2 个 `用户：` 且 ≥2 个 `模型：`。
- 相邻对 = 相邻两轮（全部相邻对）。一对「承接」= 两轮正文**剥掉所有空白**后共享 ≥1 个 8-gram。
- `real_rate = 共享对数 / 总对数`；对照 = **角色保持打乱**（只在同角色的正文位置上独立置换，
  K=5 次取平均）；`delta = real_rate - shuffled_rate`。
- 判定阈值按断言原文取 `delta > 0`。

```bash
.venv/bin/python /tmp/audit/a2_continuity.py
```

### 对照（两个方向，都必须报）
构造 A：300 个块，每块 4 轮，正文互为独立随机串（字母数字，长 64）。
构造 B：300 个块，8 轮，「唯一标记链」相邻轮互相复制（每对相邻轮共享一个 64 字符唯一标记）。

```
=== CONTROL A: independent random bodies (delta should be ~0) ===
  ctrl_random.txt  blocks_q=300 pairs=900 real=0.000% shuf=0.000% delta=0.000pp
=== CONTROL B: unique marker chain copied into adjacent turn (delta >> 0) ===
  ctrl_copy.txt    blocks_q=300 pairs=2100 real=100.000% shuf=44.000% delta=56.000pp
```

| 对照 | 期望 | 实测 | 判定 |
|---|---|---|---|
| 互为随机文本 | Δ ≈ 0 | Δ = **0.000pp** | 通过 |
| 相邻轮复制 | Δ 显著 > 0 | Δ = **+56.000pp**（real 100%、shuf 44%） | 通过 |

### 全量结果（每源 pooled，K=5；复测版）
```
belle_multiturn.txt     blocks_q= 53686 pairs= 434905 real= 5.038% shuf= 2.246% delta= 2.792pp
dailychat_dialogue.txt  blocks_q=     0 pairs=      0 real= 0.000% shuf= 0.000% delta= 0.000pp
glm_dialogue.txt        blocks_q= 30272 pairs= 184224 real= 0.068% shuf= 0.036% delta= 0.032pp
kdconv_dialogue.txt     blocks_q=  2429 pairs=  19535 real= 0.906% shuf= 0.607% delta= 0.299pp
lccc_dialogue.txt       blocks_q= 66505 pairs= 239287 real= 0.063% shuf= 0.046% delta= 0.016pp
multi_turn_dialogue.txt blocks_q=     1 pairs=      7 real= 0.000% shuf= 0.000% delta= 0.000pp
sharegpt_zh_38k.txt     blocks_q= 23268 pairs= 262603 real=16.919% shuf=10.334% delta= 6.585pp
wildchat_zh.txt         blocks_q=  3126 pairs=  39553 real=29.983% shuf=16.151% delta=13.831pp
source char share and classification:
  belle_multiturn.txt       share=21.91% delta= 2.792pp -> 有承接
  dailychat_dialogue.txt    share= 0.03% delta= 0.000pp -> 无承接
  glm_dialogue.txt          share= 4.38% delta= 0.032pp -> 有承接
  kdconv_dialogue.txt       share= 0.82% delta= 0.299pp -> 有承接
  lccc_dialogue.txt         share= 6.83% delta= 0.016pp -> 有承接
  multi_turn_dialogue.txt   share= 0.00% delta= 0.000pp -> 无承接
  sharegpt_zh_38k.txt       share=53.49% delta= 6.585pp -> 有承接
  wildchat_zh.txt           share=12.54% delta=13.831pp -> 有承接
  judged-positive char share = 100710897/113465703 = 88.759%
```

**结论**：被判「有承接」的 4 个源（sharegpt/belle/wildchat/kdconv）Δ 全部 >0；
它们的字符占比 **88.759%**（声称 88.5%，差 0.259pp）。
**判定：未能推翻。**

### 口径分歧 / 发现
1. **Δ>0 这个阈值太弱，不能唯一选出 4 个源**：`glm_dialogue` Δ=+0.032pp、
   `lccc_dialogue` Δ=+0.016pp 也 >0（按字面规则应算「有承接」）。
   真正拉开量级的是 sharegpt(+6.6pp)、wildchat(+13.8pp)、belle(+2.8pp)；
   kdconv(+0.30pp) 已属勉强。**"恰好 4 个"依赖一个未写明的阈值**（例如 Δ>0.1pp 才能排除 glm/lccc）。
2. `dailychat_dialogue` 与 `multi_turn_dialogue` 的 `blocks_q=0/1`，**无法进入该度量**：
   dailychat 622 个块全是单轮（没有 ≥2 用户+≥2 模型），multi_turn 清空到 1 块。
3. 我另算了 pooled（上面）与逐块均值两种聚合，分类结论一致（逐块均值见 `/tmp/audit/a2_out.txt`）。

---

## 断言 3 —— 高频 12-gram 覆盖率 16.4%（原始）→ 8.8%（治理后）

### 我的实现与复现命令
`/tmp/audit/coverage.c`（自写 C，流式、3 遍扫描，不用 GPU，不整库读入内存）：
- 12-gram 在**块内**形成（块 = `\n\n` 切分；窗口不跨块，也不含分隔符换行）；
- 「高频」= **全局出现次数 > 50 且出现在 ≥2 个不同块**（这是对「在别的块里也频繁出现」的直译）；
- 两个覆盖率口径都算：
  - `coverage`（window 口径）= 高频 12-gram 的**窗口起点数** / 全部窗口数；
  - `union_coverage`（字符口径）= 被至少一个高频 12-gram **覆盖的字符数** / 块内总字符数
    （即断言原文「覆盖了多少比例的**字符**」）。
- CMS 预估 + 精确候选表 + 覆盖扫描；CMS 只会高估，宽度 2^26 时假候选率可忽略（Poisson 尾部 ~1e-13）。

```bash
gcc -O3 -march=native -o /tmp/audit/coverage3 /tmp/audit/coverage.c
CL=$(tr '\n' ' ' < /tmp/audit/clean_files.txt)   # 29 个 clean 文件
RW=$(tr '\n' ' ' < /tmp/audit/raw_files.txt)     # 29 个原始对照文件
/tmp/audit/coverage3 50 $CL     # clean
/tmp/audit/coverage3 50 $RW     # raw
COV_NODISTINCT=1 /tmp/audit/coverage3 50 ...     # 变体：去掉"≥2 块"条件
COV_BUDGET=400000000 /tmp/audit/coverage3 50 ... # 变体：同一预算做尺寸匹配
```

### 对照（注入模板，两个口径都要单调且与预测量相等）
干净底本 = `clean_v3/wikipedia_cn.txt` 前 ~4M 字符；模板 = 150 字符的新鲜串（与底本不共享 12-gram），
按 0/1/2/5/10/20/40% 注入独立的模板块。预测量：
window 口径 `(base_cov*W_base + n*(L-11))/(W_base+n*(L-11))`；
union 口径 `(base_union*base_chars + n*L)/(base_chars+n*L)`。

```
window 口径（/tmp/audit/a3_control.py + coverage）
   frac  copies   measured   predicted  diff_pp
   0.00       0   0.002512    0.002512    0.000
   0.01     269   0.011925    0.011925   -0.000
   0.02     544   0.021367    0.021367    0.000
   0.05    1403   0.049730    0.049730   -0.000
   0.10    2962   0.097217    0.097217   -0.000
   0.20    6664   0.192981    0.192981   -0.000
   0.40   17771   0.387813    0.387814   -0.000
monotonic non-decreasing: True ; max |measured-predicted| = 0.000 pp

union 口径（coverage3）
   frac   measured_union  predicted_union  diff_pp
   0.00        0.003350        0.003350    -0.000
   0.01        0.013335        0.013335    -0.000
   0.02        0.023338        0.023338    -0.000
   0.05        0.053318        0.053318    -0.000
   0.10        0.103275        0.103275    -0.000
   0.20        0.203130        0.203130    -0.000
   0.40        0.402689        0.402689    -0.000
monotonic: True
```

另外还有一个**已知答案的微型对照** `tiny.txt`（两个相同的 24 字块 + 一个 13 字块）：
期望 window = 26/28 = 0.928571，union = 48/61 = 0.786885，实测逐位相等。

| 对照 | 期望 | 实测 | 判定 |
|---|---|---|---|
| 模板注入 window 覆盖率 | 与注入比例逐点相等、单调 | 逐点 diff 0.000pp、单调 | 通过 |
| 模板注入 union 覆盖率 | 同上 | 逐点 diff 0.000pp、单调 | 通过 |
| tiny.txt window/union | 0.928571 / 0.786885 | 0.928571 / 0.786885 | 通过 |

### 全量结果
```
                     chars/windows      window_coverage   union_coverage
RAW  (29 原始对照)   windows=1042510148      0.075548          0.178060
                     chars  =1063730554      union=189408221
CLEAN(29 clean)      windows= 868765539      0.029983          0.082848
                     chars  = 886845832      union= 73473498
变体（去掉 ≥2 块条件）: RAW 0.076062 / CLEAN 0.030099
变体（尺寸匹配 400M 窗口）: RAW 0.069406 / CLEAN 0.025925
```

| 口径 | 原始 | 治理后 | 声称 | 判定 |
|---|---|---|---|---|
| union（字符口径，直译原文） | **17.81%** | **8.28%** | 16.4% → 8.8% | 同一量级，方向成立；差值 1.4pp / 0.5pp |
| window（窗口起点口径） | 7.55% | 3.00% | 同上 | 方向成立，绝对值约为声称的一半 |
| 尺寸匹配（400M 窗口，union 不适用，window） | 6.94% | 2.59% | — | 方向在**同等体量**下依然成立（2.68×） |

**结论**：方向「治理后明显低于治理前」**成立且稳健**（union 口径 17.8→8.3，window 口径 7.6→3.0，
尺寸匹配 6.9→2.6，去掉「≥2 块」条件几乎无变化）。
绝对值：**只有 union（字符覆盖）口径能对上声称值**（17.81/8.28 vs 16.4/8.8），
window 口径只有一半。**判定：未能推翻（方向与量级）**，但绝对值必须写清口径。

### 口径分歧 / 发现
1. **「覆盖了多少比例的字符」是 union 口径，不是窗口占比口径**。断言原文的字面义是 union；
   我最初按窗口起点算得 7.55/3.00%，一度以为绝对值被推翻，改成 union 后才对上。
   ⇒ 报告必须写清是哪种。
2. **覆盖率强烈依赖语料体量**：同一干净文本，4M 字符时 union 覆盖率只有 **0.335%**，
   全库 886.8M 字符时是 **8.28%**。因为「出现 >50 次」在语料变大时不断有新 12-gram 达标。
   ⇒ raw（1.06B 字符）与 clean（0.887B 字符）的对比必须小心；我补做了**同预算（400M 窗口）**
   对比（6.94% vs 2.59%），方向不变。
3. 我用的「原始」集合是 29 个 clean 文件的原始对照（NLP 或 new_sources）。
   若用 `raw_all/`（27 个 .txt，不含 belle/wildchat），数字会有零点几个百分点的出入，
   无法精确复现 16.4%（见「无法验证」）。

---

## 断言 4 —— 块级去重：1,856,966 → 1,553,429

### 我的实现与复现命令
口径：块 = 文本按 `\n\n` 切分（与 manifest 的 block 定义一致，见下）；
去重 = **全局跨文件** 对块文本取 `blake2b(digest_size=16)`，保留首次出现。
三种归一化都算：`exact`（`strip("\n")`）、`raw`（原样）、`ws-norm`（空白折叠）。

```bash
.venv/bin/python /tmp/audit/a4_dedup.py
.venv/bin/python /tmp/audit/a4diag.py
```

### 对照（已知答案）
3 个完全相同的块：
```
EXPECTED: raw_blocks=3, unique=1, removed=2 (dedup keeps first occurrence)
  exact: (3, 1)
```

| 对照 | 期望 | 实测 | 判定 |
|---|---|---|---|
| 3 个重复块 | raw=3, unique=1, removed=2 | (3, 1) | 通过 |

### 全量结果
```
--- mode=exact ---
  RAW   blocks=1952360  unique=1872042  removed=80318  removed%=4.11%
  CLEAN blocks=1648377  unique=1648376  removed=1      removed%=0.00%
  claim raw=1856966 clean=1553429
  clean match claim? False   raw match claim? False
--- raw_all/ aggregation ( 28 files, 含 1 个 jsonl ) ---
  raw_all exact: blocks=1895504 unique=1815186 removed=80318
```

块定义交叉验证（我的块数 vs manifest `source_breakdown` 的 `blocks` 之和）：
lang 425,431 + know 932,436 + dlg 290,510 = **1,648,377**，
与我直接数 clean 文件得到的 **1,648,377 逐位相等**。⇒ 我的「块」定义与流水线一致。

clean 版几乎没有跨文件重复（removed=1），说明**去重本身确实跑过且生效了**。
但原始侧的**精确重复块只有 80,318（4.11%~4.24%）**，而声称的减少量是
`1,856,966 - 1,553,429 = 303,537`（16.35%）。**精确 blake2b 块去重最多解释其中 26%。**

**判定：已推翻**（就具体数字与「dedup-blocks 造成该下降」的归因而言）。我的数字：clean=1,648,377；
raw_all(27 个 .txt)=1,895,503；raw_all(含 jsonl)=1,895,504；29 个原始对照=1,952,360（去重后 1,872,042）。

### 我尝试过、仍无法复现声称数字
- `raw_all/` 里的 `qa_knowledge_274k_zh.jsonl` 只有 **1 个块**（不是 38,538），
  所以「1,895,504 − 38,538 = 1,856,966」这个假设**不成立**。
- 也无法找到任何一个子集给出 clean=1,553,429（clean 的实际块数恰好等于 manifest 之和）。
- 我**没有**读 `clean_corpus.py`（任务禁止），所以无法判断它的去重是否用了别的归一化
  （不过 `exact`/`raw`/`ws-norm` 三种口径的总块数完全相同，说明归一化不是主因）。
- 我的结论只对我的「\n\n 切块 + 精确哈希」定义成立。

---

## 断言 5 —— v3_lang 的 eos/cont = 0；know/dlg 有终止符

### 我的实现与复现命令
`.off` 为 int64 边界表（长度=块数+1，末位哨兵=bin token 数）；bin 为 uint16。
分块 `np.memmap` 流式计数。

```bash
.venv/bin/python /tmp/audit/a5a6_bin.py
```

### 对照（假 bin + .off，已知位置）
```
blocks: [[128,1,2,3,4], [10,20,128,30,40], [50,130], [60,70,80,90,100,110]]
off: [0,5,10,12,18] sentinel=18
EXPECTED: rels=[0.0,0.5,1.0]; blocks_with_special=3; frac_rel_lt_0.5=1/3; eos=2 cont=1
  eos=2 cont=1 specials=3 blocks_with_special=3 frac_rel_lt_0.5=0.3333333333333333
  hist=[1,0,1,0,1]
```

| 对照 | 期望 | 实测 | 判定 |
|---|---|---|---|
| fake bin eos/cont | 2 / 1 | 2 / 1 | 通过 |
| fake bin rels | [0.0, 0.5, 1.0] | [0.0, 0.5, 1.0]（hist=[1,0,1,0,1]） | 通过 |
| frac(rel<0.5) | 33.3333% | 33.3333% | 通过 |

### 全量结果（复测版；首测版见 `/tmp/audit/a5a6_out.txt`，数字一致到 ±0.01%）
```
lang train: tokens=280417976 blocks=421179 off_ok=True eos=0      cont=0      specials=0
lang val  : tokens= 2754290 blocks=  4252 off_ok=True eos=0      cont=0      specials=0
know train: tokens=487755407 blocks=923119 off_ok=True eos=494174 cont=174405 specials=668579
know val  : tokens= 4950588 blocks=  9317 off_ok=True eos=4955   cont=1793   specials=6748
dlg  train: tokens=113742503 blocks=287607 off_ok=True eos=622358 cont=160732 specials=783090
dlg  val  : tokens= 1098920 blocks=  2902 off_ok=True eos=6300   cont=1573   specials=7873
```
所有 6 个 bin 的 `.off` 末位哨兵都等于 bin token 数，边界单调非降。

**判定：未能推翻。**（v3_lang 训练/验证 bin 的 `<eos>`/`<cont>` 都是 0；know/dlg 都有。）

### 口径发现 / 更正
- 任务说「**id >= 117 全是 `<...>` 专用槽位**」——**这条是错的**。
  实测 `data/chinese/char_tokenizer.json`：`北京` → **[994, 1187]**，id 500=`去`、994=`北`、1187=`京`，
  即 id≥117 绝大多数是普通汉字。真正的专用槽位只是 id **29** 和 **117–139**（以及零散的 `<res*>` 占位名）。
- 我用该 tokenizer 解码 dlg bin 块 0，得到完全连贯的中文，且 `<eos>` 正好贴在整条回复末尾：
  ```
  "写一篇关于环保的文章。\n"环保意识…共同努力！"<eos>\n"提取最重要的观点。\n"重要观点：…"<eos>…
  ```
  ⇒ `128=<eos>`、`130=<cont>` 的映射**经解码验证成立**，断言 5/6 的特殊符计数有效。
- 附带发现（不属于 6 条断言）：`<unk>`=129 在三个 train bin 里分别是
  lang **5,470,428**、know **4,086,347**、dlg **1,041,121**（≈1.95% / 0.84% / 0.92%），
  说明词表对语料有可见的 OOV 比例，值得另立一条记录。

---

## 断言 6 —— 终止符位置：rel<0.5 的块占比都 <5%

### 我的实现与复现命令
对 train bin 按 `.off` 切块；`rel = 块内最后一个 <eos>/<cont> 的下标 / (块长-1)`；
块长<2 或无终止符的块排除（分别计数）。同一 `a5a6_bin.py`。

### 对照（假 bin，已知位置）
见断言 5 的对照：期望 rels=[0.0, 0.5, 1.0]，frac(rel<0.5)=1/3，实测逐位相等（通过）。

### 全量结果
```
know train: blocks_with_special=668049 without=255070
   frac(rel<0.5)=0.000000 frac(rel<0.1)=0.000000 frac(rel<0.01)=0.000000
   mean_rel=0.980334 median_rel=0.984733 hist[0,.1,.5,.9,.99,1]=[0,0,705,379270,288074]
dlg  train: blocks_with_special=287607 without=0
   frac(rel<0.5)=0.000031 frac(rel<0.1)=0.000003 frac(rel<0.01)=0.000000
   mean_rel=0.967694 median_rel=0.975309 hist[0,.1,.5,.9,.99,1]=[1,8,8543,179031,100024]
```

| bin | blocks_with_special | frac(rel<0.5) | 声称 | 判定 |
|---|---|---|---|---|
| v3_know train | 668,049 | **0.000000%**（0 块） | <5% | 未能推翻 |
| v3_dlg train | 287,607 | **0.000031%**（9 块） | <5% | 未能推翻 |

终止符确实贴在块尾：know 中位数 rel=0.9847，dlg 中位数 rel=0.9753；
落在 [0.9, 1] 的块 know 667,344、dlg 279,055。

**判定：未能推翻。**

### 口径分歧
- 断言没写「无终止符的块」怎么算。我把它们**排除**（know 有 255,070 块、dlg 有 0 块）。
  若把「无终止符」也算作「rel 不达标」，know 的「达标率」会变成 668,049/923,119 = 72.4%，
  但 `frac(rel<0.5)` 本身仍只统计有终止符的块（0 块），**断言读法不受影响**。
- rel 用的是 token 下标（不是字符/字节），与 `.off` 定义一致；对 char-level bin 二者等价。

---

## 我推翻 / 发现的东西

1. **[实测·多组] 断言 4 的数字复现不了，且机制被夸大。**
   - clean 实际 **1,648,377** 块（= 三个 manifest `blocks` 之和，逐位相等）；
   - raw 侧精确重复块只有 **80,318**（4.1%），而声称的减少量是 **303,537**（16.35%）；
     精确 blake2b 去重最多解释 **26%** 的下降。
   - 声称的 `1,856,966` 我也没复现出来（raw_all 是 1,895,504；29 原始对照是 1,952,360）。
   ⇒ 这个数字对不能标注为「dedup-blocks 的效果」；它要么来自另一个（被改过的）输入版本，
   要么混入了其他清洗规则的效果。
2. **[实测·配对] 断言 1 的「v2 对照 90.5%」不是同口径对照，且会误导。**
   v3_dlg 的 8 个源在治理**前**（raw）就已经是 **5.646%** 单轮；治理**后**是 **7.914%**
   —— **升高了**（因为整源删掉了 multi_turn_dialogue 的 100,000 个多轮块）。
   90.5% 是整库（被现在分流到 v3_know 的单轮 QA 主导）的数。把两者并排写成
   「90.5% → 7.9%」会被读成治理效果，实际主要是**源集合变了**。
3. **[实测·配对] 断言 2 的「恰好 4 个源」在 Δ>0 的字面规则下不唯一**：
   `glm_dialogue`(+0.032pp)、`lccc_dialogue`(+0.016pp) 也是正 Δ。
   要选出那 4 个必须用一个未写明的阈值（约 Δ>0.1pp）。
4. **[实测·多组] 断言 3 的绝对值只有 union（字符覆盖）口径能对上**：
   17.81%→8.28%（声称 16.4%→8.8%）；window 起点口径只有 7.55%→3.00%。
   方向在两种口径、以及尺寸匹配（6.94%→2.59%）下都成立。
5. **[实测·多组] 「id >= 117 全是 `<...>` 专用槽位」是错的**：
   id 500=`去`、994=`北`、1187=`京`；专用槽位只在 id 29、117–139 等。
   （`128=<eos>`、`130=<cont>` 经解码验证无误。）
6. **[实测] 语料在我复核期间被并发修改**：`wildchat_zh.txt` 15:13 重写、
   `train_char_v3_dlg.bin`/`.off`/`manifest_v3_dlg.json` 15:20 重建。
   我在新旧两版上各测一遍，6 条断言的头部数字变化 <0.01pp（§0 表 + §1/§2 并列）。
   ⇒ 但**任何引用这些数字的结论都应带文件哈希**，否则不可复现。
7. **[实测] 覆盖率对语料体量高度敏感**（4M 字符 0.335% vs 886.8M 字符 8.28%），
   跨 raw/clean 比较必须做尺寸匹配（我做了同预算 400M 窗口的对照）。
8. **[实测] `<unk>` 密度**：lang 1.95% / know 0.84% / dlg 0.92%——不在 6 条断言内，但建议记录。

## 我无法验证的部分

1. **v2 整库 90.5% 的确切口径与源集合**：raw 文本已用 `用户：/模型：` 格式，
   我用 turn 口径在 29 源上得 79.17%，去掉 belle/wildchat/multi_turn 得 89.85%，
   去掉再加 kdconv 得 89.97%。我无法确定原报告用的是哪个子集/是否按块数加权，
   因此不能说「90.5% 正确」或「错误」——只能说**它和 7.9% 不是同一口径**。
2. **断言 4 声称数字的确切输入版本**：我找不到任何文件组合能给出 1,856,966 / 1,553,429。
   若这两个数来自被覆盖前的中间产物（我无权访问历史），则我无法复现，只能报告现状不符。
3. **去重器的内部定义**：我按任务要求**没有读** `clean_corpus.py`，
   所以无法确认它是否用了额外归一化、是否按文档/样本而不是块去重。
   我的结论只对「\n\n 块 + blake2b 精确哈希」这一（我写明的）定义成立。
4. **断言 3 的 16.4%/8.8% 精确复现**：union 口径给 17.81%/8.28%，残差 1.4pp/0.5pp。
   可能是「原始」集合口径不同（我用 29 个 clean 的原始对照，未用 `raw_all/` 的 27 个 .txt）、
   或阈值/去重条件的细微差别。我**没有**声称已复现该数字。
5. **任务描述里的数据规模**：任务说 28 个 `*.txt`、约 787M 字符；
   我实测 29 个 `*.txt`、块内 **886,845,832** 字符（manifest 的 train+val chars 约 897M）。
   无法对齐，可能任务描述基于更早的版本。
6. **我没有验证 bin 与 clean 文本的一致性**（例如 v3_dlg bin 是否就是当前 clean 文件编码而来），
   因为 15:20 的 bin 重建发生在我首次测量之后；我只验证了 bin 自身结构自洽。

---

## 复现清单（全部命令）

```bash
cd /home/vesita/coding/my/nanoSeek
# 断言 1（含对照）
.venv/bin/python /tmp/audit/a1_singleturn.py
# 断言 2（含两个方向对照）
.venv/bin/python /tmp/audit/a2_continuity.py
# 断言 3（对照 + 两个口径 + 变体）
.venv/bin/python /tmp/audit/a3_control.py
gcc -O3 -march=native -o /tmp/audit/coverage3 /tmp/audit/coverage.c
CL=$(tr '\n' ' ' < /tmp/audit/clean_files.txt); RW=$(tr '\n' ' ' < /tmp/audit/raw_files.txt)
/tmp/audit/coverage3 50 $CL ; /tmp/audit/coverage3 50 $RW
COV_NODISTINCT=1 /tmp/audit/coverage3 50 $CL ; COV_NODISTINCT=1 /tmp/audit/coverage3 50 $RW
COV_BUDGET=400000000 /tmp/audit/coverage3 50 $CL ; COV_BUDGET=400000000 /tmp/audit/coverage3 50 $RW
# 断言 4（含对照 + 逐文件诊断）
.venv/bin/python /tmp/audit/a4_dedup.py
.venv/bin/python /tmp/audit/a4diag.py
# 断言 5/6（含假 bin 对照）
.venv/bin/python /tmp/audit/a5a6_bin.py
# v2 对照（辅助）
.venv/bin/python /tmp/audit/a1b2_v2.py
```

所有原始输出保存在 `/tmp/audit/*_out.txt` 与 `/tmp/audit/rerun_current.txt`。
