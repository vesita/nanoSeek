# 样本打包（document packing）对抗性审计报告

* 审计对象：`training/packing.py` + `training/train.py` 的 packing 集成点 + `data/chinese/*.off`
* 工作目录：`/home/vesita/coding/my/nanoSeek`
* 审计人：子代理（对抗审计）；主 AI 独立复核
* 约束遵守：**没有 import `training.train`**、**只跑了一次允许的冒烟**、**没有修改仓库任何已有文件**、没有 git commit。
* 报告落盘时仓库改动：`git status --porcelain` 里 19 个 `M` 文件全部是本次审计**之前**就存在的
  （`AGENTS.md` / `dev-notes/83-dev-notes/83最终快照.md` / `TECH_DEBT.md` / `configs/base_v2.yaml` / `training/train.py` 等，
  属于其他会话的在途改动）；本次审计**没有新增/删除任何仓库文件**（`out/_packing_audit` 已按要求删除），
  唯一新增文件就是本报告。

---

## 0. 结论速览

| # | 不变式 | 判定 | 一句话依据 | 证据强度 |
|---|---|---|---|---|
| **I1** | 注意力：窗口内位置 t 的注意力不含其他样本的 key | **成立**（生产路径 MLA+attn_sink，`base_v2.yaml` 的实际路径） | 抓真实 SDPA 的 `attn_mask` 逐元素比对：泄漏 = 0；反面 `sample_id=None` 时同一判据报 1040 个泄漏 → 判据非恒真 | [实测] 人造数据 1 组 + 真实路径 spy，配对（on/off） |
| **I2** | 标签：任一 label 不来自其他样本 | **不成立**（★ 有一个必然的漏网位置） | 对齐打包窗口**右端 `t=T-1` 的 label 100.0000% 跨样本**（4096/4096 窗口，v3_dlg/lang/know/v2 全部如此），且 `mask_cross_sample_labels` 结构上看不到它 → 每个窗口固定漏 1 个 | [实测] 4 份真实语料 × 4096 窗口 + 人造最小复现；配对（漏网 vs 内缝已屏蔽） |
| **I2b** | （同上，内缝） | **成立**（已修项真的修好了） | 内缝跨样本 label 全部 == -100；真实 y 复刻后剩的跨样本 label **位置全部是 t=T-1** | [实测] 与上同一批数据 |
| **I3** | 边界：贴首/尾/两端、窗口长=块长、退化 | **成立**（无越界、无误掩、不崩）；**但有两个退化面**：空/单块 val 会抛 `ValueError`（大声），`n_tokens < T` 时 clamp 会切出短窗口（未报错，见 §5.3） | 8 组边界用例掩码后跨样本可见对全部 = 0 | [实测] 人造 8 组 |
| **I4** | 失配：大声失败而非静默降级 | **部分成立** | 缺失/哨兵不符/首元素非 0/非严格递增 → **exit 1 + 报错原文**；**同长度但批次不同 → 静默通过（exit 0）**；bin 被同尺寸文件替换也静默通过 | [实测] 5 个场景各跑一次（执行的是 train.py:392-407 的**原文**） |
| **额外 A** | （超出四条，但直接影响即将开始的训练）对齐打包有**结构性覆盖漏洞** | **不成立（数据可见性被砍掉 2/3）** | v3_dlg/lang/know 分别有 **67.39% / 67.05% / 72.91%** 的 token **永远不会被任何窗口采到**；解析式已用"笨办法逐点标记"对照 | [实测] 4 份语料全量 + 已知答案小例 + 4 个切片对照 |
| **额外 B** | CSA 路径（`base_v2` 里 **关**）在 `sample_id` 下仍能测到跨样本影响 | **疑似泄漏（未定位到根因）** | `use_csa=True` 扰动样本 0，样本 1/2 的 `max|Δ|` 打包 2.849e-02 vs 不打包 3.005e-02 —— 几乎没被掩码挡住 | [实测] 单次、人造数据；配置非主线 |

**给即将开始的那次 warm-start + 打包训练的一句话**：I1/I3 可以依赖；**I2 有一个固定漏网位置（每个窗口 1 个 label，占 0.39%）**；
**最该先处理的是"额外 A"——默认 `pack_align=True` 下三分之二的语料永远看不到**。这两条在开训前都值得定夺。

---

## 1. 方法与复现命令

### 1.1 读了什么

* `training/packing.py` 全文（181 行）
* `training/train.py`：`grep -n` 定位后逐段读（388-407 加载、440-491 `get_batch`、762-788 `estimate_loss`、
  1351-1394 微步循环、1264-1280 checkpoint 内容），**没有 import 它**
* `tests/test_packing.py` 全文
* `model/gpt.py:130-218`（`sample_id` 透传、CE 调用点）、`model/attention.py:150-296`（MLA/标准路径掩码）、
  `model/attention.py:430-547`（CSA 路径）、`model/attention.py:573-604`（KV 记忆路径）、`training/masking.py`
* `configs/base_v2.yaml`、`analysis/doc_packing.md`、`data/chinese/*.off`（只读）

### 1.2 命令原文

```bash
# 现有单测（不 import train.py）
cd /home/vesita/coding/my/nanoSeek
HSA_OVERRIDE_GFX_VERSION=10.3.0 HSA_ENABLE_SDMA=0 .venv/bin/python -m pytest tests/test_packing.py -q
# → 33 个点（33 passed），无 F/E

# I1/I2/I3 人造最小复现
HSA_OVERRIDE_GFX_VERSION=10.3.0 HSA_ENABLE_SDMA=0 .venv/bin/python /tmp/packing_audit/run_i123.py > /tmp/packing_audit/i123.out 2>&1; echo "EXIT=$?"
# → EXIT=0

# 真实语料量化（I2 漏网率、覆盖、Q1）
HSA_OVERRIDE_GFX_VERSION=10.3.0 HSA_ENABLE_SDMA=0 .venv/bin/python /tmp/packing_audit/run_real.py > /tmp/packing_audit/real.out 2>&1; echo "EXIT=$?"
# → EXIT=0
HSA_OVERRIDE_GFX_VERSION=10.3.0 HSA_ENABLE_SDMA=0 .venv/bin/python /tmp/packing_audit/run_q1.py > /tmp/packing_audit/q1.out 2>&1; echo "EXIT=$?"
# → EXIT=0（★ 这份才是 Q1 的可信结果，real.out 里的 Q1 段有 bug，见 §6.1）
HSA_OVERRIDE_GFX_VERSION=10.3.0 HSA_ENABLE_SDMA=0 .venv/bin/python /tmp/packing_audit/run_cover_check.py > /tmp/packing_audit/cover_check.out 2>&1; echo "EXIT=$?"
# → EXIT=0（覆盖统计函数的已知答案对照）

# I4 五种失配：执行 train.py:392-407 的**原文**（源码片段提取后 exec，不 import 模块）
HSA_OVERRIDE_GFX_VERSION=10.3.0 HSA_ENABLE_SDMA=0 .venv/bin/python /tmp/packing_audit/run_i4.py > /tmp/packing_audit/i4.out 2>&1; echo "EXIT=$?"

# 补充：val 覆盖 / 顺序副作用 / CSA+KV 探边界
HSA_OVERRIDE_GFX_VERSION=10.3.0 HSA_ENABLE_SDMA=0 .venv/bin/python /tmp/packing_audit/run_extra.py > /tmp/packing_audit/extra.out 2>&1; echo "EXIT=$?"

# ★ 唯一一次真实冒烟（原样，未加任何额外参数）
HSA_OVERRIDE_GFX_VERSION=10.3.0 HSA_ENABLE_SDMA=0 .venv/bin/python training/train.py configs/base_v2.yaml \
  --out_dir=out/_packing_audit --init_from=scratch --device=cpu --compile=false \
  --batch_size=2 --gradient_accumulation_steps=1 --max_iters=2 --eval_interval=1 \
  --eval_iters=1 --eval_train_split=false --data_prefix=v3_dlg --use_doc_packing=True
# → EXIT=0（跑完已 rm -rf out/_packing_audit）
```

**脚本位置**（都在 `/tmp`，未污染仓库）：
`run_i123.py` / `run_real.py` / `run_q1.py` / `run_cover_check.py` / `run_i4.py` / `run_extra.py`，
原始输出分别在 `/tmp/packing_audit/{i123,real,q1,cover_check,i4,extra}.out`。

### 1.3 被测的反面事实（先钉住"自变量真的变了"）

真实冒烟里的原始输出（`/tmp/packing_audit/smoke.log`）：

```
样本打包：train 287,608 个 block / val 2,902 个 block（.off 已校验与 bin 对齐）
参数量：75.15M
...
step 0: train 损失 0.0000, val 损失 9.0801
step 1: train 损失 8.8857, val 损失 8.8723
✓ 新最佳 val 8.8723 → best.pt（并已更新 last.pt）
step 2: train 损失 8.8857, val 损失 8.9395
训练完成：3 步（达 max_iters 2）
  最终 best_val_loss 8.8723 · 总耗时 60.1s
```

→ `--use_doc_packing=True` 真的进了 `use_doc_packing` 分支（.off 被读取并校验），**不是空测试**。

---

## 2. I1 注意力

### 2.1 判定：**成立**（生产路径）

生产配置（`configs/base_v2.yaml`）：`use_mla: true`、`use_attn_sink: true`、`use_qk_norm: true`、
`use_csa: false`、`use_kv_memory: false`、`use_mtp: false`、`block_size: 256`。
对应 `model/attention.py:216-219` 造 `doc_allowed`，`232-247` 把 `~doc_allowed` 写进 SDPA 的 float mask
（sink 路径），`248-254` 是 bool mask 分支，`265-267` 是手动路径。

### 2.2 对抗用例

人造 `.off`：`OFF = [0, 30, 70, 120]`（3 个块，长度 30/40/50），`T=64`，窗口 `[10,74)`
⇒ 两条块边界（30、70）都落在窗口正中，`sid = [0]*20 + [1]*40 + [2]*4`。

四路判据：

* **P1** 生产路径（MLA + sink）：用 spy 替换 `F.scaled_dot_product_attention`，抓**真实传进 kernel 的 float `attn_mask`**，
  逐元素检查"跨样本且因果可见的 (t,c)"是否全为 `-inf`。**判据不是恒真**：同一模型 `sample_id=None` 时同一检查必须报出泄漏。
* **P2** `use_attn_sink=False`：抓 bool `attn_mask`，与独立算出的 `causal_boundary_mask` **逐位**比对。
* **P3** 手动路径（`flash=False` + `capture=True`）：读真实 softmax 权重 `_cap_att`，跨样本位置概率必须**精确等于 0.0**。
* **P4** 端到端扰动：只扰动样本 0（窗口位置 10..19），样本 1/2（t>=20）的 logits 必须**逐位不变**。

### 2.3 原始输出（`i123.out` 节选，完整文件见同名路径）

```
OFF = [0, 30, 70, 120]  T = 64  START = [10]
sample_id（窗口内，T=64）: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
sid 分段 (样本号, token 数): [(0, 20), (1, 40), (2, 4)]
窗口内「跨样本且因果可见」的 (t,c) 对数量 = 1040

[P1] MLA + use_attn_sink=True（= configs/base_v2.yaml 的实际路径）
[P1 packing ON] mask shape=(1, 2, 64, 65)  应屏蔽位置数=3056
[P1 packing ON] 误屏蔽(应可见却 -inf)=0  泄漏(应 -inf 却可见, 任意头即算)=0

[P1-反向对照] 同一模型 sample_id=None（等于关掉打包）
[P1-反向对照] 跨样本且因果可见的 (t,c) 共 1040 个；其中未被屏蔽（=污染可见，任意头可见即算）= 1040
[P1-反向对照] 判据非恒真 ✅

[P2] MLA + use_attn_sink=False（bool mask 分支）
[P2] mask shape=(1, 1, 64, 64) dtype=torch.bool
[P2] 与期望块对角逐位一致 = True
[P2] 泄漏(应为 False 却是 True)=0

[P3] MLA + 手动 attention（flash=False, capture=True）→ 从 _cap_att 逐元素看
[P3] _cap_att shape = (1, 2, 64, 65)
[P3] 跨样本位置注意力概率 max = 0.0 非零个数 = 0
[P3] 同一样本位置注意力概率 max = 0.8581677675247192 非零个数 = 2080
[P3-反向对照] sample_id=None 时跨样本位置概率非零个数 = 2080 max = 0.27875980734825134

[P4] 端到端扰动判据：只扰动样本0（窗口位置 10..19），看样本1/2 (t>=20) 的 logits
打包   : 样本1+2 (t>=20) max|Δ| = 0.000e+00   torch.equal=True
关掩码 : 样本1+2 (t>=20) max|Δ| = 1.509e-03   torch.equal=False
打包   : 样本0 自身 max|Δ| = 8.770e-01 （扰动确实进了模型）
```

### 2.4 证据说明

* **配对**：每个判据都有"打包 ON / 掩码 OFF"两个 arm，且 OFF arm 一定报警（1040 / 2080 / 1.509e-03）→ 判据非橡皮图章。
* [实测] P1/P2/P3/P4 各一遍；P4 与 `tests/test_packing.py::test_packing_blocks_cross_sample_contamination` 同构但用的 `use_qk_norm=True`（生产值）。
* **[未测/推断]** `--compile=true`（真实训练默认）下没有重跑这四路。掩码是静态图的一部分，[推断] 语义不变；
  但项目自己的 `analysis/doc_packing.md §3.4` 也把 `compile=true` 列为未测，我沿用这个边界。

### 2.5 反面：CSA 路径（配置关闭，仅登记）

见 §5.2（额外 B）。

---

## 3. I2 标签

### 3.1 判定：**不成立**（右端漏网） / 内缝部分成立

* 内缝（`t-1` 与 `t` 跨样本且 `t<=T-1`）：**已修好**，确实被置 `-100`。
* **右端（`t = T-1`）：结构上看不到。** `mask_cross_sample_labels`（`training/packing.py:166-181`）只能在
  `(B, T-1)` 的接缝数组上操作（`cross = sample_id[:,1:] != sample_id[:,:-1]`，长度 T-1），
  而 label `y[T-1] = data[i+T]` 属于**窗口外**的 token，`sample_id` 里没有它的样本号。
* 对齐打包（`pack_align=True`，默认）**保证**窗口结束落在块边界（`packing.py:162`：`start = off[e] - block_size`），
  于是 `y[T-1]` **必然**是下一个块的首 token ⇒ **每个窗口固定漏 1 个跨样本 label**。

### 3.2 最小复现（人造，`i123.out` 原文）

```
--- W1 start=10（窗口 [10,74)，右端落在块内） ---
x[0..4] = [10, 11, 12, 13, 14] ... y[0..4] = [11, 12, 13, 14, 15]
sid (T)  = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
y 中被置 -100 的位置 = [19, 59]
被保留的 label 数 = 62 / 64
被保留 label 的位置 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 61, 62, 63]
以 T+1 长 sid 计算的**真实**跨样本 label 位置 = [19, 59]
代码实际屏蔽的位置                        = [19, 59]
★ 漏掉的跨样本 label 位置 = []
[反向对照] sid 全 0（=不打包）时判据报告的跨样本未屏蔽 label 数 = 2 / 2 → 判据非恒真 ✅

--- W2 start=6（窗口 [6,70)，右端正好落在块边界 70） ---
x[0..4] = [6, 7, 8, 9, 10] ... y[0..4] = [7, 8, 9, 10, 11]
sid (T)  = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
y 中被置 -100 的位置 = [23]
被保留的 label 数 = 63 / 64
被保留 label 的位置 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
以 T+1 长 sid 计算的**真实**跨样本 label 位置 = [23, 63]
代码实际屏蔽的位置                        = [23]
★ 漏掉的跨样本 label 位置 = [63]
   t=63: label y[63]=70 属于样本 2，输入 x[63]=69 属于样本 1
[反向对照] sid 全 0（=不打包）时判据报告的跨样本未屏蔽 label 数 = 2 / 2 → 判据非恒真 ✅

[aligned_pack_starts(OFF, T=64, B=32)] 出现过的 start = [0, 6] （= off[e]-64，右端落在块边界；只有被 0 截断的才例外）
  start=0: 右端=64（off 成员=False） 真实跨样本 label=[29] 漏掉=[]
  start=6: 右端=70（off 成员=True） 真实跨样本 label=[23, 63] 漏掉=[63]
```

**最小复现的骨架**（4 行，与 `train.py:470-477` 等价）：

```python
yy     = torch.from_numpy(data[start + 1:start + 1 + T]) # 与 train.py:471 一致
sid    = sample_id_in_window(off, [start], T)            # 只有 T 个位置
yy     = mask_cross_sample_labels(yy, sid)               # 看不到位置 T
sid_p1 = sample_id_in_window(off, [start], T + 1)        # 多算一格才看得见
assert (sid_p1[1:] != sid_p1[:-1]).sum() >= (yy == -100).sum()   # ← 右端被漏
```

### 3.3 真实语料上的漏网率（`real.out` 原文，4 份语料）

```
  [I2 真实数据 B=4096] 代码在 T-1 个位置上屏蔽的跨样本 label = 11,681 (1.1184% of T-1)
    ★ 右端 (t=T-1) 跨样本的比例 = 100.0000% (4096/4096 个窗口)
    ★ 漏网跨样本 label 占全部 label 的比例 = 0.3906%  (每个窗口 1 个)
    每个窗口平均跨样本 label 数 = 3.852；代码屏蔽 2.852，漏 1.000
    [真实 y] 复刻 get_batch 后仍未被屏蔽的跨样本 label = 64 个（位置全部在 t=T-1: True）
  ...
  [I2 真实数据 B=4096] 代码在 T-1 个位置上屏蔽的跨样本 label = 2,245 (0.2149% of T-1)
    ★ 右端 (t=T-1) 跨样本的比例 = 100.0000% (4096/4096 个窗口)
    ★ 漏网跨样本 label 占全部 label 的比例 = 0.3906%  (每个窗口 1 个)
    每个窗口平均跨样本 label 数 = 1.548；代码屏蔽 0.548，漏 1.000
    [真实 y] 复刻 get_batch 后仍未被屏蔽的跨样本 label = 64 个（位置全部在 t=T-1: True）
  ...
  [I2 真实数据 B=4096] 代码在 T-1 个位置上屏蔽的跨样本 label = 7,940 (0.7602% of T-1)
    ★ 右端 (t=T-1) 跨样本的比例 = 100.0000% (4096/4096 个窗口)
    ★ 漏网跨样本 label 占全部 label 的比例 = 0.3906%  (每个窗口 1 个)
    每个窗口平均跨样本 label 数 = 2.938；代码屏蔽 1.938，漏 1.000
    [真实 y] 复刻 get_batch 后仍未被屏蔽的跨样本 label = 64 个（位置全部在 t=T-1: True）
  ...
  [I2 真实数据 B=4096] 代码在 T-1 个位置上屏蔽的跨样本 label = 5,844 (0.5595% of T-1)
    ★ 右端 (t=T-1) 跨样本的比例 = 100.0000% (4096/4096 个窗口)
    ★ 漏网跨样本 label 占全部 label 的比例 = 0.3906%  (每个窗口 1 个)
    每个窗口平均跨样本 label 数 = 2.427；代码屏蔽 1.427，漏 1.000
    [真实 y] 复刻 get_batch 后仍未被屏蔽的跨样本 label = 64 个（位置全部在 t=T-1: True）
```

（4 段依次是 v3_dlg / v3_lang / v3_know / v2；**4/4 语料的右端漏网率都是 100.0000%**。）

**反向对照（证明这个判据不是恒真、也不是恒假）**：
* `代码屏蔽的 = 2.852 / 1.938 / 1.427 / 0.548 个每窗口` ≠ 0 ⇒ 判据确实会报"已屏蔽"，不是无差别报警；
* 人造最小复现里把 `sid` 换成全 0（=不打包）后，同一判据立即报 `2 / 2` 个跨样本 label 未屏蔽（§3.2 末行）；
* 判据自身用 `T+1` 长的 `sid` 重算真值，而代码只拿到 `T` 长的 `sid` —— 这就是它能发现漏网的原因。

### 3.4 危害的诚实量级（不要把"漏"写成"泄漏"）

* **不是信息泄漏**：位置 `T-1` 的注意力仍被 I1 的块对角掩码限制在样本 e-1 内，
  所以不存在"样本 e 的信息流进 e-1"。这是**噪声标签**：要求模型"用 A 的结尾预测 B 的开头"，
  正是 `packing.py:166-181` 的 docstring 自己定义要消除的东西。
* **量级**：每个窗口固定 1 个 label 变噪声 = `1/256 = 0.3906%` 的 label。
  但在"跨样本 label"这个子集里，漏网占**大头**：v3_lang 每窗口平均 1.548 个跨样本 label，
  代码只屏蔽 0.548（**漏掉 65%**）；v3_know 漏 34%；v3_dlg 漏 26%；v2 漏 41%。
* **每个窗口的最后一格**都变成"预测下一个块的首 token"。对 char-level 中文语料，块首常有固定格式
  （如 `<think>`、引号、角色标签），这块噪声可能被模型学成"结尾预测某个高频起手符"——**这属于推断，我没有测**。

### 3.5 与 `analysis/doc_packing.md §5.2` 的关系

该文档已经写了这个风险（"窗口里某 block 的最后一个 token 的 label 是下一个 block 的第一个 token……
本实现没有（会轻微改变 loss 语义）。影响量级未测"）。之后加的 `mask_cross_sample_labels` 只覆盖了
**内缝**，**没有覆盖文档自己点名的那个位置**。本报告的增量是：把"影响量级未测"变成实测
（每个窗口固定 1 个、4/4 语料 100%），并给出最小复现。

---

## 4. I3 边界

### 4.1 判定：**成立**（不越界、不误掩、不崩），但登记两个退化面

### 4.2 原始输出（`i123.out`）

```
[A 边界贴窗口首（start=30=off[1]）] off=[0, 30, 70, 120] T=40 start=[30] → 窗口=[30,70) 结束 token 是 off 成员=True
  sid=[0, 0, ...40 个 0...]
  块对角掩码: 形状=(1, 40, 40) 允许=820 屏蔽=780 掩码后跨样本可见对=0 → 无跨样本可见 ✅

[B 边界贴窗口尾（start=30,T=40 → end=70=off[2]）] off=[0, 30, 70, 120] T=40 start=[30] → 窗口=[30,70) 结束 token 是 off 成员=True
  sid=[0, 0, ...40 个 0...]
  块对角掩码: 形状=(1, 40, 40) 允许=820 屏蔽=780 掩码后跨样本可见对=0 → 无跨样本可见 ✅

[C 两端同时贴边界（同 B，首=off[1] 尾=off[2]）] off=[0, 30, 70, 120] T=40 start=[30] → 窗口=[30,70) 结束 token 是 off 成员=True
  sid=[0, 0, ...40 个 0...]
  块对角掩码: 形状=(1, 40, 40) 允许=820 屏蔽=780 掩码后跨样本可见对=0 → 无跨样本可见 ✅

[D 窗口长=块长（块1=40）] off=[0, 40, 80] T=40 start=[40] → 窗口=[40,80) 结束 token 是 off 成员=True
  sid=[0, 0, ...40 个 0...]
  块对角掩码: 形状=(1, 40, 40) 允许=820 屏蔽=780 掩码后跨样本可见对=0 → 无跨样本可见 ✅

[E 边界在窗口正中（I1 的主用例）] off=[0, 30, 70, 120] T=64 start=[10] → 窗口=[10,74) 结束 token 是 off 成员=False
  sid=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
  块对角掩码: 形状=(1, 64, 64) 允许=1040 屏蔽=3056 掩码后跨样本可见对=0 → 无跨样本可见 ✅

[F 窗口完全落在同一个长块内（块长 300 > T）] off=[0, 300, 600] T=64 start=[200] → 窗口=[200,264) 结束 token 是 off 成员=False
  sid=[0, 0, ...64 个 0...]
  块对角掩码: 形状=(1, 64, 64) 允许=2080 屏蔽=2016 掩码后跨样本可见对=0 → 无跨样本可见 ✅

[G 起点=0 且 n_tokens<T（aligned 的 clamp 情形）] off=[0, 10, 20, 30] T=64 start=[0] → 窗口=[0,64) 结束 token 是 off 成员=False
  sid=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
  块对角掩码: 形状=(1, 64, 64) 允许=760 屏蔽=3336 掩码后跨样本可见对=0 → 无跨样本可见 ✅

[G-2] 用真实 aligned_pack_starts 看 clamp：off=[0,10,20,30], T=64
  starts = [0, 0, 0, 0, 0, 0, 0, 0]   n_tokens=30, start+T = [64, 64, 64, 64, 64, 64, 64, 64]
  → start+T > n_tokens；get_batch 里 data[i:i+T] 只能切出 30 个 token

[H] 窗口长 == 块长 且属于对齐打包：off=[0,64,128,192], T=64
  aligned starts = [64, 64, 0, 0, 64, 64, 64, 64]  允许的 start 集合 = off[1:]-64 = {0,64,128}
  start=0: sid 唯一值=[0] 全同=True 结束 token=64 off 成员=True
  start=64: sid 唯一值=[0] 全同=True 结束 token=128 off 成员=True

[I] 空 split / 单块 .off 的退化情形
  validate_offsets([0], 0) 通过（空 split 的 .off 合法）
  aligned_pack_starts([0], T=64)（空 split）→ ValueError: 块对齐打包至少需要 2 个 block，实际 0
  aligned_pack_starts([0,120], T=64)（单块）→ ValueError: 块对齐打包至少需要 2 个 block，实际 1
```

### 4.3 两个退化面（都"大声"或"边界外"，不是静默错掩）

1. **空 trian/val split + `pack_align=True` → 第一次 eval 会抛 `ValueError`**（`packing.py:153-154`）。
   `prepare.py` 允许空 val（`.off = [0]`，`validate_offsets` 也认它合法，见 `packing.py:69-70`），
   但 `aligned_pack_starts` 要求 `n_blocks >= 2`。**大声失败**，不会静默降级 —— 但如果你将来用
   `--val-all=false` 之类切出空 val，开打包会直接崩在 `step 0` 的 eval 上。本次 v3_* 都有非空 val，不受影响。
2. **`n_tokens < block_size`（极端小 split）**：clamp 到 `start=0` 后 `start+T > n_tokens`，
   `data[i:i+T]` 只切出 `n_tokens` 个 token → `torch.stack` 形状不一致 → 崩。真实语料（1.1亿~9.4亿 token）不可能触发。

---

## 5. I4 失配

### 5.1 判定：**部分成立** —— 长度类错配大声失败，**内容类错配静默通过**

被测代码是 `train.py:392-407` 的**原文**（脚本把它从文件里按行号提取后 `exec`，不是复刻）：

```
if use_doc_packing:
    if distill_bin and p_distill > 0:
        raise SystemExit('错误：use_doc_packing 与 distill_bin/p_distill 混采不兼容'
                         '（蒸馏流没有 .off 边界表，掩码会静默漏掉那条支路）。')
    for _split, _bn in (('train', _train_bin), ('val', _val_bin)):
        _op = os.path.join(data_dir, _offsets_name_for_bin(_bn))
        if not os.path.exists(_op):
            raise SystemExit(
                f'错误：use_doc_packing=True 但找不到样本边界文件 {_op}。\n'
                f'  .off 是 `prepare.py --emit-offsets` 的产物，与 {_bn} 逐 token 对齐。\n'
                f'  请用同一条 prepare 命令加 --emit-offsets 重建数据；'
                f'不要静默退回随机窗口（那会让人以为打包开着）。')
        _ntok = os.path.getsize(os.path.join(data_dir, _bn)) // 2
        _doc_off[_split] = _load_offsets(_op, _ntok)   # 不合法会大声抛错
    print(f"样本打包：train {len(_doc_off['train']) - 1:,} 个 block / "
          f"val {len(_doc_off['val']) - 1:,} 个 block（.off 已校验与 bin 对齐）")
```

### 5.2 五种场景的实际行为（`i4.out` 原文）

```
--- [S1_off_missing] exit code = 1 ---
[stderr last 3 lines]
错误：use_doc_packing=True 但找不到样本边界文件 /tmp/packing_audit/i4/train_char_x.off。
  .off 是 `prepare.py --emit-offsets` 的产物，与 train_char_x.bin 逐 token 对齐。
  请用同一条 prepare 命令加 --emit-offsets 重建数据；不要静默退回随机窗口（那会让人以为打包开着）。

--- [S2_sentinel_mismatch] exit code = 1 ---
[stderr last 3 lines]
  File "/home/vesita/coding/my/nanoSeek/training/packing.py", line 74, in validate_offsets
    raise ValueError(
ValueError: /tmp/packing_audit/i4/train_char_x.off: 末元素（哨兵）必须等于 bin 的 token 数 100，实际 99。常见原因：.off 与 bin 不是同一次 prepare 的产物 / --out-prefix 混用 / bin 被替换。请用 `prepare.py --emit-offsets` 重建数据。

--- [S3_bad_head] exit code = 1 ---
[stderr last 3 lines]
  File "/home/vesita/coding/my/nanoSeek/training/packing.py", line 72, in validate_offsets
    raise ValueError(f'{path}: 首元素必须为 0，实际 {off[0]} —— .off 与 bin 未对齐或文件损坏')
ValueError: /tmp/packing_audit/i4/train_char_x.off: 首元素必须为 0，实际 1 —— .off 与 bin 未对齐或文件损坏

--- [S3b_not_increasing] exit code = 1 ---
[stderr last 3 lines]
  File "/home/vesita/coding/my/nanoSeek/training/packing.py", line 82, in validate_offsets
    raise ValueError(
ValueError: /tmp/packing_audit/i4/train_char_x.off: 必须严格递增；off[1]=40 >= off[2]=40（block 非空 ⇒ 每段至少 1 token；重复/回退说明 .off 损坏或平移）

--- [S4_same_length_different_boundaries] exit code = 0 ---
[stdout] 样本打包：train 3 个 block / val 1 个 block（.off 已校验与 bin 对齐）
LOAD-OK {'train': 3, 'val': 1}

--- [S5_bin_replaced_same_size] exit code = 0 ---
[stdout] 样本打包：train 3 个 block / val 1 个 block（.off 已校验与 bin 对齐）
LOAD-OK {'train': 3, 'val': 1}
```

场景定义：

| 场景 | 构造 | 实际行为 | 判定 |
|---|---|---|---|
| S1 `.off` 缺失 | 删掉 `train_char_x.off` | `SystemExit`，exit 1，提示重建数据 | ✅ 大声 |
| S2 哨兵 != bin token 数 | `.off=[0,40,90,99]`，bin=100 token | `ValueError`，exit 1 | ✅ 大声 |
| S3 首元素非 0 / 非严格递增 | `.off=[1,40,100]` / `[0,40,40,100]` | `ValueError`，exit 1 | ✅ 大声 |
| **S4 同长度不同批次** | A 的边界面 `[0,30,70,100]` 配 B 的 bin（也 100 token） | **exit 0，正常加载，无任何警告** | ❌ **静默** |
| **S5 bin 被同尺寸替换** | bin 换成另一个等长（内容不同）的文件 | **exit 0，正常加载** | ❌ **静默** |

### 5.3 结论与已修项的验证

* **已修项真的修好了**：`.off` 缺失/长度/哨兵/单调性四类**都**在启动时大声失败（不是静默退回随机窗口）。
* **未覆盖的缺口**：校验只有"长度/哨兵/单调"，**没有内容指纹**。`manifest_*.json` 里其实记了
  `.off` 的 `sha256`（`tests/test_packing.py:448-453` 在测），但 `train.py` **从不校验 manifest 的 sha256**——
  它只把 `manifest.json` 整个文件的 hash 记进 `config['data_manifest_sha256']`（`train.py:367-374`），
  而且那是 `manifest.json`（v2 的名字），v3 是 `manifest_v3_dlg.json`，`train.py:368` 的固定路径**根本没读到**。
  这条我标 **[推断+实测]**：静默通过是实测（S4/S5），"训练不校验 manifest"是读代码得出的。
* 触发概率评估：不同批次产出 + token 数完全相同，在真实语料上概率极低（不同清洗/切分几乎必然改 token 数）；
  但 v3 三份 bin/.off 都是同一晚连续构建的（`ls -la` 时间戳 14:11~14:44；v2 是 13:57），
  **如果将来只重建其中一份 bin 而忘了重建 `.off`，且长度恰好相同，就会静默错配**。

---

## 6. 额外发现：对齐打包的**结构性覆盖漏洞**（★ 与即将开始的训练直接相关）

### 6.1 先说一个我自己的测量 bug（保持透明）

第一版 `run_real.py` 的 Q1 段用 `searchsorted` 时**没有对采样出来的 starts 排序**，
得出"每步 84% 的 token 槽位是重复"的**错误**数字。加已知答案对照后复现并修正（见 §6.4），
正确值是 **0.00~0.02%**。原始错误输出仍保留在 `real.out` 里（本报告在 §6.4 粘贴时明确标注了哪几行作废）。
§6.2/6.3 的覆盖统计用的是**排序后的 `j=arange` 全枚举**，且已用"笨办法逐点标记"逐一对过，不受这个 bug 影响。

### 6.2 现象：三分之二的 token 永远采不到

`aligned_pack_starts`（`training/packing.py:140-164`）的语义是"窗口**结束**落在块边界"：
`start = off[e] - block_size`。于是**一个 token 只有落在某个块边界前 256 个位置内才可达**。
块长 > 256 时，该块只有**尾部 256 个 token** 可达，**头部永远采不到**；
最后一个块整体不可达（没有任何窗口越过 `off[n_blocks-1]`）。

### 6.3 全量实测（`real.out` 原文，四份语料）

```
[v3_dlg] n_tokens=113,734,729  n_blocks=287,608  block 长度: mean=395.5 median=82 max=256,677
  len>T=256 的 block: 92,387 (32.12% 的 block, 88.19% 的 token)
  对齐窗口数=287,607（= n_blocks-1）；窗口结束落在 block 边界上的比例 = 100.0000%
  ★ 对齐打包的 token 覆盖: 从未被任何窗口覆盖的 token = 76,650,229 (67.39%)
    平均每个 token 被多少窗口覆盖（期望采样次数，共 287,607 个窗口）= 0.6474
    (覆盖次数 -> token 数) 前几档: 0:76,650,229, 1:25,426,200, 2:2,612,774, 3:2,197,666, 4:2,269,713, 5:2,002,699
    ★ 最后一个 block（token 区间 [113,730,539,113,734,729)，4,190 个 token）可达性: True → max(window end)=113,730,539 == off[n_blocks-1]=113,730,539，没有任何窗口越过它

[v3_lang] n_tokens=280,417,976  n_blocks=421,179  block 长度: mean=665.8 median=379 max=32,831
  len>T=256 的 block: 284,465 (67.54% 的 block, 93.01% 的 token)
  ★ 对齐打包的 token 覆盖: 从未被任何窗口覆盖的 token = 188,007,780 (67.05%)
    平均每个 token 被多少窗口覆盖（期望采样次数，共 421,178 个窗口）= 0.3845
    (覆盖次数 -> token 数) 前几档: 0:188,007,780, 1:81,535,790, 2:7,903,980, 3:1,952,914, 4:692,952, 5:211,965

[v3_know] n_tokens=487,755,407  n_blocks=923,119  block 长度: mean=528.4 median=113 max=54,821
  len>T=256 的 block: 302,345 (32.75% 的 block, 88.78% 的 token)
  ★ 对齐打包的 token 覆盖: 从未被任何窗口覆盖的 token = 355,638,708 (72.91%)
    平均每个 token 被多少窗口覆盖（期望采样次数，共 923,118 个窗口）= 0.4845
    (覆盖次数 -> token 数) 前几档: 0:355,638,708, 1:83,441,091, 2:19,239,181, 3:13,626,593, 4:8,988,328, 5:4,512,422

[v2] n_tokens=937,773,090  n_blocks=1,567,037  block 长度: mean=598.4 median=246 max=126,880
  len>T=256 的 block: 766,087 (48.89% 的 block, 91.59% 的 token)
  ★ 对齐打包的 token 覆盖: 从未被任何窗口覆盖的 token = 662,745,743 (70.67%)
    平均每个 token 被多少窗口覆盖（期望采样次数，共 1,567,036 个窗口）= 0.4278
    (覆盖次数 -> token 数) 前几档: 0:662,745,743, 1:212,765,468, 2:31,443,874, 3:14,372,593, 4:7,686,115, 5:4,289,150
```

> 说明：上面输出里的 `可达性: True` 是我脚本的措辞，含义是"**存在**永远不可达的区域"（不是"可达"）；
> 同一行的关键量是 `max(window end) == off[n_blocks-1]` —— 没有任何窗口越过最后一个块的起点，
> 所以**最后一个块整体不可达**（v3_dlg 那个块 4,190 token、v3_lang 78 token、v3_know 50 token、v2 122 token）。

（v2 的 `block 长度 mean 598.4 / median 246 / max 126,880` 与 `analysis/doc_packing.md §4` 里
"mean 598.4、median 246、max 126,880、48.89% 的 block 占 91.6% token" **逐项一致** →
说明我这份"可达性"计算用的是同一份边界表、不是另一套口径。）

### 6.4 已知答案对照（`cover_check.out` 原文）

> **反面对照的说明**：`real.out` 里还有一行 `[反向对照 pack_align=False 随机窗口] 未被覆盖 token = ...`，
> **那一行不能当对照用** —— 200,000 个随机窗口本来就只覆盖 51.2M token，剩下 63% 未覆盖是"窗口数不够"，
> 不是"结构性不可达"。用窗口数对不上的东西做对照会得出假的"检查器没问题"。
> 真正有效的对照是下面两种：(a) **手算得清的小例子**（块长 1000、T=256 ⇒ 只有 744..999 可达）；
> (b) **解析式 vs 笨办法（逐点打 bool 数组）逐一对齐**。两者都过。

```
[1] 已知答案的小例子：off=[0, 1000, 1256]（块长 1000 / 256），T=256
  aligned_pack_starts 的 20 次结果 = [744, 744, 744, 744, 744, 744, 744, 744, 744, 744, 744, 744, 744, 744, 744, 744, 744, 744, 744, 744]
  → 唯一 start 集合 = [744]   窗口 = [start, start+256)
  → 手工结论：可达 token 只有 744..999；0..743 与 1000..1255 永远采不到
  coverage_stats: 未覆盖=1000  （笨办法对照=1000） ✅一致

[2] 真实数据前 300 个 block 的切片：解析式 vs 笨办法
  [v3_dlg] off[:300] 覆盖 141,142 token；解析式未覆盖=65,257 笨办法=65,257 ✅一致；未覆盖占比=46.23%
  [v3_lang] off[:300] 覆盖 221,704 token；解析式未覆盖=162,994 笨办法=162,994 ✅一致；未覆盖占比=73.52%
  [v3_know] off[:300] 覆盖 38,617 token；解析式未覆盖=1,185 笨办法=1,185 ✅一致；未覆盖占比=3.07%
  [v2] off[:300] 覆盖 236,387 token；解析式未覆盖=174,419 笨办法=174,419 ✅一致；未覆盖占比=73.79%

[3] 真实全量：coverage_stats 的两个已知答案对照
  [v3_dlg] nwin=287,607 sum(len)=73,627,392 (应=73,627,392)  未覆盖=76,650,229 至少覆盖1次=37,084,500 合计=113,734,729 (应=n_tokens=113,734,729) ✅
  [v3_lang] nwin=421,178 sum(len)=107,821,568 (应=107,821,568)  未覆盖=188,007,780 至少覆盖1次=92,410,196 合计=280,417,976 (应=n_tokens=280,417,976) ✅
  [v3_know] nwin=923,118 sum(len)=236,318,194 (应=236,318,208)  未覆盖=355,638,708 至少覆盖1次=132,116,699 合计=487,755,407 (应=n_tokens=487,755,407) ✅
```

**第三重对照（我的全枚举实现 vs 真实的 `aligned_pack_starts`）**：用真实函数对每份语料采样 200,000 个窗口，
检查每个 `start` 是否落在我的"全枚举可达集"里（`run_consistency.py`）：

```
[v3_dlg] 真实 aligned_pack_starts 采 200,000 个窗口；start 不在"全枚举可达集"里的 = 0；可达集大小=226,837，采样命中的 distinct start=126,616
[v3_lang] 真实 aligned_pack_starts 采 200,000 个窗口；start 不在"全枚举可达集"里的 = 0；可达集大小=399,359，采样命中的 distinct start=155,673
[v3_know] 真实 aligned_pack_starts 采 200,000 个窗口；start 不在"全枚举可达集"里的 = 0；可达集大小=733,035，采样命中的 distinct start=170,903
```

（可达集大小 = "不同 start 的个数" = 窗口池；它同时是 §7-Q1 的分母。）

### 6.5 val 也一样（`extra.out` 原文）

```
[A] val 是否走同一条打包路径 + val 的覆盖率（v3_dlg）
  val n_tokens=1,113,920  n_blocks=2,902  对齐窗口=2,901
  ★ val 里从未被任何对齐窗口覆盖的 token = 743,022 (66.70%)
```

### 6.6 结论

* `pack_align=True`（默认）下 **v3_dlg 67.39% / v3_lang 67.05% / v3_know 72.91% 的 train token
  永远不会被采样**，而且被采样的是**系统性的"块尾 256 token"子集**。
* 这不是"少了一点数据"：项目当前的路线是"全量语料预训练（覆盖率 16.3×）"
  （`AGENTS.md §6`），而打包把它变回了一个**结构性有偏的子集**。
* `analysis/doc_packing.md §5.3` 只写了"首 block 可能是半截……只有前文被掩码掉，无污染"，
  **没有意识到对长块而言"半截"意味着头部永远不可达**。§4 甚至已经量出 v2 有 91.59% 的 token 在超长块里，
  但没有把这两个事实接起来。
* **修复方向（我没做，只是指出）**：`pack_align=False`（随机窗口 + 块对角掩码）覆盖是均匀的，
  代价是 `analysis/doc_packing.md §3.4` 实测的 +2.3% vs 块对齐的 +5.2%（且它对 I1 一样有效）。
  **对"以覆盖全语料为目的"的阶段一，`--pack-align=False` 可能才是正确默认**——这个取舍需要用户拍板。

---

## 7. 四个具体问题

### Q1：一个 batch 里的多个微批是否可能包含同一个块的不同部分？梯度上算不算重复？要不要去重？

**代码支撑**

* `train.py:1355-1360`：每个微步单独调 `get_batch('train')`，`gradient_accumulation_steps` 个微批**一次性取齐**
  （`micro_batches`），随后 `1366-1394` 逐微批 forward/backward，`1386` 用 `n_i / n_valid_total` 加权。
* `train.py:459-463`：打包时 `ix` 来自 `_aligned_pack_starts(...)`（或 `pack_align=False` 时的 `torch.randint`），
  **每个微批独立抽样，没有任何去重/互斥**。
* `training/packing.py:140-164`：`j = torch.randint(0, n_blocks-1, (batch_size,))`，窗口 = `[off[e]-T, off[e])`。
  两个窗口只要 `e` 相同就是**同一个窗口**；只要区间相交就共享 token。

**回答**

* **可能。** 同一个块可以出现在多个窗口里：对长块，它的**尾部 256 token** 是某个窗口的全部内容，
  同时也可以是另一个（结束在更后面边界的）窗口的**前段半截**；短块则会被多个窗口整块包含。
* **算重复，但没有去重**。同一 token 在一步里出现 k 次 ⇒ 它的梯度被累加 k 次，等价于对该样本加权 k 倍
  （`train.py:1386` 的 token 级归一化只按 `n_i` 加权，不会消除重复）。
* **实测（production 规模，`q1.out`）**：`gradient_accumulation_steps=8 × batch_size=4 = 32` 个窗口/步时，
  重复槽位占 **0.00%~0.02%**，完全相同窗口重复出现 0~2 次 / 6400 个窗口。
  可区分窗口池很大：v3_dlg **226,837**、v3_lang **399,359**、v3_know **733,035**、v2 **1,313,458**。
* **结论**：当前规模下**不需要去重**；但风险随 `gradient_accumulation_steps`/`batch_size` 上升、
  随语料变小而上升（池子小到与 32×accum 同量级时会明显）。这一点值得在 `packing.py` 的 docstring 里写明。

```
[真实数据] 500 步 × 32 窗口（= base_v2 的 gradient_accumulation_steps=8 × batch_size=4）
  [v3_dlg] n_blocks=287,608  可区分窗口池大小 = 226,837 (= distinct start)  n_blocks-1 = 287,607
    ★ 每步 32 窗口（nominal 8192 token 槽位）: 平均唯一 token = 8192.0  平均重复槽位 = 1.0 (0.01%)
    单 token 最大重叠窗口数 = 2；完全相同的窗口重复出现 = 0 次 / 6400 个窗口
  [v3_lang] n_blocks=421,179  可区分窗口池大小 = 399,359 (= distinct start)  n_blocks-1 = 421,178
    ★ 每步 32 窗口（nominal 8192 token 槽位）: 平均唯一 token = 8192.0  平均重复槽位 = 1.3 (0.02%)
    单 token 最大重叠窗口数 = 2；完全相同的窗口重复出现 = 1 次 / 6400 个窗口
  [v3_know] n_blocks=923,119  可区分窗口池大小 = 733,035 (= distinct start)  n_blocks-1 = 923,118
    ★ 每步 32 窗口（nominal 8192 token 槽位）: 平均唯一 token = 8192.0  平均重复槽位 = 0.0 (0.00%)
    单 token 最大重叠窗口数 = 1；完全相同的窗口重复出现 = 0 次 / 6400 个窗口
  [v2] n_blocks=1,567,037  可区分窗口池大小 = 1,313,458 (= distinct start)  n_blocks-1 = 1,567,036
    ★ 每步 32 窗口（nominal 8192 token 槽位）: 平均唯一 token = 8192.0  平均重复槽位 = 1.3 (0.02%)
    单 token 最大重叠窗口数 = 2；完全相同的窗口重复出现 = 1 次 / 6400 个窗口
```

（已知答案对照部分：32 个相同窗口 → 重复槽位 96.875% ✅；32 个互不重叠窗口 → 0.000% ✅。见 `q1.out` 开头。）

### Q2：val 走的是同一条打包路径吗？val 口径会不会变？代码里怎么体现？

**代码支撑**

* `train.py:455-463`（`get_batch` 内，**train/val 共用同一个分支**）：
  * `458`：`_off = _doc_off['train'] if split == 'train' else _doc_off['val']`
  * `459-460`：`if pack_align: ix = torch.from_numpy(_aligned_pack_starts(_off, block_size, batch_size)).long()`
  * `463`：`sid = _sample_id_in_window(_off, ix, block_size, device=device)`
* `train.py:762-788`：`estimate_loss` 对 `splits` 里的每个 split 调 `get_batch(split)`（`772`），
  `777` 把 `SID` 传进 `model(X, Y, sample_id=SID)`。
* `train.py:1184`：`eval_splits = ('val',) if not eval_train_split else ('train','val')`。
* `train.py:388-407`：train/val 两份 `.off` 都加载、都校验。
* `train.py:1271`：checkpoint 里存 `'config': config`，而 `use_doc_packing`/`pack_align` 在 `config_keys`（`295-298`）
  ⇒ **打包开关被写进 checkpoint**，可事后人工对比。

**回答：是，同一条路径；口径变三件事，因此不可比。**

1. **窗口采样**：val 也从"全域随机窗口"变成"块对齐窗口 + 覆盖漏洞"（`extra.out`：val v3_dlg **66.70%** 的 token 不可达）。
2. **注意力可见范围**：val 也走块对角掩码（`attention.py:216-219` 等）。
3. **loss 分母**：val 的 `y` 也被 `mask_cross_sample_labels`（`train.py:477`）与
   `build_assistant_mask`（`484-485`）改写 ⇒ `n_i = (Y != -100).sum()`（`773`）变小，CE 均值口径变了。

代码里**没有**任何"打包开关变了 ⇒ 自动警告/拒绝跨阶段比较"的机制；只有：
`analysis/doc_packing.md §5.1` 的文档约定 + checkpoint 里的 `use_doc_packing` 字段可供人工核对。
`train.py:457` 的注释 "保证 val 口径与训练一致" 说的是 **train 与 val 之间**一致，不是**打包前后**一致。

### Q3：`use_loss_masking` 与打包同时开启时，`build_assistant_mask` 与块对角掩码谁先谁后？两者相乘的语义对不对？

**代码支撑**

* `train.py:472-477`：先算 `sid`，再 `y = _mask_cross_sample_labels(y, sid)`（**跨样本 label → -100 在前**）
* `train.py:484-485`：后算 `if use_loss_masking and stage != 'pretrain': y[~build_assistant_mask(y)] = -100`（**行级 mask 在后**）
* 块对角掩码在**模型内部**：`model/attention.py:216-219` 造 `doc_allowed`，
  `237-243`（sink float mask）/ `248-254`（bool）/ `265-267`（手动）用它屏蔽 attention logits。

**回答**

* **它们不相乘，甚至不在同一个张量上。** `build_assistant_mask` 产出 `(B,T)` bool，作用在 **label `y`** 上；
  块对角掩码是 `(B,T,T)`（或 broadcast 到 `(B,nh,T,T)`），作用在 **attention logits** 上。
  任务描述里的"行级 mask × 块对角 mask"在本实现里**不存在**——没有联合乘法，也没有先后依赖。
* 唯一真正的耦合是**顺序副作用**：`477` 先把接缝 label 改成 `-100`，`485` 再用被改过的 `y`
  去算 `build_assistant_mask`（`training/masking.py:35` 的 `is_term |= (y == r)`）。
  如果某个接缝位置原本是 `<eos>`(128)/`<cont>`(130)，它会被 `-100` 抹掉，进而可能翻转行级 mask。
  实测（`extra.out`，128 个真实 v3_dlg 窗口）：

```
[B] 顺序副作用：cross-sample 屏蔽（train.py:477）先于 build_assistant_mask（:485）
  128 个真实窗口：接缝处被置 -100 的 label 数 = 367 （2.87/窗口）
  其中原本是 <eos>/<cont> 的 = 0 （样本: []）
  build_assistant_mask(raw y) vs mask(-100 之后) 不一致的 token 数 = 0 （0.000/窗口）
  结论：本批 128 个真实窗口里顺序没有造成任何差异（单批，证据=单次）。
```

  ⇒ [实测，单批] 顺序**没有**在本批数据上造成差异；[推断] 因为接缝落在**新块的首 token**，而终止符按
  v3 的约定贴在回复末尾，块首几乎不会是终止符。**证据强度只有单批 128 窗口**，不能推广成"永远无害"。
* 若要把顺序钉死，把 `build_assistant_mask` 提到 `mask_cross_sample_labels` **之前**即可（对 raw y 计算），
  语义更干净；当前顺序不是 bug，但是个隐藏耦合。

### Q4：`mask_cross_sample_labels` 的 `ignore_index` 真的被下游 `cross_entropy` 使用吗？

**代码支撑（grep 原文）**

```
training/packing.py:166:def mask_cross_sample_labels(y, sample_id, ignore_index=-100):
training/packing.py:180:    y[:, :-1][cross] = ignore_index
model/gpt.py:191:            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
model/gpt.py:254:            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size),
model/gpt.py:255:                                   tgt.view(-1), ignore_index=-100)
model/gpt.py:325:            mtp_loss = mtp_loss + F.cross_entropy(
model/gpt.py:326:                logits.view(-1, logits.size(-1)), mtp_targets.reshape(-1), ignore_index=-100)
```

* `training/train.py` **里没有任何 `cross_entropy` 调用**（grep 无匹配）——CE 全在 `model/gpt.py`。
* `train.py:477` 调用 `_mask_cross_sample_labels(y, sid)` 时**没有传 `ignore_index`** ⇒ 用默认 `-100`（`packing.py:166`）。
* 主损失走 `model/gpt.py:191`（`targets` 非 None 时），`ignore_index=-100` **硬编码**。
* `train.py:485` 的 assistant mask 也写 `-100`；`train.py:1361` 的
  `micro_counts = [float((Y != -100).sum().item()) ...]`、`train.py:773` 的 `n_i`、`train.py:1386` 的
  `loss * (n_i / n_valid_total)` 全部以 `-100` 为准。
* `model/gpt.py:317` 对 MTP 的 embedding 查找做了 `targets.clamp(min=0)`，但 MTP 的 CE 仍是
  `ignore_index=-100`（`325-326`）——不过 `base_v2.yaml` 里 `use_mtp: false`，这条路不走。

**回答：是，真的被使用，而且没有被别处覆盖。** `-100` 是唯一哨兵值，
从 `packing.py:180` → `train.py:477/485` → `gpt.py:191` 全链路一致。
`train.py:480/482` 的注释写着 "ignore_index=-1" 是**注释过期**（代码用的是 `-100`，`-1` 是合法 token id），
属于文档瑕疵，不是行为 bug。

---

## 8. 额外 B：CSA 路径的跨样本泄漏（配置关闭，仅登记）

`base_v2.yaml` 里 `use_csa: false`、`use_kv_memory: false`，所以**不影响即将开始的训练**。
但把 `use_csa=True` 打开后，同一套扰动判据测出：

```
[C] CSA / KV 记忆路径（base_v2 里 use_csa=False / use_kv_memory=False，此处仅探边界）
  [use_csa=True] packing 样本1+2 max|Δ| = 2.849e-02  no-mask max|Δ| = 3.005e-02
  [use_kv_memory=True] 无法构造/前向: AssertionError: KV 记忆注意力（P1）要求 use_csa=True（替换 HCA 槽位）
```

* [实测，单次，人造数据] packing 下 `max|Δ| = 2.849e-02`，而**不打包**是 `3.005e-02` ——
  几乎一样，说明块对角掩码在 CSA 路径上基本没挡住跨样本影响。
* [推断，未定位根因] 最可疑的是 CSA 的**可学习压缩块**（`attention.py:_compress_block` / `csa_compress=16`）：
  压缩块按**全局 token 下标**切分，一个压缩块可以横跨样本边界；块级 mask 再精确也无法把
  "已经混进同一个 compressed K/V" 的两个样本分开。`attention.py:456-460` 的 `same_sample` 是**逐 token** 的，
  用在块级路径上语义不匹配。
* 这与 `analysis/doc_packing.md §5.7`（"CSA/KV 记忆路径的新 sample_id 分支未做性能与端到端训练验证"）一致。
* **建议**：任何 `use_csa=True + use_doc_packing=True` 的训练，先当成不可信。

---

## 9. 风险清单（按对"即将开始的训练"的影响排序）

| 级别 | 风险 | 实测证据 | 建议 |
|---|---|---|---|
| **P0** | 默认 `pack_align=True` 下 **67~73% 的 train token 永远采不到**（v3 三份 + v2 全部如此），且被采样的是块尾子集 | §6，解析式 + 笨办法对照 + 已知答案小例 | 开训前定夺：改用 `--pack-align=False`（覆盖均匀、I1 一样有效、文档实测代价 +2.3% vs +5.2%），或改 `aligned_pack_starts` 让窗口起点也对齐（随机选可达区间） |
| **P1** | 每个窗口固定漏 1 个跨样本 label（`t=T-1`），4/4 语料 100% 命中 | §3，4×4096 窗口 + 最小复现 | 修法极小：`get_batch` 里把 `sid` 算成 `T+1` 长，再用 `sid[:,1:] != sid[:,:-1]` 屏蔽 `y` 的全部 T 个位置（现在只屏蔽了前 T-1） |
| **P2** | `.off` 与 `.bin` 的**内容**错配静默通过（长度相同就认） | §5，S4/S5 exit 0 | 把 manifest 里已记录的 `.off` sha256 在启动时校验一次；或至少把 `manifest_v3_*.json` 的路径也纳入 `train.py:368` |
| **P3** | val 口径同时变了三件事，打包前后 val 不可比 | §7-Q2 | 已由文档约定；建议在 checkpoint 里显式存 `use_doc_packing`/`pack_align` 并在 `results.csv` 头注释一段（现在靠人看 `config`） |
| **P4** | 空/单块 val split + `pack_align=True` 会在 step 0 eval 崩（大声） | §4.3 | 如果想支持空 val，`aligned_pack_starts` 对 `n_blocks < 2` 应返回 0 或用 `_sample_id_in_window` 的降级路径 |
| **P5** | CSA 路径疑似跨样本泄漏（当前关闭） | §8，`max|Δ| 2.849e-02` vs no-mask `3.005e-02` | 不要在 `use_csa=True` 时信任打包；若要用，需重做压缩块的样本边界处理 |
| **P6** | `train.py:480/482` 注释写 `ignore_index=-1`，实际是 `-100` | §7-Q4 | 纯注释，改掉即可 |

---

## 10. 我没能做到的部分

1. **没有在真实训练里量这两条发现的端到端影响**。I2 漏网 label 与覆盖漏洞的"训练效果代价"
   都只有结构性度量，没有配对的 val/CE 对照（那需要开两次长训练，超出本次审计的算力与"一次冒烟"约束）。
   我**没有**说"影响很大/很小"，只给了结构比例与机制。
2. **`compile=true`（真实训练默认）没有测**。四路 I1 判据都在 `flash=True`/`compile=false` 或手动路径下跑的；
   `analysis/doc_packing.md §3.4/§5.8` 也把 compile 列为未测。我沿用该边界。
3. **I4 没有跑真实 `train.py` 的端到端失败路径**。任务只允许一次冒烟，我把它用在 happy path 上；
   I4 是"把 `train.py:392-407` 的**源码原文**提取出来 exec"得到的（退而求其次，见 §5）。
   因此 I4 的"退出码/报错原文"是真实代码文本的真实行为，但**不是** `python training/train.py ...` 的退出码。
4. **没有测 MTP 路径**（`gpt.py:297-327` 的 `sample_id_mtp` 切片看起来是对的：`sample_id[:, off:off+length]`，
   但 `use_mtp=false`，且它的 `off` 错位语义只有 MTP 模块内部才说得清）。标 **[未测]**。
5. **CSA 泄漏的根因没有定位**（只测到现象，见 §8）；KV 记忆路径没测（构造要求 `use_csa=True`，会与 CSA 泄漏混在一起）。
6. **Q1 的第一版测量是错的**（84%），我已修正并在 §6.1 公开了这个过程；修正后的结论（0.00~0.02%）
   只有 200 步 × 32 窗口的采样，[证据强度 = 单一批量规模、单 seed 的蒙特卡洛，不是解析上界]。
7. **覆盖漏洞的"修复方案"没有做**，也没有测 `--pack-align=False` 下的 I1/I2（理论上 I1 相同、
   I2 的右端漏网率会从 100% 掉到"恰好落在边界"的比例，但我没量）。这是一个明确的后续实验。

---

## 附录 A：关键代码行号索引

| 位置 | 内容 |
|---|---|
| `training/packing.py:58-95` | `validate_offsets` / `load_offsets`（长度、首 0、哨兵、严格递增） |
| `training/packing.py:98-116` | `sample_id_in_window`（`searchsorted`，窗口内相对样本号） |
| `training/packing.py:119-137` | `boundary_mask` / `causal_boundary_mask` |
| `training/packing.py:140-164` | `aligned_pack_starts`（`j=randint(0,n_blocks-1)`；`start=off[e]-T`） |
| `training/packing.py:153-154` | `n_blocks < 2 → ValueError`（空/单块 split 的退化面） |
| `training/packing.py:166-181` | `mask_cross_sample_labels`（只作用在 `(B,T-1)` 上 ⇒ 右端漏网） |
| `training/train.py:295-298` | `use_doc_packing` / `pack_align` 定义在 `config_keys` 之前（铁律 8） |
| `training/train.py:388-407` | `.off` 加载 + 校验（I4 被测代码） |
| `training/train.py:455-463` | train/val 共用打包采样（`458` 选 `.off`，`459-460` 对齐，`463` `sid`） |
| `training/train.py:470-471` | `x`/`y` 错位 1 |
| `training/train.py:472-477` | 跨样本 label → `-100`（在 assistant mask 之前） |
| `training/train.py:484-485` | 行级 assistant mask → `-100` |
| `training/train.py:762-788` | `estimate_loss`（val 走同一个 `get_batch`） |
| `training/train.py:1271` | checkpoint 存 `config`（含打包开关） |
| `training/train.py:1355-1360` | 一次性取齐所有微批 |
| `training/train.py:1361-1362` | `micro_counts` / `n_valid_total`（以 `-100` 为准） |
| `training/train.py:1385-1386` | `model(X, Y, sample_id=SID)` + token 级加权 |
| `model/gpt.py:135-179` | `sample_id` 透传到每个 block |
| `model/gpt.py:191` | 主 CE，`ignore_index=-100` |
| `model/attention.py:216-219` | 造 `doc_allowed = same_sample & tril` |
| `model/attention.py:232-259` | SDPA 路径（sink float mask / bool mask） |
| `model/attention.py:265-288` | 手动路径 `masked_fill(~doc_allowed)` |
| `model/attention.py:456-467` | CSA 的 `sample_id` 分支 |
| `model/attention.py:595-604` | KV 记忆的 `new_sample` 清零 |
| `training/masking.py:35` | `is_term |= (y == r)`（受 `-100` 顺序副作用影响的那一行） |

## 附录 B：原始输出文件清单（都在 /tmp，不在仓库）

```
/tmp/packing_audit/i123.out         I1/I2/I3 人造最小复现（完整）
/tmp/packing_audit/real.out         真实语料：覆盖 / I2 漏网 / Q1（★ Q1 段作废，见 §6.1）
/tmp/packing_audit/q1.out           Q1 修正版（含已知答案对照）
/tmp/packing_audit/cover_check.out  覆盖统计的已知答案对照 + 最小复现
/tmp/packing_audit/i4.out           I4 五种失配的退出码与报错原文
/tmp/packing_audit/extra.out        val 覆盖 / 顺序副作用 / CSA 探边界
/tmp/packing_audit/smoke.log        唯一一次真实冒烟的完整日志
/tmp/packing_audit/git_after.txt    审计结束时的 git status（用于证明没改仓库文件）
```

---

# 修复与复核（2026-09-13，审计之后）

> 主 AI 已验收本报告并独立复现了 P0 覆盖漏洞与 I2 右端漏网（数字逐位一致）。
> 由同一次子代理实施修复；设计决策（默认翻转 + T+1 视图）由主 AI 给出。

## R1. 改了什么

| 文件 | 改动 | 对应发现 |
|---|---|---|
| `training/train.py` | `pack_align` 默认 **`True` → `False`**（注释里写明结构性漏洞与四个数字） | §6 / P0 |
| `training/train.py` | `get_batch`：`sid_ext = _sample_id_in_window(_off, ix, block_size + 1)`（`(B,T+1)`）**只喂标签掩码**；模型侧 `sid = sid_ext[:, :block_size]` 仍是 `(B,T)`；并加显式越界守卫 `max(start)+T+1 <= len(data)` | §3 / P1 |
| `training/packing.py` | `mask_cross_sample_labels` 改成**要求 `(B,T+1)`**、一次屏蔽全部 T 个位置；传 `(B,T)` 抛 `ValueError`（不再静默漏最后一位） | §3 / P1 |
| `training/packing.py` | 新增 `aligned_window_starts`（向量化枚举全部对齐窗口）、`unreachable_token_fraction`、`reachable_fraction` | §6 / R3 |
| `training/train.py` | 启动时把每份 `.off` 的 **sha256 写进 `config`**（→ 日志 + checkpoint），并打印 | §5 / P2 |
| `training/train.py` | 数据清单路径：`manifest_<prefix>.json` 优先、回退 `manifest.json`，**打印读了哪个/没找到** | §5.3 |
| `training/train.py` | 顺手修掉旧注释 `ignore_index=-1`（实际是 `-100`） | §7-Q4 / P6 |
| `tests/test_packing.py` | 新增 6 条测试（默认值钉死 / 已知答案小例 / 真实 `.off` 双口径 / 反向对照 / T+1 契约 / 枚举一致性） | R3 |
| `analysis/doc_packing.md` | 新增 §7「2026-09-13 修正」，作废"块对齐可作默认"的结论 | R4 |

**没有动**：`model/`、`configs/`、`data/`、`inference/`、`dev-notes/83-dev-notes/83最终快照.md`、`AGENTS.md`；
没有 git commit；没有删改任何数据文件。

### ★ 修复过程中被冒烟抓到的一次真实集成 bug（诚实记录）

第一版把 `(B,T+1)` 的 `sample_id` **直接 `return` 给调用方**（原 `return x, y, sid`），
结果它被一路传进 `model/gpt.py:179` → `attention.py:219`：

```
RuntimeError: The size of tensor a (257) must match the size of tensor b (256) at non-singleton dimension 2
```

`doc_allowed = (sample_id.unsqueeze(2) == sample_id.unsqueeze(1)) & tril.unsqueeze(0)` 里的 `tril` 是
`T×T = 256×256`，而模型输入 `x` 只有 256 个位置 ⇒ **模型侧契约必须是 `(B,T)`**。
**单测没抓到**（没有任何单测 import train.py 或跑模型），只有铁律 7 的 2 步冒烟抓到了。
改为 `sid_ext`（掩码）/ `sid = sid_ext[:, :T]`（模型）两个名字后，两次冒烟都 exit 0。
⇒ 这也说明"掩码口径"和"模型口径"必须**显式分开命名**，不能图省事共用一个变量。

## R2. 真实数据复核：修复后的 `get_batch` 还有没有跨样本 label？

方法：用 **AST 从 `train.py` 精确抠出真实的 `def get_batch` 源码**（第 470-540 行）后 `exec`
（不走 `import`），喂真实 `data/chinese/{train,val}_char_v3_dlg.{bin,off}`，
`use_loss_masking=False`（⇒ 所有 `-100` 都来自跨样本屏蔽），每格口径 4096 个窗口，
`torch.manual_seed(0)` 固定 RNG（两条采样路径都吃它）。用 spy 包住 `_sample_id_in_window`
以同时拿到模型侧 `(B,T)` 与掩码侧 `(B,T+1)` 两个视图：

```
  pack_align=False split=train 窗口=4096 真实跨样本 label=2791  ★修复后漏网=0 | 反向对照(旧 T 视图)漏网=10
  pack_align=False split=val   窗口=4096 真实跨样本 label=2784  ★修复后漏网=0 | 反向对照(旧 T 视图)漏网=11
  pack_align=True  split=train 窗口=4096 真实跨样本 label=15867  ★修复后漏网=0 | 反向对照(旧 T 视图)漏网=4096
  pack_align=True  split=val   窗口=4096 真实跨样本 label=15514  ★修复后漏网=0 | 反向对照(旧 T 视图)漏网=4096
```

* **两种口径 × train/val 全部 `漏网=0`** [实测，4096 窗口/格，单 seed]。
* **反向对照非恒真**：退回旧 `(B,T)` 视图后，对齐口径漏 **4096 = 恰好每窗口 1 个**；随机口径漏 10/11
  （= 随机窗口恰好结束在块边界的那些）。⇒ 判据能区分修好/没修。
* 契约断言（脚本里硬断言）：模型侧 `SID.shape == (B,T)`、掩码侧 `sid_ext.shape == (B,T+1)`、
  且 `SID == sid_ext[:, :T]`。
* 越界守卫两种口径都过：`max(start)+T+1 = 113,734,729 == len(data)`（随机，取等号）、
  对齐为 `113,619,629 < len`；val 同理（`1,113,920` / `1,112,702`）。

## R3. `reachable_fraction` 三种口径数字（修复后新增的纯函数）

```
  (a) 已知答案小例：off=[0,1000,1256] T=256 → 应 1000/1256=0.796178
      实测 unreachable=0.796178 reachable=0.203822
  (b) 已知答案小例：off=[0,300,600]  T=256 → 应 344/600=0.573333
      实测 unreachable=0.573333 reachable=0.426667
  (c) 真实语料（全量 .off）：
      bin           aligned 不可达   aligned 可达     random 不可达    random 可达
      v3_dlg           67.3939%     32.6061%      0.000001%    100.0000%
      v3_lang          67.0456%     32.9544%      0.000000%    100.0000%
      v3_know          72.9133%     27.0867%      0.000000%    100.0000%
      v2               70.6723%     29.3277%      0.000000%    100.0000%
```

（`unreachable_token_fraction` = **不可达**占比；`reachable_fraction` = 可达占比，二者互补。
aligned 数字与 §6.3 的审计值逐位一致 → 向量化的 `aligned_window_starts` 与原两指针实现等价；
另用真实函数抽 20 万窗口做包含性对照：`越界 start=0`。）

测试里三条钉住：手算小例、真实 `v3_dlg` 的 67.394%、以及**两种口径必须差 > 0.5**
（反向对照，防止判据恒真）。

## R4. 两次冒烟（铁律 7：`--init_from=scratch` + `configs/base_v2.yaml`）

```bash
cd /home/vesita/coding/my/nanoSeek
export HSA_OVERRIDE_GFX_VERSION=10.3.0 HSA_ENABLE_SDMA=0
# ① 默认口径（不带 --pack-align → 新默认 False = 全域随机窗口）
.venv/bin/python training/train.py configs/base_v2.yaml --out_dir=out/_packing_audit_rand \
  --init_from=scratch --device=cpu --compile=false --batch_size=2 \
  --gradient_accumulation_steps=1 --max_iters=2 --eval_interval=1 --eval_iters=1 \
  --eval_train_split=false --data_prefix=v3_dlg --use_doc_packing=True
# ② 块对齐口径（显式打开，用于确认旧路径仍能跑）
.venv/bin/python training/train.py configs/base_v2.yaml --out_dir=out/_packing_audit_align \
  --init_from=scratch --device=cpu --compile=false --batch_size=2 \
  --gradient_accumulation_steps=1 --max_iters=2 --eval_interval=1 --eval_iters=1 \
  --eval_train_split=false --data_prefix=v3_dlg --use_doc_packing=True --pack-align=True
# rand EXIT=0
# align EXIT=0
# cleaned: none        ← out/_packing_audit_rand / _align 已 rm -rf
```

① `out/_packing_audit_rand`（默认 = 随机窗口）关键原始输出：

```
数据清单：data/chinese/manifest_v3_dlg.json（sha256 c4f6f6774232305e…）
样本打包：train 287,608 个 block / val 2,902 个 block（.off 已校验与 bin 对齐）
  pack_align=False（True=块对齐，实测有覆盖漏洞；False=全域随机窗口 + 块对角掩码）
  .off sha256[train] = dcdbaaec0243d1bd993f48988c7147aadee515182db5913e2a7b113f41729514
  .off sha256[val] = b720de11aa63236776d476efd1ed2c2cc675dc0eeac04f6b68ca97f151ded2a4
step 0: train 损失 0.0000, val 损失 9.1018
step 1: train 损失 9.1157, val 损失 9.1058
step 2: train 损失 9.1157, val 损失 9.0954
训练完成：3 步（达 max_iters 2）
  最终 best_val_loss 9.0954 · 总耗时 32.9s
```

② `out/_packing_audit_align`（`--pack-align=True`）关键原始输出：

```
Overriding: pack_align = True
数据清单：data/chinese/manifest_v3_dlg.json（sha256 c4f6f6774232305e…）
样本打包：train 287,608 个 block / val 2,902 个 block（.off 已校验与 bin 对齐）
  pack_align=True（True=块对齐，实测有覆盖漏洞；False=全域随机窗口 + 块对角掩码）
  .off sha256[train] = dcdbaaec0243d1bd993f48988c7147aadee515182db5913e2a7b113f41729514
  .off sha256[val] = b720de11aa63236776d476efd1ed2c2cc675dc0eeac04f6b68ca97f151ded2a4
step 0: train 损失 0.0000, val 损失 9.0798
step 1: train 损失 8.8818, val 损失 8.8712
step 2: train 损失 8.8818, val 损失 8.9390
训练完成：3 步（达 max_iters 2）
  最终 best_val_loss 8.8712 · 总耗时 32.0s
```

* 两次 `EXIT=0`；`--eval_interval=1` ⇒ **val 的 `get_batch` 也被真的跑过**（T+1 路径覆盖到 val）。
* `数据清单：…manifest_v3_dlg.json` 证明 §5.3 的修复生效（原来只找 `manifest.json`，读的是 v2 时代的旧文件）。
* `.off sha256` 已进 `config`（`train.py` 的 `config['doc_off_sha256']`，随 checkpoint 落盘）。
* 两次 step 0 的 val 不同（9.1018 vs 9.0798）符合 `doc_packing.md §3.3` 记录的打包路径抖动，**不**作为效果结论。
* 附带：默认口径比块对齐口径 **32.9s vs 32.0s**（3 步、CPU、单次）—— 方向上与 §3.4 的 GPU 测量相反，
  这是 CPU 小步数噪声，**不**作为性能结论（[实测/单次]，证据强度不足）。

## R5. 本次修复**没有**做到的

1. **没有量训练效果**：默认翻转与 T+1 掩码只做了"结构正确性 + 冒烟能跑"，
   没有跑配对 val/CE 看"覆盖从 1/3 回到 100% + 少 0.39% 噪声 label"对 loss 的实际影响。
2. **没有跑 `--compile=true`**（真实训练默认）下的冒烟 —— 与审计 §2.4 同一个未测边界。
3. **I4 的"同长度不同批次"仍是静默通过**：本次只加了 sha256 记录（事后可查），
   **没有**加"启动时与 manifest 登记的 sha256 比对/拒绝"。原报告 §5.2 的 S4/S5 缺口仍在。
4. **CSA 路径的跨样本泄漏（额外 B）没有修**（配置关闭，未在本次范围内）。
5. `aligned_pack_starts` **没有删**（只把默认关掉）：保留用于复现旧实验；它现在带着
   "结构性覆盖漏洞"的醒目警告 docstring + 一条一致性测试，但**没有任何运行时阻止**——
   谁显式写 `--pack-align=True` 仍然会丢掉 2/3 语料（只是现在会看到警告日志与文档）。
6. 第一次修复的 `(B,T+1)` 直传模型**崩在冒烟里**（见 §R1 末尾）；已改为
   `sid_ext`（掩码）/ `sid = sid_ext[:, :T]`（模型）双视图后两次冒烟 exit 0。
   这次事故本身**只能**靠冒烟发现——单测不覆盖 train.py 与模型的接口。

## R6. 门禁（最终代码状态，改完 `train.py` 之后重跑）

```bash
cd /home/vesita/coding/my/nanoSeek
HSA_OVERRIDE_GFX_VERSION=10.3.0 HSA_ENABLE_SDMA=0 .venv/bin/python -m pytest -q -m 'not slow'
```

```
........................................................................ [ 17%]
........................................................................ [ 34%]
........................................................................ [ 52%]
........................................................................ [ 69%]
........................................................................ [ 86%]
......................................................                   [100%]
```

（6 行共 **414** 个点 = 414 passed，无 F/E；`tests/test_packing.py` 单跑是 **39** 个点，
比修复前多 6 条。整份 pytest 输出**没有 summary 行**是本仓库 pytest 配置如此，点号即结果。）

```bash
.venv/bin/python -m ruff check .
# All checks passed!
```

## R7. `git diff --stat` 原文（及其**局限**）

```
$ git diff --stat -- training/train.py
 training/train.py | 205 +++++++++++++++++++++++++++++++++++++++++++-----------
 1 file changed, 166 insertions(+), 39 deletions(-)
```

⚠ **这份 stat 不等于"本次修复的改动量"**：
* `training/train.py` 在本次任务**开始之前**就已经是 `M`（其他会话/上一轮 doc packing 的在途改动），
  所以这 166/39 是**与 HEAD 的总 diff**，混着别人的改动。
* 本次修复改的三个文件里，`training/packing.py` 与 `tests/test_packing.py` 是
  **untracked（从未 commit 的新文件）**，`git diff` **看不到**它们：

```
$ git status --porcelain -- training/packing.py tests/test_packing.py training/train.py analysis/doc_packing.md analysis/packing_audit.md
 M training/train.py
?? analysis/doc_packing.md
?? analysis/packing_audit.md
?? tests/test_packing.py
?? training/packing.py
```

* 因此本次修复的改动量用**行数**表示：`training/packing.py` **185 → 279 行**、
  `tests/test_packing.py` **589 → 718 行**、`training/train.py` 改了 5 处
  （`pack_align` 默认 + `.off` sha + manifest 路径 + `get_batch` 双视图 + 注释修正）。
* 没有 git commit（按要求）；没有碰任何其它路径。
