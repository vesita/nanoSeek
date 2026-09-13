# 样本打包 + 块对角注意力掩码（document packing）

> 任务：让每个 token 只能注意自己所在的样本，杜绝同一窗口内跨样本的注意力污染。
> 交付：`training/packing.py`（新）、`tests/test_packing.py`（新）、
> `data/chinese/prepare.py` / `training/train.py` / `model/attention.py`（改，
> 另连带 `model/block.py` / `model/gpt.py` 透传 `sample_id`）。
> 状态：**代码+测试+四条对照全部完成**；默认关（`use_doc_packing=False`），
> 主线 `configs/base_v2.yaml` 未改（配方键不变）。

---

## 0. TL;DR

1. `prepare.py --emit-offsets` 现在额外产出与 bin 逐 token 对齐的 block 边界表
   `<bin 名>.off`（int64 裸数组：每个元素 = block 起始 token 下标，末尾补 `len(bin)` 哨兵），
   并登记进 manifest（sha256 + elements）。默认关，不影响既有产物（逐字节自检）。
2. 训练侧新键 `use_doc_packing`（默认 **False**）与 `pack_align`（默认 **True**）。
   打开后从 `.off` 算 `(B,T)` 的 `sample_id` 传进模型；MLA/普通因果路径把
   「同一样本」与因果掩码相与。`.off` 缺失/对不上 → **明确报错**，不静默退回随机窗口。
3. **污染判据（决定性）**：同一窗口内扰动样本 A，样本 B 的 logits **逐位差 0.000e+00**；
   关掉掩码后同一扰动改变 B 的 logits（MLA 1.6e-3 / 标准注意力 5.7e-2）——
   反向对照有信号，测试不是橡皮图章。
4. **性能**：`s/step` 基线 3.617（3 次重复极差 0.002）；掩码本身（随机窗口+掩码）
   **+2.3%**；默认的块对齐打包 **+5.2%**（单次范围 +2.2%~+6.9%）；峰值显存不变（2.0G）。
5. **向后兼容**：`use_doc_packing=False` 时模型侧与 git HEAD 的实现**逐位一致**
   （loss / logits / 272,280 个梯度全部 `np.array_equal=True`）。

---

## 1. 问题与设计

已查证的事实（不重复发现）：

* 语料被压成**一条扁平 token 流**，block 之间只有 `\n\n`，回复结束处插 `<eos>`/`<cont>`。
* 采样是**全域随机窗口**（`train.py` 的 `torch.randint(len(data) - block_size, ...)`），
  256-token 窗口经常横跨多个样本。
* 模型侧唯一的「连续」是窗口内因果注意力；窗口左边界硬截断、无跨窗状态。
* 原有的边界阻断（`attention.py` 里 `win_causal & same_sample`）只接在
  `_csa_forward`（需 `use_csa=true`）与 `_kv_memory_forward`（需 `use_kv_memory=true`）上；
  主线 MLA 普通因果路径**不接收 `is_eos`** ⇒ `sample_boundary_reset` 是死旋钮。
* `<eos>` 不足以当边界：非对话 block（c4_zh / wikipedia / 名著）整块没有 `<eos>`。

**真实 v2 语料上的污染面**（[实测]，`/tmp` 一次性脚本，4000 个窗口）：

| 窗口类型 | 每个窗口含样本数（均值） | 跨 >1 样本 | 跨 ≥3 | 跨 ≥5 |
|---|---|---|---|---|
| 全域随机（现状） | 1.42 | **28.4%** | 7.6% | 1.4% |
| 块对齐打包（新） | 2.43（max 11） | 50.7% | — | — |

⇒ 现状下约 28% 的训练窗口存在跨样本注意力；块对齐打包把多个完整 block 塞进一个窗口后，
跨样本窗口比例升到 51%（掩码工作量更大，也更省 token）。

设计要点：

* 边界表由**与写 bin 同一次 `tokenizer.encode`** 的字符区间映射出来，
  从根上排除「bin 与 .off 用两套编码逻辑」的错位。
* 模型侧只在**显式传入 `sample_id`** 时启用掩码；`is_eos` 的旧语义一字不动
  （否则 base_v2 的行为会被偷偷改掉）。
* `.off` 缺失/不合法一律报错：绝不静默退回随机窗口（那会让人以为打包开着）。

---

## 2. 接口怎么用

### 2.1 `prepare.py --emit-offsets`

```bash
.venv/bin/python data/chinese/prepare.py --char-level --val-all --val-ratio 0.01 \
    --seed 20260910 --out-prefix v2 --emit-offsets
```

* 产物：`train_char_v2.bin` + **`train_char_v2.off`**、`val_char_v2.bin` + **`val_char_v2.off`**。
* 命名规则：`.off` 由 **bin 名派生**（`os.path.splitext(bin)[0] + '.off'`），
  所以 char 模式是 `train_char_<prefix>.off`（不是 `train_<prefix>.off`）；
  与 `training/packing.py:offsets_name_for_bin` 严格同一规则（有测试钉住）。
* 与 `--source-dir` / `--out-prefix` / `--pretrain` 正交（pretrain 的 block = 每个文件的片段，分隔符长 0）。
* `--byte-level` 明确不支持（抛 `SystemExit`，不产出一个错的 .off）。
* manifest：`prepare_args.emit_offsets=true`，`artifacts["train_char_v2.off"] =
  {size_bytes, sha256, elements}`（elements = block 数 + 1）。
* 默认关：不传 `--emit-offsets` 时 bin 逐字节与旧实现相同（`tests/test_packing.py` 里
  用同语料开/关两次比对字节验证）。

### 2.2 `training/packing.py`（纯函数，CPU 可单测）

| 函数 | 语义 |
|---|---|
| `offsets_name_for_bin(bin_name)` | `train_char_v2.bin` → `train_char_v2.off` |
| `load_offsets(path, n_tokens)` | 读 int64 裸数组并校验；不合法抛 `ValueError`（信息含路径+诊断） |
| `validate_offsets(off, n_tokens, path)` | `[0]==0`、末元素==n_tokens、严格递增 |
| `sample_id_in_window(off, starts, T, device)` | `(B,T)` 窗口内样本号（0 起），`torch.searchsorted` 向量化、无 python 循环 |
| `boundary_mask(sid)` | `(B,T,T)` bool：`sid[:,:,None]==sid[:,None,:]` |
| `causal_boundary_mask(sid)` | `boundary_mask & tril`，可直接当因果掩码用 |
| `aligned_pack_starts(off, T, B, generator)` | 块对齐打包的窗口起点（torch RNG，续训可复现） |

### 2.3 模型侧

```python
# model/attention.py
def forward(self, x, rope_offset=0, is_eos=None, sample_id=None): ...
# model/block.py  Block.forward / _mhc_forward / _mhc_sublayer / MTPModule.forward 同样透传
# model/gpt.py    GPT.forward(idx, targets=None, rope_offset=0, sample_id=None)
```

* **标准/MLA 路径**（`configs/base_v2.yaml` 走的就是这条）：`sample_id` 非 None 时
  `doc_allowed = (same_sample & tril)`，然后
  * SDPA 无 sink：`attn_mask=doc_allowed.unsqueeze(1)`（bool `(B,1,T,T)`，不物化 float 掩码）；
  * SDPA 带 sink：显式 float `(B,nh,T,T+1)` 掩码的 `:T` 部分填 `~doc_allowed`；
  * 手动路径：`att.masked_fill_(~doc_allowed.unsqueeze(1), -inf)`。
* `_csa_forward` 与 `_kv_memory_forward` 也新增可选 `sample_id`：传入时用
  `.off` 样本号（比 `<eos>` 准）阻断滑窗/重置记忆；`sample_id=None` 时走原来的
  `is_eos` 分支，**CSA/KV 既有行为逐位不变**（`has_sample_id` 只是多一条 guarded 分支）。
* **向后兼容**：`sample_id=None` 时 `attention.py` 不执行任何新张量操作，
  `is_eos` 在 MLA/标准路径依旧被忽略（有单元测试逐位钉住）。

### 2.4 `training/train.py` 接线

```bash
# 默认（不打包，逐位兼容）
.venv/bin/python training/train.py configs/base_v2.yaml ...
# 打开块对齐打包（默认 pack_align=True）
    --use_doc_packing=True
# 只加掩码、窗口仍全域随机（A/B 对照用）
    --use_doc_packing=True --pack-align=False
```

* 两个键都定义在 `config_keys` 快照**之前**（铁律 8，有 AST 测试钉住）。
* `.off` 路径从 `pick_bin_names()` 的产物名派生（唯一解析点）；
  train/val 各一份，val 也按块对齐窗口 + 掩码（口径与训练一致）。
* `get_batch(split)` 现在返回 `(x, y, sid)`，`sid=None` 表示未启用；
  4 个调用点（`estimate_loss` / `ndb_eval` / 首 batch / 微步循环）都改传 `sample_id=SID`。
* `distill_bin` + `p_distill>0` 与打包互斥 → 直接报错（蒸馏流没有 `.off`，静默漏掩码更危险）。
* `.off` 缺失/与 bin token 数对不上 → `SystemExit`，提示用 `--emit-offsets` 重建数据。

---

## 3. 四条必做对照

### 3.1 ★ 污染判据（决定性）——[实测，单元测试，逐位]

`tests/test_packing.py::test_packing_blocks_cross_sample_contamination`：
一个窗口装 A（位置 0..15）与 B（16..31），只扰动 A 的输入 token，比较 B 段 logits。

| 分支 | 打包（传 sample_id） | 关掉掩码（sample_id=None） |
|---|---|---|
| MLA + SDPA | `max|ΔB| = 0.000e+00`，`torch.equal=True` | `max|ΔB| = 1.605e-03`，不相同 |
| MLA + 手动 | 0.000e+00 | 1.605e-03 |
| 标准注意力 + SDPA | 0.000e+00 | 5.689e-02 |
| 标准注意力 + 手动 | 0.000e+00 | 5.689e-02 |

* 覆盖矩阵：`use_mla` × {SDPA, 手动} × {sink on}；另有
  `test_packing_three_samples_no_leak_between_any_pair`（三样本窗口，只扰动中间样本，
  前后两段都不变）与 `test_packing_blocks_with_attn_sink_float_mask`（sink 的 float 掩码分支）。
* **反向对照是这条判据的一半**：关掉掩码后同一扰动必须改变 B 的 logits（上表第 3 列非零），
  否则测试是橡皮图章 —— 断言写在同一个测试里，不可能只过一半。

### 3.2 对齐判据——[实测，单元测试 + 真实语料交叉核对]

小语料（5 个 block，含一个**完全没有 `<eos>`/`<cont>`** 的非对话 block）：

* `test_emit_offsets_aligns_with_bin_and_decodes_back`：
  跑真的 `prepare.main([... --emit-offsets])`，断言
  `off == 独立手算的逐 block token 长度`；
  `len(off) == block 数 + 1`；`off[0]==0`；`off[-1]==bin token 数`；
  并且**用 `.off` 切出的每个区间 `decode(skip_special_tokens=False)` 恰好等于原 block
  （分隔符 `\n\n` 归前一段）**；还显式断言"至少有一个无终止符 block 仍然有正确边界"。
* `test_offsets_valid_on_nonempty_val_split`：`--val-all --val-ratio 0.4` 下 train/val
  两份 `.off` 都与各自 bin 对齐。
* **负向对照** `test_negative_control_shifted_offsets_are_rejected`：
  `.off` 整体 +1 → `off[0]!=0` 报错；整体 −1 → 报错；只动末位哨兵 → 报"哨兵"错；
  截断一个元素 → 报错；重复元素 → 报"严格递增"错。
* 真实 v2 语料交叉核对（`data/chinese/train_char_v2.off`，本任务为跑性能对照生成）：
  `elements = 1,567,038 == manifest.train_samples + 1`；`off[-1] = 937,773,090 == bin tokens`；
  严格递增；**前 64M token 内 bin 里每一对相邻 `(0,0)` 的位置与 `.off` 完全一一对应**
  （`np.array_equal` 为 True）；抽查 3 个边界，前一段末尾是 `\n\n`，后一段解码为新的 block 首句。
  *注：这份 v2 的 `.off` 是用"扫 bin 找 `\n\n`"生成的**测量用副本**（不是 `--emit-offsets` 跑出来的，
  因为重建 1.9GB bin 不值得）；它与 `--emit-offsets` 的产出契约完全一致，
  且已被上面的对照独立验证。正式重建数据请用 `--emit-offsets`。*

### 3.3 向后兼容判据——[实测，逐位]

三条证据，强度从硬到软：

1. **模型级逐位（决定性）**：把 git HEAD 的 `model/{attention,block,gpt}.py` 取到
   `/tmp/oldmodel`，同一脚本、同一种子、同一 batch（含 `<eos>`、开 MLA + MoE(aux-free)
   + attn_sink + qk_norm + sample_boundary_reset），**不传 sample_id**：

   ```
   loss:   bitwise_equal=True  max_abs_diff=0.000e+00
   logits: bitwise_equal=True  max_abs_diff=0.000e+00
   grads:  bitwise_equal=True  max_abs_diff=0.000e+00   (272,280 个梯度)
   ```
2. **训练级（冒烟）**：改动前后同一命令、同种子（`configs/base_v2.yaml`，CPU，2 步）：

   | | step0 val | step1 val | step2 val | train EMA(s1/s2) |
   |---|---|---|---|---|
   | 改动前基线 | 9.0804 | 9.0425 | 9.0727 | 9.0270 |
   | 改动后 `use_doc_packing=False` | 9.0804 | 9.0426 | 9.0726 | 9.0188 |
   | 改动后同代码重跑 #2 | 9.0804 | 9.0426 | 9.0727 | 9.0237 |
   | 改动后同代码重跑 #3 | 9.0804 | 9.0425 | 9.0722 | 9.0287 |

   ⇒ **CPU 训练层做不到逐位**：同一份代码、同一 seed 重跑三次，train EMA 就在
   9.0188~9.0287（±5e-3）、val 在 ±5e-4 抖动，量级 ≥ 改动前后差。
   机制已定位：把 `use_moe=False` 后两次重跑**完全一致**（9.0539/9.1091/9.1189 两遍逐位相同）
   ⇒ 抖动来自 **MoE aux-free 路由偏置的原地更新**（与本改动无关；
   模型级对照①里它也是逐位相等的，因为那是单次前向）。
   step0 val 在所有 6 次运行里都是 9.0804（训练前的前向完全确定），也支持"抖动在更新之后"。
3. **单元级**：`test_is_eos_is_still_ignored_on_standard_path` 断言
   `attn(x) == attn(x, is_eos=random) == attn(x, is_eos=random, sample_id=None)` 逐位相等
   —— `is_eos` 仍是 MLA/标准路径里的死参数（没有把死旋钮变成默认生效）。

### 3.4 性能判据——[实测，GPU，单卡 gfx1030，各 3 次重复]

配置：`configs/base_v2.yaml`（char、block 256、bs 4、grad_accum 8、8192 token/step、
MLA+MoE+Muon）、`--compile=false`、`--max_iters=12`、`--eval_interval=10000`（不评估）、
`--ndbg_debug_mem=false`；取**末 3 步稳态**，`s/step = 8192 / (tqdm 吞吐 t/s)`。

| 配方 | run1 | run2 | run3 | 均值 | 相对基线 |
|---|---|---|---|---|---|
| `use_doc_packing=False`（基线） | 3.616 | 3.618 | 3.617 | **3.617** | — |
| `use_doc_packing=True`（默认块对齐） | 3.695 | 3.851 | 3.866 | **3.804** | **+5.2%**（逐次 +2.2%/+6.4%/+6.9%） |
| `use_doc_packing=True --pack-align=False`（随机窗口+掩码） | 3.644 | 3.734 | 3.720 | **3.699** | **+2.3%**（逐次 +0.7%/+3.2%/+2.8%） |

* 峰值显存三者**都是 2.0G**（基线也 2.0G）→ 无显存代价。
* 基线 3 次重复极差 **0.002s（0.06%）**，非常干净；两个打包配方的重复极差
  0.09~0.17s（2~5%），说明打包路径本身比基线更"抖"。
* **掩码本身的代价 ≈ +2.3%**（随机窗口 + 掩码 vs 基线；两者只有掩码一个变量）。
  默认块对齐比随机窗口+掩码再慢 ~2.9%（两者掩码机器完全相同，只有窗口构成不同：
  2.43 vs 1.42 样本/窗）——这个差额**机制未定位**，[推断] 与不同对角块结构下
  SDPA 的 kernel 行为/每步计算量波动有关，未做进一步实验。
* `--compile=true`（真实训练默认）未测：编译预热 ~1.5 分钟，本次只做少量步的性能对照。
  这属于**已知未覆盖**：真实训练里 mask 的图会一起进 Inductor，代价可能不同（见 §5）。

---

## 4. 真实数据事实（v2）

| 指标 | 值 |
|---|---|
| train blocks | 1,567,037（mean 598.4 token，median 246，max 126,880） |
| train tokens | 937,773,090 |
| block 长度 > 256 的 | 48.89%（占全部 token 的 91.6%） |
| val blocks / tokens | 15,818 / 9,417,945 |

⇒ 一半的 block 本身就能填满一个窗口，但按 token 质量看 91.6% 的 token 落在超长 block 里；
真正跨样本的窗口占 28%（随机采样）。

---

## 5. 已知限制与风险

1. **`val` 口径会变（重要）**：打开打包后 val 也用块对齐窗口 + 掩码，
   于是 val loss 同时改了「采样窗口」与「注意力可见范围」两件事，
   **不可与打包前的历史 val 相比**。要在同一阶段内配对比较（`scripts/ckpt_paired_eval.py`
   也需同样打开打包）。建议：切换配方时把新旧 val 各立一次基线，或只在打包内部做 A/B。
2. **边界标签泄漏（未处理）**：掩码只管注意力，`y = x` 错位 1 仍是跨样本的：
   窗口里某 block 的**最后一个 token 的 label 是下一个 block 的第一个 token**。
   主流实现有时把这些边界 label 置 `-100`，本实现没有（会轻微改变 loss 语义）。
   影响量级未测；如需严格，可在 `get_batch` 里对 `sid[:,1:] != sid[:,:-1]` 的位置置 -100。
3. **块对齐窗口的首 block 可能是半截**：窗口结束落在 block 边界（"末尾不截断"），
   但起点可能在前一个 block 中间（只有前文被掩码掉，无污染）。随机模式两端都会切。
4. **`byte_level` 不支持 `.off`**（显式报错）。`--pretrain`/BPE/char 支持。
5. **`distill_bin` 混采与打包互斥**（报错）。
6. **`.off` 与 bin 的一致性只校验 token 数**：长度相同但内容不同的错配（例如拿 A 语料的
   `.off` 配 B 语料、且 token 数恰好相同）无法自动发现；manifest 的 sha256 是人工核对的抓手。
7. **CSA / KV 记忆路径的新 `sample_id` 分支未做性能与端到端训练验证**（主线不开）；
   只有纯逻辑的 guarded 分支 + 既有路径逐位不变的保证。
8. **性能只测了 `compile=false`、12 步、单 seed、单机**：证据强度=单机少量步配对重复；
   真实 `compile=true` 长跑下的 mask 代价未测（AGENTS §5.5：不要写成"没有代价"）。
9. **两份 `.off` 的数据来源差异**：仓库里 `data/chinese/train_char_v2.off` /
   `val_char_v2.off` 是本任务用 bin 扫描生成的测量用副本（已交叉验证）；
   `clean_v3/*.off` 是父会话用 `--emit-offsets` 正常产出的。

---

## 6. 复现命令与证据

```bash
cd /home/vesita/coding/my/nanoSeek
export HSA_OVERRIDE_GFX_VERSION=10.3.0 HSA_ENABLE_SDMA=0

# 门禁
.venv/bin/python -m pytest -q -m 'not slow'      # 398 passed（含新 tests/test_packing.py 30 条）
.venv/bin/python -m ruff check .                 # All checks passed!

# 必需的 2 步冒烟（带 config + --init_from=scratch）
.venv/bin/python training/train.py configs/base_v2.yaml \
  --out_dir=out/_smoke_pack --init_from=scratch --device=cpu --compile=false \
  --batch_size=2 --gradient_accumulation_steps=1 --max_iters=2 \
  --eval_interval=1 --eval_iters=1 --eval_train_split=false --use_doc_packing=True

# 性能（每条 ~65s）
.venv/bin/python training/train.py configs/base_v2.yaml --out_dir=out/_perf_on \
  --init_from=scratch --device=cuda --compile=false --batch_size=4 \
  --gradient_accumulation_steps=8 --max_iters=12 --eval_interval=10000 \
  --log_interval=1 --eval_train_split=false --ndb_debug_mem=false \
  --mem_snapshot_gb=0.0 --use_doc_packing=True
```

关键输出留痕：

```
样本打包：train 1,567,037 个 block / val 15,818 个 block（.off 已校验与 bin 对齐）
step 0: train 损失 0.0000, val 损失 9.0637
step 1: train 损失 8.9669, val 损失 9.0839
step 2: train 损失 8.9669, val 损失 9.0655
```

（对照 `use_doc_packing=False`：9.0804 / 9.0426 / 9.0726；差异来自打包后的窗口构成，
不是 bug —— 见 §3.3 的重跑抖动与 §5.1 的口径说明。）

---

## 7. ★★ 2026-09-13 修正：块对齐模式（`pack_align=True`）**不能当默认**

> 这一节**作废**本文档 §2 / §5.3 / §6 里"默认块对齐打包"的说法。修正来自一次对抗审计
> （`analysis/packing_audit.md`，主 AI 独立复算过数字）。

### 7.1 结论

**`pack_align` 默认已由 `True` 翻成 `False`**（`train.py`）。开打包必须走
**全域随机窗口 + 块对角掩码**：

```bash
--use_doc_packing=True        # pack_align 默认就是 False，不用再显式关
```

块对齐模式**不是"有 bug 可修"**，而是结构上做不到全覆盖：窗口只有 T 长且必须**结束在块边界**
（`start = off[e] - T`），于是位置 p 可达 ⟺ p 落在某个块边界前 T 个 token 内
⇒ **比 T 长的块，其前 L−T 个 token 永远进不了任何窗口**；最后一个 block 整体不可达。
改成"锚在块首"只是把不可达集**镜像**过去，同样漏一半。

### 7.2 实测覆盖（`training/packing.unreachable_token_fraction` 可复算，[实测]）

| bin | aligned 不可达 | aligned 可达 | random 不可达 | random 可达 |
|---|---:|---:|---:|---:|
| `v3_dlg` | **67.3939%** | 32.6061% | 0.000001% | 100.0000% |
| `v3_lang` | **67.0456%** | 32.9544% | 0.000000% | 100.0000% |
| `v3_know` | **72.9133%** | 27.0867% | 0.000000% | 100.0000% |
| `v2` | **70.6723%** | 29.3277% | 0.000000% | 100.0000% |

（random 的 0.000001% ≈ 1/n，就是最后一个 token；`n_tokens=113,734,729`。）

对照组：`tests/test_packing.py::test_unreachable_token_fraction_known_answer` 用**手算**的
小例钉住（块长 1000、T=256 ⇒ 不可达 1000/1256；块长 300 ⇒ 344/600），
`test_unreachable_token_fraction_real_offsets_pins_known_bad` 把真实 `v3_dlg` 的 67.394% 钉住，
`test_pack_align_default_is_false_pinned` 用 AST 读 `train.py` 钉住默认值。

### 7.3 同时修掉的第二个问题：I2 右端 label 漏网

`get_batch` 里 `sample_id` 原来是 `(B,T)`；但位置 t 的 label 是 token t+1，
所以最后一位 `t=T-1` 没有 `t+1` 可比 ⇒ **每个块对齐窗口固定漏 1 个跨样本 label**
（实测 v3_dlg/lang/know/v2 的右端跨样本比例都是 **100.0000%**，占全部 label 的 **0.3906%**）。
现在 `sample_id` 算 **`(B, T+1)`**、`mask_cross_sample_labels` 要求 T+1 视图
（传 T 会**大声抛 `ValueError`**），一次覆盖内部接缝与右端。
真实 `get_batch` 复算（4096 窗口 × 两种口径 × train/val）：修复后漏网 **0**；
退回旧 T 视图时 aligned 口径漏 **4096**（= 每窗 1 个）、random 口径漏 7~11。

### 7.4 仍然成立的部分

* I1（注意力块对角）**没有变**：掩码在 random 口径下与 aligned 完全相同，审计里逐元素验证过
  （泄漏 0，反向对照报 1040 个泄漏）。见 `analysis/packing_audit.md §2`。
* 代价：random+掩码比基线 **+2.3%**、块对齐 +5.2%（§3.4）⇒ 这次翻转**还顺带省了约 2.8% 步时**
  （3.804 → 3.699 s/step，同一批测量的两个配方）。
* `val` 口径仍然会变（同一路径），跨打包开关的 val 仍不可比（§5.1 仍有效）。

### 7.5 §5.3 那条"首 block 可能是半截"的说法

原文说"只有前文被掩码掉，无污染" —— **只在短块情况下成立**。对长块，被"切掉"的不是可有可无的前文，
而是**该块 2/3 以上的 token**（§7.2）。这条已作废。
