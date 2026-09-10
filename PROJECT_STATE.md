# nanoSeek 项目状态（上下文压缩后的唯一恢复入口）

> **上下文被压缩后，第一件事就是读这个文件。** 不要凭记忆重启训练、不要凭记忆改配置。
> 最后更新：2026-09-10 23:2x（**主线基座训练已启动** + NDB v7 原型落地）
>
> 配套文档：`TECH_DEBT.md`（技术债清单与现有闸门）、`tests/README.md`（测试布局）。

---

## 0. 一句话现状

🟢 **主线基座训练正在跑**（`configs/base_v2.yaml`，**WSD 调度**，70000 步，ETA 2.73 天）。
NDB 方向已从"神经元级"重定位到"可读可写的后缀 n-gram 库"；全量容量探针
**推翻了 §12.3 的覆盖率结论**（§6.3），并已落地 **NDB v7 原型**
（no_grad 表 + 可学习读写门控，`model/ngram_ndb.py`，§12）。
工程侧有 **289 条单元测试 + lint 门禁**（§11）。

### 恢复上下文后先做这三件事
```bash
cd /home/vesita/coding/my/nanoSeek
tail -c 1500 out/base_v2_train.log      # ① 训练还在跑吗（日志在 out_dir **之外**！见 §8 铁律 9）
pgrep -f "training/train.py configs/base_v2" && echo 在跑
cat PROJECT_STATE.md TECH_DEBT.md       # ② 恢复记忆
.venv/bin/python -m pytest tests/ -q    # ③ 289 条测试应当全绿（跳过 slow 时 <1 秒）
```

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

**为什么巡检里带 `prune_ckpts.sh`**：`train.py` 逢 1000 步写一个 `ckpt_step_<N>.pt`（≈0.6GB）,
70000 步 ⇒ 70 个 ⇒ **42GB**。保留策略已加进 `train.py`（`keep_step_ckpts`，默认 5，
见 `training/checkpoints.py` + `tests/test_checkpoints.py`），**但只对下次重启后生效**；
当前这个运行进程是旧代码，所以由外部脚本按「每 5000 步留一个 + 最新 2 个」稀疏化。
脚本幂等、只删归档、绝不碰 `best.pt`/`last.pt`。

**★ 巡检动作必须是一个脚本，不能是每次现敲的长命令。**
2026-09-11 事故：我把 prune 写进了**文档里**的巡检命令，但实际挂出去的定时器只有
`ls | wc -l`（只数个数）→ **看起来在清理，实际没清**。
现在巡检 = `bash scripts/watch.sh`（一条不可分割、可审计的脚本），定时器只负责
`sleep N && bash scripts/watch.sh`，没机会漏步骤。

**★ 守夜人（独立于 agent 定时器）**：`scripts/ckpt_janitor.sh`，
`setsid nohup` 出去、每 1800s 稀疏化一次、**训练进程消失即自动退出**。
存在理由：训练是 `setsid nohup` 出去的，**会话/定时器链死了训练还会继续**，
那时归档会以 ~0.63GB/小时 无限增长。停止：`pkill -f ckpt_janitor.sh`。

**节律（以 300s = 5 分钟起步，逐次翻倍，5 小时 = 18000s 封顶）**：
```
300 → 600 → 1200 → 2400 → 4800 → 9600 → 18000 → 18000 → …
```
👉 **当前档位：4800s**（下一轮结束挂 9600s，再下一轮 18000s 后固定不变）
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
→ **训练侧 2.150 ≈ val 2.177**（一致，无过拟合）；而 `results.csv` 的 train 列 1.8050
与两者都差 0.35，**背离出在 train-eval 列本身**。
佐证：step 6000 的 train 列 2.2194 甚至**高于**同期窗口均值 2.1722 —— 方向都不稳定。

**[推断，未验证]** train-eval 采样 800 个窗口，若语料存在重复/近似重复文档，
模型会先记住它们 → train-eval 被少数被记住的文档拉低。若后续要查，方向是
**量语料重复率**，不是调正则。

**推论（对未来 NDB A/B 至关重要）**：Δ 只有 0.02~0.07 量级，**参照量必须用 val**
或「1000 步聚合的 window_mean」；拿 `results.csv` 的 train 列做对照会被 ±0.2 的噪声淹掉。

---

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

### 3.1 结论
| 判据 | 旧 | **v2** |
|---|---|---|
| 终止符密度 val/train | **3.15×** | **0.97×** ✅ |
| 有效 token 占比 train / val | 6.29% / 27.39% | **6.24% / 6.30%** ✅ |
| bigram CE 差（train-未见 vs val）| −0.1202 nats | **+0.0268 nats** ✅ |
| train token | 835.8M | 937.8M |
| val token | 16.1M | 9.4M |

### 3.2 旧 bug（已修）
`prepare.py` 的 `DIALOGUE_FILES` 只硬编码 5 个文件名，但目录里有 13+ 个 `*_dialogue.txt`：
- 名单内 → **10% 进 val**；名单外（含 573MB 的 deepseek、267MB 的 qwen3）→ **1% 进 val**
- 结果 val 把对话样本放大约 5 倍 → **train/val 测的不是同一个任务**

### 3.3 附带发现
- `encode_to_bin` 无界缓冲：`c4_zh.txt`（1.89 亿字符、无 `<eos>`）整段堆内存 → **RSS 7.1GB、可用内存剩 469MB、swap 打满**。已改为 1M 字符分块 flush + 逐字节等价自检。
- 旧 `train_char.bin` 少 94,773 个 block（~1.0 亿字符），**已被定量反推为 `--source-ratio c4_zh=0.5`**（预测 1,488,082 = 实际 1,488,082，并用首个 `<eos>` 位置独立验证）。字面命令不可考。
- v2 **不复制**该降采样，全来源全量。

### 3.4 产物
```
data/chinese/train_char_v2.bin   1,875,546,180 B / 937,773,090 token
data/chinese/val_char_v2.bin        18,835,890 B /   9,417,945 token
data/chinese/meta_char_v2.pkl       {vocab 8192, char_level}
data/chinese/manifest_v2.json       含完整 argv + 25 条逐源占比 + sha256
data/chinese/DATASET_REPORT.md      完整报告
```
复现：`.venv/bin/python data/chinese/prepare.py --char-level --val-all --val-ratio 0.01 --out-prefix v2 --seed 20260910`
**旧 `train_char.bin` / `val_char.bin` 一个字节没动**（mtime 仍是 2026-09-08 22:35）。

### 3.5 语料混合（建议，未动手）
v2 train 占比 >15%：`deepseek` 27.33% · `c4_zh` 20.25% · `wikipedia` 19.67%；`qwen3` 14.41%。
大类：蒸馏对话 41.74% · 网页/百科 44.75% · **人类自然对话仅 13.51%**。

---

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

## 5. 新基座配置：已就绪，**尚未启动**

**文件：`configs/base_v2.yaml`**（放在 `out_dir` **之外** —— 见 §7 的坑）

| 键 | 旧 | 新 | 依据 |
|---|---|---|---|
| `data_prefix` | — | `v2` | 验收通过的新数据集 |
| `out_dir` | `out/ndb_run` | `out/base_v2` | — |
| `init_from` | resume | **scratch** | 数据集分布变了，且已确认可放弃旧基线 |
| `batch_size` / `gradient_accumulation_steps` | 8 / 4 | **4 / 8** | −9.7% 步时，显存 −30% |
| `use_mhc` | true | **false** | 已证明逐位无效 |
| `use_mtp` | true | **false** | −10.3% 步时 + 省 7.9% 参数 |
| `swiglu_clamp` | 0.0 | **10.0** | 实测有异常值（最大 49.6）|
| `muon_ns_steps` / `muon_ns_aggressive` | 10 / — | **7 / 4** | 残差 8.79e-06 vs 4.53e-05，配对 t=−10.5 |
| `lr_decay_iters` | 30000 | **70000** | 修掉"退火在 30000 结束、之后 40000 步平在 1e-4" |
| `ndb_store` / `ndb_heldout` | store5mA | **`""`** | 这一轮先做纯基座 |

- **冒烟测试已通过**：参数量 75.15M、step0 loss 9.0689（≈ln8192=9.011）、**3.38 s/it**、显存 1.7G
- **数据开关已验证**：`--data_prefix=v2` → 读 `train_char_v2.bin`；`--data_prefix=''` → 读旧文件
- ETA：70000 × 3.371 s = **65.5 小时 = 2.73 天**；0.61 epoch

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

```bash
cd /home/vesita/coding/my/nanoSeek
mkdir -p out/base_v2
HSA_OVERRIDE_GFX_VERSION=10.3.0 HSA_ENABLE_SDMA=0 \
  setsid nohup .venv/bin/python -u training/train.py configs/base_v2.yaml \
  > out/base_v2_train.log 2>&1 < /dev/null &
```

**日志必须放在 `out_dir` 之外**（这里是 `out/base_v2_train.log`，不是 `out/base_v2/train.log`）。
原因见 §8 铁律 9：shell 在进程启动前就创建了 `out/base_v2/train.log`，而
`_backup_old_run` 会把 `out_dir` 里的**所有**文件挪进 `old/` —— 进程的 fd 跟着被挪走，
`out/base_v2/train.log` 永远是空的（人会被这个假象骗很久）。

**当前状态**：PID 297961，`pgrep -f "training/train.py configs/base_v2"` 可查。

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

## 6. NDB：方向已重定位为「可读可写的后缀 n-gram 库」

### 6.1 完整实验史在 `dev-notes/78-残差神经数据库P0P2落地与数学优化路径.md`（1065 行）

| 阶段 | 写什么 | Δ | 结论 |
|---|---|---|---|
| ResidualNeuralDB + PKM | 惊讶门控写残差向量 | ≈0 | 失败 |
| P0/P1'/P4' 冻结基座 | 同上 | −0.046（冻结）/ −0.0025（co-train）| 见下 |
| **0.7-epoch co-train** | 同上，**在线写** | **−0.0002** | **DB 被基座吸收** |
| v5 神经元级 | 建库一次，**只读** | −0.0046（oracle −0.124，捕获 3.7%）| 弱 |
| v6 三级 backoff | 同上 | −0.0046 | 覆盖 24.2%→32.6%，Δ 仅 1.18× |
| **§12.3 纯检索 n-gram** | **下一 token（零梯度）** | **−0.0161** | **比残差式好 50×** |

### 6.2 ★ 为什么"写残差"注定失败 —— 定理级（§11.4 / §12.2）
> 写目标 `r = e_y − E_p[e]`，槽位读出 ≈ `E[r|h]`。基座在 h 条件下校准 ⇒ `E[e_y|h] = E_p[e]`
> ⇒ **`E[r|h] = 0`** ⇒ 注入 ≈ 0。
> *残差 DB 只能学"基座的条件系统偏差"，而梯度下降正在把这份偏差清零；
> **它对"基座无法表示的关联"没有任何入口，因为写目标被 `E_p` 过滤掉了。***

实测印证（§10.2）：注入幅度 **11.5%（冻结旧基座）→ 1.0%@2000 → 0.60%@4000**，Δ 同步归零。

### 6.3 ★ 什么才有效 —— **全量探针实测（2026-09-10，已推翻 §12.3）**

**§12.3 的覆盖率结论作废。** 它用 100M 顺序前缀建表，得到 L=8 覆盖率 4.9%；
全量 937.8M token 建表后，**同样 M=67M 的覆盖率是 99.99%** ——
§13.1 警告的"前缀采样严重低估"被实测坐实（差 20 倍）。

```
【覆盖率（全量流式，位图）】val 位置的后缀在 train 里出现过的比例
  L \ M      4.2M     16.8M     67M      268M
   6       100.00%  100.00%   99.96%   91.86%
   8       100.00%  100.00%   99.99%   93.49%
  12       100.00%  100.00%  100.00%   94.95%
  ★ 小 M 的 100% 是「槽已饱和」（负载因子 = 937.8M/M），不是「覆盖好」。
    覆盖率不是瓶颈 —— 见下表：M 从 67M 加到 268M，覆盖率**降**了 6.5 个点，
    收益却几乎翻倍。

【插值 Δ（kNN-LM 凸组合，base = out/ndb_run/last.pt step19000）】
  口径说明：A_ungated = §12.3 原口径（无门控）；B_oracle 用到标签，是**离线上界**；
            C_uncert / D_disagree 只用推理时可获得的信息，**可部署**。
  配置(L/M)     槽命中  覆盖内top1命中  最优λ(无门控)  无门控Δ    可部署Δ(C)
  L=8  /  67M   99.80%     18.52%          0.03       −0.0354    −0.0380 @λ=0.05
  L=8  / 268M   93.31%     23.27%          0.05       −0.0685    −0.0723 @λ=0.08   ← 最优
  L=12 / 268M   94.68%     15.72%          0.03       −0.0418    −0.0459 @λ=0.05

  ★ 三组都在 `--lams 0.01,0.02,0.03,0.05,0.08,0.12` 上扫过；最优 λ 都落在网格内部
    （不是边界），所以这几个 λ* 可信。注意 §12.3 报的 λ*=0.2 **不可用**。
```

**结论（四条，全部反直觉）**
1. **瓶颈是槽碰撞，不是覆盖率。** M=67M 装 937.8M 个位置 → 负载因子 14，
   一个槽的"多数续写"混了约 14 个不同 8-gram → top-1 命中率只有 18.5%。
   把表加到 268M（负载因子 3.5，已填 89.6%，接近"一槽一个 8-gram"）→ 命中率升到 23.3%，
   **Δ 从 −0.035 翻到 −0.069**（覆盖率反而从 99.8% 降到 93.3%）。
2. **更长上下文（L=12）反而更差**：命中率掉到 15.72%，Δ 从 −0.069 退到 −0.042。
   原因：12 字后缀太"平凡"，同一后缀几乎不重复 → 每个槽平均观测量很低 →
   所谓"多数续写"退化成个别样本的噪声。
   ⇒ **L=8 是甜点，不要往上加。**（§12.3 报 L=12 略优于 L=8，那也是前缀采样的假象。）
3. **23.27% 是 L=8 后缀法的本征上限**，不是表的限制。268M 槽已接近无碰撞，
   剩下的 77% 是**语言本身的歧义**（8 个汉字不能唯一确定下一个汉字）。
   要再往上只能换机制：**软检索（用 top-K 计数分布取代 hard one-hot）** —— 
   §12.3 早就测过 `w_soft = count/total` 比 hard 好。当前探针只实现了 hard。
4. **可部署的置信度门控几乎能吃到 oracle 的全部收益**：
   C_uncert（只按模型自身最大概率门控，推理时可得）−0.0723 vs
   B_oracle（用标签，不可实现）−0.0789 → **捕获 92%**。
   这直接支撑 NDB v7 的读接口设计（§6.5）。

**最佳配置（下一步实现 NDB v7 用这个）**
```
L = 8              槽位 M = 268M（约 2.1GB，本机 11GB 可用内存装得下）
λ = 0.05（无门控）/ 0.08（带 C_uncert 置信度门控）
门控：λ_eff = λ · (1 − p_max) · 1[槽非空]        ← 可部署，捕获 92% oracle 收益
```

**⚠ 本次 Δ 是**保守估计**：基座 `out/ndb_run/last.pt` 是在 **v1 语料**上训的，
却在 **v2 val** 上评（base CE = 2.795，而它在自己 val 上的纯 CE 是 1.237）。
换成 v2 基座后 Δ 会变多少**未知**，必须重测（见 §10）。

**产物**：`out/ngram_full/result.json`（原始覆盖率）、
`out/ngram_full/delta_L8_M67M_finelam.json`、`delta_L8_M268M.json`、
`out/ngram_full/sweep.log`（汇总）；
复现脚本 `scripts/run_delta_sweep.sh`，探针 `scripts/ngram_capacity_probe.py`。

> 注：`out/ngram_full/top1_*.npz` 是 top-1 表的磁盘缓存（合计 **~4.7GB**），
> 留着可以让 Δ 实验省掉 90 秒建表；不需要时可直接删（会自动重建）。

### 6.4 已确认的负结果（不要重犯）
| 负结果 | 证据 |
|---|---|
| **神经元向量检索** | 检索到的 h_j 与当前 h_t **cos = 0.824**（82% 冗余），注入**有害**；唯一略有用的是状态转移 nn_d（−0.0012），弱 600×（§14.8）|
| **未来多步隐藏态作 value** | 最强方向 p1 只有梯度的 **1/15**（§14.7）|
| **注入层提前 h[-2]→h[-6]** | oracle ×2.35，但 R² 腰斩（3.1%→1.4%），**端到端打平**（§14.13）|
| **多向量槽 mixture K=2** | 内存 2×，收益 **0.54×**（§14.12）|
| **缩表堆 count** | count 1.57→2.48，但覆盖 24.2%→17.6%，**Δ 反而差 2.8×**（§14.9）|
| **`Δh` 作 value** | 比 `−∂CE/∂h` 弱 **20×**，且与梯度方向 cos ≈ −0.02（§14.2）|
| **惊讶作写判据** | q0.1（只写最惊讶 10%）**最差**，q0.5 最好（§8.7）|

### 6.5 用户判定的 NDB 形态：**必须支持读和写**
现有 v5/v6 是「`--mode build` 建库一次 + 只训读接口」，**在线写被去掉了**。
用户明确要求补上写。

**建议的 v7 形态**（避开上表所有陷阱）：
```
【写】训练/推理时增量，no_grad，零额外前向
   位置 t：key = hash(tokens[t-L+1 : t+1])，多级 L ∈ {12, 8, 6}
          槽内 value = {token_id → count} 的 top-K + total
   不做惊讶筛选

【读】训练与推理同一套
   q = hash(当前后缀) → 多级回退 → p_ng
   λ = σ(gate(h)) × conf(覆盖, count) × max(0, 1 − p_model)
   p = (1−λ)·p_model + λ·p_ng       ← 凸组合，恒合法、不发散

【表】16.8M 槽起步（token 计数 3–6 B/槽 ≈ 100MB）
```
**为什么避开所有陷阱**：写的是**经验续写 token**（不经过 `E_p` 过滤）；键是**离散后缀**、
值是**离散 token**（与 h 无关 → 不重蹈 §14.8）；库是**数据统计量**（不随基座收敛归零 → 不重蹈 §10）；
用**计数累加**（不是梯度式槽更新 → 不重蹈 §8.4 的发散）。

**已有代码雏形**：`model/residual_neural_db.py:210 read_knn`、`model/product_key_memory.py:106 token_ids`
buffer、`local/ngram_memory_probe.py`、`local/ngram_sample_capacity.py`、`local/ngram_value_capacity.py`。

### 6.6 ⚠ 一个必须正面处理的矛盾
§14.1 记着用户判定：**"token 级 DB 与 harness 记忆工具无本质区别"**，于是转向神经元级。
**但数据说反了**：token 级 −0.0161 / 神经元级 −0.0046 / 残差式 −0.0003 —— **token 级好 3.5~50 倍**。
诚实结论：**机制本身不新**（kNN-LM 2020 / RETRO 2022）；**真正原创的是"否定面"** ——
校准定理、λ 判据 `A/p−1`、以及 §6.4 那张负结果表。

---

## 7. 任务状态

### ✅ 任务 1（已完成 2026-09-10）：全量 n-gram 容量探针
**结论见 §6.3。** 一句话：覆盖率从 §12.3 的 4.9% 修正到 **99.99%**，
但真正的瓶颈是**槽碰撞**；槽位 67M→268M 让 Δ 从 −0.035 翻到 **−0.069**。

**执行记录（给下一个人少踩坑）**：
- 原脚本 `hashes()` 一次性建 (836M,) uint64 + `lexsort` ≈ 17GB → 必 OOM。
  改成**位图**（固定槽表，正是真实 NDB 的形态），内存从 O(N) 降到 O(M/8)。
- **GPU Hang 事故**：第一次上 GPU 时把显卡跑挂，内核 MODE1 reset，
  `VRAM is lost`，**连带把桌面一起打死**。根因是显存峰值（槽表常驻 537MB +
  每个 λ 克隆整个 (N,8192) 概率矩阵）。加固后峰值 **0.93GB**：
  查表移到 numpy 侧、目标概率从 O(N·V) 降到 O(N)、每 N batch 打进度、
  top-1 表落盘缓存、默认 `--device cpu`（CPU 前向只要 ~10 分钟）。
- **证伪 NaN 假说**：同一 checkpoint 前向 `finite=True`、`max|logit|=13.9`，
  两次成功运行 `nonfinite_logits = 0`。
- 全量跑完约 4 分钟（覆盖率 3 分钟 + 评测 1 分钟）。

### ⏭ 下一步（NDB 方向，按优先级）

1. **换 v2 基座重测 Δ**（**必须**）：本次用的 `out/ndb_run/last.pt` 是 v1 语料训的，
   却在 v2 val 上评（base CE 2.795 vs 它自己的 1.237）。Δ 会变多少未知。
   新基座（`configs/base_v2.yaml`，70000 步）起来后重跑 `scripts/run_delta_sweep.sh`。
2. **软检索取代 hard top-1**：§6.3 结论 2 说 23.27% 是 L=8 的本征上限，
   但那是 **top-1 硬 one-hot** 的命中率。§12.3 早就测过 **top-3 计数的软权重**
   （`w_soft = count/total`）比 hard 好。当前探针只实现了 hard —— 加软权重是最直接的提升方向。
3. **L=12 / 更长 + 大表**：B 组（L=12, M=268M）在跑，看更长后缀能否突破 23% 歧义上限。
4. **然后才是 NDB v7 读+写实现**（§6.5 的形态；避开 §6.4 全部负结果）。

---

## 8. 运维铁律（违反过，别再犯）

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
   harness 的超时会 SIGTERM 整条命令，**连带杀掉刚 `setsid nohup` 出去的后台作业**。
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
     --out_dir=out/_smoke --device=cpu --compile=false \
     --batch_size=2 --gradient_accumulation_steps=1 \
     --max_iters=2 --eval_interval=1 --eval_iters=1 --eval_train_split=false
   ```
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

### 11.1 单元测试：0 → 289 条

```bash
uv run pytest                 # 289 条，跳过 slow 时 < 1 秒
uv run pytest -m 'not slow'
uv run ruff check             # lint 门禁（同时也是 pytest 里的一条测试）
```

布局与"每条为什么值得测"见 **`tests/README.md`**。测试抓住的真 bug：

| 测试文件 | 抓到的真问题 |
|---|---|
| `test_model_smoke.py` | 参数量被 `state_dict()` 虚高；`targets=None` 只返回最后一位（**我的因果性测试因此一度是空测试**） |
| `test_ngram_delta.py` | Δ 评测忘了把 `(B,T,V)` 展平成 `(N,V)`；λ 门控语义被偷换 |
| `test_config_loader.py` | `extends` + `base` 同时出现时 `base` 泄漏成全局变量 |
| `test_lint.py` | `GATES`→`GATE_NAMES` 改名漏改，4 分钟评测跑完才崩 |
| `test_diag.py` | 显存诊断用错守卫，`--device=cpu` 时崩在 step 0 |

### 11.2 lint 门禁：只开"一定是 bug"的四条

`pyproject.toml` 的 `[tool.ruff.lint] select = ["E9", "F821", "F811", "F823"]`。
全项目实测这四条合计只有 1 处真实违规 → 可以设**零容忍**而不需要 `noqa`。
故意不开 `F401`（82 处）、`F841`（27 处）与风格规则 —— 噪声会让门禁变成摆设。

### 11.3 结构优化（都是行为保持的）

- **抽纯函数出模块级脚本**：`training/schedules.py`（`lr_at` / `with_data_prefix` /
  `pick_bin_names`）、`training/diag.py`（`mem_debug_line` / `should_dump_snapshot`）。
  原因：`train.py` 是 1466 行模块级脚本，**`import` 它就等于开训**，里面任何函数都无法单测。
- **消除 5 处重复的数据文件名解析**：那个
  `'train_char.bin' if char_level else 'train_byte.bin' if byte_level else 'train.bin'`
  以前重复 5 遍，加一个新模式要改 5 处、漏一处就让"换数据集"静默失效。
  现在唯一解析点是 `pick_bin_names()`，并有 AST 测试保证 `train.py` 里没有第二处硬编码。
- **删死代码**：`main_bin`（赋值后从未使用，一度让人误以为 `data_prefix` 没生效）、
  `_ds`（收敛后无调用点）、`model/attention.py::k_glob`（HCA 路径算了两次用零次）、
  以及 3 处测试里的未使用导入。
- **配置零漂移核实**：`configs/base_v2.yaml` 与旧基线 `out/ndb_run/config.yaml`
  逐键比对，13 处差异**完全等于** §5 记录的 10 项 + 2 个新键（`data_prefix`、
  `muon_ns_aggressive`），无意外改动。现在有测试把这个对照固化
  （`test_base_v2_matches_documented_decisions`）。

### 11.4 技术债清单：`TECH_DEBT.md`

本轮已偿还 6 类、待还 9 项（按"代价 ÷ 修复成本"排序）。最大的一笔仍是
**`train.py` 的 1466 行模块级脚本**，其次是 `local/` 57 个一次性脚本无索引、
`training/rl/`（2704 行）与本项目当前目标无关。

---

## 12. NDB v7 原型：**可读可写、no_grad、模型自己决定**（2026-09-10 新增）

### 12.1 在哪
```
model/ngram_ndb.py           模块（no_grad 表 + 可学习读写门控），约 440 行
tests/test_ngram_ndb.py      20 条测试
scripts/ndb_online_ab.py     在线 A/B（三臂：off / frozen / joint）
```

### 12.2 与 v5/v6 的根本区别（三处，都针对已被证伪的旧范式）
```
              v5/v6（残差式 / 神经元级）           v7
  键          隐藏态 h                             离散 token 后缀哈希
  值          e_y − E_p[e]（残差向量）              token 计数的 top-K
  寻址        可微（PKM 的 key_proj 可学）          离散 + no_grad（hash 直查）
  写          建库一次 / 写残差（被 E_p 过滤）      在线增量计数（数据统计量）
```
理由就是 §6.2 的校准定理 + §6.4 那张负结果表。**表的内容永远是数据统计量，
不进 `parameters()` / `state_dict()`** → "不增加模型大小"成立（部署模型仍 75.15M）。

### 12.3 「模型自己决定读写」落在哪
```
【写】模型决定写多重   w_t = σ(W_w · h_t)              ← 可学习写门控
【读】模型决定信多少   g_t = σ(W_r · [h_t; 槽统计量; w_t]) ← 可学习读门控
【级】模型决定信哪级   α = softmax(level_weight)        ← 可学习多级混合
       p = (1−g_t)·p_model + g_t·Σ_L α_L·p_ng^(L)
      槽统计量 = [log(1+total), top1占比, 槽是否非空]（全是推理时可得、无标签）
```

**梯度路径（诚实说明，别假装是严格端到端）**
```
✓ ∂L/∂W_r ≠ 0           读门控直接可微，路径干净
✓ ∂L/∂level_weight ≠ 0  多级混合直接可微
△ ∂L/∂W_w ≠ 0           **但不是通过"写"来的**：离散表阻断了跨 batch 的梯度链。
      本实现让 w_t 同时作为 read_gate 的输入特征 → W_w 从读侧拿到真实梯度。
      是合理的归纳偏置（"想读的地方就多写"），**不是**严格的写信用分配。
```

### 12.4 已做的验证
- **20 条单测全绿**（含"增量写 == 离线建表逐位对照"）
- **端到端冒烟通过**：`scripts/ndb_online_ab.py` 用退化配置
  （M=4M / 预热 20M token / 3 步）跑完 off+frozen 两臂，32 秒，
  输出配对 Δ 与 JSON。⚠ 那个 Δ 是**正的**，因为配置退化（碰撞严重 + 只训 3 步），
  **不是结论**；要看结论必须用 `--slots 268435456 --prewarm_m 900 --steps 300`。
- 测试期间抓到 **4 个真 bug + 2 个设计缺陷**，全部会静默给出错误检索结果：
  1. `p_ng` 未归一化（top-K 计数之和 < total）→ `p_new` 总和只有 0.88
  2. 未覆盖位置的读门控没归零 → `p_new = (1−g)·p_model` 总和 < 1
  3. 表初始化成 0 而非 -1 → 空槽里 token 0 是**合法词表 id** →
     第一次 flush 时每槽凭空多出 K 个 `(token=0, count=0)` **幽灵候选**
  4. int32 计数每次 flush 四舍五入 → 累积丢精度（5.0 vs 5.848）。改 float32，内存不变
  5. `read_gate.weight` 全零初始化 ⇒ `∂g/∂w_t = g(1−g)·W_r[:,−1] = 0`
     ⇒ **写门控在 init 时梯度恒为零**（死启动）
  6. `flush` 里多余的 `_toks[us_s[sel]] = 0`（`_cnts` 无对应清理）→ 又一批幽灵条目

  **其中 3/4/6 只有靠"增量写 == 离线建表逐位对照"才抓得到** —— 它们不崩、
  不让 loss 异常，只是让表里多几个看起来无害的条目。这是 skill 里
  "对照必须能区分"那条纪律的直接价值。

### 12.5 已知未做（下一步，见 §10）
- **没有接进 `train.py`**。接入时的头号风险是 **val 泄漏**：`train.py` 的同一个
  forward 同时服务训练 / `estimate_loss` / 健康体检，若把"写"挂在 forward 上，
  验证集会写进训练库。模块已设计成**写路径默认关闭**，必须显式
  `with ndb.write_enabled():` 只包训练 micro-batch
  （`tests/test_ngram_ndb.py::test_write_is_disabled_by_default` 是防线）。
- **表不进 checkpoint**（2.4GB 太大）。所以**续训会丢 DB**，
  目前靠"离线预热可重建"绕过。这需要在正式接入前决定策略。
- **在线 vs 离线的 Δ 对照还没跑**（`scripts/ndb_online_ab.py` 只冒烟过）。
  这是 NDB 方向最关键的一个数字：离线的 −0.0723 能否兑现。
