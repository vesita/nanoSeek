> ⚠ **本文件已被 `PROJECT_STATE.md`（仓库根目录）取代为总入口。**
> 那份包含：新数据集 v2、新基座配置 `configs/base_v2.yaml`、架构审计（mHC 逐位无效）、
> 速度归因（−32%）、NDB 方向（**当前 = RETRO-lite，见 `dev-notes/79`**；本文件不再维护方向结论）、运维铁律、下一步任务。
> **上下文压缩后请先读 `PROJECT_STATE.md`，本文件只保留 NDB 共训阶段的历史细节。**

# NDB 长跑状态 —— 压缩上下文后**先读这一个文件**，不要凭记忆重启

## 一句话目标
接续基座训练（step 12000 起）共训 NDB 记忆读取接口，跑到 max_iters=70000。
核心问题：让基座自己学会「用记忆」，而不是把内容吸进权重。

## ★★★ 免训练判别探针（2026-09-10 晚，`local/eval_ndb_probe.py`，GPU）
用 step 19000 检查点，**只改推理期读取结构，不训练**。256 打包 val 窗口 =
39034 有效 token，base_off = 1.2370。日志：`probe.log` / `probe_gate_conf.log` / `probe_sim.log`。

**P1 读取深度饱和在 k≈8**（Δ 相对 base_off）
| k | 1 | 2 | 4 | **8** | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| Δ | −0.0269 | −0.0486 | −0.0609 | **−0.0642** | −0.0637 | −0.0604 | −0.0550 |
k=1 已拿到 44%，k=8 见顶，再大反而变差 → **加 k 不是容量解法**。

**P2 单个坏条目的稀释远慢于 1/k**
| k | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| 1 槽污染代价 | +0.1123 | +0.0947 | +0.0857 | +0.0691 | +0.0609 |
16 倍 k 只降 46%（1/k 应降 94%）。即使 k=64，错 1 个槽（1.6%）记忆仍净有害。

**P3 k=4 时模型完全不认排名**
逐槽污染代价：槽0 +0.1209 / 槽1 +0.1200 / 槽2 +0.1175 / 槽3 +0.1214（**几乎相同**）；
全 4 槽随机才 +0.2215（1.83×）。**一个坏条目比完全不用记忆还差 2 倍**（关记忆仅 +0.0592）。

**P4 现门控部分有效但无法拒收**
gate mean 0.360 std 0.244；corr(gate, Δ) = **−0.221**；最高 gate 分位（20% token）吃掉
**69.7%** 总增益。但 corr(gate, base_loss) 仅 +0.074，最低分位仍有增益。

**P5 检索相似度不预测增益**
corr(sim_top, 增益) = **+0.003**（分桶完全平坦）；用 sim_top 阈值当门控：留出集 −0.0643
vs 全用 −0.0615（**无效**）。完美逐 token oracle 门控上界 Δ = **−0.1032**（仅 1.68×）。

**P6 修法验证：把检索相关性送回注意力 logit（λ 扫描，推理期直接设）**
| λ | 干净 Δ | 1 槽污染 Δ | 全随机 Δ |
|---|---|---|---|
| 0.0 | −0.0612 | **+0.0455（有害）** | +0.1329 |
| 1.0 | −0.0593 | +0.0170 | +0.1237 |
| 2.0 | −0.0542 | **−0.0075（转正）** | +0.1127 |
| 4.0 | −0.0453 | −0.0346 | +0.1030 |
| 8.0 | −0.0390 | **−0.0492** | +0.0997 |

鲁棒曲线 λ=2 vs λ=0：10% 错误保留 **0.62 vs 0.35**；25% 错误 **+0.24 vs −0.36**。
→ **用 11% 干净增益换到「检索出错时记忆仍然有益」。**

**根因一句话**：检索用 sim 选出条目后就把 sim 扔了，注意力只用学出来的 wq/wk 重打分，
所以随机条目和正确条目抢注意力的机会几乎相同（P3 的排名无差别即此）。

**代码改动**（均在 `model/memory_cross_attn.py` + `training/train.py`）
- `sim_gain`（可训标量，`ndb_att_sim` 配置，初值 0）：把「条目与查询余弦相似度的 z 分数」加进注意力 logit。
- `retr_noise`（`ndb_retr_noise` 配置）：训练期按概率把槽换成随机条目（train-aware）。
  **必须用按步种子 `noise_seed` 的独立 Generator**——本仓库开了 gradient_checkpointing，
  反向会重算前向，用全局 RNG 会让重算抽到不同掩码 → 梯度出错。
- 两道保险：`estimate_loss` / `ndb_eval` 期间把 `retr_noise` 置 0（监控口径干净）；
  NDB 优化器状态与参数组不匹配时重建（新增 `sim_gain` 时实测会触发）。


## ★★★ 续训陷阱与数据异常（2026-09-10 晚，动手改配置前必读）

### 1. `model_args` 强制覆盖：续训时改 config.yaml **有一半不生效**
`_build_model_from_checkpoint`（`training/train.py:476-492`）对一长串架构开关执行
`model_args[k] = checkpoint_model_args.get(k, model_args[k])` —— **checkpoint 里有值就用 checkpoint 的**。

| 键 | 续训改 config 有效？ |
|---|---|
| `dropout` | ✅ 有效 |
| `gradient_checkpointing` | ✅ 有效 |
| `ndb_store` / `ndb_top_k` / `ndb_att_sim` / `ndb_retr_noise` | ✅ 有效（NDB 模块直接读当前 config） |
| **`use_mtp` / `mtp_weight`** | ❌ 被 checkpoint 覆盖 |
| **`swiglu_clamp`** | ❌ 被覆盖（checkpoint 里就是 0.0） |
| `hc_mult` / `use_moe` / `n_experts` … | ❌ 被覆盖 |

**要测这些必须先改 checkpoint 的 `model_args`，或把键从强制列表里移出。**

### 2. train / val 分布不匹配（`local/diag_trainval.py`，同采样器同窗口数）
| 数据集 | 纯 CE | 混合 loss | 有效 token | 每批有效 token |
|---|---|---|---|---|
| train | **1.7292** | 2.3349 | 24120 | 753.8 |
| val | **1.2370** | 1.6561 | 39034 | 1219.8 |
| 差 | **+0.4922** | +0.6788 | | |

**训练数据比验证数据难 0.49 纯 CE，且 val 每窗有效 token 多 60%。**
val 上的绝对 loss 系统性偏乐观（模型真实水平更接近 1.73）。
**Δ 是配对的，仍然可用；但绝对数字不能当基准，跨集合外推要谨慎。**

### 3. dropout **不是** train/val gap 的元凶（直接测量否定了这个假设）
`local/diag_loss_parts.py` 同一批 train 数据、6 批 × 8 次：
train 模式（dropout 0.2 开）2.5702 vs eval 模式（关）2.4964
→ **纯 dropout 只贡献 +0.0737（3.0%）**。原来那个「gap +0.44 是 dropout 造成的」判断是错的。

### 4. `mfu` 列不可信
results.csv 里的 0.27 是**假的**：ROCm 上 `props.clock_rate` 读不到 →
`flops_promised` 回退成 A100 的 312 TFLOPS（`model/gpt.py:396-399`）。别用它判断有没有空间。

### 5. DSpark 不能加速训练（结论：不做）
推测解码省的是**自回归解码**的串行前向；训练是 teacher forcing、所有位置并行，
没有那个串行循环可省。论文自己写明 DSpark 用于 *online serving 和 RL rollout*。


## ★★ 评估结果（2026-09-10，step 19000 检查点，CPU / 纯 CE 口径，`eval_all.log`）
256 个打包 val 窗口 = **39034 有效 token**，全部配对（同批比四条件）。

### 1｜容量扩展：换更大的库有没有用 → **没有**
| 条件 | loss | Δ |
|---|---|---|
| base_off（关记忆） | 1.2369 | — |
| **mem 5M** | 1.1757 | **−0.0612** |
| mem 10M | 1.1799 | −0.0570 |
| held-out 5M（未见库） | 1.1879 | −0.0490 |
| held-out 10M | 1.1873 | −0.0496 |
| rand（随机检索） | 1.3736 | +0.1366 |

**库从 5M 翻到 10M，Δ 反而略差 0.004（可忽略）→ 记忆容量不是瓶颈，5M 已饱和。**
这与冻结基座期的结论一致，说明**共训也没能让「更大的库」变得有用**。
`Δ_held/Δ = 0.80`（10M 对照 0.87）→ 读策略可迁移。

### 2｜检索鲁棒性：**悬崖式**
| top-k 被随机替换的比例 | loss | Δ | 占 Δ(0) |
|---|---|---|---|
| 0%（干净） | 1.1765 | −0.0611 | 1.00 |
| 10% | 1.2164 | −0.0212 | 0.35 |
| 25% | 1.2597 | **+0.0222** | −0.36 |
| 50% | 1.3076 | +0.0701 | −1.15 |
| 100% | 1.3884 | +0.1509 | −2.47 |

**只要 10% 的检索出错，就损失 65% 的收益；25% 出错时记忆净有害。**
模型是「无条件信任检索内容」，没有学会「不确定就不注入」——这是当前方案最大的工程风险。

### 3｜增益落在哪：**几乎全部落在基座不确定处**
| base loss 区间 | token 占比 | Δ | 占总增益 |
|---|---|---|---|
| [0, 0.5) | 64.0% | +0.002 | −1.8% |
| [0.5, 1) | 5.0% | −0.046 | 3.8% |
| [1, 2) | 8.2% | −0.120 | 16.0% |
| **[2, 4)** | 11.2% | −0.238 | **43.6%** |
| **[4, 8)** | 9.5% | −0.199 | **30.7%** |
| [8, ∞) | 2.1% | −0.222 | 7.7% |

base loss ≥2 的 token 只占 **22.8%**，却贡献 **82%** 的总增益；占 64% 的「容易 token」上 Δ≈0。
= 记忆被用在了正确的地方（补基座不知道的），这也是期望行为。

### 4｜生成质量（开/关记忆，prompt 取自语料真实位置）
| 条件 | 平均长度 | EOS 收尾率 | rep3 | 出现轮次率 |
|---|---|---|---|---|
| 开记忆 | 57.8 | **83%** | 0.069 | 0% |
| 关记忆 | 48.5 | 67% | 0.063 | 0% |

开记忆更长、显著更会正常收尾；重复率相当。定性看两边都还只是「勉强通顺」，
个别 prompt 有退化重复（如 prompt#2 的「光、光、光…」）。**生成质量不是本项目的强项。**

### 结论一句话
NDB 确实在加有效容量（Δ=−0.061，且落在基座不确定处、可迁移到未见库），
但**加库不能加容量**，且**读得极脆**（10% 检索错误就废掉 65% 收益）。
下一步要提容量，瓶颈在**读取分辨率**（chunk 均值 + top-4 太粗），不在库大小。

---

## 当前 job（2026-09-10 20:4x，★ 与上面旧记录不同：长跑已被人手动停掉）
**长跑已停在 step 19781（用户手动关闭），最后检查点 `ckpt_step_19000.pt` / `last.pt` 均在 19000。**
现在跑的是**短 pilot（各 2000 步，串行）**，目的是几小时内回答设计问题，胜者再回灌长跑。

| 角色 | 状态 | 说明 |
|---|---|---|
| Pilot 1 | `out/pilot_sim/` | λ=2.0（可训）· top_k=8 · 噪声 0 · 19000→21000 步 |
| Pilot 2 | `out/pilot_sim_noise/` | 同上 + 检索噪声 0.15（**待 Pilot 1 跑完再启动**） |
| 监控 | 后台 `sleep` 定时器 | 300→600→1200→2400→4800→9600→18000，**只能后台，禁前台等** |

配置生成器：`local/make_pilot.py <名字> --att_sim X --retr_noise Y --top_k K --steps N`
（会自动设 out_dir / init_from=resume / max_iters / eval_interval）。
**pilot 的 `last.pt` 必须先由 `ckpt_step_19000.pt` 复制而来。**

> 旧记录里的 bash-74/78/79 已随会话重启消失（上下文压缩/模型切换会杀掉后台任务）。
> 找 pid：`ps -eo pid,etime,pcpu,args | grep '\.venv/bin/python -u training/train.py' | grep -v grep`

> 注意：定时器「完成时刻」和「我被唤醒的时刻」可能差好几个小时（实测 sleep 4800
> 11:18 就完成了，19:26 才收到通知）——唤醒实际取决于会话活跃度，不是定时器本身。

## ★ 损失口径（极易读错，务必先看）
`results.csv` / `ndb.csv` / 日志里的 loss **不是纯下一个 token 交叉熵**，而是
**`纯 CE + MoE aux + 0.3 × MTP`**（`model/gpt.py: forward` 里累加的）。
同一批 val 数据实测：纯 CE = 1.3715，模型返回 loss = 1.8375，**差 +0.466**。
所以看绝对值得先想清楚是哪个口径；`ndb.csv` 的 Δ 与纯 CE 的 Δ 一致（−0.074 vs −0.058~−0.064）。
`local/eval_ndb.py` 报的是**纯 CE**。

### ★ 三成分实测拆分（2026-09-10，`local/diag_loss_parts.py`，step 19000，256 打包 val 窗口 = 39034 token）
| 成分 | 数值 | 占报告 loss |
|---|---|---|
| 纯 CE（真正的 LM 信号） | **1.2369** | 74.7% |
| MoE aux（Σ 12 个 block） | **0.0000** | **0.0%** |
| MTP × 0.3 | **0.4191** | **25.3%** |
| 合计（= 报告 loss） | 1.6560 | 100% |

- **MoE aux 是 0**：`use_aux_free_balance: true` 已生效，`moe_aux_weight: 0.01` 不起作用。
  之前担心的「aux 比论文大 100 倍」是**错的**，别再往那儿修。
- **MTP 独占报告 loss 的 25%**。论文 V4.1 明确「**omit the MTP module during backbone
  pre-training**」（V3 是全程联合训）。**去掉 MTP 会让报告 loss 直接掉 ~0.42，那与语言
  建模质量无关** —— 跨配置比较必须用纯 CE。

## 评估脚本（`local/eval_ndb.py`）
四个模式，只读、不写任何检查点，可在 **CPU** 上跑（实测 batch8×block256 一次前向约 1s，
全套约 40 分钟，**不用停训**）：
```
--mode cap     容量扩展：Δ(5M) / Δ(10M) / Δ(heldout5M) / Δ(heldout10M) / Δ(rand)
--mode robust  检索鲁棒性：逐查询随机替换 top-k 的比例 0→1 扫描
--mode pos     Δ 的位置分布：按「关记忆时」per-token loss 分桶，看增益落在哪
--mode gen     生成质量：开/关记忆对比，prompt 取自语料真实位置（q_pos 对齐、防泄漏）
--mode all     依次全部
```
默认用「非空窗口打包」抽样（与训练的 pack_nonempty 同口径）；
加 `--uniform` 可切回均匀抽样对齐旧口径。

## 每次唤醒：一条命令 + 立刻挂定时器
```bash
cd /home/vesita/coding/my/nanoSeek && bash local/ndb_status.sh      # 非阻塞，秒回
```
然后**马上**挂下一个后台定时器（`run_in_background: true`）。
节奏：`300 → 600 → 1200 → 2400 → 4800 → 9600（当前）→ 18000 → 18000…`

**禁止**：前台 sleep、`job_output(wait=true)`、两次唤醒之间前台干等。
等待期间做不用 GPU 的事（读代码、分析 CSV、写文档）。

## 重启命令（整条复制）
```bash
cd /home/vesita/coding/my/nanoSeek
HSA_OVERRIDE_GFX_VERSION=10.3.0 HSA_ENABLE_SDMA=0 \
.venv/bin/python -u training/train.py out/ndb_run/config.yaml \
  >> out/ndb_run/train.log 2>&1
```
配置里已含 `init_from=resume`、`out_dir=out/ndb_run`，直接跑即断点续训。
重启前先确认没有别的 train.py 进程。

### 日志管道三条铁律（2026-09-10 连踩 3 次，别再犯）
1. **管道里绝不放行导向过滤器**（grep/awk/sed）。它们要等 `\n` 才放行，tqdm 只写 `\r`
   永不换行 → 输出全卡缓冲区，日志滞后几十步、看着像卡死。
   实测 `stdbuf -o0 grep … >> file` 跑 6 秒输出 **0 字节**（producer 已产 9.6KB），
   `-oL` / `--line-buffered` 同样无效。
2. **tqdm 默认写 `stderr`**，不是 stdout。所以 `>> log 2>> err` 会把进度条和事件拆两个文件。
3. 正解 = **双重定向同一个文件**，`python -u … >> train.log 2>&1`，逐写实时落盘（已实测）。
   amdgpu 警告会混进来，一个 run 只有 2~3 行，事后 `grep -v amdgpu` 即可。

## 文件索引
| 路径 | 用途 |
|---|---|
| `out/ndb_run/train.log` | **当前 run** 的日志（stdout+stderr 合并，实时） |
| `out/ndb_run/train_run1..4.log` | 历次 run 的历史日志（排障用） |
| `out/ndb_run/train_loss_window.csv` | **训练侧窗口均值**（每 20 步一行，看趋势看这个） |
| `out/ndb_run/ndb.csv` | NDB 监控：delta / delta_rand / delta_held（每 1000 步） |
| `out/ndb_run/results.csv` | train/val loss、lr、mfu（每 1000 步） |
| `out/ndb_run/last.pt` | 每 1000 步存，续训入口 |
| `out/ndb_run/ckpt_step_N.pt` / `best.pt` | 归档 / 最优 |
| `out/ndb_run/mem_snap_*.pickle` | 峰值 >6GB 时自动落盘（含调用栈） |
| `local/ndb_status.sh` | 一键状态（唤醒后跑它） |
| `local/diag_mask_stats.py` | mask 统计（窗口全 mask 比例、集中度） |
| `local/validate_pack_nonempty.py` | 打包非空窗口的离线校验 |
| `local/parse_mem_snapshot.py` | 解析显存快照，定位尖峰 |

## 关键配置
| 项 | 值 |
|---|---|
| 基座 | `out/base_probe/best.pt`（step 12000，81.58M，bf16 autocast） |
| 记忆库 | `out/mem_store/store5mA.pt`（78124 chunks，mean） |
| held-out 库 | `out/mem_store/store5m_heldout.pt` |
| 挂载层 / top_k / chunk / exclude_radius | -6 / 4 / 64 / 512 |
| 记忆 dropout / 接口 lr | 0.1 / 3e-4（固定，独立 AdamW） |
| 基座 lr | 余弦 3e-4→1e-4，lr_decay_iters=30000 |
| batch | 8 × block 256 × grad_accum 4 = 8192 token/步 |
| **pack_nonempty** | **true**（只抽含终止符的窗口） |
| compile / eval_interval / log_interval | True / 1000 / 20 |
| 速度 / 显存 | 稳态 ~3.6 s/步 → 58000 步约 2.5~3.5 天；peak ~3.1GB |

## 实验结果（共训，截至 step 19000）★ 这是当前最硬的正面结果
| step | base_off | mem | **Δ** | Δ_rand | Δ_held | Δ_held/Δ | gate | \|Δ\| |
|---|---|---|---|---|---|---|---|---|
| 13000 | 2.0043 | 1.9630 | −0.0413 | +0.0316 | −0.0391 | 0.95 | +0.074 | 4.18 |
| 14000 | 1.8735 | 1.8236 | −0.0499 | +0.0763 | −0.0435 | 0.87 | +0.053 | 5.24 |
| 15000 | 1.7387 | 1.6820 | −0.0567 | +0.1037 | −0.0490 | 0.86 | +0.032 | 5.14 |
| 16000 | 1.6986 | 1.6373 | −0.0613 | +0.1124 | −0.0520 | 0.85 | +0.015 | 5.81 |
| 17000 | 1.8246 | 1.7585 | −0.0662 | +0.1809 | −0.0543 | 0.82 | −0.002 | 5.78 |
| 18000 | 1.7370 | 1.6659 | −0.0711 | +0.1425 | −0.0602 | 0.85 | −0.020 | 5.80 |
| 19000 | 1.8060 | 1.7322 | **−0.0738** | +0.1793 | −0.0608 | 0.82 | −0.036 | 6.29 |

**读法（4 条）**
1. **Δ 单调变负 −0.041→−0.074**（6000 步）→ 基座在学会用记忆，越来越依赖。
2. **Δ_held/Δ = 0.82~0.95**（判据 >0.8）→ 学到的是**可迁移读策略**，不是背下 5M 库。
3. **对比冻结基座基线 Δ=−0.0348 → 共训 −0.0738，提升 2.1×。**
4. **Δ_rand 强烈为正且增长（+0.03→+0.18）** → 随机检索主动伤害模型。`ndb_dropout=0.1`
   已排除「只看有没有注入」的伪影，说明模型在**读内容**；代价是它选择「信任记忆」而非
   「无关就不注入」，抗检索错误能力弱。

**增长在减速**：每千步 Δ 增量 −0.0086→−0.0068→−0.0046→−0.0049→−0.0049→−0.0027，
可能在 −0.08~−0.10 附近饱和。

**口径提醒（别读错）**
- `results.csv` 的 `val/loss` 是**带记忆**的（`estimate_loss` 跑在 `ndb_eval` 之前，
  `_ndb_state['on']` 仍为 True）→ 对应 `ndb.csv` 的 `mem` 列，**不是 base_off**。
- `ndb.csv` 的 `gate` 是 `sigmoid(gate_proj(h) + gate)` **内部的标量偏置**（初始化 0.1），
  变负只是「平均门控略低于 0.5」，经 sigmoid 永远翻不了号，不是 bug。
- `|Δ|` 是注入向量逐 token L2 范数均值，6/25 ≈ 24%，与冻结期基线一致。

其他：peak 3.09GB、OOM 0、NaN 0、归档检查点齐全、`best.pt` = val 1.7181（带记忆口径）。
速度约 5.0 s/步（含每 1000 步评估），step 19534 时已跑 10.5 小时，剩余约 2.9 天。

## 判据（每 1000 步看 ndb.csv）
- `delta`（mem − base_off）< 0 且越来越负 → 记忆有用
- `delta_rand` ≈ 0 → 学到「无关就不注入」
- `delta_held / delta` > 0.8 → 学到的是**可迁移读策略**
- `base_off` 单独大降而 `delta` 变小 → 基座在**吸收**内容（NDB 变装饰）
- 出现「显存不足，跳过该步」→ 尖峰又来了，看累计次数 + 是否落快照
- 出现「非有限值」→ NaN 步，看 §已修 bug
- `delta` 是**配对评估**（四条件同一批 val batch），不再有 ±0.15 采样噪声

---

## 本轮两个修复（2026-09-10 第二轮）

### 背景：进度条 loss 在 0.77 / 3.40 / 2.25 之间乱跳，像不收敛
`local/diag_mask_stats.py`（20 万窗口）实测：

| 指标 | 值 |
|---|---|
| 全 mask 窗口 | 82.6% |
| 全库参与 loss 的 token | 6.76% |
| 单步（32 窗口）有效 token | mean 549 / p5 118 / p95 1061 |
| 单步最大窗口占该步有效 token | mean **41%** / p99 **100%** |
| 前 10% 窗口占全部有效 token | **90.7%** |

即「单步 loss」= 2~5 个随机回复文档的 loss。叠加 `log_interval=20`（tqdm 每 20 步才重设），
逐点看就是噪声。参考量级：`out/nanoseek_100m/results.csv` 7800→24800 步 train 3.70→2.94，
**每 1000 步只降 ~0.045** → 几十步本来就不该报「不下降」。

### 修复 1：`pack_nonempty`（`training/train.py: _sample_nonempty_ix`）
终止符 `<eos>/<cont>` 自身必然有效（`training/masking.py`：`next_term=自身 < next_nl`），
故「窗口内含终止符」⟺「窗口有 ≥1 有效 token」。抽样改拒绝采样只抽这种窗口 →
**目标函数逐位不变**，只是不把算力花在全 mask 窗口。
校验（`local/validate_pack_nonempty.py`）：6400 窗口 0 个空；**有效 token/步 549→3174（×5.8）**；
条件分位数与「均匀∩非空」重合。连带：eval 有效 token 2.8 万→16 万，噪声 ÷2.4。
代价：不再有整批被 `n_i==0` 跳过的 microbatch，步时 +~30%。

### 修复 2：窗口均值日志
tqdm `损失=` 改为「最近 log_interval 步的 token 加权均值」，每 20 步落一行
`train_loss_window.csv`（`step, window_mean, window_steps, last_step_loss, grad_norm, lr`）。
顺带把 `[mem]` 调试行从每步改成每 20 步（日志长度减半）。

---

## 已修 bug 清单（全在 2026-09-10）
1. **loss 归一化**：一步内先取齐所有 microbatch → 按 `n_i/n_total` 加权 → 全步严格 token 级均值；
   `n_i==0` 跳过；`estimate_loss`/`ndb_eval` 同口径。
   **→ 因此 train/val loss 与历史数字不可直接比。**
2. **续训重复存档**：加 `_resume_iter` 跳过载入那步的评估/存档（否则白写 ~2GB + 消耗 RNG）。
3. **q_pos 回归**：逐 microbatch 复位 `micro_qpos[micro_step]`，否则 `exclude_radius` 失效 → 答案泄漏。
4. **NDB eval 配对**：四条件同一批 val batch，Δ 不再是纯噪声。
5. **pack_nonempty**（本轮）。
6. **窗口均值日志 + 管道实时化**（本轮）。

另有：OOM 安全网（`torch.OutOfMemoryError` → 清梯度/缓存 → 跳过该步）、
NaN 步同时清 NDB 梯度、`grad_norm` 初始化。

## 未解决：基座前向的一次性显存尖峰
- 现象：NDB-off 对照里某步 peak 从 4.5GB 一步跳到 **7.36GB**；`dynamo.unique_graphs` 没变
  → 不是重编译；合成退化 batch 不复现；NDB 只占 0.2GB。
- 重建 RNG 流复现 3 次都没对上（NDB 模块的 `nn.Linear` init 也消耗 CPU RNG）。
- **策略：不猜、不打补丁。** 快照已武装（`mem_snapshot_gb: 6.0`），尖峰再现会自动落
  `mem_snap_<step>.pickle`，用 `local/parse_mem_snapshot.py <pickle>` 解析。
- `model/mlp.py` 已恢复原样（不再 disable 编译）。

### ★★★ 2026-09-10 21:0x 审查：为什么「原因没排查出来」—— 两个诊断机制同时沉默
不是运气问题，是**两条信息通道都被堵死了**：

1. **OOM 处理器丢弃 traceback**（微步循环的 `except torch.OutOfMemoryError`）：旧版只打一行
   `⚠ 显存不足，跳过该步` 就 `break`，**分配现场的操作栈根本没进日志**。历史日志里清一色
   `oom=0`、`peak=3.06~4.79`，看不出任何线索。
2. **快照触发条件只看 allocated**：`max_memory_allocated() > mem_snapshot_gb(6.0)`。
   实测 peak 只有 3.06–4.79GB → **永远不会触发**。核查确认：全仓库 `mem_snap_*.pickle`
   **一个都没有**。

**碎片型 OOM**（`reserved` 涨满、`allocated` 没到阈值）恰好让这两个机制**同时沉默** ——
正对应 PyTorch 报错里那句 `try setting expandable_segments:True`。
**推论（待证）：那次"莫名 OOM"很可能是碎片，而不是真申请了 7GB。**

已修（本轮）：
- OOM 时前 3 次落盘 `oom_dump_<step>.txt`（完整 traceback + allocated/reserved/peak +
  batch/grad_accum/grad_ckpt/use_moe/use_aux_free_balance/use_mhc/use_mtp 现场）
  + `oom_snap_<step>.pickle`；
- 快照触发改为 `max(max_memory_allocated, memory_reserved) > 阈值`，日志同时打印两者。

⚠ **因此「峰值 3.10G / 7.98G = 61% 空闲 ⇒ 可以加 batch」这个推论暂缓**：
3.10G 是**常态**峰值，不是**安全**峰值。加 batch 之前必须先解释尖峰。

## 已确认基线（冻结基座，2000 步）
- Δ(5M) = −0.0346 ≈ Δ(10M) = −0.0348 → **库在 5M 已饱和**
- 库轮换 Δ_held/Δ = 0.95 → 冻结基座阶段已是「读策略」
- |Δ|/|h| ≈ 24%，注入尺度扫描最优 scale≈1.0
- 库 key 有效秩 PR=43.9；top-1 cos 0.862@10M / 0.855@5M

## 其他
- `training/train.py`、`model/memory_cross_attn.py` 有改动；NDB 与 pack_nonempty 全部 opt-in，
  `ndb_store=''` + `pack_nonempty=false` 时与原版逐位一致。
- 暂停：kill 训练 job 即可，`last.pt` 每 1000 步落盘。
- 经验已固化：skill `longrun-train-monitor`（§0 后台定时器铁律、§3 日志管道坑、
  §5 显存尖峰方法论、§7 续训确定性、§8 loss mask 陷阱 + 窗口打包）。

---

## ★★★ 步时归因与三个被推翻/被证伪的假设（2026-09-10 21:0x 审查）

### 1｜参数量真值是 **81.58M**，不是 90.19M
`state_dict()` 求和会**重复计数**：`gpt.py:83` 把
`transformer.wte.weight = lm_head.weight`，`gpt.py:87` 又把
`mtp_head.weight = lm_head.weight` —— 三名字同一张量，却列 3 遍，每次多算 2×4.19M。
判据：训练日志自己打 `参数量：81.58M`；`sum(p.numel() for p in parameters())` = 81.58M。

| 组 | 参数 | 占比 |
|---|---|---|
| FFN / MoE 专家 | 62.88 M | 77.1% |
| Attention (MLA) | 8.06 M | 9.9% |
| MTP 模块（第 13 层，不含共享头）| 6.44 M | 7.9% |
| 词嵌入+输出头（三者共享 1 张量）| 4.19 M | 5.1% |
| 1D 归一化/缩放 | 0.01 M | 0.0% |
| **合计** | **81.58 M** | |

去掉 MTP 的主干 = 75.15M。

### 2｜`gradient_checkpointing: true` **从来没生效过**（本轮最重要的发现）
`gpt.py:166-168`：
```python
use_ckpt = getattr(self.config, 'gradient_checkpointing', False) and self.training
if use_ckpt and self.config.use_moe and getattr(self.config, 'use_aux_free_balance', False):
    use_ckpt = False
```
而本配置 `use_moe=true` 且 `use_aux_free_balance=true` → **恒为 False**。
原因（注释里写了）：aux-free 的 `router_bias` 在前向里就地更新（no_grad 副作用），
与 checkpoint 重算不兼容（重算时 bias 已变 → 路由不一致 → CheckpointError）。

**后果 1**：`local/bench_train_variants.sh` 的 `nockpt` 变体是**空测试** —— 两个 arm 都是关，
测出 4.99 vs 5.01 是必然的，不构成"梯度检查点无杠杆"的结论。**该结论作废。**
**后果 2**：峰值显存 3.10G / 7.98G = **只用了 39%**，而且这是**不开检查点**的数字。
→ **micro-batch 形状是一条从未探索的杠杆**（见下）。

### 3｜config 层四个变体全部无杠杆（60 步，单 seed，误差约 ±0.5%）
| 变体 | s/it | 峰值显存 | 可信度 |
|---|---|---|---|
| base | 5.01 | 3.10 G | — |
| `gradient_checkpointing=false` | 4.99 | 3.09 G | ✗ **空测试**（见上） |
| `dropout=0.0` | 4.98 | 2.98 G | 差 0.6%，单次跑不能断定非噪声 |
| `ndb_store=""` | 崩溃 | — | `--set ndb_store=` 写出 YAML `None`，被 `config_loader.py:69` 类型检查拒；正确写法 `--set 'ndb_store=""'` |

### 4｜两个不能信的指标
- `mfu`（`gpt.py:396-399`）：ROCm 读不到 `clock_rate` → `flops_promised` 回退成 A100 的 312 TFLOPS。
- tqdm 的 `吞吐`（`train.py:1351`）：用**单步** `dt`，是瞬时值，实测抖到 2× 误差（1614 vs 713 t/s
  而 s/it 都是 5.0）。**只看 `s/it`。**

### 5｜续训时架构键被 checkpoint 强制覆盖（`train.py:476-492`）
改 config.yaml 对 `use_mtp` / `mtp_weight` / `swiglu_clamp` / `muon_ns_steps` / `use_moe` /
`hc_mult` 等**无效**。生效的只有 `dropout` / `gradient_checkpointing` / `ndb_*`。
**纯数据解法**：`local/fork_ckpt_arch.py`（改 `model_args` + 裁 `state_dict` + 裁优化器）。
已验证被移除参数**全部位于各 param_group 尾部**，位置 zip 安全，**Muon 动量不用丢**；
被裁权重自动备份到 `<dst>.dropped.pt`，可回滚。

### 6｜新工具
- `local/prof_step.py`：逐阶段计时（取数/前向/反向/裁剪/Muon-NS/AdamW/归零），两种模式
  （`total` 真 s/it；`phase` 占比），可覆盖 `--ns_steps` `--bs` `--grad_accum` `--compile` `--grad_ckpt`。
  **注意**：必须「一个 micro-batch 前向完立刻反向」。先攒 4 个图再统一反向会让 4 份激活同时驻留，
  8GB 卡必 OOM（实测吃到 7.53G，而真实训练峰值 3.10G）。
- `local/prof_suite.sh`、`local/bench_nondb.sh`：排队作业，等 GPU 空再跑。
