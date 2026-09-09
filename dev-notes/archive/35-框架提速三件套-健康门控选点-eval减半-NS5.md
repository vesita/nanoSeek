# 35-框架提速三件套：健康体检门控选点 + 评估开销减半 + Muon NS=5 A/B（2026-08-20）

> 归档说明：本笔记属于 2026-08-19/20 的 V4 框架线，编号与 2026-09-02 批次的 31–35 冲突，已移入 `archive/`。

背景：用户从优化分析（数据/训练动力学/工程提速三组杠杆）中选定「框架提速三件套」落地——
全部是纯框架改动（零模型参数）：① best.pt 健康分复合选点（防「val 骗低、采样坍缩」）；
② 评估开销削减；③ Muon NS 迭代 10→5 A/B（archive/33 实测 NS 是全栈每步慢 40% 的主因）。

## 1. 健康体检门控选点（train.py `health_*` 配置）

### 动机
- dev-notes/14 实证「val 继续降但采样崩」；archive/34 的 val 冠军（0.7776）采样反而是混合的
  （rep3 0.0222 为 reb 9 倍、回复变长 100.6、1/10 seed 自续半轮）——best.pt 只看 val 会把
  坍缩/劣化模型当冠军。
- 目标：选点 = 「val 创新低 **且** 采样体检合格」，体检不合格时不覆盖旧 best。

### 实现
- 每个评估点（`health_eval_interval` 默认跟随 eval_interval）跑固定 prompt 体检：
  3 prompt（你好/你是谁/你在哪）× `health_seeds`(5) × ≤`health_max_new`(120) token，
  temp 0.8 / topk 200 / rep 1.2（与 training/health_check_hello.py 同口径，原始生成不截断）。
- 指标：EOS 自吐率、平均 len、rep3、续轮率；合格阈值 `health_min_eos_rate`(0.6) 且
  `health_max_rep3`(0.1)（rep3>0.1 即坍缩线）。冠军模型复测通过（9/10 EOS、rep3 0.0222）。
- 门控：`is_best = val 创新低 and health_ok`；`best_val_loss` = 门控后选点（checkpoint 实际
  保存的 val）；`raw_best_val` = 原始 val 最优（早停判断用，与门控解耦——坍缩模型 val 仍可能
  降，但 best.pt 不再跟）。
- 产物：`out/health.csv`（新文件，results.csv 列不变，archive.py 不受影响）。
- 开关：`health_enabled`（train_chinese.yaml 默认开，test.yaml 继承开）。

### 关键实现点（踩坑）
- **体检必须用未编译模型**：`raw_model = model.module if ddp else model` 在 `torch.compile`
  之后拿到的是编译包装——变长生成的每步长度变化触发 inductor 逐长度重编译，冒烟实测
  直接打满 recompile_limit、体检耗时 ~30s。正确写法：`unoptimized_model if compile else raw_model`。
- **不修改共享代码**：`sample_py.generate_ids` 的 `torch.tensor([context_ids])` 只接受 list，
  传 CUDA 张量会 TypeError。在 train.py 内写本地 `_health_generate`（镜像同一生成语义）。
- **不污染训练 RNG**：体检采样用独立 `torch.Generator(device).manual_seed(seed)`，而不是
  health_check_hello 的全局 `torch.manual_seed`——训练循环内重置全局种子会让评估点后的
  batch 流与历史重复相关（同一 RNG 状态重新走一遍）。seed 值相同，结果口径仍可比。
- **体检后恢复训练态**：`model.eval()` → finally `model.train(was_training)`。

## 2. 评估开销削减（`eval_train_split` + eval_iters）

- 原评估开销：每 250 步 `eval_iters 100 × (train+val)` = 200 次前向，占训练总计算 ~25-30%。
- 改法：`eval_train_split=false` 时评估点只评 val（50 次前向），results.csv 的 train/loss 列
  改用训练侧 EMA（每 log_interval 步 0.9 滑动平均）代替；train_chinese.yaml `eval_iters 100→50`。
- 效果：评估开销降到原 1/4，省 ~10-15% 墙钟时间（同预算多跑步数）。

## 3. Muon NS A/B（1500 步 · A1 数据 · 同配置双跑，新框架设置）

| 臂 | NS | val@1500 | MFU | 墙钟 | EOS@1500 | rep3@1500 | avg_len@1500 |
|---|---|---|---|---|---|---|---|
| chinese-ns10 | 10 | **0.8815** 🏆 | 0.33–0.42 | 665.5s | 73% | 0.0288 | 92.0 |
| chinese-ns5 | 5 | 0.9858 | **0.48–0.50** | **528.8s** | 60% | 0.0163 | 87.5 |

- **结论：NS=10 保持默认（yaml 无需改）**。NS=5 每步快 ~21% 墙钟（MFU +~30-40%，
  确认 NS 迭代是全栈主要耗时，archive/33 归因正确），但 val@1500 差 0.104——衰减段爆发
  不如 NS=10（1250→1500：NS=10 1.178→0.882，NS=5 1.318→0.986），与 GLM-5「正交化迭代
  越足注意力越稳」方向一致。速度杠杆真实存在，但 1500 步预算下质量买单。
- 健康分对比：两者全程体检合格（turns 均 0%）；NS=5 末端 rep3 更干净（0.0163 vs 0.0288）
  但 EOS 率更低（60% vs 73%）——采样质量混合，val 是这里更可信的信号。
- 方法注意：新 eval 设置（eval_iters 50 + 仅 val）改变评估点的 RNG 消耗路径 → 训练 batch 流
  与旧 run 不同，**ns10 的 0.8815 与旧冠军 0.7776 不可逐位比较**（同设置跑出的另一个样本）；
  后续 A/B 必须同 eval 设置对比。
- 体检轨迹（health.csv）验证了健康门控的价值：rep3 从 0.0085（step 1000）爬到 0.0236→0.0288
  （step 1250/1500），回复变长 64→87→92——val 之外可见的坍缩前兆；turns_rate 全程 0%
  （多轮缺口与优化器正交，archive/31/34 复现）。

## 结论

1. 体检门控选点：纯框架、零参数，直接对抗已知失效模式「val 降但采样崩」；health.csv 成为
   每实验标准产物（EOS 率/rep3/回复长度的训练内轨迹，比 val 单点更早暴露坍缩）。
2. eval 减半：同预算白赚 ~10-15% 训练步。
3. **NS=5：确认提速（~21% 墙钟）但 1500 步预算下 val 差 0.104 → 保持 NS=10 默认**；
   速度杠杆留给需要快速扫参的场景（可临时 `--muon_ns_steps=5`）。
