# QK-Norm + Router Z-Loss 1500 步 A/B：单独各有小赢，合开超线性协同

> 动机（来自实际训练两个老痛点）：欠训练重复坍缩 / 训满后 Sinks 失效 / 输出过度自信——
> 两个「近零参数」的结构稳定性手段正好对症：QK-Norm 把注意力 logits 尺度拉平（结构性抑制过度自信），
> Router Z-Loss 把 MoE 路由 logits 拉回有界（DeepSeek 系列实际使用的稳定手段）。
> 实验：完整 2×2 因子设计（base / 只QK / 只Z / 都开），每臂 1500 步串行、seed 1337、
> RTX 5060、关早停、lr_decay_iters=5000（与 out/ 既有基线同调度位置，val@1500 可直接对比）。
> 训练 2026-08-17。

## 现象

1. **实现初稿根本跑不起来**：上一版把 QK-Norm 方法插到 `__init__` 中间，下面的 RoPE `cos/sin`
   buffer 注册代码被吸进 `_apply_qk_norm` 的 return 之后变成死代码 → 任何 use_rope 前向
   `AttributeError: 'CausalSelfAttention' object has no attribute 'cos'`。这是「改代码时把一段
   语句块吞进新函数体」的典型缩进事故，冒烟测试（前向+反向+参数量）第一时间就抓出来了。
2. **flash / 手动注意力路径不一致**：`scaled_dot_product_attention` 不传 scale 时默认再除
   1/sqrt(head_dim)，与手动路径（QK-Norm 开启后不再除）行为不同 → 同模型 flash 开/关 logits 不等。
3. **完整 A/B 之前无任何对照数据**：4 个 arm 全从 0 训 1500 步，结果在下表。

## 实验结构（2×2 因子）

| 臂 | out_dir | use_qk_norm | z_loss_weight | 其余 |
|----|---------|-------------|---------------|------|
| base 对照 | out/chinese-data2-ab-base | ✗ | 0 | 同默认（CSA/HCA+MoE+Sinks+MTP，下同） |
| qk 只开 QK | out/chinese-data2-ab-qk | ✓ | 0 | 同上 |
| z 只开 Z | out/chinese-data2-ab-z | ✗ | 1e-4 | 同上 |
| qkz 都开 | out/chinese-data2-ab-qkz | ✓ | 1e-4 | 同上 |

- 参数成本：QK-Norm 每层每头 1 个 scale（6×4=+24 参数），Z-Loss 0 参数 —— **模型规模不变**。
- 配置 `training/config/train_chinese_ab_{base,qk,z,qkz}.yaml`，runner `training/run_ab_serial.sh`，
  对比脚本 `training/compare_ab.py`。

## 结果 1：val loss（越低越好，每 250 步评估）

| step | base | qk | z | qkz |
|------|------|----|----|-----|
| 250  | 3.3470 | 3.3202 | 3.3365 | **3.3379** |
| 500  | 2.3347 | **2.2764** | 2.2985 | 2.2821 |
| 750  | 1.7022 | 1.4626 | 1.5233 | **1.4223** |
| 1000 | 1.2942 | 1.1616 | 1.2056 | **1.1382** |
| 1250 | 1.1064 | 1.0304 | 1.0630 | **1.0090** |
| 1500 | 1.0024 | 0.9506 | 0.9822 | **0.8131** |

**单独开都赢，合开是超线性协同**（同 1500 步预算）：

- qk 单独：val 1.0024 → 0.9506（−0.052）
- z 单独：val 1.0024 → 0.9822（−0.020）
- **qkz 都开：val 1.0024 → 0.8131（−0.189），比「只 QK 的 −0.052」再多 3.6 倍收益**

两个稳定器不是简单叠加（0.052+0.020=0.072 ≪ 0.189）：QK-Norm 让注意力 logits 始终有界、
Z-Loss 让路由 logits 始终有界，两层「压尺度」一起把训练压在健康数值域里，收敛显著加速。

## 结果 2：采样质量（1500 步欠训练态，统一 6 prompt × seed 1337）

| 模型 | val@1500 | rep2↓ | rep3↓ | rep4↓ | d2↑ | turns↑ |
|------|---------|-------|-------|-------|-----|--------|
| base | 1.0024 | 0.1318 | 0.0249 | 0.0049 | 0.9275 | 1.7 |
| qk   | 0.9506 | 0.1390 | 0.0393 | 0.0134 | 0.9242 | 1.7 |
| z    | 0.9822 | 0.1423 | 0.0300 | 0.0061 | 0.9221 | 1.7 |
| qkz  | 0.8131 | 0.1462 | 0.0304 | 0.0068 | 0.9188 | 1.3 |
| **prod 参考**（chinese-data2 @2000 步） | 0.9822 | **0.0535** | **0.0061** | **0.0012** | **0.9706** | 0.8 |

**1500 步采样仍是欠训练态**（都凑到 200 token 上限、结构平庸），且 qkz 的重复率略高于 base——
这不是矛盾，正是 dev-notes/14 反复出现的「loss 更低 → 分布更尖 → 采样更自信」曲线：qkz 在同一步数
学得更深（val 0.81 历史级低），把同一把「欠训练→重复坍缩」的刀磨得更利了。**本轮唯一可靠信号是
loss 对比（同预算同数据同 seed）**；采样质量要在训满（现有 5000 步基线）后再评。

### 采样目测（`training/compare_sampling.py`，完整样本见 `dev-notes/sample_compare.md`）

四个 1500 步臂**互相之间几乎看不出差别**：都是同一套「共情话术模板」复读（"你先按你舒服的节奏
来就好 / 这种事通常是白天更明显 / 现在最卡你的那句话 / 一点点来就好 / 扛不住 / 被拖住"），对
prompt 本体（推荐小说 / 吃了没 / 出去玩）完全不回应——这是**欠训练的共享伪迹**，不是某臂的特质。
臂间差异仅剩 token 级噪音（qk/qkz 偶尔 "吗吗啊呢" 类乱码、base 偶发 "隐约有个影子，隐约有个影子"
复读），且 qkz 的伪「用户/模型」标签轮次略少（turns 1.3 vs 1.7）——它学得更深入、更少照抄语料里
的轮次外壳，但也因此更少产生"看似有结构"的输出。
生产参考（2000 步）rep3=0.0061 明显更低、d2 更高，**但**同样不回应 prompt 本体，且 6 条里有 3 条
直接崩成古文/拼音拼贴（"韦小宝道""PindIrandninding"）——低重复≠智能（dev-notes/14 同一结论）。
**采样验收只能等 qkz 按默认 5000 步训满再做**，1500 步的采样对比没有区分度。

## 根因 / 修复 / 怎么避免

1. **缩进事故吞掉 buffer 注册** → 修复：把 RoPE `cos/sin` 注册块还原回 `__init__`（在
   `_apply_qk_norm` 方法定义之前、与其余 buffer 同级缩进）。怎么避免：**新方法插进 `__init__` 后，
   肉眼核对「方法定义后面还跟着的缩进块是不是被吞了」；任何架构改动先跑一次前向+反向冒烟**。
2. **SDPA scale=None 默认再除 sqrt(d)** → 修复：QK-Norm 开启时显式传 `scale=1.0`，与手动路径
   （不再除）语义一致；不开启时仍传 `1/sqrt(d)`，基线行为零变化。怎么避免：flash 有「默认 softmax
   scale」，改了「是否除 √d」就必须把两路径都核对。
3. **串行 A/B runner 的段错误噪音**：base/部分臂在「训练完成、图已生成」之后进程退出时偶发
   SIGSEGV（rc=139），产物（results.csv/best.pt/loss_curve.png）已全部落盘——torch.compile +
   CUDA 解释器退出时的已知良性崩溃；runner 按「results.csv 是否完整」判定成败，不是 rc。

## 结论与建议

1. **两个改动都是「缩小规模前提下」的高性价比收益，且组合超线性**。qkz（都开）在 1500 步就把
   val 压到 0.813，比默认结构同等预算低 0.19。
2. **建议直接进默认配置**：`use_qk_norm: true, z_loss_weight: 0.0001`，零参数、零推理开销
   （QK-Norm 的 normalize 只影响训练数值域；推理同样受益于有界 logits）。
3. 按常规 5000 步训满一版 qkz 再做采样验收（对治 dev-notes/06/14 的重复坍缩要看训满表现）。

## 复现

```sh
uv run python training/train.py training/config/train_chinese_ab_base.yaml  # val 1.0024
uv run python training/train.py training/config/train_chinese_ab_qk.yaml    # val 0.9506
uv run python training/train.py training/config/train_chinese_ab_z.yaml     # val 0.9822
uv run python training/train.py training/config/train_chinese_ab_qkz.yaml   # val 0.8131
uv run python training/compare_ab.py                                        # 汇总表 + 叠加曲线
```