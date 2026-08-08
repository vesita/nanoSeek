# 11 · Attention Sinks 单独开就能打破重复坍缩（A/B 归因实证）

## 现象
默认配置模型（`out/chinese`，val 1.65）训练完，采样却是"悟空空空空…"×300 的死循环；
全开版（`out/chinese-all`，val 3.13）loss 更高，但采样不重复、内容多样。

**矛盾**：loss 低的反而不"能用"，loss 高的反而"能用"。

## 根因
6 个 V4 开关**全开**时，采样不重复的真实来源是其中**一个**开关——**Attention Sinks**。
它给 softmax 追加一个可学习标量偏置列（value 为零），相当于给注意力预算一个"垃圾桶"：
模型不需要把概率全押在单一 token 上（那是"重复坍缩"的自我强化循环），
可以把多余的注意力预算"倒掉"，从而避免陷入"看到高频 token 就继续押它"的捷径。

全开版 loss 高的**拖累**来自其它开关（最可疑：hash 路由放弃了 2/4 层的语义路由、
Lightning Indexer 在 256 上下文下纯属负担），不是 Sinks。

## A/B 归因证据（固定种子 1337，同配置，仅差 use_attn_sink）
> 复现命令：
> ```
> uv run python training/train.py training/config/train_chinese.yaml --out_dir=out/ab-base-1000 --max_iters=1000
> uv run python training/train.py training/config/train_chinese.yaml --use_attn_sink=True --out_dir=out/ab-sink-1000 --max_iters=1000
> ```

| 指标 | baseline-1000 | sink-1000（只开 Sinks） |
|------|--------------|------------------------|
| val loss | 4.2666 | **4.2146**（略低） |
| 采样① | 悟空空来来来来…（单字重复） | 悟空，有，有，让，可叫何太太…（多样） |
| 采样② | 悟空意意念念形气气气…（单字重复） | 悟空意意时时时，后，而见，可不耐。不杀我。 |
| 采样③ | 悟空声声声道…大师大哥师叔姊妹…（词重复） | 悟空！…爸爸说：“是啊，说道：“吴氏母子… |

**结论：Sinks 单独开就打破重复，且 loss 不升反降。** 这是"免费"改进——每头一个标量，参数可忽略，无性能损失。

> 注意：1000 步 vs 5000 步的对照——baseline-1000 与 baseline-5000 在相同步数下 loss 几乎一致
> （250 步 6.619 vs 6.607），说明固定种子下训练可复现，A/B 对照有效。
> 但 200 步太短，重复坍缩还没形成，baseline 与 sink 采样看不出差异（都碎片化）。

## 三重验证（2026-08-08 补充）
> 加深+MTP+RoPE 1e6（`out/chinese-v2`，6×80+MTP，无 Sinks）训 5500 步、val 1.71，
> 采样**仍"空空空"重复**——加深/MTP/rope 都治不了重复，重复是注意力机制的病，只有 Sinks 治。
> v2 结构 + Sinks 从 0 训 1000 步（`out/chinese-v2-sink0`）：采样变"人生的声音在哪？李文秀心想：啊哟。"
> ——成句叙事。**Sinks 是打破重复的必要条件**（开 Sinks 的一律不重复，不开的一律重复，
> 与层数/loss 无关）。⚠️ Sinks 必须从 0 训，后训练加不进去（见 [12-后训练不能改架构](12-后训练不能改架构.md)）。

## 怎么用
- **`use_attn_sink` 已写进默认配置**（`train_chinese.yaml`），是经得起 A/B 的真改进。
- 评估模型别只看 val loss：**loss 低 ≠ 能用**。重复坍缩是 loss 骗低的产物
  （模型发现押"高频 token"能抄近道），必须采样目测 + 固定 prompt 对比。
- 全开 6 开关不可取：收益集中在 Sinks，拖累来自 hash/indexer。要做就**逐开关归因**，
  别一口气全开。

## 关联
- [06-重复坍缩诊断](06-重复坍缩诊断.md)：重复坍缩怎么诊断（loss 曲线 + logits 分布）
- Sinks 实现见 `model/model.py` CausalSelfAttention.use_attn_sink（每头可学习标量偏置）
