# 对数放缩残差（LSE）：log-sum-exp 域合并替代线性相加，零参数抗数值尖刺

> 用户想测试「对数放缩网络」（log-scaled intermediate representation）——把子层中间表示放到对数域运算。选用最贴近实验意图、又对当前病根最对症的实现：**残差连接从线性相加换成对数域 soft-max（log-sum-exp）合并**。数值连接实验，零参数，与 block_order / no_attn_layers 同类。

## 动机

dev-notes/14 确认了核心病根：**「loss 骗低」**——交叉熵只惩罚不确定、不惩罚自信地错。SwiGLU 门控乘积会冒出极端值（smoke 里同权重关钳制 max 高达 6000+），这些尖刺在线性相加的残差里被**逐层放大**，最终让模型押高频低信息 token（全角空格）抄近道压 loss。

对数放缩残差针对的正是这条放大链路本身：

```
普通残差:  x ← x + F(x)              # 线性相加，极端值可无限涨大
LSE 残差:  x ← log(exp(x)+exp(F(x))) # 对数域 soft-max，有界收缩
           = max(x, F(x)) + log2(1 + exp(-|x - F(x)|))
```

三个性质（diff 里已逐个断言验证）：
1. **有界收缩**：`LSE(x,F) ≤ max(x,F) + log2`，永远压在最强的分支附近，不像加法那样随层数涨大。smoke 实测：极端输入下 LSE max = 356 vs 线性相加 max = 570。
2. **软选择偏置**：某分支占优时输出 ≈ 该分支 + log 小项，而不是把两条信号（包括那个极端尖刺）**直接相加**。模型被迫「选一个主路」，无法再靠叠加便宜极端来骗低 loss。
3. **数值稳定**：`logaddexp` 内部按 max 平移，负输入也稳、梯度干净。

## 改动

**Python-only，零参数**（LSE残差 102,528 参数 = 标准路径同规模，严格不变）。

- `model/model.py`
  - 顶层加 `logsumexp_residual(x, y)`：对数域 soft-max 合并两个残差分支。
  - `GPTConfig` 加 `use_lse_residual: bool = False`。
  - `Block.__init__`：`assert not (use_mhc and use_lse_residual)`——LSE v1 只服务标准单流路径；mHC 已有自己的 4 流残差拓扑，两者互斥。
  - `Block.forward`：抽了个 `res = lse if use_lse_residual else add` 的小 lambda，替换标准路径的线性相加（attn/ffn 两个子层都换）。
- `model/__init__.py`：导出 `logsumexp_residual`。
- `training/train.py`：全局 `use_lse_residual` + 进 `model_args` + checkpoint 兜底键（老 ckpt 无此键 → 默认 False，兼容不破）。
- `inference/scripts/smoke_test.py`：加 `LSE残差` / `LSE残差+CSA` 用例 + 三段数值断言（有界性 / 软选择 / 梯度干净）。

## 验证（已跑）

`uv run python inference/scripts/smoke_test.py` 通过 ✅：

- LSE残差 forward/backward 正常、梯度无 NaN；参数量 = 标准路径（零参数新增）。
- 有界性：`|LSE - max| = 0.653 ≤ log2≈0.693`；线性相加 max 570 → LSE 356（抑制极端值生效）。
- 软选择：大分支主导时 `|LSE - a| max = 0.0000`（被抑制分支贡献归零）。
- 梯度：logaddexp 处处可微，`backward` 无 NaN。
- mHC 与 LSE 互斥断言生效（同时开启会报错）。

## 怎么跑 A/B（用户跑，seed 1337，唯一变量 = use_lse_residual）

对照臂 = dev-notes 既有 af 全注意力路径（block_order=attn_ffn，无 LSE）：

```sh
# 基线臂（无 LSE）：用 --out_dir 避开覆盖当前 best 模型
uv run python training/train.py training/config/train_chinese.yaml --out_dir=out/chinese-data2-af
# 实验臂（LSE）：注意与训练配置里的 use_mhc 互斥，需先关掉 mHC
uv run python training/train.py training/config/train_chinese.yaml --use_mhc=false --use_lse_residual=true --out_dir=out/chinese-data2-lse
```

同步数（建议先 2000 与 chinese-data2 参照对齐），各跑 3 个 prompt 采样目测对比。

**判据**：
1. val loss 别只盯低——重点看训满时**采样是否重复/坍缩到空格**（LSE 的价值是训练目标的对症，不是快）。
2. 用 `--repeat-penalty` 压噪音后，LSE 臂的**空白/重复率**是否显著低于 af 臂。
3. loss 骗低是否缓解：训练后期 logits 极端值（抽样 max|logits|）是否被压住。

## 可借鉴的研究方向（后续）

- 这个性质本质是 **soft-max 做残差混合**——思想同源可参考 log-linear / Bayesian 更新里的 `log-sum-exp` 聚合（如 PGM 的因子图消息传递），以及 Vision Transformer 的 **register token**（固定可学习 token 会产生 OOD artifact——同类「非负压缩缓冲」陷阱）。
- **放缩放调**：当前纯 `LSE`（`max+log2`）是最激进的对数域，最"软选择"。若想向加法平滑退化做消融，可把合并改成 `α·max + logaddexp` 的混合或加一个可学习 offset——但 v1 先测纯形式，别加参数污染「零参数数值连接」的对照条件。
- **与 swiglu_clamp 的关系**：clamp 是**幅度**上硬钳制（保住符号、砍幅度）；LSE 是**连接方式**上软压缩（不保留加法）。两者正交，可就同一离群输入做 A/B 对比谁的抗尖刺更天然、副作用更小。
