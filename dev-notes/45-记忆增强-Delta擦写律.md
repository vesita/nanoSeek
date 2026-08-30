# 45-记忆增强#2：Delta 擦写律（P2，先擦后写）

## 动机（dev-notes/39-#2）
- P1 写入是**叠加式**：S_t = r⊙S_{t-1} + β·k_t v_tᵀ（rank-1 累积）。
- 问题：不同键写入同一黑板位置会**混叠**——S·k 的检索值是历史写入的线性叠加，
  键冲突时互相污染 → 记忆精度低 → 接话相关性差（答非所问）。
- Delta 擦写（DeltaNet 思路）：写入前先用当前键检索已有内容，从 v 中减去再写：
  S_t = r⊙S_{t-1} + β·(v_t − S_{t-1}·k_t)·k_tᵀ
  新状态对键 k 的检索 = 旧值 + β·(v − 旧值) → 精确逼近 v，消除混叠。

## 设计
- 新开关 `kv_memory_delta`（bool，默认 False = P1 叠加式；True = Delta 擦写）。
- 逐 token 顺序递推（与 P1 chunk 并行同语义，**写后读**，自身写入合法）：
  ```
  S = S_in
  for t:
      u  = S·k_t                  # 检索：当前键下已存值
      S  = r_t ⊙ S + w_t·(v_t − u)·k_tᵀ   # 先擦后写（w 每头标量或逐通道）
      o_t = (S + persist)·q_t     # 写后读
  ```
- **梯度检查点**：块内顺序递推提为 `_mem_delta_chunk`，块边界存 S，
  backward 重算块内（同 dev-notes/39-#4 思路）→ 内存安全（T=256 保存每步 S 会 1.5GB）。
- 数值：β=sigmoid ∈(0,1) 限写幅；r 衰减提供遗忘；S fp32。
  擦写项 −(S·k)kᵀ 是"投影"（S·k 向量 ⊗ k 向量）→ S 范数受 r 与 β 控制。

## 兼容性
- kv_memory_delta=False 时行为与 P1 完全一致（默认不变）。
- 推理：逐 token 递推 O(1) 状态/步，与 P1 同复杂度；sample_py 无需改动
  （kv_memory_delta 入 checkpoint model_args 即可重建）。

## 验证
1. fp32：Delta 版 vs 手写参考逐位一致；梯度全通（含 mem_forget）。
   kv_memory_delta=False 的顺序版 vs P1 chunk 版 ~3.7e-7（浮点求和顺序）。
2. 3 步训练冒烟（GPU）。
3. 300 步 A/B：A=out/b_full_300（P1，val 5.3096 已有）；B=out/c2_delta_300
   （--kv_memory_delta=true）。验收：B val 优于 A（Delta 是**增强**方向，
   不要求 300 步就胜出——若持平也做 1500 步阶段性验证）。

## 结果：△ 300 步持平但效率低 → 用户放弃（2026-08-30）
| 配置 | val @300 | 速度 |
|---|---|---|
| A P1 叠加式（对照） | 5.3096 | ~0.9s/it |
| B Delta 擦写（k 归一化+checkpoint） | **5.3033**（−0.006） | ~2.3s/it（3×慢） |

- 300 步持平（微弱胜出，噪声级）、无 NaN → 但**训练效率低 3 倍**
  （块内 256 步顺序小 kernel 无法并行）。
- **决策（用户指示）：效率太低，不考虑。** 代码保留（kv_memory_delta 参数，
  默认 False 行为不变），若未来要做需 DeltaNet chunkwise 并行化（三角逆+低秩因子，
  成本高，当前不值得）。
- 效率教训：结构增强若带来 >2× 训练开销，300 步预算内收益不明确时不值得。

## 状态
- [x] 实施（config.py / attention.py / train.py）
- [x] fp32 验证：Delta 方法 vs 手写参考**逐位一致（max|Δ|=0）**；全参数梯度（除既有死参数 c_attn）
- [x] 3 步训练冒烟（GPU，loss 正常下降）
- [x] 速度：稳态 ~2.3s/it（P1 ~0.9s，慢 2.6×——块内 256 步顺序小 kernel；
      300 步 ~12 分钟，快速测试预算内可接受）
- [x] 300 步 A/B：持平（−0.006）→ **用户以效率为由放弃**

## 实施踩坑（v1 失败 → v2 修复）
- **v1 失败**（bash-10）：step 30 后 loss=NaN + step 31 OOM（7.6GB allocated）。
- **NaN 根因**：Delta 擦写正反馈——S·k 大 → 擦写残差 v−u 大 → 外积 (v−u)kᵀ 更大
  （|k| 未约束，放大 10×+）→ S 爆炸。
- **修复① k 归一化**（DeltaNet 标准做法）：F.normalize(k_m) → 外积范数 ≤ |v−u|，
  且擦写自校正（S·k̂ 逼近 v），正反馈消除。
- **修复② 梯度检查点**：--kv_memory_checkpoint=true（块边界存 S，backward 重算块内）
  → 训练 autograd 中间量从 ~4GB 降到安全水位（OOM 根因：256 步×6 层 S 快照全保存）。
- 教训：新结构训练前先跑 30+ 步看数值稳定性（3 步冒烟掩盖了 30 步后的爆炸）；
  重跑前删除失败 out_dir（train.py 自动从 best.pt 续训，污染 checkpoint 会传染）。
