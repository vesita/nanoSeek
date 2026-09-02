# 62-治理后重训冒烟：Muon 3D 断言 / bf16 dtype 泄漏 / ROCm 异步拷贝竞态

日期 2026-09-03
状态: 已修复（三个问题全部定位并解决，GPU 冒烟通过）
涉及: `model/gpt.py` / `model/attention.py` / `model/optimizer.py` / `model/utils.py` / `training/train.py` / 训练环境变量

---

## 0. 背景

数据治理（dev-notes/60）+ 三区词表与待续符（dev-notes/61）全部落地后，开始治理后重训的
冒烟验证。目标：跑通「新 4533 词表 + <eos>/<cont> 标注 + 去标签 masking」的完整训练链路，
为 1-epoch 正式训练铺路。

冒烟配置（与正式训练一致）：
`--char-level=true --factorized-emb-dim=24 --n-layer=6 --kv-memory-output-gate=true
--n-experts=8 --n-top-k=2 --moe-hidden-scale=1.33333333 --n-mtp=2 --kv-memory-checkpoint=true
--use-muon=true --muon-split=true --device=cuda --compile=false`

依次踩到三个独立问题，逐个修复后 GPU 12 步冒烟通过（loss 10.95 → 10.88，正常下降）。

---

## 1. Muon `assert G.ndim == 2` 断言崩溃（优化器参数路由）

### 现象
`AssertionError: assert G.ndim == 2`，栈在 `model/utils.py` zeropower_via_newtonschulz，
由 `optimizer.py:60` 整块正交化分支触发。

### 根因
KV 记忆的持久工作台 `mem_persist` 是 **3D 参数** `(n_head, kv_memory_latent, kv_memory_latent)`
即 `(4, 16, 16)`（attention.py:86 `nn.Parameter(torch.zeros(config.n_head, l, l))`）。
`configure_optimizers` 的 Muon 分支判定 `p.dim() >= 2` → 进 Muon 整块正交化，NS 迭代要求
2D 矩阵 → 断言炸。

旧 checkpoint（`eos_fix_1epoch`）不炸是因为它 `use_muon: False`（AdamW 无此约束）。
所有层 + MTP 块共 8 个 `mem_persist`，全部中招。

### 修复（model/gpt.py configure_optimizers）
3D+ 参数**没有矩阵结构，整块正交化无意义**，与 embedding/lm_head 同理由 → 路由到 AdamW（带衰减）：

```python
elif p.dim() > 2:
    # 3D+ 参数（如 kv 记忆的 mem_persist (nh,l,l)）不是矩阵，整块
    # 正交化无意义且会触发 NS 的 ndim==2 断言 → 走 AdamW 带衰减。
    adamw_decay.append(p)
```

### 验证
参数清单脚本确认：除 `mem_persist` 外无其他 ndim>2 参数；修复后优化器构建通过。

---

## 2. bf16 前向 dtype 泄漏（`float != c10::BFloat16`）

### 现象
GPU（autocast bf16 / 直接 .to(bfloat16)）前向报
`RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16`；
CPU（float32）完全正常。旧 checkpoint 未触发是因为旧代码路径不同 + 之前 GPU 冒烟死得更早。

### 根因（两处，逐一定位）
1. **`_kv_memory_forward`（attention.py）**：KV 记忆内部**有意保持 fp32**（状态累加防精度
   丢失，注释明确），但最后 `self.mem_up(o.reshape(B,T,nh*l))` 把 fp32 的 `o` 直接喂给
   bf16 权重的 Linear → 崩。delta / block 分支同病。
2. **CSA 块路径（attention.py:413）**：`y_comp = y_comp * has_prior.unsqueeze(...).float()`
   —— `has_prior` 是 float32，乘法把 bf16 的 `y_comp` 提升成 fp32，一路带进
   `c_proj`（bf16 权重）→ 崩。

### 修复（model/attention.py）
```python
# 1. 三个 mem_up 调用点统一：进线性层前对齐投影权重 dtype
return self.mem_up(o.reshape(B, T, nh * l).to(self.mem_up.weight.dtype))

# 2. CSA 块掩码乘数对齐 x.dtype，不再把 y_comp 提升成 fp32
y_comp = y_comp * has_prior.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(x.dtype)
```

### 验证
隔离脚本覆盖 4 组合（fp32/bf16 × eval/train，batch 2 和 64）：全部通过，loss 值正常。
注意：`mem_out_norm`（RMSNorm）bf16 权重 × fp32 输入不会崩（torch 自动 promotion），
所以只需修线性层输入。

---

## 3. HIP `invalid device function`：ROCm 异步 H2D 拷贝竞态

### 现象
GPU 冒烟（无任何特殊环境变量）确定性崩溃：
`torch.AcceleratorError: HIP error: invalid device function`，栈指到第一个
`estimate_loss` 的 embedding（`F.embedding`）——但 HIP 错误是异步上报，真正炸的 kernel
在更早。

诡异点：**同样配置的隔离脚本（手搓 batch 2/64 前向+反向）完全不崩**，只有 train.py 崩。

### 定位过程
| 尝试 | 结果 |
|------|------|
| compile=true + GPU | inductor 编译期 HIP 错误（老问题，已知） |
| compile=false | 仍在 estimate_loss 崩（不是 compile 的锅） |
| 隔离脚本 fp32/bf16 × eval/train × batch 2/64 | **全过** → 问题在 train.py 特有环节 |
| `AMD_SERIALIZE_KERNEL=3` | **通过**（262s/12步，慢） |
| `AMD_SERIALIZE_KERNEL=1` | **通过**（114s/12步） |
| `HSA_ENABLE_SDMA=0` | **通过**（114s/12步，同样速度） |

### 根因
train.py 的 `get_batch` 用 `x.pin_memory().to(device, non_blocking=True)` 异步 H2D 拷贝
（走 ROCm SDMA 引擎），与并发 kernel 执行存在驱动级竞态 → 偶发 kernel 加载/执行失败，
HIP 以 `invalid device function` 异步上报。隔离脚本直接 `device="cuda"` 造数据，没有
pin_memory 异步拷贝路径，所以不崩。

### 修复（运行环境，不改代码）
```bash
export HSA_ENABLE_SDMA=0   # 禁用 SDMA 异步拷贝引擎，H2D 走计算拷贝，消除竞态
```
- 效果与 `AMD_SERIALIZE_KERNEL=1` 相同但语义更精确（只关拷贝引擎，不串行化全部 kernel）
- 速度：稳态 ~1.9s/it（与 SERIALIZE=1 一致；旧训练 compile=true 时 1.33 it/s，
  本次 compile=false + Muon 开销，可接受）
- 若未来某天换驱动/ROCm 版本后此问题消失，可去掉该变量

### 怎么避免
- 排查 HIP 异步错误时，先想「同步点 vs 真实失败点」：栈位置不可信
- 隔离复现优先复刻 train.py 的数据搬运路径（pin_memory + non_blocking），而不只是模型前向
- 手头已有两个可绕过的环境变量，记录于此，别重复踩

---

## 4. 冒烟结果（GPU，12 步）

```
step 0: train 损失 0.0000, val 损失 10.9548
  体检: EOS 自吐 0% | 平均 len 120.0 | rep3 0.0016 | 续轮 0% | ⚠ 不合格（随机初始化正常）
step 12: train 损失 10.9373, val 损失 10.8816
✓ 新最佳 val 10.8816 → best.pt（并已更新 last.pt）
训练完成：13 步（达 max_iters 12）· 总耗时 114.3s（含首次评估/体检）
```

- loss 单调下降、无 NaN、无断言 → 新词表 + <cont> 标注 + 去标签 masking + Muon 全链路通
- 体检「不合格」是随机初始化下的正常现象（训练早期模型只会乱吐），min_eos_rate 门控
  在正式训练的后期评估点才会体现价值
- CPU 冒烟（12 步）同样通过：`step 12: train 10.9320 → val 10.8787`，说明与 GPU 无关的
  逻辑（masking/词表/标注）也验证过一遍

---

## 5. 正式训练配置（out/cont_v1_1epoch）

- 数据：治理后 train_char.bin（45,827,969 token，≈1 epoch）
- `max_iters=2800` ≈ steps_per_epoch（1 epoch，Chinchilla 最优预算：2.7M 参数 × 20 ≈ 54M token）
- WSD 调度（stable_frac 0.8 → 2240 步起衰减），warmup 100
- 环境：`HSA_OVERRIDE_GFX_VERSION=10.3.0 HSA_ENABLE_SDMA=0 TMPDIR=/home/vesita/AI/scratch`
- 早停关（训满 2800 步），eval/体检每 280 步
- 实际完成：2800 步 · 总耗时 4058.6s（~1.13h，1.32s/it）

### 训练结果（2026-09-03 03:32 完成）

```
val: 10.9547 → 7.4812 → 6.1064 → 4.8960 → 3.7573 → 2.8009 → 2.1928
    → 1.8641 → 1.5294 → 1.3179 → 1.2020（best，WSD 衰减末端）
体检: 280 步起 EOS 自吐 100% · rep3 0.0000 · 全程合格（10/10 评估点）
```

- 体检自 280 步起就 100% EOS 自吐且 rep3=0 —— 新词表 + <cont> 标注下模型快速学会
  「收尾 + 不复读」，没有重复坍缩迹象
- 采样验证（16 条，4 prompt × 4 seed）：**eos 9 / cont 7，全部自然终止**（0 截断）——
  模型已能在 `<eos>`（收尾）与 `<cont>`（递回）间自主选择，待续符机制生效
- 文本仍有碎片感（2800 步小模型正常水平），但结构健康：无死循环、无重复、回复
  长度适中（avg len ~10-30）
- 对比：旧 eos_fix_1epoch（17000 步，val 0.9413）；本次仅 2800 步 val 1.2020，
  收敛方向一致且数据量更少——后续若继续训练（如 2-3 epoch）还有明显下降空间

### 遗留（后续 RL 阶段）
- RL 采样仍以 `<eos>` 为唯一停止符，`<cont>` 会被当普通 token 继续生成
- 需按 dev-notes/61 §5：采样循环加入 `<cont>` 停止、重构 `r_quiet_eos`、加 `r_control`
  自控奖励 —— 这是治理后 SFT 基座就绪后的下一步工作

---

## 6. 文件改动清单

| 文件 | 改动 |
|------|------|
| `model/gpt.py` | configure_optimizers：3D+ 参数（mem_persist）→ AdamW，不再进 Muon |
| `model/attention.py` | 3 处 `mem_up` 输入对齐 `mem_up.weight.dtype`；CSA `y_comp` 掩码乘数对齐 `x.dtype` |
| （无代码改动） | ROCm 竞态用 `HSA_ENABLE_SDMA=0` 环境变量绕过 |
