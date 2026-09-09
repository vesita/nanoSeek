# 62-治理后重训冒烟：Muon 3D 断言 / bf16 dtype 泄漏 / ROCm 异步拷贝竞态

日期 2026-09-03 · 状态: 已修复（三问题全定位解决，GPU 冒烟通过）

## 0. 背景
dev-notes/60 数据治理 + dev-notes/61 三区词表/待续符落地后，冒烟「4533 词表 + <eos>/<cont> 标注 + 去标签 masking」全链路，为 1-epoch 训练铺路。
配置（同正式训练）：`--char-level=true --factorized-emb-dim=24 --n-layer=6 --kv-memory-output-gate=true --n-experts=8 --n-top-k=2 --moe-hidden-scale=1.33333333 --n-mtp=2 --kv-memory-checkpoint=true --use-muon=true --muon-split=true --device=cuda --compile=false`
三个独立问题依次踩到，修复后 GPU 12 步通过（loss 10.95 → 10.88）。

## 1. Muon `assert G.ndim == 2` 崩溃（优化器参数路由）
- 现象：`AssertionError: assert G.ndim == 2`，栈 `model/utils.py` zeropower_via_newtonschulz（`optimizer.py:60` 整块正交化分支）。
- 根因：`mem_persist` 是 3D 参数 `(n_head, latent, latent)` = `(4, 16, 16)`（attention.py:86）；Muon 判 `p.dim() >= 2` → 进正交化，NS 迭代要求 2D → 炸。旧 checkpoint `eos_fix_1epoch` 不炸因 `use_muon: False`；所有层 + MTP 共 8 个 `mem_persist` 全中招。
- 修复（`model/gpt.py` configure_optimizers）：3D+ 无矩阵结构 → 路由 AdamW（带衰减）：`elif p.dim() > 2: adamw_decay.append(p)`。
- 验证：参数清单确认除 `mem_persist` 外无 ndim>2 参数。

## 2. bf16 前向 dtype 泄漏（`float != c10::BFloat16`）
- 现象：GPU（autocast bf16）前向报 `RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16`；CPU（float32）正常。
- 根因 1：`_kv_memory_forward`（attention.py）内部有意保持 fp32（防精度丢失），但 `self.mem_up(o.reshape(B,T,nh*l))` 把 fp32 `o` 喂给 bf16 权重 Linear → 崩；delta/block 分支同病。
- 根因 2：CSA 块路径（attention.py:413）`y_comp * has_prior.unsqueeze(...).float()`——float32 乘数把 bf16 `y_comp` 提升成 fp32，带进 `c_proj` → 崩。
- 修复（`model/attention.py`）：3 个 `mem_up` 输入 `.to(self.mem_up.weight.dtype)`；CSA 掩码乘数 `.to(x.dtype)`。
- 验证：隔离脚本 4 组合（fp32/bf16 × eval/train × batch 2/64）全过；`mem_out_norm`（RMSNorm）bf16 × fp32 不崩（torch 自动 promotion），只需修线性层输入。

## 3. HIP `invalid device function`：ROCm 异步 H2D 拷贝竞态
- 现象：GPU 冒烟（无特殊环境变量）确定性崩 `HIP error: invalid device function`，栈指第一个 `estimate_loss` 的 `F.embedding`——HIP 异步上报，真炸 kernel 更早。同配置隔离脚本（手搓 batch 2/64 前向+反向）不崩，只有 train.py 崩。
- 定位：compile=true 为 inductor 编译期 HIP 错误（老问题）；compile=false 仍崩；隔离脚本全过 → train.py 特有环节；`AMD_SERIALIZE_KERNEL=3`（262s/12步）、`=1`（114s/12步）、`HSA_ENABLE_SDMA=0`（114s/12步）均通过。
- 根因：`get_batch` 的 `x.pin_memory().to(device, non_blocking=True)` 异步 H2D（ROCm SDMA 引擎）与并发 kernel 驱动级竞态 → 偶发 kernel 加载/执行失败；隔离脚本直接 `device="cuda"` 造数据，无此路径。
- 修复（环境变量，不改代码）：`export HSA_ENABLE_SDMA=0`（禁 SDMA 异步拷贝，H2D 走计算拷贝）。与 `AMD_SERIALIZE_KERNEL=1` 同效但语义更精确；稳态 ~1.9s/it（旧 compile=true 时 1.33 it/s，本次 compile=false + Muon 开销，可接受）；换驱动/ROCm 后问题消失可去掉。
- 避免：HIP 异步错误栈位置不可信；隔离复现要复刻数据搬运路径（pin_memory + non_blocking），不只模型前向。

## 4. 冒烟结果（GPU，12 步）
- step 0: train 0.0000, val 10.9548 | 体检 EOS 0% | len 120.0 | rep3 0.0016 | 续轮 0% | ⚠ 不合格（随机初始化正常）；step 12: train 10.9373, val 10.8816 → best.pt（已更新 last.pt）；13 步 · 114.3s
- loss 单调下降、无 NaN/断言 → 新词表 + <cont> 标注 + masking + Muon 全链路通；体检「不合格」是随机初始化正常现象（min_eos_rate 门控在后期评估点才体现价值）。
- CPU 冒烟同样通过：`step 12: train 10.9320 → val 10.8787`（验 masking/词表/标注）。

## 5. 正式训练配置与结果（out/cont_v1_1epoch）
- 数据：治理后 train_char.bin（45,827,969 token，≈1 epoch）；`max_iters=2800` ≈ steps_per_epoch（Chinchilla 最优预算 2.7M 参数 × 20 ≈ 54M token）。
- WSD（stable_frac 0.8 → 2240 步起衰减），warmup 100；环境 `HSA_OVERRIDE_GFX_VERSION=10.3.0 HSA_ENABLE_SDMA=0 TMPDIR=/home/vesita/AI/scratch`；早停关，eval/体检每 280 步；实际 2800 步 · 4058.6s（~1.13h，1.32s/it）。
```
val: 10.9547 → 7.4812 → 6.1064 → 4.8960 → 3.7573 → 2.8009 → 2.1928 → 1.8641 → 1.5294 → 1.3179 → 1.2020（best，WSD 衰减末端）
体检: 280 步起 EOS 自吐 100% · rep3 0.0000 · 全程合格（10/10）
```
- 280 步起即 100% EOS 自吐且 rep3=0 → 快速学会「收尾 + 不复读」，无重复坍缩。
- 采样 16 条（4 prompt × 4 seed）：eos 9 / cont 7，全部自然终止（0 截断）——能在 `<eos>`（收尾）/`<cont>`（递回）间自主选择。
- 文本有碎片感（2800 步小模型正常），结构健康：无死循环/重复，avg len ~10-30。
- 对比旧 eos_fix_1epoch（17000 步，val 0.9413）：本次 2800 步 val 1.2020，收敛方向一致且数据更少——继续训（2-3 epoch）仍有下降空间。
- 遗留（后续 RL）：RL 采样仍以 `<eos>` 为唯一停止符，`<cont>` 被当普通 token；需按 dev-notes/61 §5 加 `<cont>` 停止、重构 `r_quiet_eos`、加 `r_control` 自控奖励。
- 文件改动：`gpt.py` 3D+ 参数 → AdamW；`attention.py` 3 处 mem_up 输入对齐权重 dtype、CSA 掩码乘数对齐 `x.dtype`；ROCm 竞态用 `HSA_ENABLE_SDMA=0`（无代码改动）。
