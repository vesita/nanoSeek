# 64-连续坍缩根因（KL符号Bug + 短空壳漏门）与稳定重训

日期 2026-09-03（接 63）
涉及: `training/rl/updater.py`（KL 修正）/ `training/rl/grpo_char.py`（反坍缩门补强）

## 症状
`rl_cont_v1..v4` 每轮都从**健康 SFT 基座** `out/cont_v1_1epoch/best.pt` 重跑，却逐一坍缩：
- v3 组均分到最后打到 **-3271**（exp 负尾爆炸，还没 max_neg 上限）
- v4 修了负尾上限，但到 step 150 退化到 `均长 1.0`、全组 `"` 单字符；
  probe 复测 `rl_cont_v4/best.pt`：**64/64 全部输出单字符 `"`，数学命中 0/48**。

坍缩路径（v4 实测量）：`Step1 均长16 → Step20 均长4.2 → Step30 均长1.0(cont)`
→ 之后在 1~9 之间振荡、趋势向下，门栏只挡住 `len<=1`，抓不住 `len 2~4` 短空壳。

## 根因 1（决定性）：重构把 KL 方向写反了
旧 `grpo_char.py` 用规范 KL：
```
kl = F.kl_div(curr_logp, ref_logp, log_target=True)   # 策略→参考，有界下界=0，最小在"策略==参考"
```
重构抽到 `updater.py` 后写成了：
```
kl_loss = (reflogprobs - new_logprobs).mean()          # ← 裸对数差，无下界
```
- `F.kl_div(input=策略logp, target=参考logp)`：逐点 = `ref_logp*(ref_logp-policy_logp)`，
  策略==参考时严格 0，偏离越远越正 → 退化时是"拉回基座"的主力。
- 裸 `(reflog-news_logp)`：策略越自信值越负、无下界，优化器最小化 total 时**反而驱动
  置信度爆炸/模式坍缩**。实锤：重构前日志 KL 恒**正**（5~44），重构后 KL 恒**负**
  （-0.36 ~ -2.85）。负 KL = 它已不是 KL，而是"强化当前输出"的滑坡项。

## 根因 2：反坍缩门只挡 len<=1 单字空壳
`content_collapse` 只判"组最佳 len<=1"。len 2~4 的短空壳（`可以10"` / `不我,"`）长度>1
逃过旧门，又被组内相对归一化当"最不烂"给正优势 → 均长被一步步拖向 1。配合根因 1 的
负 KL，整组一旦退化就被"越陷越深"，门内门外都救不回来。

## 修复
1. **`updater.py`**：KL 改回规范散度
   ```
   kl_tensor = F.kl_div(new_logprobs, reflogprobs, log_target=True, reduction='none')
   kl_loss   = (kl_tensor * mask).sum() / mask.sum().clamp(min=1)
   ```
   单测：两分布相同 → 0；策略比参考更自信 → +0.12（正、有界）。日志 KL 回到 ~0 量级。
2. **`grpo_char.py` 反坍缩门补强**：除"组最佳 len<=1"外，再判 `组均长<=4`
   → 整组被拖成短空壳即判死、归零优势（退化时只剩正确 KL + 静默拉回基座）。
3. **`updater.py` device 兼容 bug**（改完上面后暴露的**既有隐患**）：`quiet_losses`
   列表里 `compute_quiet_loss_from_hidden` 早退分支返回 CPU `0.0`，与 CUDA 候选混在
   一起 `torch.stack` 会崩 `Expected all tensors on same device`。修复：stack 前统一
   `.to(advantages.device)`。正常训练（cont/eos 多样终止、空回复）都会走这条 CPU
   分支，之前坍缩后全 eos 恰好绕过它。<br><br>

## 重训（阶段一 rl_cont_v5）
- 从健康基座 `out/cont_v1_1epoch/best.pt` 重跑（固定 `--ref_base` 同基座）+ `beta_kl` 0.8。
- 结果：20 步健康（0% 坍缩），但 150 步终态仍缓慢坍缩到单引号 `"`（81~100%）。
  热探针 `dev_scripts/probe_v3.py` 复测 + 0.6~1.4 温度扫描确认：**终态确实坍缩，
  与训练日志"均长 18~28"矛盾的根因是：坍缩发生在被记录步之间/末尾，训练只存在
  单点终态保存，看不到中途趋势**。三个修复（KL 方向 + 反坍缩门 + device）把训练
  从"爆炸性坍缩"(v3/v4) 稳定成"缓慢坍缩"，但未根治 `"` 吸引子。

## 根因 3（阶段二要根治的深水区）：`"` 单引号低方差吸引子
- 2.7M 模型长句**高方差**：偶尔命中关键词/长度奖励(+5)，但多数长句不完美、净分低；
  `"`+<eos> **近乎零方差、概率近 1**，原始 reward ~-1.4 但组内归一化下可与"更烂"
  候选**互相掩护**——每人分数接近，`"` 的优劣不明显，其高确定性使其逐步累积概率质量。
- 纯绝对惩罚在"整组都是 `"`"时被归一化抵消，单靠绝对惩罚救不回来 → 必须同时
  (a)奖励层拉开差距 + (b)逐候选硬压制，让退化候选**永远**拿不到正优势。

## 阶段二 加固（reward/grpo 三层）
1. `reward/dimensions.py r_natural`：L<5 短回复惩罚由固定 -2.5 改为 **随短缺陡增**
   `-1.0*(6-L)`（L=1 → -5.0 … L=4 → -2.0），数学正确短答仍软化（-0.5）。
2. `reward/dimensions.py r_anti_collapse`：对**非 <cont> 型单字空壳**（`"`+eos 等）
   增加 -3.0（L=1）/ -1.5（L=2），专治最顽固坍缩形态。
3. `grpo_char.py` 新增 **Anti-Dwarf 逐候选硬压制**：空壳/单字/二字非数字候选的
   优势直接压到组最负（`adv_bottom-1`），杜绝被归一化强化；单数字回复（数学努力）
   保留。
- 实测奖励：`"` → raw -10.4（压倒性最差），garbled 20 → +3.4，good 24 → +5.1；
  长句稳定收益、`"` 稳定垫底。
- 输出 `out/rl_v6_*/`（60步 smoke 验证 → 全量）。