# 64-连续坍缩根因（KL符号Bug + 短空壳漏门）与稳定重训

日期 2026-09-03（接 63）; 涉及 `updater.py` / `grpo_char.py`

## 症状
`rl_cont_v1..v4` 每轮都从健康基座 `out/cont_v1_1epoch/best.pt` 重跑却逐一坍缩：
- v3 组均分 **-3271**（exp 负尾爆炸，尚无 max_neg 上限）
- v4 修了负尾上限，step 150 退化到 `均长 1.0`、全组 `"`；probe 复测 **64/64 全输出 `"`，数学命中 0/48**
- 坍缩路径 `Step1 均长16 → Step20 4.2 → Step30 1.0` → 后在 1~9 振荡、趋势向下；门栏只挡 `len<=1`，抓不住 `len 2~4` 短空壳

## 根因 1（决定性）：重构把 KL 方向写反
- 旧：`F.kl_div(curr_logp, ref_logp, log_target=True)`（策略→参考，下界=0，最小在"策略==参考"）。
- 重构后：`kl_loss = (reflogprobs - new_logprobs).mean()` ← 裸对数差、无下界；策略越自信值越负 → 最小化 total 反驱动置信度爆炸/坍缩。
- 实锤：重构前日志 KL 恒**正**（5~44），重构后恒**负**（-0.36 ~ -2.85）。

## 根因 2：反坍缩门只挡 len<=1
`content_collapse` 只判"组最佳 len<=1"；len 2~4 短空壳逃过旧门，被组内归一化当"最不烂"给正优势 → 均长拖向 1；配合负 KL 即"越陷越深"。

## 修复
1. `updater.py` KL 改回规范散度 `F.kl_div(..., log_target=True, reduction='none')` 按 mask 归一。单测：同分布 → 0、策略更自信 → +0.12（正、有界）；日志 KL 回到 ~0。
2. `grpo_char.py` 门补强：除"组最佳 len<=1"外再判 `组均长<=4` → 整组短空壳即判死、归零优势。
3. `updater.py` device bug：`quiet_losses` 中 `compute_quiet_loss_from_hidden` 早退返回 CPU `0.0`，与 CUDA 候选 `torch.stack` 崩 `Expected all tensors on same device`；修法：stack 前统一 `.to(advantages.device)`（正常训练都走该分支，全 eos 恰好绕过）。

## 重训（rl_cont_v5）
- 从 `out/cont_v1_1epoch/best.pt` 重跑（固定 `--ref_base`）+ `beta_kl` 0.8。
- 20 步健康（0% 坍缩），150 步终态仍缓慢坍缩到 `"`（81~100%）。热探针 `probe_v3.py` + 0.6~1.4 温度扫描确认坍缩；日志"均长 18~28"矛盾根因：坍缩在被记录步之间/末尾，只存单点终态。
- 三修复把"爆炸性坍缩"(v3/v4) 稳定成"缓慢坍缩"，未根治 `"` 吸引子。

## 根因 3（深水区）：`"` 低方差吸引子
- 2.7M 长句**高方差**（偶尔命中关键词/长度奖励 +5，多数净分低）；`"`+<eos> **近乎零方差、概率近 1**，raw reward ~-1.4，组内归一化下与"更烂"候选互相掩护 → 累积概率质量。
- 纯绝对惩罚在"整组都是 `"`"时被归一化抵消 → 必须 (a) 奖励层拉开差距 + (b) 逐候选硬压制，让退化候选**永远**无正优势。

## 阶段二（三层）
1. `reward/dimensions.py r_natural`：L<5 惩罚由 -2.5 改为随短缺陡增 `-1.0*(6-L)`（L=1 → -5.0 … L=4 → -2.0）；数学正确短答软化（-0.5）。
2. `r_anti_collapse`：非 <cont> 型单字空壳（`"`+eos 等）加 -3.0（L=1）/ -1.5（L=2）。
3. `grpo_char.py` **Anti-Dwarf 硬压制**：空壳/单字/二字非数字候选优势压到组最负（`adv_bottom-1`）；单数字回复保留。
- 实测：`"` -10.4（最差），garbled 20 +3.4，good 24 +5.1。
- 输出 `out/rl_v6_*/`（60 步 smoke → 全量）。
