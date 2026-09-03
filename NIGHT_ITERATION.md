# 醒来看这里 —— 2026-09-04 夜间 RL 迭代速览

> 详细见 dev-notes/66(根因) 67(判死链+成果) 68(能力边界+方向建议)。
> 全部改动已 commit (5ee106e → 76e8ca2, 6个)。无残留训练进程。

## 一句话结论
你要的"数学 RL"经 4 套独立实验证明：**2.7M 模型学不会算术是容量天花板，不是奖励/步数问题**。
同时我修复了奖励梯度、把语感(coherence)做成可用奖励、并产出了一个比基座**明显更好**的对话模型。

## 最佳模型（可直接用）
```
out/rl_coh_dialog/best.pt    # 300步 coherence RL 对话增强
# 特点: EOS自吐100%, len<=1从基座6%→0%, 乱码5%<基座8%, 会说通顺的话
# 部署: uv run python inference/scripts/chat.py --out_dir out/rl_coh_dialog
```
基座 `out/cont_v1_1epoch` 是治理后 SFT 基线(只会碎片话术); rl_coh_dialog 学会自然对话。

## 你关心的两个问题怎么回答的
| 你的问题 | 结论 | 证据 |
|---|---|---|
| 数学能不能 RL 学会? | **不能(2.7M容量)** | 纯RL1000+300步0%; SFT 700步泛化0%; 全覆盖81加法表仅~10%记忆; 数字位概率与输入解耦(见66§6-7) |
| 增加语法/语感奖励? | **做了,有效** | r_coherence 需 z-score 化(--coherence_w), 乱码-1.94/正常+1.8; 但只抑乱码,不能评内容(见66§3) |
| 分析/理论? | **写了** | reward向量分解diag + 容量probe + SFT判别 + logprob数字位分析 + dev-notes/66-68 |

## 本轮落地代码(git)
- r_arith 拒答惩罚: 废话流不给数字重罚, 剔避难所
- has_answer_intent: 严格判定"有无作答"(用阿拉伯数字信号)
- grpo_char --coherence_w: coherence z-score 奖励开关(默认0兼容)
- 工具: dev_scripts/{diag_arith_reward, probe_capacity, rl_probe_math, sft_arith, dialog_quality}.py

## 方向建议(详见 68)
1. 目标=稳定对话小模型 → 用 rl_coh_dialog, 已是实践边界
2. 目标=真理解/数学/多轮 → 需换更大模型, 2.7M 上调参无突破(4套铁证)
3. 数学硬目标 → 走工具化/格式化(模型输出算式, 外部算), 不追求真算
