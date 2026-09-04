# 醒来速览 · 奖励引擎×词表混合训练（2026-09-04 日间）

你在睡前提的方案：**"奖励引擎可以和词表混合着学"** —— 已完成验证，结论如下。

## 一句话结论
**混合着学成立，但机制必须是 GRPO（组相对优势），不是 RWR（softmax 加权模仿）**。
前者可"压制"坏输出，后者只会"模仿"好输出——2.7M 小模型采样出的次优壳一旦被
模仿就自我强化，三次实测全塌。换成 GRPO + 语料 CE 同场后，产出**新最佳对话模型**。

## 结果对比（同起点 cont_v1_1epoch SFT 基座，均 300 步，dialog_quality 两轮）
| 指标 | 基座 | rl_coh_dialog (纯GRPO) | **rl_mixed_v4 (GRPO+语料CE)** |
|---|---|---|---|
| 乱码字占比 | 9% | 5~7% | **2~3%** ✅ 最低 |
| 均长 | 11.5 | 19~20.4 | **26.1~26.8** ✅ 最稳 |
| 多样性(8话题) | 100% | 100% | 100% |
| 温和确认腔 | 0/8 | 2/8 | 1~2/8 |
| 退化步/300 | - | 稀疏 | ~6/300 健康 |

**关键洞察**：混入语料 CE 的 GRPO = "更稳的 GRPO"。词表学习全程在场给模型更强的
语言锚，RL 漂移更少 → 乱码减半、均长更稳。收益在**语言稳定性**，不在理解/算术
（2.7M 容量天花板不变，见 dev-notes/68）。

## 踩坑记录（已写入 dev-notes/69，供以后参考）
1. **RWR 必塌**：softmax 恒正权重只学正样本不压制负样本 → v1/v2/v3 三次全塌（均长→0）
2. 语料 CE 必须 assistant-mask（与基座 SFT 同目标）；全 token CE 让模型学"预测用户轮次"，目标错位
3. 退化时 KL+EOS 静默必须全程在场（不能随 RL 权重一起清零），否则模型无锚点漂移
4. masked CE + MTP 辅助 loss：窗口有效 token 极少时 MTP targets 全 -100 → 0/0=NaN，
   需 min_valid 窗口过滤（>=32 token）
5. 退化门控"组最佳"用 argmax 奖励候选长度，不是组内最短（v2 用 min 导致每步误判退化）

## 产物
- **最佳模型**: `out/rl_mixed_v4/best.pt` → `chat.py --out_dir out/rl_mixed_v4` 直接部署
- 训练脚本: `training/rl/rwr_mixed_char.py`（--rl_mode grpo 推荐 / rwr 对照）
- 文档: dev-notes/69 + out/README-MODELS.md 已更新
- 验证: dev_scripts/test_arith_reward.py 全部通过
- commit: 21200b6（main，未推送）

## 复现命令
```bash
# 冒烟 (30步, ~5分钟)
HSA_ENABLE_SDMA=0 HSA_OVERRIDE_GFX_VERSION=10.3.0 TMPDIR=/home/vesita/AI/scratch \
PYTHONUNBUFFERED=1 .venv/bin/python training/rl/rwr_mixed_char.py \
  --ckpt out/cont_v1_1epoch/best.pt --out out/rl_mixed_v4_smoke --steps 30 \
  --rl_mode grpo --coherence_w 0.5

# 正式 (300步, ~40分钟)
... 同上, --out out/rl_mixed_v4 --steps 300

# 评估
... .venv/bin/python dev_scripts/dialog_quality.py --ckpt_dir out/rl_mixed_v4 --label MIXED
```
