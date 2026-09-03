# dev_scripts 分析 / 探针 / 判别工具说明

本项目探针多为一次性诊断, 但核心方法可复用。本 README 说明每个工具"查什么、怎么跑",
方便后续会话/复现。运行统一加 ROCm 环境变量(dev-notes/62):
`HSA_ENABLE_SDMA=0 HSA_OVERRIDE_GFX_VERSION=10.3.0 TMPDIR=/home/vesita/AI/scratch`

## 奖励 / 数学判别
| 文件 | 用途 |
|---|---|
| `test_arith_reward.py` | **回归测试**: 配置驱动数学题生成↔识别闭环 + 答对/答错/拒答奖惩。改 reward 必跑。无 GPU, 秒级。 |
| `diag_arith_reward.py` | **reward 向量分解探针**: 对同一道 40+27=67 喂不同作答(对/错/废话流/空/复述), 打印各维度加权分 + exp_R + GRPO 组内相对优势。验证"废话流避难所"是否剔除。 |
| `rl_probe_math.py` | **训练后模型命中率/健康度**: 配置驱动 80 题 × G4 判组级命中, + 对话题看均长/空/EOS/乱码/d1。`--base_dir` 可并排基座对照。 |
| `probe_capacity.py` | **容量判别**: 基座 argmax 对最简个位数加法能否输出数字。判"模型有无算术能力基础"。 |
| `sft_arith.py` | **算术 SFT 判别**: 从基座 warm-start 监督教 "a加b→答案", 测训练记忆 vs unseen 泛化。判容量天花板。 |

## 坍缩探查(历史)
| 文件 | 用途 |
|---|---|
| `probe_v3.py` | 对指定 ckpt 自采样, 量化空回复/单字壳/数学命中(历史 v3-v5 坍缩用)。改 CKPT 常量。 |

## 结论速查(2026-09-04, dev-notes/66-67)
- 数学 RL 学不会是 **2.7M 容量天花板**(SFT 全覆盖加法表 81 组仍仅 ~10% 记忆, 铁证)。
- `diag_arith_reward` 证明奖励修复(拒答惩罚)让废话流 exp_R 从 +6.2 降到 +0.5、答对 → +14。
- coherence 需 **z-score 化**才能当奖励(原始 mean-logp ~-8 全拉成 -5); 只能抑乱码不能评内容。
- 详见 `dev-notes/66`(根因)与 `dev-notes/67`(判死链+决策建议)。
