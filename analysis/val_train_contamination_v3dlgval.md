# val/train 污染率审计（32-gram 定长哈希等值匹配）

- val 窗口：`out/val_windows_v3dlg.json`（= `per_source_ce_probe.py` **实际用的那一批**，block=256，seed=20260913）
- val bin：`data/chinese/val_char_v3_dlg.bin`
- 判据：窗口前 **32** 个 token 全是普通字符（id < 2075）才可查；含 `<eos>`/`<cont>`/`<unk>` 等的窗口跳过并单独计数
- 碰撞上界 ≈ 查询数 × 训练 n-gram 数 / 2^64 ≈ 173 × 1.1e8 / 1.8e19 ≈ 1.1e-09 ⇒ 可视为精确等值匹配

## 对 `data/chinese/train_char_v3_dlg.bin`：命中 **0/153 = 0.00%**

**已知答案对照**（同一次扫描、同一条代码路径）：ctrl_raw: 0/5; ctrl_shuffled: 0/5; train_raw: 5/5; train_shuffled: 0/5 —— `_raw` 应全中、`_shuffled` 应全不中

| 源 | 命中 | 有效窗口 | 命中率 | 窗口总采样数 | 跳过 |
|---|---:|---:|---:|---:|---:|
| belle_multiturn.txt | 0 | 23 | 0.0% | 32 | 9 |
| dailychat_dialogue.txt | 0 | 21 | 0.0% | 32 | 11 |
| escov_zh.txt | 0 | 19 | 0.0% | 32 | 13 |
| glm_dialogue.txt | 0 | 22 | 0.0% | 32 | 10 |
| kdconv_dialogue.txt | 0 | 23 | 0.0% | 32 | 9 |
| lccc_dialogue.txt | 0 | 20 | 0.0% | 32 | 12 |
| sharegpt_zh_38k.txt | 0 | 20 | 0.0% | 32 | 12 |
| wildchat_zh.txt | 0 | 5 | 0.0% | 32 | 27 |

## 对 `data/chinese/train_char_v2.bin`：命中 **0/153 = 0.00%**

**已知答案对照**（同一次扫描、同一条代码路径）：ctrl_raw: 5/5; ctrl_shuffled: 0/5; train_raw: 0/5; train_shuffled: 0/5 —— `_raw` 应全中、`_shuffled` 应全不中

| 源 | 命中 | 有效窗口 | 命中率 | 窗口总采样数 | 跳过 |
|---|---:|---:|---:|---:|---:|
| belle_multiturn.txt | 0 | 23 | 0.0% | 32 | 9 |
| dailychat_dialogue.txt | 0 | 21 | 0.0% | 32 | 11 |
| escov_zh.txt | 0 | 19 | 0.0% | 32 | 13 |
| glm_dialogue.txt | 0 | 22 | 0.0% | 32 | 10 |
| kdconv_dialogue.txt | 0 | 23 | 0.0% | 32 | 9 |
| lccc_dialogue.txt | 0 | 20 | 0.0% | 32 | 12 |
| sharegpt_zh_38k.txt | 0 | 20 | 0.0% | 32 | 12 |
| wildchat_zh.txt | 0 | 5 | 0.0% | 32 | 27 |
