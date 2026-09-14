# val/train 污染率审计（32-gram 定长哈希等值匹配）

- val 窗口：`out/val_windows_v2.json`（= `per_source_ce_probe.py` **实际用的那一批**，block=256，seed=20260913）
- val bin：`data/chinese/val_char_v2.bin`
- 判据：窗口前 **32** 个 token 全是普通字符（id < 2075）才可查；含 `<eos>`/`<cont>`/`<unk>` 等的窗口跳过并单独计数
- 碰撞上界 ≈ 查询数 × 训练 n-gram 数 / 2^64 ≈ 389 × 1.1e8 / 1.8e19 ≈ 2.4e-09 ⇒ 可视为精确等值匹配

## 对 `data/chinese/train_char_v3_dlg.bin`：命中 **0/369 = 0.00%**

**已知答案对照**（同一次扫描、同一条代码路径）：ctrl_raw: 0/5; ctrl_shuffled: 0/5; train_raw: 5/5; train_shuffled: 0/5 —— `_raw` 应全中、`_shuffled` 应全不中

| 源 | 命中 | 有效窗口 | 命中率 | 窗口总采样数 | 跳过 |
|---|---:|---:|---:|---:|---:|
| c4_zh.txt | 0 | 16 | 0.0% | 32 | 16 |
| classical_poetry.txt | 0 | 23 | 0.0% | 32 | 9 |
| code_alpaca_dialogue.txt | 0 | 19 | 0.0% | 32 | 13 |
| coig_code_dialogue.txt | 0 | 21 | 0.0% | 32 | 11 |
| coig_cqia_dialogue.txt | 0 | 26 | 0.0% | 32 | 6 |
| coig_logic_dialogue.txt | 0 | 19 | 0.0% | 32 | 13 |
| coig_math_dialogue.txt | 0 | 19 | 0.0% | 32 | 13 |
| coig_other_dialogue.txt | 0 | 21 | 0.0% | 32 | 11 |
| coig_wiki_dialogue.txt | 0 | 21 | 0.0% | 32 | 11 |
| dailychat_dialogue.txt | 0 | 17 | 0.0% | 32 | 15 |
| deepseek_r1_distill_dialogue.txt | 0 | 13 | 0.0% | 32 | 19 |
| glm_dialogue.txt | 0 | 16 | 0.0% | 32 | 16 |
| gsm8k_cot_dialogue.txt | 0 | 18 | 0.0% | 32 | 14 |
| kdconv_dialogue.txt | 0 | 22 | 0.0% | 32 | 10 |
| lccc_dialogue.txt | 0 | 18 | 0.0% | 32 | 14 |
| muice_dialogue.txt | 0 | 14 | 0.0% | 32 | 18 |
| multi_turn_dialogue.txt | 0 | 16 | 0.0% | 32 | 16 |
| qwen3_235b_distill_dialogue.txt | 0 | 15 | 0.0% | 32 | 17 |
| wikipedia_cn.txt | 0 | 8 | 0.0% | 32 | 24 |
| zhihu_kol_dialogue.txt | 0 | 17 | 0.0% | 32 | 15 |
| 三国演义.txt | 0 | 2 | 0.0% | 32 | 30 |
| 水浒传.txt | 0 | 5 | 0.0% | 32 | 27 |
| 红楼梦.txt | 0 | 2 | 0.0% | 32 | 30 |
| 西游记.txt | 0 | 1 | 0.0% | 32 | 31 |

## 对 `data/chinese/train_char_v2.bin`：命中 **0/369 = 0.00%**

**已知答案对照**（同一次扫描、同一条代码路径）：ctrl_raw: 5/5; ctrl_shuffled: 0/5; train_raw: 0/5; train_shuffled: 0/5 —— `_raw` 应全中、`_shuffled` 应全不中

| 源 | 命中 | 有效窗口 | 命中率 | 窗口总采样数 | 跳过 |
|---|---:|---:|---:|---:|---:|
| c4_zh.txt | 0 | 16 | 0.0% | 32 | 16 |
| classical_poetry.txt | 0 | 23 | 0.0% | 32 | 9 |
| code_alpaca_dialogue.txt | 0 | 19 | 0.0% | 32 | 13 |
| coig_code_dialogue.txt | 0 | 21 | 0.0% | 32 | 11 |
| coig_cqia_dialogue.txt | 0 | 26 | 0.0% | 32 | 6 |
| coig_logic_dialogue.txt | 0 | 19 | 0.0% | 32 | 13 |
| coig_math_dialogue.txt | 0 | 19 | 0.0% | 32 | 13 |
| coig_other_dialogue.txt | 0 | 21 | 0.0% | 32 | 11 |
| coig_wiki_dialogue.txt | 0 | 21 | 0.0% | 32 | 11 |
| dailychat_dialogue.txt | 0 | 17 | 0.0% | 32 | 15 |
| deepseek_r1_distill_dialogue.txt | 0 | 13 | 0.0% | 32 | 19 |
| glm_dialogue.txt | 0 | 16 | 0.0% | 32 | 16 |
| gsm8k_cot_dialogue.txt | 0 | 18 | 0.0% | 32 | 14 |
| kdconv_dialogue.txt | 0 | 22 | 0.0% | 32 | 10 |
| lccc_dialogue.txt | 0 | 18 | 0.0% | 32 | 14 |
| muice_dialogue.txt | 0 | 14 | 0.0% | 32 | 18 |
| multi_turn_dialogue.txt | 0 | 16 | 0.0% | 32 | 16 |
| qwen3_235b_distill_dialogue.txt | 0 | 15 | 0.0% | 32 | 17 |
| wikipedia_cn.txt | 0 | 8 | 0.0% | 32 | 24 |
| zhihu_kol_dialogue.txt | 0 | 17 | 0.0% | 32 | 15 |
| 三国演义.txt | 0 | 2 | 0.0% | 32 | 30 |
| 水浒传.txt | 0 | 5 | 0.0% | 32 | 27 |
| 红楼梦.txt | 0 | 2 | 0.0% | 32 | 30 |
| 西游记.txt | 0 | 1 | 0.0% | 32 | 31 |
