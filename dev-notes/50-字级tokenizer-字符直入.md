# 50-字级 tokenizer（字符直入：汉字=1 token）

## 动机（用户指示 2026-08-30）
- 字节直入失败（dev-notes/48）：3M 模型学字节统计不学语义，3 并行头组合爆炸出生僻字。
- 用户要求：**对齐字符 + 高效率**。字级同时满足：1 汉字 = 1 token（无组内组合/字节爆炸）、序列 ≈ 汉字数（≈BPE）、训练 ≈19 分钟（非字节 20-30 分钟）、词表只有字无 BPE 话术片段合并。

## 设计
词表 ~5000、生僻字 → `<unk>`、复用标准 GPT 架构（**无聚合层/并行头**）；管线 `prepare.py --char-level` → `train.py` `char_level` 联动（vocab 从 meta、block_size 256、use_mtp 可开）→ 推理 load_tokenizer 加 char 分支（自动检测 model_args.char_level）。

## 验证
1. 词表覆盖语料 ~99% 字符；2. fp32 smoke 前向/梯度；3. 300 步 A/B（vs BPE 基线）——**决定性：生成是否正常中文**；4. 好则 1500 步（~19 分钟）+ 对话评估（话术是否比 BPE 更少）。

## 与字节直入
字节代码保留（byte 分支 + 推理适配），字级是替代路线（同"无 BPE 话术固化"目标，更适配小模型）；若字级也一般 → BPE 主线 + 数据 v2（分句器 + 日常权重）兜底。

## 实施记录（2026-08-30）
- **统一到标准 tokenizers 库**（用户要求"由 tokenizer 实现"）：字级 = WordLevel 词表（每字一词条），pre_tokenizer = Split([\s\S]) 逐字符切分，全链路 Tokenizer.from_file 复用——模型零改动（标准 GPT + vocab 4623），只换 tokenizer + mask ids。
- `train_tokenizer.py --char` 构建词表（4623 项：特殊+换行+ASCII96+全角标点+常用汉字4500）；`data/chinese/char_tokenizer.py` 已删除（用户指示）。
- prepare `--char-level` → train_char.bin（52.9M token = 1 字 1 token）；mask ids 从 WordLevel get_vocab 取"模型：/用户：/换行"。
- 踩坑：data_dir 在联动后定义（用 data/dataset 拼路径）；GPTConfig 补 char_level 字段。
- 3 步 GPU 冒烟 ✅（2.77M 参数、loss 正常、mask 生效）。

## 状态
- [x] 记录 / char tokenizer / prepare 联动 / 推理 load_tokenizer / 3 步冒烟
- [ ] 300 步 A/B（后台 bash-30，对照 bpe_daily_300）；[ ] 1500 步 + 对话评估
