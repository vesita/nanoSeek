# 主题 C · 数据与词表治理

> 吸收笔记：[04](04-换词表模型作废.md) [05](05-小模型嵌入表占大头.md) [07](07-分词器压缩率要实测.md)
> [21](21-碎片拼贴根源是数据污染.md)

## 核心事实

1. **换词表 = 模型作废**（04）：BPE 重训后 token→id 映射全变（同一句"你好世界"旧 `[4285,8957,27833]` 新 `[5298,1152]`），
   旧权重对新 tokenizer 无意义。换词表必须：重训 tokenizer → prepare 重编码 → **模型从头重训**。
2. **小模型嵌入表占大头**（05）：嵌入 = 词表×维度。30k×96=2.88M 占 4.32M 的 **67%**（推理能力只剩 1/3）；
   压到 8k 词表 → 嵌入 0.77M（35%），"生产线"占比反超。设计模型先算 `vocab×n_embd÷总参数`，>50% 要警惕；
   **词表大小决定模型尺寸下限**。
3. **压缩率必须实测**（07）：README 写 ~4.2x，实测 1.50x。子集上训 BPE 会系统性低估压缩率（只可看相对趋势）。
   报告数字 = 全量词表 + 同一份评估文本实测。
4. **采样崩先查数据分布**（21）：af 采样全崩成任务模板拼贴（对对联/实体识别/热评碎片）→ 排查链
   排除采样器/模型/checkpoint 后定位到数据：**zhuangxialie（149MB 单轮指令）占 train 54.7%**，
   模型自由生成先验 = "接碎片"；val 只验对话所以 val 低但采样崩——又一个"loss 骗低"，根源在数据。

## 治理动作（21，已落地）

```sh
uv run python data/chinese/prepare.py --task-ratio 0   # 剔除任务碎片，train 只剩对话
# 结果：train 30.0M token（原 72.4M），94,097 对话条进 train / 10,457 进 val
# 原数据备份：train.bin.bak-20260812-zhuangxialie / val.bin.bak-20260812（--task-ratio 归一化治理）
```

- 每文件独立 seed（`1337 + sum(ord(c) for c in fn)`）保证切分可复现。
- 重训产物：`out/chinese-data2-clean-af5k`（纯对话数据，采样验证碎片消除）。

## 怎么避免

- 后续加语料：**先量化各文件 token 占比**（不是文件个数），对话类/任务类分开配比。
- 新数据/新词表都是高成本操作（隐含模型重建）：动手前备份数据、记 manifest 哈希（prepare.py 已写）。
- 排查采样质量：怀疑采样器 bug 先确认 `build_model_from_checkpoint` 是否 `.eval()`（本项目是 eval 的）；
  用权威采样器 `inference/sample.py` 交叉验证最可靠。

## 本次新的数据来源（2026-08-17，工具已入库）

- `training/extract_agent_dialogues.py`：**只读**提取 Claude Code（`~/.claude/projects`）与
  DSH Harness（`~/.dsh/sessions`，zstd 流式解压）真实对话日志 → 清洗成 `用户：/模型：` 语料。
  - 只取纯文本轮次对，剥掉工具调用/代码块/系统上下文；长度/中文占比/去重过滤。
  - 2026-08-17 干跑：1804 候选 → **805 对 ≈104K 字符**（真实问答、有收尾，正好是"治喋喋不休"示范数据）。
  - 产物 `data/chinese/agent_dialogue.txt` 已 gitignore；接入训练需并入 prepare.py（未做）。