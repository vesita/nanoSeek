# 32-本地 DSH 会话导出对话语料

## 现象/动机
- chinese 数据管线一直没有真实语料（`data/chinese/` 只有脚本，`train.bin` 从未生成）；
- 本地 DSH（`~/.dsh/sessions/`）存了 47 个真实开发会话（压缩 115MB），内容就是
  中文技术对话——正好是"用户：/模型："训练格式。

## 做法
- `data/dsh/export.py`：扫描 `~/.dsh/sessions/*/*/session.jsonl.zstd`（zstd 压缩 JSONL），
  抽取 `user/message`（data.content[].text）与 `assistant/message`（data.message.content[] 中
  type=="text" 的块），按序配对成 `用户：…\n模型：…` 交换；
- 过滤：丢 reasoning（思维链）/ tool-call / tool/result；丢压缩摘要与系统注入
  （"This is an automatically generated checkpoint"、"Current runtime context"、
  "<system-reminder>" 等标记）；单条限长、交换限最短长度、全局去重。

## 结果
- 292 个交换、52 万字符（≈ 预期 12~15 万 BPE token）；`data/dsh/dialogue.txt`；
- 与 chinese 管线格式完全一致（同 download_dialogue.py 输出），可直接进
  train_tokenizer.py / prepare.py。

## 注意
- 会话是**活的**：导出用一次性快照（zstdcat 只读），别用软链接挂进 data/ 防污染；
- 过滤词表要跟着 DSH 注入消息的变体走（已踩：runtime-context 变体漏网导致 26 个假配对）；
- 语料里有大量工具调用上下文被滤掉后留下的"半截"对话，质量需抽查；后续可加
  --max-samples / 按工作区筛选。
