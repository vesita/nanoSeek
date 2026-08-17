# 让模型自己学会「话说完 → 吐 <eos>」（turn-level EOS 训练验证）

> 目标（用户，2026-08-17）：不是解码端硬截断，而是**让模型在训练分布里学到"话说完吐终止符"**。
> 采用 chat 微调的 turn-level EOS 惯例（Llama-3 `<|eot_id|>` / Qwen `<|im_end|>` 同思路）。
> 实验：给纯对话数据每条「模型：」回复后插 `<eos>`（id=3），qkz 配置从 0 训 1500 步（`out/chinese-data2-ab-qkz-eos`）。

## 改动

- `data/chinese/prepare.py`：新增 `--insert-eos`。行级状态机 `insert_eos_after_replies` 在每条
  模型回复最后一行后插 `<eos>`（多行回复只在末尾插；用户轮次/空行不插）。`prepare_args` 记入 manifest。
  - 数据重生：train 30.04M → 31.04M token（+100 万 EOS，+3.3%）。备份 `*.bak-20260817-noeos`。
  - 编码确认：`"<eos>"` 字面量经 tokenizer 编成 special token id=3；`<eos>` 解码为空串 → 检测必须在 token 级。
- `inference/scripts/sample_py.py`：抽出 `generate_ids()`（返回完整 token + `eos_pos`），generate 保持旧接口；
  未开截断时也自动在模型吐出的首个 `<eos>` 处收尾。
- `training/compare_verbosity.py`：加第三策略「EOS 学习探测」——原始生成下 token 级统计模型自吐
  `<eos>` 的命中率和位置。

## 验证（同一模型、同 6 prompt、同 seed，token 级检测）

| 模型 | v| 现状 avg_len | 现状 rep3 | **EOS 自吐** | 平均位置 |
|------|-----|------------|----------|-------------|---------|
| qkz-EOS（1500 步） | 0.4663 | 78 | 0.015 | **6/6** | 77.8 token |
| qkz 非 EOS（1500 步） | 0.8131 | 200 | 0.042 | **0/6** | 0 |

**结论：模型真的学会了吐终止符。** qkz-EOS 在无任何截断的原始生成（`stop_on_turn`/`stop_on_eos` 全关）下，
6 个 prompt 全部自己吐出 `<eos>` 结束（平均位置 77.8 token），而非 EOS 训练的对照臂一个都不吐、
顶格 200 token 续写。首次效果就达成主目标。

## 附带收益与诚实标注

1. **EOS 训练同时压了喋喋不休**：现状（不截断）策略下 qkz-EOS avg_len 200→78（自身在 EOS 处停）、rep3 降 2.7 倍、d2 0.918→0.960。只用 stop_on_eos 解码即拿到"说话有收尾"——比纯解码端硬截断更本质。
2. **无 loss masking**：本项目是纯因果 LM 对全序列算 loss（user 轮次和特殊 token 也参与预测），EOS 信号比 chat 微调（只对 assistant 算 loss）稀释。本次仍显著学成；若想更强，标准做法是在 train.py 加"只对 `模型：` 之后算 loss"（中等成本，作为后续可选加强）。
3. **内容仍欠训练**：输出仍是共情话术（不回答"推荐小说"）——EOS 解决"停"，不解决"学得好"，符合本项目一贯判断。
4. 跨数据集 val 0.4663 不与非 EOS 的 0.8131 直接可比（val 集也含 EOS，且样本构成不同），本实验的判定依据是采样侧 EOS 自吐率。

## 复现 / 下一步

```sh
# 数据：确认纯对话数据上重跑（改前默认 task_ratio=1.0 需显式 --task-ratio 0）
uv run python data/chinese/prepare.py --task-ratio 0 --insert-eos
# 训练（qkz 配置 + EOS 数据）
uv run python training/train.py training/config/train_chinese_ab_qkz.yaml --out_dir=out/chinese-data2-ab-qkz-eos
# 验证模型自吐 EOS 的比例/位置
uv run python training/compare_verbosity.py --dirs out/chinese-data2-ab-qkz-eos
```

- 下一步可选：① 加 loss masking（只对 assistant token 算 loss，更贴 chat 惯例）② 把 EOS 数据写进默认
  `train_chinese.yaml` 数据链路 ③ Rust 端 `model.rs` 采样加"遇 `<eos>` 即停"（Rust 当前是否尊重 EOS 待确认）。
