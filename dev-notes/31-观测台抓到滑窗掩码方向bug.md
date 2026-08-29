# 31-观测台抓到滑窗掩码方向 bug（+ KV/注意力观测台工具）

## 现象
- `probe --stats`（KV/注意力观测台）第一版冒烟时，滑窗注意力距离衰减曲线全为 0；
- 直接打印权重发现：query 50 的 top-5 key 是 91/113/93/106/77 —— **全部 > 50**，
  即滑窗注意力在看**未来** token，违反因果。

## 根因
- `model/attention.py` 滑窗因果掩码构造为
  `win_causal[r, c] = (r <= c) & (c - r <= win)`：行 r 允许列 c ∈ [r, r+win]；
- flash（SDPA attn_mask）与手动（masked_fill）两条路径都按 **[query, key]** 读它
  → 实际允许 key ∈ [q, q+win]，是"前视窗"而非"回看窗"；
- 后果：训练时模型可偷看未来 token（数据泄漏），推理生成时不存在未来 token
  → **train/inference 不一致**，生成的模型吃的是训练时"作弊"学到的分布；
- 此前所有 A/B / 基线数值都带这个 bug（相对对比可能仍近似成立，绝对值不可信）。

## 修复
- 一行翻转方向：
  `win_causal = (i.unsqueeze(0) <= i.unsqueeze(-1)) & (i.unsqueeze(-1) - i.unsqueeze(0) <= win)`
- 修复后验证（同一 checkpoint 前向）：
  - q=50 top5 keys = [25, 28, 49, 21, 22]（全部 ≤ 50 ✓）；
  - 距离衰减 δ=0: 0.023 → δ=1: 0.023 → δ=8: 0.020 → δ=32: 0.015 → δ=63: 0.014，平滑衰减 ✓。

## 怎么避免
- 掩码/矩阵方向类 bug 肉眼极难看穿：先做**不变量测量**（"权重是否违反因果"）再谈结构探索；
- 新增任何注意力路径，冒烟时必须检查 top-k key 索引 ≤ query 索引（观测台 dist_curve + top1 检查即可）；
- 修完后基线需**重训**：out/obs_kv 旧 run 已作废归档。

## 工具：KV/注意力观测台（probe --stats）
- 统计项：KV 范数分布/随位置漂移、KV 有效秩（参与比）、注意力距离衰减曲线、
  头熵（坍缩检测）、sink 使用率、头间余弦相似度、CSA 三路路径贡献（comp/win/glob 平均 token 范数）；
- 输出 `out/<dir>/stats/stats.json` + 6 张 PNG（kv_norm_pos / kv_effrank / dist_decay /
  attn_entropy / head_sim / csa_paths）；
- 用法：`uv run python cli.py probe --out_dir=out/obs_kv --stats --num_batches=8`；
- 实现：`model/attention.py` 加 `capture` 开关（前向记录 q/k/v、softmax 权重、路径范数、多头输出），
  `model/probe.py` 加 `run_stats()`（聚合 + 绘图）。
