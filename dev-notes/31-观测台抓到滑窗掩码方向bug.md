# 31-观测台抓到滑窗掩码方向 bug（+ KV/注意力观测台工具）

## 现象
- `probe --stats` 第一版冒烟：滑窗注意力距离衰减曲线全为 0。
- 打印权重：query 50 的 top-5 key 为 91/113/93/106/77 —— **全部 > 50**，滑窗注意力在看**未来** token，违反因果。

## 根因
- 掩码 `win_causal[r,c] = (r <= c) & (c - r <= win)`：行 r 允许列 c ∈ [r, r+win]；flash（SDPA attn_mask）与手动（masked_fill）都按 **[query, key]** 读 → 实际 key ∈ [q, q+win]，是"前视窗"非"回看窗"。
- 后果：训练偷看未来 token（数据泄漏），推理无未来 token → **train/inference 不一致**；此前所有 A/B / 基线数值都带此 bug（相对对比近似成立，绝对值不可信）。

## 修复
- 一行翻转方向：`win_causal = (i.unsqueeze(0) <= i.unsqueeze(-1)) & (i.unsqueeze(-1) - i.unsqueeze(0) <= win)`
- 验证（同 ckpt 前向）：q=50 top5 keys = [25, 28, 49, 21, 22]（全 ≤ 50 ✓）；距离衰减 δ=0: 0.023 → δ=1: 0.023 → δ=8: 0.020 → δ=32: 0.015 → δ=63: 0.014，平滑 ✓。

## 怎么避免
- 掩码/矩阵方向 bug 肉眼难看出：先做**不变量测量**（"权重是否违反因果"）。
- 新增注意力路径冒烟须检查 top-k key 索引 ≤ query 索引（dist_curve + top1）。
- 修完后基线需**重训**：`out/obs_kv` 旧 run 已作废归档。

## 工具：KV/注意力观测台（probe --stats）
统计 KV 范数/位置漂移、有效秩、距离衰减、头熵（坍缩）、sink 使用率、头间余弦、CSA 三路贡献；输出 `out/<dir>/stats/stats.json` + 6 张 PNG（kv_norm_pos / kv_effrank / dist_decay / attn_entropy / head_sim / csa_paths）；用法 `uv run python cli.py probe --out_dir=out/obs_kv --stats --num_batches=8`；实现：`attention.py` 加 `capture`、`probe.py` 加 `run_stats()`。
