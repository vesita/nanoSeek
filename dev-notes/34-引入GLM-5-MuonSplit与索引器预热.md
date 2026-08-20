# 34-引入 GLM-5 Muon Split + 索引器预热（DSA 配方）（2026-08-20）

背景：用户调研前沿技术（Kimi K3 / GLM-5.x / Qwen3.8 / 小模型技术）后选定两项落地：
1. **Muon Split**（GLM-5 技术报告 2026-02）：修 dev-notes/33 实测的「Muon @1500 步收敛差」
   （val 1.6522 vs AdamW 1.2700）。
2. **索引器预热**（GLM-5 DSA 配方）：复活 lightning indexer（dev-notes/13 曾因 1500 步拖慢而关）。

## 1. Muon Split（model/optimizer.py + utils.py + gpt.py）

- 动机（GLM-5 报告原话）：Muon 配 MLA/注意力时，整块投影矩阵做 NS 正交化追不上简单方案
  （GLM-5 实测 Muon+MLA 弱于 GQA-8）；改成按「注意力头」分块、每头单独正交化后追平，
  且注意力分数训练中自动稳定（不需要额外 clamp）。
- 实现：`zeropower_via_newtonschulz_split(G, n_heads, head_first)`——把投影权重按头切片，
  逐头做 NS 正交化（内部统一 (H, d, n) 布局批处理，窄侧 d 小，XX 是 d×d）。
  三种布局全覆盖（与逐头调用原函数逐位对拍 max_err=0.00）：
  - 行方向按头（q/k/v 投影，含融合 c_qkv_csa 的 3·H 头）→ head_first=True
  - 列方向按头（输出投影 c_proj）→ head_first=False
  - MLA 的 q_proj/k_up/v_up → head_first=True；kv_down 是低秩潜在（无头结构）→ 整块
- 踩坑：SwiGLU 下投影也叫 `c_proj`（(n_embd, hidden) 非方形）——误匹配会按头拆 FFN 权重
  且形状报错（80×213）。修复：c_proj 必须 `.attn.` 路径 + 方形 (n_embd, n_embd) 双条件。
- 开关：`use_muon=true` 时 `muon_split=true/false`（默认 false，A/B 后定）；零参数、零前向
  影响、checkpoint 兼容（resume 时 configure_optimizers 重建 split_heads，不依赖 id 持久化）。
- 验证：数值对拍 0.00；60 步冒烟 loss 正常下降；checkpoint 记录 muon_split。

## 2. 索引器预热（train.py `indexer_warmup_steps`）

- 配方（GLM-5 技术报告）：DSA 稀疏注意力适配 = 先 1000 步**只训练索引器、主模型冻结**，
  再做 20B token 稀疏适配——20B token 追平 DeepSeek 943.7B token 的 DSA 训练效果（~47×）。
- 实现：`indexer_warmup_steps=N` 时训练循环开头冻结主模型（只 idx_q/idx_k 可训练），
  N 步后解冻联合训练。关键：先建优化器再冻结（configure_optimizers 过滤 requires_grad）。
  预热期优化器仍持有全部参数，冻结参数 grad=None 自动跳过。
- 注意：aux-free 路由偏置在 MoE.forward 原地更新（不经过优化器），预热期仍会微调
  （幅度 balance_factor=0.001，自校正，可接受，相当于路由偏置顺带预热）。
- 前置条件：`use_lightning_indexer=true`（无索引器时 assert 报错，防止全模型冻结成 NaN）。
- 验证：预热期只有 idx_q/idx_k 有梯度（4 个/2 层），解冻后全参数恢复；80 步冒烟通过。

## 3. A/B 结果（1500 步，A1 数据 41.8M/3.46M，与 dev-notes/33 同基线）

| 模型 | 优化器 | 索引器 | val@1500 | 你好体检 EOS | 多轮 EOS |
|---|---|---|---|---|---|
| reb | AdamW | 关 | 1.2700 | 9/10 | 3/9（33%） |
| reb+Muon 整块（dev-notes/33） | Muon | 关 | 1.6522 | 10/10 | 4/9（44%） |
| **reb+Muon Split** | Muon Split | 关 | **0.7776** 🏆 | 9/10 | 3/9（33%） |
| reb+Muon Split+索引器预热 | Muon Split | 开（200 步预热） | 1.5076 | — | — |

### Muon Split：决定性胜出 ✅ → 翻默认

- val 0.7776 vs 整块 1.6522 vs AdamW 1.2700：**不仅修好 Muon 短预算收敛差，还大幅超越
  AdamW 冠军**（全项目历史最优）。后段爆发（750: 2.28 → 1000: 1.77 → 1250: 1.06 → 1500: 0.78），
  WSD 衰减期（1200 后）收益最大——与 GLM-5 报告「按头正交化让注意力分数自动稳定」吻合。
- 采样质量混合：EOS 9/10（与 reb 持平）、回复变长（100.6 vs 66.5）、rep3 0.0222（绝对仍低，
  但比 reb 0.0025 高 9 倍）、seed 2 出现 1 次自续轮。val 大幅下降 ≠ 采样全面变好（延续
  dev-notes/14「val 与质量解耦」规律），但离坍缩（rep3>0.1）很远。
- 多轮：EOS 33%（与 reb 持平），第 2/3 轮仍全灭——优化器与多轮能力正交，模型层缺口不变。
- 已翻默认：train_chinese.yaml `muon_split: true`。

### 索引器预热：救不回小模型 ❌ → 保持默认关

- val 1.5076（200 步预热 + Muon Split）vs 无索引器 0.7776：索引器（即便用 GLM-5 的
  冻结预热配方）在 2.85M/1500 步预算下仍净负——比整块 Muon（1.6522）好、比 AdamW（1.2700）
  差。预热只是把「索引器拖慢」改成「先预热再拖慢」，预算太小没机会回本。
- 机制本身工作正常（预热期只有 idx_q/idx_k 有梯度、MFU 0.60 验证冻结省反向、解冻正常）。
- 开关保留：`indexer_warmup_steps` 供长预算（WSD 全周期）A/B，默认 0。
- 踩坑：索引器路径内存峰值更高（batch 64 OOM，需 batch 32×grad_accum 2 等效 batch 重试）。

## 结论

1. **Muon Split 是 GLM-5 报告里最值得搬进小模型框架的技术**：零参数、纯优化器、一改即赢。
2. 索引器类 DSA 技术在 2.85M/1500 步下继续维持负面结论（dev-notes/13 复现），GLM-5 的
   20B token 配方是「大模型 + 长预算」的玩法，小模型不适用。
3. 多轮能力缺口（dev-notes/31）与优化器/注意力无关，仍需数据侧或训练侧专项解决。
