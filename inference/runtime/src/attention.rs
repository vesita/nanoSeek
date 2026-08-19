//! 因果自注意力（对应 model.py 的 CausalSelfAttention）。
//! 两种模式：
//!   · 标准路径：合并 c_attn 投影 + 因果掩码（支持全头 RoPE）
//!   · CSA（压缩稀疏注意力）/ HCA（重度压缩注意力）：独立 Q/K/V 投影 +
//!     部分 RoPE + 块级 KV 压缩 + 稀疏 top-k 块选择 + 滑窗 + 全局信号
//! 小模型上 CPU 推理足够快。
use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use crate::model::{linear, topk_last, Config};

pub struct CausalSelfAttention {
    // 标准路径权重（use_csa 时不加载）
    c_attn_w: Option<Tensor>, // (3*n_embd, n_embd)
    c_attn_b: Option<Tensor>,
    // CSA 路径权重（use_csa 时加载，bias=False）
    q_proj_csa_w: Option<Tensor>, // (n_embd, n_embd)
    k_proj_csa_w: Option<Tensor>,
    v_proj_csa_w: Option<Tensor>,
    // 输出投影（两条路径共享）
    c_proj_w: Tensor,
    c_proj_b: Option<Tensor>,
    n_head: usize,
    n_embd: usize,
    head_dim: usize,
    rope_head_dim: usize, // 部分 RoPE 的旋转维数（CSA 时 = qk_rope_head_dim）
    cos: Option<Tensor>,  // (block_size, rope_head_dim)，use_rope 时才有
    sin: Option<Tensor>,
    // CSA 参数
    use_csa: bool,
    csa_compress: usize,
    csa_topk: usize,
    csa_window: usize,
    use_hca: bool,
    // V4 Attention Sinks：每头一个可学习标量偏置，softmax 末尾追加一列（value 为零）
    use_attn_sink: bool,
    attn_sink: Option<Tensor>, // (n_head,)
    // V4 QK-Norm：L2 归一化 q/k，q 乘每头可学习 scale（初始 sqrt(head_dim)）。
    // 开启后三处注意力 scale（块/滑窗/标准）都不再除 sqrt(d)——q 已自带 scale。
    use_qk_norm: bool,
    qk_scale: Option<Tensor>, // (n_head,)
    // V4 可学习门控池化：压缩 Linear + sigmoid gate 替代平均池化（K/V 共享权重）
    use_csa_learnable: bool,
    compress_w: Option<Tensor>, // (d, m*d)
    gate_w: Option<Tensor>,     // (d, m*d)
    // V4 Lightning Indexer（简版）：学习型块选择。分数只决定选哪些块，不进注意力。
    use_lightning_indexer: bool,
    idx_q_w: Option<Tensor>, // (n_head, n_embd)
    idx_k_w: Option<Tensor>, // (1, head_dim)
}

impl CausalSelfAttention {
    pub fn new(vb: &candle_nn::VarBuilder, prefix: &str, config: &Config) -> Result<Self> {
        let n_embd = config.n_embd;
        let n_head = config.n_head;
        let head_dim = n_embd / n_head;
        let rope_head_dim = if config.use_csa {
            config.qk_rope_head_dim
        } else {
            head_dim
        };

        // 标准路径或 CSA 路径的 Q/K/V 权重（二选一）
        let (c_attn_w, c_attn_b, q_proj_csa_w, k_proj_csa_w, v_proj_csa_w) = if config.use_csa {
            (
                None,
                None,
                Some(vb.get_unchecked(&format!("{prefix}.q_proj_csa.weight"))?),
                Some(vb.get_unchecked(&format!("{prefix}.k_proj_csa.weight"))?),
                Some(vb.get_unchecked(&format!("{prefix}.v_proj_csa.weight"))?),
            )
        } else {
            (
                Some(vb.get_unchecked(&format!("{prefix}.c_attn.weight"))?),
                vb.get_unchecked(&format!("{prefix}.c_attn.bias")).ok(),
                None,
                None,
                None,
            )
        };
        let c_proj_w = vb.get_unchecked(&format!("{prefix}.c_proj.weight"))?;
        let c_proj_b = vb.get_unchecked(&format!("{prefix}.c_proj.bias")).ok();

        let (cos, sin) = if config.use_rope {
            let (c, s) = precompute_rope_freqs(
                config.block_size,
                rope_head_dim,
                config.rope_theta,
                vb.device(),
            )?;
            (Some(c), Some(s))
        } else {
            (None, None)
        };

        // V4 Attention Sinks：每头一个可学习标量偏置（垃圾回收多余的注意力预算）
        let attn_sink = if config.use_attn_sink {
            Some(vb.get_unchecked(&format!("{prefix}.attn_sink"))?)
        } else {
            None
        };
        // V4 QK-Norm：每头一个可学习 scale（q = normalize(q) * scale，k = normalize(k)）
        let qk_scale = if config.use_qk_norm {
            Some(vb.get_unchecked(&format!("{prefix}.qk_scale"))?)
        } else {
            None
        };
        // V4 可学习门控池化：压缩块内 m 个 token 的 K/V（K/V 共享权重）
        let (compress_w, gate_w) = if config.use_csa && config.use_csa_learnable {
            (
                Some(vb.get_unchecked(&format!("{prefix}.csa_compress_linear.weight"))?),
                Some(vb.get_unchecked(&format!("{prefix}.csa_gate_linear.weight"))?),
            )
        } else {
            (None, None)
        };
        // V4 Lightning Indexer：学习型块选择（替代 raw top-k）
        let (idx_q_w, idx_k_w) = if config.use_csa && config.use_lightning_indexer {
            (
                Some(vb.get_unchecked(&format!("{prefix}.idx_q.weight"))?),
                Some(vb.get_unchecked(&format!("{prefix}.idx_k.weight"))?),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            c_attn_w,
            c_attn_b,
            q_proj_csa_w,
            k_proj_csa_w,
            v_proj_csa_w,
            c_proj_w,
            c_proj_b,
            n_head,
            n_embd,
            head_dim,
            rope_head_dim,
            cos,
            sin,
            use_csa: config.use_csa,
            csa_compress: config.csa_compress,
            csa_topk: config.csa_topk,
            csa_window: config.csa_window,
            use_hca: config.use_hca,
            use_attn_sink: config.use_attn_sink,
            attn_sink,
            use_qk_norm: config.use_qk_norm,
            qk_scale,
            use_csa_learnable: config.use_csa_learnable,
            compress_w,
            gate_w,
            use_lightning_indexer: config.use_lightning_indexer,
            idx_q_w,
            idx_k_w,
        })
    }

    /// QK-Norm（对应 model.py 的 _apply_qk_norm）：
    /// q = normalize(q, dim=-1) * qk_scale[head]，k = normalize(k, dim=-1)。
    /// q/k 形状 (B,T,nh,d)；qk_scale (nh,) 按 head 广播。RoPE 是范数保持的旋转，
    /// 前后顺序等价（Python 在 RoPE 前应用，这里同样在 RoPE 前应用）。
    fn apply_qk_norm(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let scale = self
            .qk_scale
            .as_ref()
            .expect("use_qk_norm 缺 qk_scale")
            .reshape((1, 1, self.n_head, 1))?; // (1,1,nh,1)
        let q_norm = normalize_last(q)?.broadcast_mul(&scale)?;
        let k_norm = normalize_last(k)?;
        Ok((q_norm, k_norm))
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if self.use_csa {
            let y = self.csa_forward(x)?;
            return linear(&y, &self.c_proj_w, self.c_proj_b.as_ref());
        }

        // —— 标准路径 ——
        let (b, t, _) = x.dims3()?;
        let device = x.device();

        // 一次性投影出 q/k/v：y = x @ W^T (+ b)
        let qkv = linear(
            x,
            self.c_attn_w.as_ref().expect("标准路径缺 c_attn"),
            self.c_attn_b.as_ref(),
        )?;
        let q = qkv.narrow(2, 0, self.n_embd)?;
        let k = qkv.narrow(2, self.n_embd, self.n_embd)?;
        let v = qkv.narrow(2, 2 * self.n_embd, self.n_embd)?;

        // 拆成多头：head 维保留在第 2 维，便于先做 RoPE 再转置
        let q = q.reshape((b, t, self.n_head, self.head_dim))?;
        let k = k.reshape((b, t, self.n_head, self.head_dim))?;
        let v = v.reshape((b, t, self.n_head, self.head_dim))?;

        // QK-Norm：L2 归一化 q/k，q 乘每头 scale（RoPE 范数保持，前后顺序等价）
        let (q, k) = if self.use_qk_norm {
            self.apply_qk_norm(&q, &k)?
        } else {
            (q, k)
        };

        // RoPE（全头旋转）：只对 q 和 k 旋转（v 不参与）
        let (q, k) = if let (Some(cos), Some(sin)) = (&self.cos, &self.sin) {
            let cos = cos.narrow(0, 0, t)?.unsqueeze(0)?.unsqueeze(2)?; // (1, T, 1, hd)
            let sin = sin.narrow(0, 0, t)?.unsqueeze(0)?.unsqueeze(2)?;
            let q = q.broadcast_mul(&cos)?.add(&rotate_half(&q)?.broadcast_mul(&sin)?)?;
            let k = k.broadcast_mul(&cos)?.add(&rotate_half(&k)?.broadcast_mul(&sin)?)?;
            (q, k)
        } else {
            (q, k)
        };

        // 把 head 维挪到第 1 维：(B, n_head, T, head_dim)
        let q = q.permute((0, 2, 1, 3))?;
        let k = k.permute((0, 2, 1, 3))?;
        let v = v.permute((0, 2, 1, 3))?;

        // 注意力分数 = q @ k^T / sqrt(head_dim)；QK-Norm 时 q 已自带 scale，不再除
        let scale = if self.use_qk_norm {
            1.0
        } else {
            1.0 / (self.head_dim as f64).sqrt()
        };
        let mut att = q.matmul(&k.transpose(2, 3)?)?.affine(scale, 0.0)?; // (B, nh, T, T)

        // 因果掩码：上三角（j > i）置 -inf
        let mask = causal_mask(t, device)?; // (T, T)
        att = att.broadcast_add(&mask.unsqueeze(0)?.unsqueeze(0)?)?;

        // V4 Attention Sink：softmax 前追加一列 sink[h]（对应零 value 行），
        // 模型借此把多余的注意力预算倒进"垃圾桶"。flash 不支持加列 → 手动注意力。
        let (att, v) = if self.use_attn_sink {
            let sink = self.attn_sink.as_ref().expect("use_attn_sink 缺 attn_sink");
            let sink_col = sink.reshape((1, self.n_head, 1, 1))?.broadcast_as((b, self.n_head, t, 1))?;
            let att = Tensor::cat(&[&att, &sink_col], 3)?; // (B, nh, T, T+1)
            let zero_row = Tensor::zeros((b, self.n_head, 1, self.head_dim), DType::F32, device)?;
            let v = Tensor::cat(&[&v, &zero_row], 2)?; // (B, nh, T+1, head_dim)
            (att, v)
        } else {
            (att, v)
        };

        let att = candle_nn::ops::softmax(&att, 3)?;

        let y = att.matmul(&v)?; // (B, nh, T, head_dim)
        let y = y.permute((0, 2, 1, 3))?.reshape((b, t, self.n_embd))?;

        // 输出投影
        linear(&y, &self.c_proj_w, self.c_proj_b.as_ref())
    }

    /// CSA + HCA 混合注意力（V4 简化教育版，逐位对齐 model.py 的 _csa_forward）。
    /// 注意开销从 O(T²) 降到 O(T·(nb + win))。
    fn csa_forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        let nh = self.n_head;
        let d = self.head_dim;
        let m = self.csa_compress;
        let win = self.csa_window.min(t);
        let device = x.device();

        // —— 1) Q/K/V 独立投影 + 部分 RoPE ——
        let q = linear(x, self.q_proj_csa_w.as_ref().expect("use_csa 缺 q_proj_csa"), None)?
            .reshape((b, t, nh, d))?;
        let k = linear(x, self.k_proj_csa_w.as_ref().expect("use_csa 缺 k_proj_csa"), None)?
            .reshape((b, t, nh, d))?;
        let v = linear(x, self.v_proj_csa_w.as_ref().expect("use_csa 缺 v_proj_csa"), None)?
            .reshape((b, t, nh, d))?;

        // QK-Norm：L2 归一化 q/k，q 乘每头 scale（在部分 RoPE 前，范数保持顺序等价）
        let (q, k) = if self.use_qk_norm {
            self.apply_qk_norm(&q, &k)?
        } else {
            (q, k)
        };

        // 部分 RoPE：只旋转前 rope_head_dim 维（与 MLA 一致），其余是"无位置"内容维
        let (q, k) = if let (Some(cos), Some(sin)) = (&self.cos, &self.sin) {
            let rope = self.rope_head_dim;
            let cos = cos.narrow(0, 0, t)?.unsqueeze(0)?.unsqueeze(2)?; // (1,T,1,rope)
            let sin = sin.narrow(0, 0, t)?.unsqueeze(0)?.unsqueeze(2)?;
            let q_rope = q.narrow(3, 0, rope)?;
            let q_nope = q.narrow(3, rope, d - rope)?;
            let q_rope = q_rope.broadcast_mul(&cos)?.add(&rotate_half(&q_rope)?.broadcast_mul(&sin)?)?;
            let q = Tensor::cat(&[&q_rope, &q_nope], 3)?;
            let k_rope = k.narrow(3, 0, rope)?;
            let k_nope = k.narrow(3, rope, d - rope)?;
            let k_rope = k_rope.broadcast_mul(&cos)?.add(&rotate_half(&k_rope)?.broadcast_mul(&sin)?)?;
            let k = Tensor::cat(&[&k_rope, &k_nope], 3)?;
            (q, k)
        } else {
            (q, k)
        };

        // —— 2) 块级压缩 + 稀疏块选择 + HCA ——
        // 短序列（块数 nb < topk）时 topk 会越界，用 k_eff = min(topk, nb) 兜底；
        // nb = 0（不足一个块）时块路径整个跳过，只靠滑窗。
        let t_ok = (t / m) * m;
        let nb = t_ok / m;
        // QK-Norm 时 q 已自带 qk_scale，不再除 sqrt(d)（对应 model.py 块/滑窗路径）
        let scale = if self.use_qk_norm {
            1.0
        } else {
            1.0 / (d as f64).sqrt()
        };
        let q_p = q.permute((0, 2, 1, 3))?; // (B,nh,T,d)，滑窗也要用

        let mut y_comp = Tensor::zeros((b, t, nh, d), DType::F32, device)?;
        if nb > 0 {
            // 块级压缩：平均池化（基线）或 V4 可学习门控池化（K/V 共享权重）
            let k_blocks = if self.use_csa_learnable {
                self.compress_block(&k.narrow(1, 0, t_ok)?, b, nb, nh, d, m)?
            } else {
                k.narrow(1, 0, t_ok)?.reshape((b, nb, m, nh, d))?.mean(2)?
            }; // (B,nb,nh,d)
            let v_blocks = if self.use_csa_learnable {
                self.compress_block(&v.narrow(1, 0, t_ok)?, b, nb, nh, d, m)?
            } else {
                v.narrow(1, 0, t_ok)?.reshape((b, nb, m, nh, d))?.mean(2)?
            };

            // 因果块掩码：query 只能看「它所在块之前」的块
            let (causal_block, has_prior) = csa_masks(t, m, device)?; // (T,nb), (T,)

            // 注意力分数 s_blk = einsum('bthd,bnhd->bthn', q, k_blocks) / sqrt(d)
            let kb_p = k_blocks.permute((0, 2, 1, 3))?; // (B,nh,nb,d)
            let s_blk = q_p.matmul(&kb_p.transpose(2, 3)?)?.affine(scale, 0.0)?; // (B,nh,T,nb)
            let s_blk = s_blk.permute((0, 2, 1, 3))?; // (B,T,nh,nb)

            // 因果 mask：不允许的块 → -inf（where_cond 的条件必须是 U8）
            let causal_inv = causal_block
                .affine(-1.0, 1.0)?
                .to_dtype(DType::U8)?
                .unsqueeze(0)?
                .unsqueeze(2)?; // (1,T,1,nb)
            let causal_inv = causal_inv.broadcast_as(s_blk.shape())?;
            let (neg_inf, zeros) = inf_zeros(s_blk.shape().dims().to_vec(), device)?;
            let s_blk = causal_inv.where_cond(&neg_inf, &s_blk)?;

            // 稀疏：只保留每个 query 得分最高的 k_eff 个块（其余 -inf，softmax 后为 0）。
            // use_lightning_indexer 时用学习型选择，否则 raw top-k。
            let k_eff = self.csa_topk.min(nb);
            let s_blk = if self.use_lightning_indexer {
                // --- V4 Lightning Indexer：学习型块选择（对应 model.py:266-298）---
                // idx_q 把 query 投影到「选择空间」(B,T,nh)，idx_k 给每块每头打标量分。
                // idx_scores = q_idx @ k_idx^T 收缩头维 → (B,T,nb)。分数只决定选哪几块，
                // 真实注意力分数 s_blk 仍用于注意力值（推理时不做 KL 桥接损失）。
                let q_idx = linear(x, self.idx_q_w.as_ref().expect("indexer 缺 idx_q"), None)?; // (B,T,nh)
                let k_idx = linear(
                    &k_blocks.reshape((b * nb * nh, d))?,
                    self.idx_k_w.as_ref().expect("indexer 缺 idx_k"),
                    None,
                )?
                .reshape((b, nb, nh))?; // (B,nb,nh)
                let mut idx_scores = q_idx.matmul(&k_idx.transpose(1, 2)?)?; // (B,T,nb)

                // 因果 mask：不允许的块 → -inf
                let causal_inv_idx = causal_block
                    .affine(-1.0, 1.0)?
                    .to_dtype(DType::U8)?
                    .unsqueeze(0)?
                    .broadcast_as(idx_scores.shape())?; // (1,T,nb)
                let (idx_inf, idx_zeros) = inf_zeros(idx_scores.shape().dims().to_vec(), device)?;
                idx_scores = causal_inv_idx.where_cond(&idx_inf, &idx_scores)?;
                // 无历史块的 query：整行置 0（不是 -inf，避免 topk 对全 -inf 产生 NaN）
                let has_prior_inv_idx = has_prior
                    .affine(-1.0, 1.0)?
                    .to_dtype(DType::U8)?
                    .unsqueeze(0)?
                    .unsqueeze(2)?
                    .broadcast_as(idx_scores.shape())?; // (1,T,1)
                idx_scores = has_prior_inv_idx.where_cond(&idx_zeros, &idx_scores)?;

                // topk → 阈值 → sel_mask = idx_scores >= 阈值（保留并列）
                let (idx_topk, _) = topk_last(&idx_scores, k_eff)?; // (B,T,k_eff)
                let thr = idx_topk
                    .narrow(idx_topk.shape().rank() - 1, k_eff - 1, 1)?
                    .broadcast_as(idx_scores.shape())?; // (B,T,nb)
                let sel = idx_scores.ge(&thr)?; // U8 (B,T,nb)
                let sel = sel.unsqueeze(2)?.broadcast_as(s_blk.shape())?; // (B,T,nh,nb)

                // 选中块保留真实注意力分数，未选中置 -inf（替代 raw top-k）
                let (sel_inf, _) = inf_zeros(s_blk.shape().dims().to_vec(), device)?;
                sel.where_cond(&s_blk, &sel_inf)?
            } else {
                // --- 基线：直接对真实注意力分数取 top-k（raw 稀疏）---
                let (topk_vals, _) = topk_last(&s_blk, k_eff)?; // (B,T,nh,k_eff)
                let thr = topk_vals
                    .narrow(topk_vals.shape().rank() - 1, k_eff - 1, 1)?
                    .broadcast_as(s_blk.shape())?; // 第 k_eff 大值广播成阈值
                let lt = s_blk.lt(&thr)?;
                lt.where_cond(&neg_inf, &s_blk)?
            };

            // 没有历史块的 query：整行置 0（否则 softmax 对全 -inf 产生 NaN）
            let has_prior_inv = has_prior
                .affine(-1.0, 1.0)?
                .to_dtype(DType::U8)?
                .unsqueeze(0)?
                .unsqueeze(2)?
                .unsqueeze(3)?
                .broadcast_as(s_blk.shape())?; // (1,T,1,1)
            let mut s_blk = has_prior_inv.where_cond(&zeros, &s_blk)?;

            // V4 Attention Sink：块注意力也追加一列 sink[h]（v_blocks 补零块占位）
            let mut v_blocks = v_blocks;
            if self.use_attn_sink {
                let sink = self.attn_sink.as_ref().expect("use_attn_sink 缺 attn_sink");
                let sink_col = sink.reshape((1, 1, self.n_head, 1))?.broadcast_as((b, t, self.n_head, 1))?;
                s_blk = Tensor::cat(&[&s_blk, &sink_col], 3)?; // (B,T,nh,nb+1)
                let zero_blk = Tensor::zeros((b, 1, self.n_head, d), DType::F32, device)?;
                v_blocks = Tensor::cat(&[&v_blocks, &zero_blk], 1)?; // (B,nb+1,nh,d)
            }

            let a_blk = candle_nn::ops::softmax(&s_blk, 3)?; // (B,T,nh,nb) 或 (B,T,nh,nb+1)
            // y_comp = einsum('bthn,bnhd->bthd', a_blk, v_blocks)
            let a_p = a_blk.permute((0, 2, 1, 3))?; // (B,nh,T,nb) 或 (B,nh,T,nb+1)
            let vb_p = v_blocks.permute((0, 2, 1, 3))?; // (B,nh,nb,d) 或 (B,nh,nb+1,d)
            y_comp = a_p.matmul(&vb_p)?.permute((0, 2, 1, 3))?; // (B,T,nh,d)
            let hp = has_prior.unsqueeze(0)?.unsqueeze(2)?.unsqueeze(3)?; // (1,T,1,1)
            y_comp = y_comp.broadcast_mul(&hp)?; // 无历史块的 query 贡献清零

            // —— HCA：重度压缩的全局信号（可选）——
            // 把所有允许的压缩块再平均成一个全局潜在，每个 query 加上它
            if self.use_hca {
                // sink 模式下 v_blocks 末尾多了一个占位零块，只用真实块
                let v_blocks_real = if self.use_attn_sink {
                    v_blocks.narrow(1, 0, nb)?
                } else {
                    v_blocks
                };
                let n_allowed = causal_block.sum_keepdim(1)?.clamp(1.0, f64::INFINITY)?.flatten_all()?; // (T,)
                let v_flat = v_blocks_real.reshape((b, nb, nh * d))?; // (B,nb,nh*d)
                // v_glob = einsum('tn,bnhd->bthd', causal_block, v_blocks) / n_allowed
                let v_glob = causal_block
                    .unsqueeze(0)?
                    .matmul(&v_flat)?
                    .reshape((b, t, nh, d))?; // (B,T,nh,d)
                let n_b = n_allowed.unsqueeze(0)?.unsqueeze(2)?.unsqueeze(3)?; // (1,T,1,1)
                let v_glob = v_glob.broadcast_div(&n_b)?;
                y_comp = y_comp.add(&v_glob)?;
            }
        }

        // —— 3) 滑窗：最近 win 个原始 token 的局部因果注意力 ——
        // 滑窗允许看自己（j ≤ i），保证每个位置至少有一个合法键，避免全 -inf
        let win_mask = window_mask(t, win, device)?; // (T,T)，-inf 表示不合法
        let k_p = k.permute((0, 2, 1, 3))?; // (B,nh,T,d)
        let mut s_win = q_p.matmul(&k_p.transpose(2, 3)?)?.affine(scale, 0.0)?; // (B,nh,T,T)
        let win_b = win_mask.unsqueeze(0)?.unsqueeze(0)?.broadcast_as(s_win.shape())?; // (1,1,T,T)
        let (win_inf, _) = inf_zeros(s_win.shape().dims().to_vec(), device)?;
        s_win = win_b.where_cond(&win_inf, &s_win)?;

        // V4 Attention Sink：滑窗也追加一列（v 补零行占位）
        let mut v_p = v.permute((0, 2, 1, 3))?; // (B,nh,T,d)
        if self.use_attn_sink {
            let sink = self.attn_sink.as_ref().expect("use_attn_sink 缺 attn_sink");
            let sink_col = sink.reshape((1, self.n_head, 1, 1))?.broadcast_as((b, self.n_head, t, 1))?;
            s_win = Tensor::cat(&[&s_win, &sink_col], 3)?; // (B,nh,T,T+1)
            let zero_row = Tensor::zeros((b, self.n_head, 1, d), DType::F32, device)?;
            v_p = Tensor::cat(&[&v_p, &zero_row], 2)?; // (B,nh,T+1,d)
        }

        let a_win = candle_nn::ops::softmax(&s_win, 3)?; // (B,nh,T,T) 或 (B,nh,T,T+1)
        let y_win = a_win.matmul(&v_p)?.permute((0, 2, 1, 3))?; // (B,T,nh,d)

        let y = y_comp.add(&y_win)?;

        // 合并 head：(B,T,nh,d) → (B,T,C)。
        // 逐位复刻 Python 的 y.transpose(1,2).contiguous().view(B,T,C)——
        // 当 y 是 (B,T,nh,d) 时这个 view 不是干净 reshape，而是置换：
        //   result[b][t][h*d+dd] = y[b][t'][h'][dd]，其中 h'*T + t' = t*nh + h。
        // 标准注意力路径的 y 是 (B,nh,T,d)，transpose 后 view 无置换；
        // 这是 CSA 独有的布局怪癖，模型权重按它训出来的，必须原样复刻。
        let y_flat: Vec<f32> = y.flatten_all()?.to_vec1()?;
        let mut out = vec![0f32; b * t * c];
        for bb in 0..b {
            for tt in 0..t {
                for hh in 0..nh {
                    // h' = V // T, t' = V % T，其中 V = tt*nh + hh
                    let v = tt * nh + hh;
                    let h_idx = v / t;
                    let t_idx = v % t;
                    for dd in 0..d {
                        let src = ((bb * t + t_idx) * nh + h_idx) * d + dd;
                        let dst = (bb * t + tt) * c + hh * d + dd;
                        out[dst] = y_flat[src];
                    }
                }
            }
        }
        let y = Tensor::new(out.as_slice(), device)?.reshape((b, t, c))?; // (B,T,C)
        Ok(y)
    }

    /// V4 可学习门控池化（对应 model.py 的 _compress_block）：把块内 m 个 token 的
    /// K/V 压成 1 个潜在。compress 线性压缩 × sigmoid 门控，K/V 共享同一组权重。
    /// x_block: (B, T_ok, nh, d)，T_ok = nb*m；返回 (B, nb, nh, d)。
    fn compress_block(&self, x_block: &Tensor, b: usize, nb: usize, nh: usize, d: usize, m: usize) -> Result<Tensor> {
        // (B, nb*m, nh, d) → (B, nb, m, nh, d) → (B, nb, nh, m, d) → 展平每块 (B*nb*nh, m*d)
        let flat = x_block
            .reshape((b, nb, m, nh, d))?
            .permute((0, 1, 3, 2, 4))?
            .reshape((b * nb * nh, m * d))?;
        let h = linear(&flat, self.compress_w.as_ref().expect("use_csa_learnable 缺 compress"), None)?; // (B*nb*nh, d)
        let gate = candle_nn::ops::sigmoid(&linear(
            &flat,
            self.gate_w.as_ref().expect("use_csa_learnable 缺 gate"),
            None,
        )?)?; // (B*nb*nh, d)
        Ok(h.mul(&gate)?.reshape((b, nb, nh, d))?)
    }
}

/// rotate_half：后半段搬到前半段并取负，和 model.py 完全一致。
fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let d = x.dim(3)?;
    let x1 = x.narrow(3, 0, d / 2)?;
    let x2 = x.narrow(3, d / 2, d - d / 2)?;
    Ok(Tensor::cat(&[&x2.neg()?, &x1], 3)?)
}

/// 沿最后一维做 L2 归一化（对应 torch.nn.functional.normalize 的默认 eps=1e-12）：
/// x / max(||x||₂, eps)。candle 无原生 normalize，手动算（张量都很小）。
fn normalize_last(x: &Tensor) -> Result<Tensor> {
    let eps = 1e-12f32;
    let norm = x.sqr()?.sum_keepdim(x.shape().rank() - 1)?.sqrt()?;
    let denom = norm.clamp(eps, f32::INFINITY)?;
    Ok(x.broadcast_div(&denom)?)
}

/// 预计算 RoPE 的 cos/sin 表：(block_size, rope_dim)。
/// rope_dim 可以是全 head_dim（标准路径）或 qk_rope_head_dim（CSA 部分 RoPE）。
fn precompute_rope_freqs(
    block_size: usize,
    rope_dim: usize,
    theta: f64,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    // 逐位对齐 model.py 的 precompute_rope_freqs：全程 f32 运算（PyTorch 里
    // arange.float() / head_dim 是 f32，theta ** exp 也是 f32 pow）。之前用 f64 算
    // 完再转 f32，会和 PyTorch 的 f32 结果差 1 ulp，被 MoE 路由放大成 logits 偏差。
    let theta32 = theta as f32;
    let mut freqs = Vec::with_capacity(block_size * rope_dim);
    for t in 0..block_size {
        // 每个二维平面一个频率：指数 = 2j/head_dim（对应 torch.arange(0, head_dim, 2)）
        for j in 0..rope_dim / 2 {
            let exp = (2.0 * j as f32) / rope_dim as f32;
            let inv = 1.0f32 / theta32.powf(exp);
            freqs.push(t as f32 * inv);
        }
        for j in 0..rope_dim / 2 {
            let exp = (2.0 * j as f32) / rope_dim as f32;
            let inv = 1.0f32 / theta32.powf(exp);
            freqs.push(t as f32 * inv);
        }
    }
    let freqs = Tensor::new(freqs.as_slice(), device)?.reshape((block_size, rope_dim))?;
    Ok((freqs.cos()?, freqs.sin()?))
}

/// 因果掩码：(T, T)，j > i 的位置为 -inf（query i 不能看 key j）。
fn causal_mask(t: usize, device: &Device) -> Result<Tensor> {
    // candle 的比较运算不做广播，先把两边都展开成 (T, T)
    let query = Tensor::arange(0u32, t as u32, device)?
        .unsqueeze(1)?
        .broadcast_as((t, t))?; // 每行都是 0..t
    let key = Tensor::arange(0u32, t as u32, device)?
        .unsqueeze(0)?
        .broadcast_as((t, t))?; // 每列都是 0..t
    let invalid = key.gt(&query)?; // [i][j] = (j > i)
    // 不能用 invalid * (-inf)：0 * (-inf) = NaN。改用 where 显式选择
    let neg_inf = Tensor::new(vec![f32::NEG_INFINITY; t * t], device)?.reshape((t, t))?;
    let zeros = Tensor::zeros((t, t), DType::F32, device)?;
    let mask = invalid.where_cond(&neg_inf, &zeros)?;
    Ok(mask)
}

/// 滑窗掩码：(T, T)，U8 类型，1 = 屏蔽（where_cond 的条件必须是 U8）。
/// 逐位对齐 model.py：win_causal[m][n] = (m <= n) & (n - m <= win)。
/// 注意：这是「向前看」的窗口（query m 允许看 key n ∈ [m, m+win]），
/// 与常规因果相反——但权重就是按这个语义训出来的，必须原样复刻。
fn window_mask(t: usize, win: usize, device: &Device) -> Result<Tensor> {
    let mut mask = vec![0u8; t * t];
    for i in 0..t {
        for j in 0..t {
            mask[i * t + j] = if i <= j && j - i <= win { 0 } else { 1 };
        }
    }
    Ok(Tensor::new(mask.as_slice(), device)?.reshape((t, t))?)
}

/// CSA 的块级掩码：
///   · causal_block (T, nb)：[i][n] = 1 表示 query i 允许看块 n（块 n < 所在块）
///   · has_prior (T,)：该 query 有没有历史块（用于 NaN 守卫）
fn csa_masks(t: usize, m: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let nb = t / m;
    let mut has_prior = vec![0f32; t];
    let mut causal = vec![0f32; t * nb];
    for i in 0..t {
        let bi = i / m; // 该 query 所在的块索引
        has_prior[i] = if bi > 0 { 1.0 } else { 0.0 };
        for n in 0..nb {
            causal[i * nb + n] = if bi > n { 1.0 } else { 0.0 };
        }
    }
    let causal_t = Tensor::new(causal.as_slice(), device)?.reshape((t, nb))?;
    let has_prior_t = Tensor::new(has_prior.as_slice(), device)?;
    Ok((causal_t, has_prior_t))
}

/// 生成指定形状的全 -inf 张量和全 0 张量（where_cond 的 on_true/on_false）。
fn inf_zeros(dims: Vec<usize>, device: &Device) -> Result<(Tensor, Tensor)> {
    let n: usize = dims.iter().product();
    let neg_inf = Tensor::new(vec![f32::NEG_INFINITY; n], device)?.reshape(dims.clone())?;
    let zeros = Tensor::zeros(dims, DType::F32, device)?;
    Ok((neg_inf, zeros))
}
