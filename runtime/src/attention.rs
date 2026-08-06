//! 因果自注意力（对应 model.py 的 CausalSelfAttention）。
//! 支持 RoPE（可选）；用因果掩码实现，小模型上 CPU 推理足够快。
use anyhow::Result;
use candle_core::{DType, Device, Tensor};

pub struct CausalSelfAttention {
    // 注意：PyTorch 的 Linear 是 y = x @ W^T + b，这里同样处理
    c_attn_w: Tensor, // (3*n_embd, n_embd)
    c_attn_b: Option<Tensor>,
    c_proj_w: Tensor, // (n_embd, n_embd)
    c_proj_b: Option<Tensor>,
    n_head: usize,
    n_embd: usize,
    head_dim: usize,
    cos: Option<Tensor>, // (block_size, head_dim)，use_rope 时才有
    sin: Option<Tensor>,
}

impl CausalSelfAttention {
    pub fn new(
        vb: &candle_nn::VarBuilder,
        prefix: &str,
        n_embd: usize,
        n_head: usize,
        block_size: usize,
        use_rope: bool,
        rope_theta: f64,
    ) -> Result<Self> {
        let head_dim = n_embd / n_head;
        let c_attn_w = vb.get_unchecked(&format!("{prefix}.c_attn.weight"))?;
        let c_attn_b = vb.get_unchecked(&format!("{prefix}.c_attn.bias")).ok();
        let c_proj_w = vb.get_unchecked(&format!("{prefix}.c_proj.weight"))?;
        let c_proj_b = vb.get_unchecked(&format!("{prefix}.c_proj.bias")).ok();
        let (cos, sin) = if use_rope {
            let (c, s) = precompute_rope_freqs(block_size, head_dim, rope_theta, vb.device())?;
            (Some(c), Some(s))
        } else {
            (None, None)
        };
        Ok(Self {
            c_attn_w,
            c_attn_b,
            c_proj_w,
            c_proj_b,
            n_head,
            n_embd,
            head_dim,
            cos,
            sin,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let device = x.device();

        // 一次性投影出 q/k/v：y = x @ W^T (+ b)
        let qkv = crate::model::linear(x, &self.c_attn_w, self.c_attn_b.as_ref())?;
        let q = qkv.narrow(2, 0, self.n_embd)?;
        let k = qkv.narrow(2, self.n_embd, self.n_embd)?;
        let v = qkv.narrow(2, 2 * self.n_embd, self.n_embd)?;

        // 拆成多头：head 维保留在第 2 维，便于后面先做 RoPE 再转置
        let q = q.reshape((b, t, self.n_head, self.head_dim))?;
        let k = k.reshape((b, t, self.n_head, self.head_dim))?;
        let v = v.reshape((b, t, self.n_head, self.head_dim))?;

        // RoPE：只对 q 和 k 旋转（v 不参与），这样 q·k 的点积携带相对位置
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

        // 注意力分数 = q @ k^T / sqrt(head_dim)
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let att = q.matmul(&k.transpose(2, 3)?)?.affine(scale, 0.0)?; // (B, nh, T, T)

        // 因果掩码：上三角（j > i）置 -inf
        let mask = causal_mask(t, device)?; // (T, T)
        let att = att.broadcast_add(&mask.unsqueeze(0)?.unsqueeze(0)?)?;
        let att = candle_nn::ops::softmax(&att, 3)?;

        let y = att.matmul(&v)?; // (B, nh, T, head_dim)
        let y = y.permute((0, 2, 1, 3))?.reshape((b, t, self.n_embd))?;

        // 输出投影
        crate::model::linear(&y, &self.c_proj_w, self.c_proj_b.as_ref())
    }
}

/// rotate_half：后半段搬到前半段并取负，和 model.py 完全一致。
fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let d = x.dim(3)?;
    let x1 = x.narrow(3, 0, d / 2)?;
    let x2 = x.narrow(3, d / 2, d - d / 2)?;
    Ok(Tensor::cat(&[&x2.neg()?, &x1], 3)?)
}

/// 预计算 RoPE 的 cos/sin 表：(block_size, head_dim)。
fn precompute_rope_freqs(
    block_size: usize,
    head_dim: usize,
    theta: f64,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let mut freqs = Vec::with_capacity(block_size * head_dim);
    for t in 0..block_size {
        for j in 0..head_dim / 2 {
            let inv = 1.0 / theta.powf(j as f64 / head_dim as f64);
            freqs.push(t as f64 * inv);
        }
        for j in 0..head_dim / 2 {
            let inv = 1.0 / theta.powf(j as f64 / head_dim as f64);
            freqs.push(t as f64 * inv);
        }
    }
    let freqs = Tensor::new(freqs.as_slice(), device)?.reshape((block_size, head_dim))?;
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
