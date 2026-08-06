//! GPT 推理实现，对应 model.py 的前向逻辑。
//! 支持 base GPT-2 + 现代化开关（RMSNorm / RoPE / SwiGLU）。
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use rand::{Rng, RngExt};

use crate::attention::CausalSelfAttention;

/// 模型配置，从 model_config.json 读取。未声明的键由 serde 自动忽略。
#[derive(serde::Deserialize, Debug, Clone)]
pub struct Config {
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
    pub vocab_size: usize,
    pub block_size: usize,
    #[serde(default)]
    pub use_rmsnorm: bool,
    #[serde(default)]
    pub use_rope: bool,
    #[serde(default)]
    pub use_swiglu: bool,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
}

fn default_rope_theta() -> f64 {
    10000.0
}

impl Config {
    pub fn load(path: &str) -> Result<Self> {
        Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
    }
}

/// GPT 模型：权重全部以原始 Tensor 存储，lm_head 与 wte 共享权重（weight tying）。
pub struct GPT {
    config: Config,
    device: Device,
    wte: Tensor,          // (vocab_size, n_embd)
    wpe: Option<Tensor>,  // (block_size, n_embd)，use_rope 时无
    blocks: Vec<Block>,
    ln_f: Norm,
}

impl GPT {
    /// 从 safetensors + config 加载权重。
    pub fn load(model_path: &str, config: &Config, device: &Device) -> Result<Self> {
        let tensors = candle_core::safetensors::load(model_path, device)?;
        let vb = candle_nn::VarBuilder::from_tensors(tensors, DType::F32, device);

        let wte = vb.get_unchecked("transformer.wte.weight")?;
        let wpe = if config.use_rope {
            None
        } else {
            Some(vb.get_unchecked("transformer.wpe.weight")?)
        };

        let mut blocks = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            let prefix = format!("transformer.h.{i}");
            let ln1 = Norm::new(&vb, &format!("{prefix}.ln_1"), config.use_rmsnorm)?;
            let attn = CausalSelfAttention::new(
                &vb,
                &format!("{prefix}.attn"),
                config.n_embd,
                config.n_head,
                config.block_size,
                config.use_rope,
                config.rope_theta,
            )?;
            let ln2 = Norm::new(&vb, &format!("{prefix}.ln_2"), config.use_rmsnorm)?;
            let mlp = Mlp::new(&vb, &format!("{prefix}.mlp"), config)?;
            blocks.push(Block { ln1, attn, ln2, mlp });
        }
        let ln_f = Norm::new(&vb, "transformer.ln_f", config.use_rmsnorm)?;

        Ok(Self {
            config: config.clone(),
            device: device.clone(),
            wte,
            wpe,
            blocks,
            ln_f,
        })
    }

    /// 前向：token id 序列 → 最后一个位置在所有词表上的 logits（形状 (vocab,)）。
    pub fn forward(&self, tokens: &[u32]) -> Result<Tensor> {
        let device = &self.device;
        let t = tokens.len();
        // candle 的 embedding 要求 1-D 索引，先取 (T,) 再补 batch 维
        let idx = Tensor::new(tokens, device)?; // (T,)

        let mut h = self.wte.embedding(&idx)?.unsqueeze(0)?; // (T, n_embd) → (1, T, n_embd)
        if let Some(wpe) = &self.wpe {
            let pos = Tensor::arange(0u32, t as u32, device)?; // (T,)
            let pos_emb = wpe.embedding(&pos)?.unsqueeze(0)?; // (1, T, n_embd)
            h = h.add(&pos_emb)?;
        }

        for block in &self.blocks {
            h = block.forward(&h)?;
        }
        let h = self.ln_f.forward(&h)?;

        // lm_head 与 wte 共享权重：logits = h @ wte^T
        let logits = linear(&h, &self.wte, None)?; // (1, T, vocab)
        let last = logits.narrow(1, t - 1, 1)?.flatten_all()?; // (vocab,)
        Ok(last)
    }

    /// 续写：从 prompt 生成 n 个新 token，返回新生成的 token（不含 prompt）。
    pub fn generate<R: Rng>(
        &self,
        prompt: &[u32],
        n: usize,
        temperature: f64,
        top_k: Option<usize>,
        rng: &mut R,
    ) -> Result<Vec<u32>> {
        let mut new_tokens = Vec::with_capacity(n);
        let mut tokens = prompt.to_vec();
        for _ in 0..n {
            // 上下文过长时按 block_size 裁剪
            let ctx = if tokens.len() > self.config.block_size {
                &tokens[tokens.len() - self.config.block_size..]
            } else {
                &tokens[..]
            };
            let logits = self.forward(ctx)?; // (vocab,)
            let next = sample(&logits, temperature, top_k, rng)?;
            tokens.push(next);
            new_tokens.push(next);
        }
        Ok(new_tokens)
    }
}

/// 单个 transformer 块：attn + mlp，残差连接。
struct Block {
    ln1: Norm,
    attn: CausalSelfAttention,
    ln2: Norm,
    mlp: Mlp,
}

impl Block {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.attn.forward(&self.ln1.forward(x)?)?;
        let x = x.add(&h)?;
        let h = self.mlp.forward(&self.ln2.forward(&x)?)?;
        Ok(x.add(&h)?)
    }
}

/// 归一化：RMSNorm（modern）或 LayerNorm（原始 GPT-2）。
struct Norm {
    weight: Tensor,
    bias: Option<Tensor>,
    use_rmsnorm: bool,
}

impl Norm {
    fn new(vb: &candle_nn::VarBuilder, prefix: &str, use_rmsnorm: bool) -> Result<Self> {
        let weight = vb.get_unchecked(&format!("{prefix}.weight"))?;
        // RMSNorm 没有 bias 参数
        let bias = if use_rmsnorm {
            None
        } else {
            vb.get_unchecked(&format!("{prefix}.bias")).ok()
        };
        Ok(Self {
            weight,
            bias,
            use_rmsnorm,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if self.use_rmsnorm {
            rms_norm(x, &self.weight)
        } else {
            layer_norm(x, &self.weight, self.bias.as_ref())
        }
    }
}

/// 前馈网络：SwiGLU（modern）或标准 GELU MLP。
struct Mlp {
    c_fc_w: Tensor,
    c_fc_b: Option<Tensor>,
    c_fc2_w: Option<Tensor>, // SwiGLU 的门控分支
    c_fc2_b: Option<Tensor>,
    c_proj_w: Tensor,
    c_proj_b: Option<Tensor>,
    use_swiglu: bool,
}

impl Mlp {
    fn new(vb: &candle_nn::VarBuilder, prefix: &str, config: &Config) -> Result<Self> {
        let c_fc_w = vb.get_unchecked(&format!("{prefix}.c_fc.weight"))?;
        let c_fc_b = vb.get_unchecked(&format!("{prefix}.c_fc.bias")).ok();
        let (c_fc2_w, c_fc2_b) = if config.use_swiglu {
            (
                Some(vb.get_unchecked(&format!("{prefix}.c_fc2.weight"))?),
                vb.get_unchecked(&format!("{prefix}.c_fc2.bias")).ok(),
            )
        } else {
            (None, None)
        };
        let c_proj_w = vb.get_unchecked(&format!("{prefix}.c_proj.weight"))?;
        let c_proj_b = vb.get_unchecked(&format!("{prefix}.c_proj.bias")).ok();
        Ok(Self {
            c_fc_w,
            c_fc_b,
            c_fc2_w,
            c_fc2_b,
            c_proj_w,
            c_proj_b,
            use_swiglu: config.use_swiglu,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let a = linear(x, &self.c_fc_w, self.c_fc_b.as_ref())?;
        let h = if self.use_swiglu {
            // SwiGLU：silu(值分支) * 门控分支
            let g = linear(
                x,
                self.c_fc2_w.as_ref().expect("use_swiglu 但缺 c_fc2 权重"),
                self.c_fc2_b.as_ref(),
            )?;
            candle_nn::ops::silu(&a)?.mul(&g)?
        } else {
            // GELU：PyTorch nn.GELU() 默认是 erf 近似，用 gelu_erf 保持一致
            a.gelu_erf()?
        };
        linear(&h, &self.c_proj_w, self.c_proj_b.as_ref())
    }
}

// -----------------------------------------------------------------------------
// 基础算子
// -----------------------------------------------------------------------------

/// y = x @ W^T + b（和 PyTorch Linear 一致）。
/// candle 的 matmul 不支持「高维 @ 低维」的广播，这里先把所有前导维拍平成 2 维再算。
pub(crate) fn linear(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
    let shape = x.shape();
    let rank = shape.rank();
    let n_last = shape.dims().last().copied().unwrap();
    let n_rest: usize = shape.dims()[..rank - 1].iter().product();
    let mut y = x.reshape((n_rest, n_last))?.matmul(&w.t()?)?; // (n_rest, out)
    if let Some(b) = b {
        y = y.broadcast_add(b)?;
    }
    let out = w.dim(0)?;
    let mut dims = shape.dims().to_vec();
    dims[rank - 1] = out;
    Ok(y.reshape(dims)?)
}

/// RMSNorm：x / sqrt(mean(x^2) + eps) * w
fn rms_norm(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let x = x.to_dtype(DType::F32)?;
    let var = x.sqr()?.mean_keepdim(2)?; // (B, T, 1)
    // candle 没有标量加法，用 affine(1.0, eps) = x*1.0 + eps
    let x = x.broadcast_div(&(var.affine(1.0, 1e-5)?.sqrt()?))?;
    Ok(x.broadcast_mul(weight)?)
}

/// LayerNorm：(x - mean) / sqrt(var + eps) * w + b
fn layer_norm(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let x = x.to_dtype(DType::F32)?;
    // candle 的 sub 不做广播，先把 mean 展开成和 x 一样的形状
    let mean = x.mean_keepdim(2)?.broadcast_as(x.shape())?; // (B, T, n_embd)
    let centered = x.sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(2)?; // (B, T, 1)
    let x = centered.broadcast_div(&(var.affine(1.0, 1e-5)?.sqrt()?))?;
    let x = x.broadcast_mul(weight)?;
    if let Some(b) = bias {
        return Ok(x.broadcast_add(b)?);
    }
    Ok(x)
}

/// 温度 + top-k + softmax + 多项式采样，和 model.py 的 generate 一致。
fn sample<R: Rng>(
    logits: &Tensor,
    temperature: f64,
    top_k: Option<usize>,
    rng: &mut R,
) -> Result<u32> {
    let mut logits: Vec<f32> = logits.to_vec1()?;

    // 温度缩放
    for l in logits.iter_mut() {
        *l /= temperature as f32;
    }

    // top-k：只保留概率最高的前 k 个 token
    if let Some(k) = top_k {
        let k = k.min(logits.len());
        if k > 0 {
            let mut sorted = logits.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let threshold = sorted[k - 1]; // 第 k 大的值
            for l in logits.iter_mut() {
                if *l < threshold {
                    *l = f32::NEG_INFINITY;
                }
            }
        }
    }

    // softmax（数值稳定：减最大值）
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f64> = logits.iter().map(|l| ((*l as f64) - max as f64).exp()).collect();
    let sum: f64 = exps.iter().sum();
    let probs: Vec<f64> = exps.iter().map(|e| e / sum).collect();

    // multinomial 采样：均匀随机数落在累计概率区间里
    let u: f64 = rng.random_range(0.0..1.0);
    let mut cum = 0.0;
    for (i, p) in probs.iter().enumerate() {
        cum += p;
        if u < cum {
            return Ok(i as u32);
        }
    }
    Ok((logits.len() - 1) as u32)
}
