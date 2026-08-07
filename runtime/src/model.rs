//! GPT 推理实现，对应 model.py 的前向逻辑。
//! 支持 base GPT-2 + 现代化开关（RMSNorm / RoPE / SwiGLU）。
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use rand::{Rng, RngExt};

use crate::attention::CausalSelfAttention;

/// 模型配置，从 model_config.json 读取。未声明的键由 serde 自动忽略。
/// 全部 V4 字段带 #[serde(default)]：旧 checkpoint 缺字段时自动回退默认值。
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
    // --- MoE（V3）---
    #[serde(default)]
    pub use_moe: bool,
    #[serde(default = "default_n_experts")]
    pub n_experts: usize,
    #[serde(default = "default_n_top_k")]
    pub n_top_k: usize,
    #[serde(default)]
    pub use_anticipatory_routing: bool,
    // 推理时不做 EMA 更新，ar_momentum 仅为反序列化完整性保留
    #[serde(default)]
    #[allow(dead_code)]
    pub ar_momentum: f64,
    // --- SwiGLU Clamp（V4）---
    #[serde(default)]
    pub swiglu_clamp: f64,
    // --- CSA / HCA（V4）---
    #[serde(default)]
    pub use_csa: bool,
    #[serde(default = "default_csa_compress")]
    pub csa_compress: usize,
    #[serde(default = "default_csa_topk")]
    pub csa_topk: usize,
    #[serde(default = "default_csa_window")]
    pub csa_window: usize,
    #[serde(default)]
    pub use_hca: bool,
    // --- 部分 RoPE（CSA/MLA 共用）---
    #[serde(default = "default_qk_rope_head_dim")]
    pub qk_rope_head_dim: usize,
    // --- mHC（V4）---
    #[serde(default)]
    pub use_mhc: bool,
}

fn default_rope_theta() -> f64 {
    10000.0
}
fn default_n_experts() -> usize {
    8
}
fn default_n_top_k() -> usize {
    2
}
fn default_csa_compress() -> usize {
    16
}
fn default_csa_topk() -> usize {
    4
}
fn default_csa_window() -> usize {
    64
}
fn default_qk_rope_head_dim() -> usize {
    16
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
            let attn = CausalSelfAttention::new(&vb, &format!("{prefix}.attn"), config)?;
            let ln2 = Norm::new(&vb, &format!("{prefix}.ln_2"), config.use_rmsnorm)?;
            // 前馈：MoE > SwiGLU > 标准 GELU MLP（优先级和 model.py 一致）
            let mlp = if config.use_moe {
                Ffn::MoE(MoE::new(&vb, &format!("{prefix}.mlp"), config)?)
            } else {
                Ffn::Mlp(Mlp::new(&vb, &format!("{prefix}.mlp"), config)?)
            };
            // mHC 的 2×2 混合矩阵 logits（经 Sinkhorn 投影成双随机矩阵）
            let mix = if config.use_mhc {
                Some(vb.get_unchecked(&format!("{prefix}.mix"))?)
            } else {
                None
            };
            blocks.push(Block { ln1, attn, ln2, mlp, mix });
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

        // mHC 的记忆流 z：None 时第一层 Block 内部会用 x 初始化
        let mut z: Option<Tensor> = None;
        for block in &self.blocks {
            if self.config.use_mhc {
                let (xn, zn) = block.forward_mhc(&h, z.as_ref())?;
                h = xn;
                z = Some(zn);
            } else {
                h = block.forward(&h)?;
            }
        }
        if self.config.use_mhc {
            // 最终用「记忆流」解码：它累积了所有层的块输出（和 model.py 一致）
            h = z.expect("use_mhc 但记忆流 z 为空");
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

/// 单个 transformer 块：attn + ff，残差连接（或 mHC 双流混合）。
struct Block {
    ln1: Norm,
    attn: CausalSelfAttention,
    ln2: Norm,
    mlp: Ffn,
    mix: Option<Tensor>, // mHC 的 2×2 混合矩阵 logits（经 Sinkhorn 投影）
}

/// 前馈网络：MoE（V3）或单一 MLP/SwiGLU。
enum Ffn {
    MoE(MoE),
    Mlp(Mlp),
}

impl Ffn {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Ffn::MoE(m) => m.forward(x),
            Ffn::Mlp(m) => m.forward(x),
        }
    }
}

impl Block {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.attn.forward(&self.ln1.forward(x)?)?;
        let x = x.add(&h)?;
        let h = self.mlp.forward(&self.ln2.forward(&x)?)?;
        Ok(x.add(&h)?)
    }

    /// mHC 双流前向：返回 (新工作流 x, 新记忆流 z)。
    /// z 为 None 时（第一块）用 x 初始化记忆流。
    fn forward_mhc(&self, x: &Tensor, z: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        let z = z.unwrap_or(x); // 第一块：记忆流从工作流初始化
        // 块输出 z_block = FFN(LN2(ATTN(LN1(x))))，两条流都从它混合
        let z_block = self.mlp.forward(&self.ln2.forward(&self.attn.forward(&self.ln1.forward(x)?)?)?)?;
        // 2×2 双随机矩阵：谱范数恒为 1，信号经过每个超连接最多不被放大
        let m: Vec<f32> = sinkhorn_knopp(self.mix.as_ref().expect("use_mhc 缺 mix 参数"), 10)?
            .flatten_all()?
            .to_vec1()?;
        let (m00, m01, m10, m11) = (m[0], m[1], m[2], m[3]);
        let x_new = x.affine(m00 as f64, 0.0)?.add(&z_block.affine(m01 as f64, 0.0)?)?;
        let r_new = z.affine(m10 as f64, 0.0)?.add(&z_block.affine(m11 as f64, 0.0)?)?;
        Ok((x_new, r_new))
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
/// 也作为 MoE 的单个专家复用（前缀不同，逻辑一致）。
struct Mlp {
    c_fc_w: Tensor,
    c_fc_b: Option<Tensor>,
    c_fc2_w: Option<Tensor>, // SwiGLU 的门控分支
    c_fc2_b: Option<Tensor>,
    c_proj_w: Tensor,
    c_proj_b: Option<Tensor>,
    use_swiglu: bool,
    swiglu_clamp: f64, // V4 钳制半宽；0 = 关
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
            swiglu_clamp: config.swiglu_clamp,
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
            let mut h = candle_nn::ops::silu(&a)?.mul(&g)?;
            // V4 钳制：在门控乘积之后、c_proj 之前钳（异常值就产生在门控乘积里）
            if self.swiglu_clamp > 0.0 {
                h = h.clamp(-self.swiglu_clamp, self.swiglu_clamp)?;
            }
            h
        } else {
            // GELU：PyTorch nn.GELU() 默认是 erf 近似，用 gelu_erf 保持一致
            a.gelu_erf()?
        };
        linear(&h, &self.c_proj_w, self.c_proj_b.as_ref())
    }
}

/// MoE（V3）：混合专家。每个 token 只路由到 top-k 个专家——参数量随专家数
/// 线性增长，但每个 token 的计算量不变。推理时不计算负载均衡辅助损失。
struct MoE {
    gate: Tensor,              // (n_experts, n_embd) 路由打分权重
    gate_slow: Option<Tensor>, // (n_experts, n_embd) 预判路由的 EMA 副本（buffer）
    experts: Vec<Mlp>,         // n_experts 个完整 FFN（SwiGLU 或 GELU）
    n_top_k: usize,
    use_anticipatory_routing: bool,
}

impl MoE {
    fn new(vb: &candle_nn::VarBuilder, prefix: &str, config: &Config) -> Result<Self> {
        let gate = vb.get_unchecked(&format!("{prefix}.router.weight"))?;
        // 预判路由的慢路由 buffer：推理时直接用（不再做 EMA 漂移——那是训练行为）
        let gate_slow = if config.use_anticipatory_routing {
            Some(vb.get_unchecked(&format!("{prefix}.router_slow"))?)
        } else {
            None
        };
        let mut experts = Vec::with_capacity(config.n_experts);
        for j in 0..config.n_experts {
            experts.push(Mlp::new(vb, &format!("{prefix}.experts.{j}"), config)?);
        }
        Ok(Self {
            gate,
            gate_slow,
            experts,
            n_top_k: config.n_top_k,
            use_anticipatory_routing: config.use_anticipatory_routing,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        let x_flat = x.reshape((b * t, c))?; // (N, C)，把 batch 和序列压平逐个 token 路由
        let n = b * t;
        let n_experts = self.gate.dim(0)?;

        // 路由打分：softmax 得到每个 token 在每个专家上的概率
        let gate_logits = linear(&x_flat, &self.gate, None)?; // (N, n_experts)
        let (top_k_probs, top_k_indices) = if self.use_anticipatory_routing {
            // 预判路由（V4）：离散选择用慢路由（旧参数），门控用当前路由
            let slow_logits =
                linear(&x_flat, self.gate_slow.as_ref().expect("预判路由缺 router_slow"), None)?;
            let (_, indices) = topk_last(&softmax_last(&slow_logits)?, self.n_top_k)?;
            let router_probs = softmax_last(&gate_logits)?;
            let probs = gather_last(&router_probs, &indices)?; // (N, k)
            let denom = probs.sum_keepdim(1)?.affine(1.0, 1e-6)?; // +1e-6 防除零
            (probs.broadcast_div(&denom)?, indices)
        } else {
            let router_probs = softmax_last(&gate_logits)?;
            let (vals, indices) = topk_last(&router_probs, self.n_top_k)?;
            let denom = vals.sum_keepdim(1)?;
            (vals.broadcast_div(&denom)?, indices)
        };

        // 逐个专家计算：收集路由到它的 token 过 FFN，按门控权重加回原处
        let idx: Vec<u32> = top_k_indices.flatten_all()?.to_vec1()?;
        let w: Vec<f32> = top_k_probs.flatten_all()?.to_vec1()?;
        let mut output = x_flat.zeros_like()?;
        for i in 0..n_experts {
            let mut token_ids: Vec<u32> = Vec::new();
            let mut weights: Vec<f32> = Vec::new();
            for row in 0..n {
                for slot in 0..self.n_top_k {
                    if idx[row * self.n_top_k + slot] == i as u32 {
                        token_ids.push(row as u32);
                        weights.push(w[row * self.n_top_k + slot]);
                    }
                }
            }
            if token_ids.is_empty() {
                continue;
            }
            let ids = Tensor::new(token_ids.as_slice(), x.device())?; // (num,)
            let expert_in = x_flat.index_select(&ids, 0)?; // (num, C)
            let expert_out = self.experts[i].forward(&expert_in)?; // (num, C)
            let w_t = Tensor::new(weights.as_slice(), x.device())?.unsqueeze(1)?; // (num, 1)
            let contrib = expert_out.broadcast_mul(&w_t)?;
            output = output.index_add(&ids, &contrib, 0)?;
        }
        Ok(output.reshape((b, t, c))?)
    }
}

/// Sinkhorn-Knopp：把 2×2 logits 投影成双随机矩阵（每行每列和都为 1、元素非负）。
/// 双随机矩阵的谱范数恒为 1——信号经过每个 mHC 超连接最多不被放大。
fn sinkhorn_knopp(logits: &Tensor, iters: usize) -> Result<Tensor> {
    // softplus(x) = log(1 + exp(x))：非负且处处可导
    let mut m = logits.exp()?.affine(1.0, 1.0)?.log()?;
    for _ in 0..iters {
        m = m.broadcast_div(&m.sum_keepdim(1)?)?; // 行归一：每行和为 1
        m = m.broadcast_div(&m.sum_keepdim(0)?)?; // 列归一：每列和为 1
    }
    Ok(m)
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

/// 沿最后一维做 softmax。
fn softmax_last(x: &Tensor) -> Result<Tensor> {
    Ok(candle_nn::ops::softmax(x, x.shape().rank() - 1)?)
}

/// 沿最后一维取 top-k，返回 (值, 下标)。candle 没有 topk，自己实现（张量都很小）。
pub(crate) fn topk_last(t: &Tensor, k: usize) -> Result<(Tensor, Tensor)> {
    let shape = t.shape();
    let n_last = shape.dims().last().copied().unwrap();
    let rows = shape.elem_count() / n_last;
    let data: Vec<f32> = t.flatten_all()?.to_vec1()?;
    let mut vals = vec![0f32; rows * k];
    let mut idxs = vec![0u32; rows * k];
    for r in 0..rows {
        let base = r * n_last;
        let mut pairs: Vec<(f32, usize)> = (0..n_last).map(|i| (data[base + i], i)).collect();
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap()); // 按值降序
        for j in 0..k {
            vals[r * k + j] = pairs[j].0;
            idxs[r * k + j] = pairs[j].1 as u32;
        }
    }
    let mut out_dims = shape.dims().to_vec();
    *out_dims.last_mut().unwrap() = k;
    let vals_t = Tensor::new(vals.as_slice(), t.device())?.reshape(out_dims.as_slice())?;
    let idxs_t = Tensor::new(idxs.as_slice(), t.device())?.reshape(out_dims.as_slice())?;
    Ok((vals_t, idxs_t))
}

/// 沿最后一维 gather：out[row, j] = t[row, idx[row, j]]，模拟 PyTorch 的 gather。
fn gather_last(t: &Tensor, indices: &Tensor) -> Result<Tensor> {
    let n_last = t.shape().dims().last().copied().unwrap();
    let k = indices.shape().dims().last().copied().unwrap();
    let data: Vec<f32> = t.flatten_all()?.to_vec1()?;
    let idx: Vec<u32> = indices.flatten_all()?.to_vec1()?;
    let mut out = vec![0f32; idx.len()];
    for (i, &ix) in idx.iter().enumerate() {
        let row = i / k;
        out[i] = data[row * n_last + ix as usize];
    }
    Ok(Tensor::new(out.as_slice(), t.device())?.reshape(indices.shape())?)
}

/// RMSNorm：x / sqrt(mean(x^2) + eps) * w
/// 用 x * (var+eps)^(-0.5) 而非 x / sqrt(...)，逐位对齐 PyTorch 的 rsqrt（避免舍入差异）。
fn rms_norm(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let x = x.to_dtype(DType::F32)?;
    let var = x.sqr()?.mean_keepdim(2)?; // (B, T, 1)
    // candle 没有标量加法，用 affine(1.0, eps) = x*1.0 + eps；powf(-0.5) = 1/sqrt
    let inv = var.affine(1.0, 1e-5)?.powf(-0.5)?;
    Ok(x.broadcast_mul(&inv)?.broadcast_mul(weight)?)
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
