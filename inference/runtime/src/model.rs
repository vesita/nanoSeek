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
    // --- V4 结构设计升级（默认全关，缺字段的旧配置自动回退）---
    #[serde(default = "default_hc_mult")]
    pub hc_mult: usize,              // mHC 残差流数（Python GPTConfig 默认 4）
    #[serde(default)]
    pub use_attn_sink: bool,         // Attention Sinks：每头可学习标量偏置
    #[serde(default)]
    pub use_lightning_indexer: bool, // 学习型块选择替代 CSA raw top-k
    #[serde(default)]
    pub num_hash_layers: usize,      // 前 N 层用 hash 路由（0=禁用）
    #[serde(default)]
    pub use_shared_expert: bool,     // MoE 始终激活的共享专家
    #[serde(default)]
    pub use_csa_learnable: bool,     // 门控池化替代平均池化（旧配置缺字段按 false=平均池化最安全）
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
fn default_hc_mult() -> usize {
    4
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
            // Hash 路由：前 num_hash_layers 层的 MoE 用确定性 hash 分配（对应 model.py 的 layer_idx < num_hash_layers）
            let mlp = if config.use_moe {
                Ffn::MoE(MoE::new(&vb, &format!("{prefix}.mlp"), config, i < config.num_hash_layers)?)
            } else {
                Ffn::Mlp(Mlp::new(&vb, &format!("{prefix}.mlp"), config)?)
            };
            // mHC 4-copy：6 个 raw 参数（attn 子层 + FFN 子层的 A/B/C），use_mhc 时加载
            let (raw_a_attn, raw_b_attn, raw_c_attn) = if config.use_mhc {
                (
                    Some(vb.get_unchecked(&format!("{prefix}.raw_A_attn"))?),
                    Some(vb.get_unchecked(&format!("{prefix}.raw_B_attn"))?),
                    Some(vb.get_unchecked(&format!("{prefix}.raw_C_attn"))?),
                )
            } else {
                (None, None, None)
            };
            let (raw_a_ffn, raw_b_ffn, raw_c_ffn) = if config.use_mhc {
                (
                    Some(vb.get_unchecked(&format!("{prefix}.raw_A_ffn"))?),
                    Some(vb.get_unchecked(&format!("{prefix}.raw_B_ffn"))?),
                    Some(vb.get_unchecked(&format!("{prefix}.raw_C_ffn"))?),
                )
            } else {
                (None, None, None)
            };
            blocks.push(Block {
                ln1,
                attn,
                ln2,
                mlp,
                use_mhc: config.use_mhc,
                hc_mult: config.hc_mult,
                raw_A_attn: raw_a_attn,
                raw_B_attn: raw_b_attn,
                raw_C_attn: raw_c_attn,
                raw_A_ffn: raw_a_ffn,
                raw_B_ffn: raw_b_ffn,
                raw_C_ffn: raw_c_ffn,
            });
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

        // mHC：4 个残差流从同一嵌入出发（在流维广播扩展），层后取均值回到 1 流
        if self.config.use_mhc {
            h = h.unsqueeze(2)?.broadcast_as((1, t, self.config.hc_mult, self.config.n_embd))?;
        }
        for block in &self.blocks {
            h = block.forward(&h)?; // Block::forward 内部按 use_mhc 分派
        }
        if self.config.use_mhc {
            h = h.mean(2)?; // (1, T, hc, d) → (1, T, d)
        }
        let h = self.ln_f.forward(&h)?;

        // lm_head 与 wte 共享权重：logits = h @ wte^T
        let logits = linear(&h, &self.wte, None)?; // (1, T, vocab)
        let last = logits.narrow(1, t - 1, 1)?.flatten_all()?; // (vocab,)
        Ok(last)
    }

    /// 续写（流式）：从 prompt 生成 n 个新 token，每生成一个调用 on_token，
    /// 返回新生成的 token（不含 prompt）。
    pub fn generate_stream<R: Rng, F: FnMut(u32)>(
        &self,
        prompt: &[u32],
        n: usize,
        temperature: f64,
        top_k: Option<usize>,
        repeat_penalty: f64,
        eos_id: Option<u32>,
        rng: &mut R,
        mut on_token: F,
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
            // 把当前上下文作为"已见 token"传给采样：模型越倾向于复述结构标签，压得越狠
            let next = sample(&logits, temperature, top_k, repeat_penalty, ctx, rng)?;
            // 遇到 <eos> 立即停止：不输出、也不拼进上下文，否则会继续生成垃圾
            if Some(next) == eos_id {
                break;
            }
            on_token(next);
            tokens.push(next);
            new_tokens.push(next);
        }
        Ok(new_tokens)
    }

    /// 续写（一次性）：生成 n 个新 token 后一起返回（流式的薄包装）。
    /// main.rs 现在走 generate_stream，这里作为非流式 API 保留。
    #[allow(dead_code)]
    pub fn generate<R: Rng>(
        &self,
        prompt: &[u32],
        n: usize,
        temperature: f64,
        top_k: Option<usize>,
        repeat_penalty: f64,
        eos_id: Option<u32>,
        rng: &mut R,
    ) -> Result<Vec<u32>> {
        self.generate_stream(prompt, n, temperature, top_k, repeat_penalty, eos_id, rng, |_| {})
    }
}

/// 单个 transformer 块：attn + ff，残差连接（或 mHC 4 流混合）。
#[allow(non_snake_case)] // raw_A_attn 等字段名与 Python 权重名一致，方便 vb 加载
struct Block {
    ln1: Norm,
    attn: CausalSelfAttention,
    ln2: Norm,
    mlp: Ffn,
    // mHC 4-copy（V4）：X' = B·X + C·F(A·X)。attn 和 FFN 子层各一组 A/B/C。
    use_mhc: bool,
    hc_mult: usize,
    raw_A_attn: Option<Tensor>, // (1, hc) → sigmoid，把 4 流压成 1 流给子层
    raw_B_attn: Option<Tensor>, // (hc, hc) → softplus → Sinkhorn（双随机混合）
    raw_C_attn: Option<Tensor>, // (hc, 1) → sigmoid，把子层输出展开回 4 流
    raw_A_ffn: Option<Tensor>,
    raw_B_ffn: Option<Tensor>,
    raw_C_ffn: Option<Tensor>,
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
        if self.use_mhc {
            return self.forward_mhc(x);
        }
        let h = self.attn.forward(&self.ln1.forward(x)?)?;
        let x = x.add(&h)?;
        let h = self.mlp.forward(&self.ln2.forward(&x)?)?;
        Ok(x.add(&h)?)
    }

    /// mHC 4-copy 前向：X' = B·X + C·F(A·X)，逐位对齐 model.py 的 Block._mhc_forward。
    /// x: (B, T, hc, d) 4 个并行残差流；返回同样形状的新 4 流。
    /// A（1×hc，sigmoid）把 4 流压成 1 流给子层 F；子层只算 1 次；
    /// C（hc×1，sigmoid）把子层输出展开回 4 流；B（hc×hc，双随机）混合残差流。
    fn forward_mhc(&self, x: &Tensor) -> Result<Tensor> {
        let (_, _, hc, _) = x.dims4()?;
        debug_assert_eq!(hc, self.hc_mult);

        // --- 子层 1：注意力 ---
        let x = self.mhc_sublayer(x, hc, true)?;

        // --- 子层 2：FFN（MoE 或 SwiGLU）---
        self.mhc_sublayer(&x, hc, false)
    }

    /// 单个 mHC 子层：A 压缩 → F 计算 → C 展开 → B 混合残差。
    fn mhc_sublayer(&self, x: &Tensor, hc: usize, is_attn: bool) -> Result<Tensor> {
        // A = sigmoid(raw_A)：(1, hc)
        let raw_a = if is_attn { &self.raw_A_attn } else { &self.raw_A_ffn };
        let a = candle_nn::ops::sigmoid(raw_a.as_ref().expect("use_mHC 缺 raw_A"))?;
        // h_in = Σ_h x[...,h,:] * A[h]：(B, T, d)
        let h_in = x.broadcast_mul(&a.reshape((1, 1, hc, 1))?)?.sum(2)?;

        // 子层 F 只跑 1 次
        let h_out = if is_attn {
            self.attn.forward(&self.ln1.forward(&h_in)?)?
        } else {
            self.mlp.forward(&self.ln2.forward(&h_in)?)?
        };

        // C = sigmoid(raw_C)：(hc, 1)；delta = h_out.unsqueeze(2) * C：(B, T, hc, d)
        let raw_c = if is_attn { &self.raw_C_attn } else { &self.raw_C_ffn };
        let c = candle_nn::ops::sigmoid(raw_c.as_ref().expect("use_mhc 缺 raw_C"))?;
        let delta = h_out.unsqueeze(2)?.broadcast_mul(&c.reshape((1, 1, hc, 1))?)?;

        // B = sinkhorn(softplus(raw_B))：(hc, hc)，双随机
        let raw_b = if is_attn { &self.raw_B_attn } else { &self.raw_B_ffn };
        let b_ds = sinkhorn_knopp(raw_b.as_ref().expect("use_mhc 缺 raw_B"), 20)?;

        // x' = einsum('bthd,hc->btcd', x, B) + delta
        //     = (x 的 h/d 轴对调) @ B 再对调回来，等价于对 h 维做矩阵乘。
        // candle 的 matmul 不支持 4D @ 2D，把 (B,T,d,hc) 拍平成 2D 乘完再还原。
        let (b, t, hc, d) = x.dims4()?;
        let x_pt = x.permute((0, 1, 3, 2))?; // (B,T,d,hc)
        let mixed = x_pt
            .reshape((b * t * d, hc))?
            .matmul(&b_ds)?
            .reshape((b, t, d, hc))?
            .permute((0, 1, 3, 2))?; // (B,T,hc,d)
        Ok(mixed.add(&delta)?)
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
/// V4：浅层可用 hash 确定性路由（use_hash），深层用学习路由；可叠加共享专家。
struct MoE {
    gate: Tensor,              // (n_experts, n_embd) 路由打分权重
    gate_slow: Option<Tensor>, // (n_experts, n_embd) 预判路由的 EMA 副本（buffer）
    experts: Vec<Mlp>,         // n_experts 个完整 FFN（SwiGLU 或 GELU）
    shared_expert: Option<Mlp>, // V4：始终激活的共享专家（捕获共性特征）
    n_top_k: usize,
    use_anticipatory_routing: bool,
    use_hash: bool, // V4：浅层用 hash(token 第一维) 确定性分配，不学习
}

impl MoE {
    fn new(vb: &candle_nn::VarBuilder, prefix: &str, config: &Config, use_hash: bool) -> Result<Self> {
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
        // V4 共享专家：始终激活，捕获所有 token 的共性特征（语法、常见搭配）
        let shared_expert = if config.use_shared_expert {
            Some(Mlp::new(vb, &format!("{prefix}.shared_expert"), config)?)
        } else {
            None
        };
        Ok(Self {
            gate,
            gate_slow,
            experts,
            shared_expert,
            n_top_k: config.n_top_k,
            use_anticipatory_routing: config.use_anticipatory_routing,
            use_hash,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        let x_flat = x.reshape((b * t, c))?; // (N, C)，把 batch 和序列压平逐个 token 路由
        let n = b * t;
        let n_experts = self.gate.dim(0)?;

        // 路由：hash 确定性分配（浅层）或学习打分（深层）
        let (top_k_probs, top_k_indices) = if self.use_hash {
            // --- V4 Hash 路由：确定性分配，不学习（对应 model.py:438-450）---
            // Knuth 乘法哈希：用 token 嵌入的第一维作签名，均匀映射到各专家。
            // 用 k 个不同的乘子生成 k 个专家槽位。
            let col0: Vec<f32> = x_flat.narrow(1, 0, 1)?.flatten_all()?.to_vec1()?;
            let mut indices = vec![0u32; n * self.n_top_k];
            for i in 0..self.n_top_k {
                let prime: i64 = 2654435761 + 97 * i as i64;
                for row in 0..n {
                    // .long() 截断向零 → Rust `as i64`；rem_euclid 保证和 Python % 一样恒非负
                    let v = col0[row] as i64;
                    let h = v.wrapping_mul(prime) >> 16;
                    indices[row * self.n_top_k + i] = h.rem_euclid(n_experts as i64) as u32;
                }
            }
            let indices_t = Tensor::new(indices.as_slice(), x.device())?
                .reshape((n, self.n_top_k))?;
            let probs_t = Tensor::new(vec![1.0 / self.n_top_k as f32; n * self.n_top_k], x.device())?
                .reshape((n, self.n_top_k))?;
            (probs_t, indices_t)
        } else {
            // 路由打分：softmax 得到每个 token 在每个专家上的概率
            let gate_logits = linear(&x_flat, &self.gate, None)?; // (N, n_experts)
            if self.use_anticipatory_routing {
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
            }
        };

        // 逐个专家计算：收集路由到它的 token 过 FFN，按门控权重加回原处
        let idx: Vec<u32> = top_k_indices.flatten_all()?.to_vec1()?;
        let w: Vec<f32> = top_k_probs.flatten_all()?.to_vec1()?;
        // V4 共享专家：所有 token 都先过一遍（捕获共性），再叠加路由专家的差异化输出
        let mut output = if let Some(shared) = &self.shared_expert {
            shared.forward(&x_flat)?
        } else {
            x_flat.zeros_like()?
        };
        for i in 0..n_experts {
            // 收集路由到该专家的 (row, weight)。PyTorch 的 `output[token_ids] += contrib`
            // 是 fancy-index +=，对重复 row（同一 token 被 hash 选进同一专家两次）会「后写覆盖
            // 先写」——只保留每个 row 最后一个 slot 的贡献，而不是累加两次。这里显式模拟：
            // 用 row → weight 的覆盖映射，重复 row 时后者覆盖前者。
            let mut per_row: Vec<(u32, f32)> = Vec::new();
            for row in 0..n {
                for slot in 0..self.n_top_k {
                    if idx[row * self.n_top_k + slot] == i as u32 {
                        // 直接更新（覆盖）：重复 row 只保留最后一次的权重
                        if let Some(e) = per_row.iter_mut().find(|(r, _)| *r == row as u32) {
                            e.1 = w[row * self.n_top_k + slot];
                        } else {
                            per_row.push((row as u32, w[row * self.n_top_k + slot]));
                        }
                    }
                }
            }
            if per_row.is_empty() {
                continue;
            }
            let token_ids: Vec<u32> = per_row.iter().map(|(r, _)| *r).collect();
            let weights: Vec<f32> = per_row.iter().map(|(_, wgt)| *wgt).collect();
            let ids = Tensor::new(token_ids.as_slice(), x.device())?; // (num,)
            let expert_in = x_flat.index_select(&ids, 0)?; // (num, C)
            let expert_out = self.experts[i].forward(&expert_in)?; // (num, C)
            let w_t = Tensor::new(weights.as_slice(), x.device())?.unsqueeze(1)?; // (num, 1)
            let contrib = expert_out.broadcast_mul(&w_t)?;
            // 每个 row 只出现一次，用 index_add（等价于加一次）
            output = output.index_add(&ids, &contrib, 0)?;
        }
        Ok(output.reshape((b, t, c))?)
    }
}

/// Sinkhorn-Knopp：把 hc×hc 矩阵投影成双随机矩阵（每行每列和都为 1、元素非负）。
/// 双随机矩阵的谱范数 ≤ 1——信号经过每个 mHC 超连接最多不被放大。
/// 逐位对齐 model.py：sinkhorn_knopp(F.softplus(raw_B))，内部先 exp(softplus(·)) 再 20 次行列归一。
fn sinkhorn_knopp(raw: &Tensor, iters: usize) -> Result<Tensor> {
    // softplus(x) = log(1 + exp(x))；exp(softplus(raw)) = 1 + exp(raw)
    let softplus = raw.exp()?.affine(1.0, 1.0)?.log()?; // log(1+exp(raw))
    let mut m = softplus.exp()?;                        // exp(softplus(raw))
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

/// 温度 → 重复惩罚 → top-k → softmax → 多项式采样，和 model.py 的 generate 一致。
fn sample<R: Rng>(
    logits: &Tensor,
    temperature: f64,
    top_k: Option<usize>,
    repeat_penalty: f64,
    seen: &[u32], // 已出现的 token（重复惩罚要压的对象）
    rng: &mut R,
) -> Result<u32> {
    let mut logits: Vec<f32> = logits.to_vec1()?;

    // 温度缩放
    for l in logits.iter_mut() {
        *l /= temperature as f32;
    }

    // 重复惩罚（CTRL 标准做法）：已出现的 token，正 logits 除以、负 logits 乘以，
    // 两者都把它压向低概率——抑制模型复述"模型：""用户："这类结构标签。
    // 同一个 token 出现几次就压几次（出现越频繁压得越狠）。
    // penalty > 1 才生效；=1.0 时完全等价于原逻辑，零开销。
    if repeat_penalty > 1.0 {
        let p = repeat_penalty as f32;
        for &t in seen {
            let l = &mut logits[t as usize];
            if *l >= 0.0 {
                *l /= p;
            } else {
                *l *= p;
            }
        }
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
