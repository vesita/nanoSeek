//! nanoSeek 对话推理 CLI。
//! 用法（在 inference/runtime/ 目录下）：
//!   cargo run --release -- --prompt "悟空"                # 一次性生成
//!   cargo run --release -- --print-logits --prompt "悟空" # 调试：打印 top-10 logits
//!   cargo run --release                                   # 进入对话 REPL
use std::io::Write;

use anyhow::Result;
use model::Config;
use rand::SeedableRng;
use tokenizer::BpeTokenizer;

mod attention;
mod model;
mod tokenizer;

fn main() -> Result<()> {
    // --- 简单参数解析（默认文件在 runtime/ 目录下） ---
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut model_path = "model.safetensors".to_string();
    let mut config_path = "model_config.json".to_string();
    let mut tokenizer_path = "tokenizer.json".to_string();
    let mut prompt: Option<String> = None;
    let mut max_new_tokens = 300usize;
    let mut temperature = 0.8f64;
    let mut top_k: Option<usize> = Some(200);
    let mut seed = 1337u64;
    let mut print_logits = false;
    let mut dump_logits: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--print-logits" => print_logits = true,
            "--dump-logits" => {
                i += 1;
                dump_logits = Some(args[i].clone());
            }
            "--model" => {
                i += 1;
                model_path = args[i].clone();
            }
            "--config" => {
                i += 1;
                config_path = args[i].clone();
            }
            "--tokenizer" => {
                i += 1;
                tokenizer_path = args[i].clone();
            }
            "--prompt" => {
                i += 1;
                prompt = Some(args[i].clone());
            }
            "--max-new-tokens" => {
                i += 1;
                max_new_tokens = args[i].parse()?;
            }
            "--temperature" => {
                i += 1;
                temperature = args[i].parse()?;
            }
            "--top-k" => {
                i += 1;
                top_k = Some(args[i].parse()?);
            }
            "--seed" => {
                i += 1;
                seed = args[i].parse()?;
            }
            _ => {}
        }
        i += 1;
    }

    // --- 加载模型 ---
    let device = candle_core::Device::Cpu;
    println!("正在加载模型……");
    let config = Config::load(&config_path)?;
    let gpt = model::GPT::load(&model_path, &config, &device)?;
    let tok = BpeTokenizer::load(&tokenizer_path)?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    println!(
        "模型就绪：{} 层 · {} 头 · {} 维 · {} 词表",
        config.n_layer, config.n_head, config.n_embd, config.vocab_size
    );
    // 打印启用的架构特性（方便确认部署的是哪套架构）
    let mut feats: Vec<String> = Vec::new();
    if config.use_moe {
        feats.push(format!("MoE({}专家/top{})", config.n_experts, config.n_top_k));
        if config.use_anticipatory_routing {
            feats.push("预判路由".to_string());
        }
        if config.use_shared_expert {
            feats.push("共享专家".to_string());
        }
        if config.num_hash_layers > 0 {
            feats.push(format!("Hash路由(前{}层)", config.num_hash_layers));
        }
    }
    if config.use_csa {
        feats.push(format!(
            "CSA(块{}/top{}/窗{})",
            config.csa_compress, config.csa_topk, config.csa_window
        ));
        if config.use_hca {
            feats.push("HCA".to_string());
        }
        if config.use_csa_learnable {
            feats.push("门控池化".to_string());
        }
        if config.use_lightning_indexer {
            feats.push("Indexer".to_string());
        }
    }
    if config.use_mhc {
        feats.push(format!("mHC({}流)", config.hc_mult));
    }
    if config.use_attn_sink {
        feats.push("Sinks".to_string());
    }
    if config.swiglu_clamp > 0.0 {
        feats.push(format!("Clamp({})", config.swiglu_clamp));
    }
    if !feats.is_empty() {
        println!("特性: {}", feats.join(" + "));
    }

    // --- 一次性生成（给了 --prompt） ---
    if let Some(p) = prompt {
        let ids = tok.encode(&p)?;
        if print_logits {
            // 调试：打印最后一个位置的 top-10 logits（用于与 Python 端对拍验证）
            let logits = gpt.forward(&ids)?; // (vocab,)
            let v: Vec<f32> = logits.to_vec1()?;
            let mut pairs: Vec<(usize, f32)> = v.iter().copied().enumerate().collect();
            pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            for (idx, val) in pairs.iter().take(10) {
                println!("{idx}\t{val:.6}");
            }
            return Ok(());
        }
        if let Some(path) = dump_logits {
            // 调试：把全部 logits 落盘（每行一个，用于 numpy 精确 diff）
            let logits = gpt.forward(&ids)?; // (vocab,)
            let v: Vec<f32> = logits.to_vec1()?;
            let body = v
                .iter()
                .map(|x| format!("{x}"))
                .collect::<Vec<_>>()
                .join("\n");
            std::fs::write(&path, body)?;
            return Ok(());
        }
        // 流式生成：逐 token 打印
        stream_print(&gpt, &tok, &ids, max_new_tokens, temperature, top_k, &mut rng)?;
        println!();
        return Ok(());
    }

    // --- 对话 REPL ---
    println!("nanoSeek 对话推理（输入 `退出` / `exit` / Ctrl-D 结束）");
    let mut context: Vec<u32> = Vec::new();
    let stdin = std::io::stdin();
    loop {
        print!("用户: ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        if stdin.read_line(&mut line)? == 0 {
            break;
        }
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        if line == "退出" || line == "exit" || line == "quit" {
            break;
        }
        // 把这一轮用户输入拼进上下文，让模型接着往下写。
        // 注意：训练数据用的是全角冒号（用户：/模型：），这里必须保持一致，
        // 否则模型认不出对话结构。数据里 "用户：...\n模型：" 之后就是模型回答，无空格。
        context.extend(tok.encode(&format!("用户：{line}\n模型："))?);
        print!("模型: ");
        let new_tokens = stream_print(&gpt, &tok, &context, max_new_tokens, temperature, top_k, &mut rng)?;
        println!();
        context.extend(new_tokens);
    }
    Ok(())
}

/// 流式生成：逐 token 生成 + 逐 token 解码打印（处理跨 token 的多字节字符）。
/// 返回新生成的 token（不含 prompt），供调用方继续拼上下文。
fn stream_print(
    gpt: &model::GPT,
    tok: &BpeTokenizer,
    ids: &[u32],
    max_new_tokens: usize,
    temperature: f64,
    top_k: Option<usize>,
    rng: &mut rand::rngs::StdRng,
) -> Result<Vec<u32>> {
    let eos_id = tok.eos_id(); // 遇到 <eos> 就停止，避免生成垃圾
    let mut decoder = tok.stream();
    let new_tokens = gpt.generate_stream(
        ids,
        max_new_tokens,
        temperature,
        top_k,
        eos_id,
        rng,
        |t| {
            if let Ok(Some(chunk)) = decoder.push(t) {
                print!("{chunk}");
                let _ = std::io::stdout().flush();
            }
        },
    )?;
    let tail = decoder.finish()?;
    print!("{tail}");
    let _ = std::io::stdout().flush();
    Ok(new_tokens)
}
