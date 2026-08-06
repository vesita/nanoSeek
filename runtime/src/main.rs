//! nanoGPT 对话推理 CLI。
//! 用法（在 runtime/ 目录下）：
//!   cargo run --release -- --prompt "悟空"                # 一次性生成
//!   cargo run --release                                   # 进入对话 REPL
use std::io::Write;

use anyhow::Result;
use model::Config;
use rand::SeedableRng;
use tokenizer::CharTokenizer;

mod attention;
mod model;
mod tokenizer;

fn main() -> Result<()> {
    // --- 简单参数解析（默认文件在 runtime/ 目录下） ---
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut model_path = "model.safetensors".to_string();
    let mut config_path = "model_config.json".to_string();
    let mut vocab_path = "vocab.json".to_string();
    let mut prompt: Option<String> = None;
    let mut max_new_tokens = 300usize;
    let mut temperature = 0.8f64;
    let mut top_k: Option<usize> = Some(200);
    let mut seed = 1337u64;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model_path = args[i].clone();
            }
            "--config" => {
                i += 1;
                config_path = args[i].clone();
            }
            "--vocab" => {
                i += 1;
                vocab_path = args[i].clone();
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
    let tok = CharTokenizer::load(&vocab_path)?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    println!(
        "模型就绪：{} 层 · {} 头 · {} 维 · {} 词表",
        config.n_layer, config.n_head, config.n_embd, config.vocab_size
    );

    // --- 一次性生成（给了 --prompt） ---
    if let Some(p) = prompt {
        let ids = tok.encode(&p);
        let new_tokens = gpt.generate(&ids, max_new_tokens, temperature, top_k, &mut rng)?;
        println!("{}", tok.decode(&new_tokens));
        return Ok(());
    }

    // --- 对话 REPL ---
    println!("nanoGPT 对话推理（输入 `退出` / `exit` / Ctrl-D 结束）");
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
        // 把这一轮用户输入拼进上下文，让模型接着往下写
        context.extend(tok.encode(&format!("用户: {line}\n模型: ")));
        let new_tokens = gpt.generate(&context, max_new_tokens, temperature, top_k, &mut rng)?;
        let text = tok.decode(&new_tokens);
        println!("模型: {text}");
        context.extend(new_tokens);
    }
    Ok(())
}
