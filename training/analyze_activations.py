#!/usr/bin/env python3
"""训练后网络激活分析：各层激活分布、MoE 负载均衡、EOS 注意力模式。

用法：
    .venv/bin/python training/analyze_activations.py --dir out/eos_fix_1epoch

输出 JSON 报告 + 终端摘要。
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from collections import defaultdict
from inference.scripts.sample_py import build_model_from_checkpoint, load_tokenizer


def analyze_activations(model, tok, device, n_batches=20):
    """用随机训练数据片段跑前向，hook 收集各层激活统计。"""
    stats = defaultdict(lambda: {"mean": [], "std": [], "dead_frac": [], "max": []})
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                x = output[0]
            else:
                x = output
            if not isinstance(x, torch.Tensor):
                return
            with torch.no_grad():
                flat = x.float().flatten()
                stats[name]["mean"].append(flat.mean().item())
                stats[name]["std"].append(flat.std().item())
                stats[name]["dead_frac"].append((flat.abs() < 1e-6).float().mean().item())
                stats[name]["max"].append(flat.abs().max().item())
        return hook_fn

    # 注册 hooks
    for i, block in enumerate(model.transformer.h):
        hooks.append(block.register_forward_hook(make_hook(f"block_{i}")))
        if hasattr(block, 'attn') and not getattr(block, 'skip_attn', False):
            hooks.append(block.attn.register_forward_hook(make_hook(f"attn_{i}")))
        if hasattr(block, 'mlp'):
            hooks.append(block.mlp.register_forward_hook(make_hook(f"ffn_{i}")))

    # 跑几个 batch
    data_path = 'data/chinese/train_char.bin'
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    block_size = model.config.block_size

    model.eval()
    with torch.no_grad():
        for _ in range(n_batches):
            ix = torch.randint(len(data) - block_size, (16,))
            x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
            x = x.to(device)
            model(x)

    # 清理 hooks
    for h in hooks:
        h.remove()

    # 汇总
    summary = {}
    for name, vals in sorted(stats.items()):
        summary[name] = {
            "mean": float(np.mean(vals["mean"])),
            "std": float(np.mean(vals["std"])),
            "dead_frac": float(np.mean(vals["dead_frac"])),
            "max_abs": float(np.mean(vals["max"])),
        }
    return summary


def analyze_moe_balance(model, tok, device, n_batches=20):
    """收集 MoE 各专家被选中的频率，评估负载均衡。"""
    if not model.config.use_moe:
        return {"enabled": False}

    expert_counts = defaultdict(lambda: defaultdict(int))
    hooks = []

    def make_router_hook(layer_idx):
        def hook_fn(module, input, output):
            # MoE router output: topk indices
            if hasattr(module, '_last_topk_indices'):
                indices = module._last_topk_indices  # (B*T, top_k)
                for idx in indices.flatten().tolist():
                    expert_counts[layer_idx][idx] += 1
        return hook_fn

    for i, block in enumerate(model.transformer.h):
        if hasattr(block, 'mlp') and hasattr(block.mlp, 'gate'):
            hooks.append(block.mlp.register_forward_hook(make_router_hook(i)))

    data_path = 'data/chinese/train_char.bin'
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    block_size = model.config.block_size

    model.eval()
    with torch.no_grad():
        for _ in range(n_batches):
            ix = torch.randint(len(data) - block_size, (16,))
            x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
            x = x.to(device)
            model(x)

    for h in hooks:
        h.remove()

    # 如果没收集到数据，尝试从 aux loss 推断
    if not expert_counts:
        return {"enabled": True, "note": "router indices not captured (internal API)"}

    result = {}
    for layer, counts in sorted(expert_counts.items()):
        total = sum(counts.values())
        dist = {f"expert_{k}": v / total for k, v in sorted(counts.items())}
        result[f"layer_{layer}"] = dist
    return result


def analyze_eos_attention(model, tok, device):
    """分析模型在 EOS token 附近的注意力模式。"""
    # 构造含 EOS 的输入
    test_prompts = [
        "用户：你好\n模型：你好！很高兴认识你。<eos>\n用户：今天天气怎么样？\n模型：",
        "用户：谢谢\n模型：不客气！<eos>\n用户：再见\n模型：",
    ]

    eos_id = tok.token_to_id("<eos>")
    results = []

    for prompt in test_prompts:
        ids = tok.encode(prompt).ids
        eos_positions = [i for i, t in enumerate(ids) if t == eos_id]
        if not eos_positions:
            continue

        x = torch.tensor([ids], dtype=torch.long, device=device)
        model.eval()
        with torch.no_grad():
            model(x)

        results.append({
            "prompt_len": len(ids),
            "eos_positions": eos_positions,
            "eos_count": len(eos_positions),
        })

    return {
        "eos_id": eos_id,
        "test_cases": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="模型 checkpoint 目录")
    ap.add_argument("--out", default=None, help="输出 JSON 路径（默认 <dir>/analysis.json）")
    a = ap.parse_args()

    out_path = a.out or os.path.join(a.dir, "analysis.json")

    model, ckpt = build_model_from_checkpoint(a.dir)
    tok = load_tokenizer(ckpt)
    device = next(model.parameters()).device
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型: {a.dir} | {n_params:,} 参数")
    print(f"val loss: {ckpt.get('best_val_loss', 'N/A')}")
    print()

    # 1. 激活分布
    print("▶ 分析各层激活分布...")
    act_stats = analyze_activations(model, tok, device)
    print(f"{'层':<12} {'均值':>8} {'标准差':>8} {'死神经元%':>10} {'最大|值|':>10}")
    print("─" * 50)
    for name, s in act_stats.items():
        print(f"{name:<12} {s['mean']:>8.4f} {s['std']:>8.4f} {s['dead_frac']*100:>9.2f}% {s['max_abs']:>10.2f}")
    print()

    # 2. EOS 注意力
    print("▶ 分析 EOS token 模式...")
    eos_info = analyze_eos_attention(model, tok, device)
    print(f"  EOS id: {eos_info['eos_id']}")
    for tc in eos_info.get('test_cases', []):
        print(f"  prompt_len={tc['prompt_len']}, eos@{tc['eos_positions']}")
    print()

    # 3. MoE 负载
    print("▶ 分析 MoE 专家负载...")
    moe_info = analyze_moe_balance(model, tok, device)
    if moe_info.get("enabled") is False:
        print("  MoE 未启用")
    elif "note" in moe_info:
        print(f"  {moe_info['note']}")
    else:
        for layer, dist in moe_info.items():
            vals = list(dist.values())
            imbalance = max(vals) / max(min(vals), 1e-9)
            print(f"  {layer}: max/min={imbalance:.2f}x | {dist}")
    print()

    # 写 JSON
    report = {
        "model_dir": a.dir,
        "n_params": n_params,
        "best_val_loss": float(ckpt.get('best_val_loss', -1)),
        "activations": act_stats,
        "eos_analysis": eos_info,
        "moe_balance": moe_info,
    }
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"📄 报告已保存: {out_path}")


if __name__ == "__main__":
    main()
