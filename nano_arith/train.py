"""
Training script for nano_arith.

Features:
- Monitors exact match (EM) accuracy on held-out validation set
- Compares train loss vs val accuracy to detect "grokking" (loss drops -> val accuracy delays then jumps to 100%)
- Supports reverse-output vs forward-output comparison
- Supports 2-digit to 4-digit addition/subtraction
- High-efficiency training loop tailored for APU / GPU
"""

import os
import time
import argparse
import json
import torch
from torch.utils.data import DataLoader

from dataset import (
    ArithmeticDataset, VOCAB_SIZE, EOS_ID, PAD_ID,
    encode_str, decode_ids
)
from model import NanoArithTransformer, ModelConfig

def evaluate_exact_match_by_digits(
    model,
    device,
    min_digits: int = 1,
    max_digits: int = 4,
    samples_per_digit: int = 50,
    reverse_output: bool = True,
    use_scratchpad: bool = False,
    literal_noise: bool = False,
    literal_noise_prob: float = 0.8
):
    """
    Evaluates exact match accuracy broken down by digit length combinations:
    1-digit, 2-digit, 3-digit, 4-digit, etc.
    """
    model.eval()
    results = {}
    total_correct = 0
    total_count = 0
    
    for d in range(min_digits, max_digits + 1):
        d_correct = 0
        from dataset import (
            generate_addition_sample,
            generate_scratchpad_addition_sample,
            generate_literal_noisy_addition_sample,
            parse_scratchpad_to_number
        )
        for _ in range(samples_per_digit):
            if literal_noise:
                prompt, target = generate_literal_noisy_addition_sample(
                    min_digits=d,
                    max_digits=d,
                    use_scratchpad=use_scratchpad,
                    reverse_output=reverse_output,
                    noise_prob=literal_noise_prob
                )
            elif use_scratchpad:
                prompt, target = generate_scratchpad_addition_sample(min_digits=d, max_digits=d)
            else:
                prompt, target = generate_addition_sample(min_digits=d, max_digits=d, reverse_output=reverse_output)
                
            prompt_ids = torch.tensor([encode_str(prompt)], dtype=torch.long, device=device)
            generated = model.generate(prompt_ids, max_new_tokens=len(target) + 4, eos_id=EOS_ID)
            pred_ids = generated[0, len(prompt):].tolist()
            pred_str = decode_ids(pred_ids)
            
            if use_scratchpad:
                pred_num = parse_scratchpad_to_number(pred_str)
                gt_num = parse_scratchpad_to_number(target)
                if pred_num == gt_num:
                    d_correct += 1
            else:
                if pred_str == target:
                    d_correct += 1
                    
        acc = d_correct / samples_per_digit
        results[f"{d}d"] = acc
        total_correct += d_correct
        total_count += samples_per_digit
        
    results["overall"] = total_correct / total_count
    model.train()
    return results

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.1)  # weight decay helps grokking!
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--min_digits", type=int, default=1)
    parser.add_argument("--max_digits", type=int, default=3)
    parser.add_argument("--ops", type=str, default="+")
    parser.add_argument("--reverse_output", action="store_true", default=True)
    parser.add_argument("--no_reverse", dest="reverse_output", action="store_false")
    parser.add_argument("--use_scratchpad", action="store_true", default=False)
    parser.add_argument("--literal_noise", action="store_true", default=False,
                        help="训练中注入运算符/等号/问句/全角数字等字面噪声")
    parser.add_argument("--literal_noise_prob", type=float, default=0.8,
                        help="每个样本注入字面噪声的概率")
    parser.add_argument("--save_dir", type=str, default="out/nano_arith")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (ROCm/CUDA: {torch.cuda.is_available()})")
    print(f"Settings: reverse_output={args.reverse_output}, use_scratchpad={args.use_scratchpad}, literal_noise={args.literal_noise} (p={args.literal_noise_prob}), digits={args.min_digits}-{args.max_digits}, ops={args.ops}")

    os.makedirs(args.save_dir, exist_ok=True)
    
    # Datasets
    ops = tuple(args.ops.split(","))
    train_ds = ArithmeticDataset(
        num_samples=40000,
        min_digits=args.min_digits,
        max_digits=args.max_digits,
        ops=ops,
        reverse_output=args.reverse_output,
        use_scratchpad=args.use_scratchpad,
        literal_noise=args.literal_noise,
        literal_noise_prob=args.literal_noise_prob,
        max_seq_len=64,
        seed=123
    )
    val_ds = ArithmeticDataset(
        num_samples=2000,
        min_digits=args.min_digits,
        max_digits=args.max_digits,
        ops=ops,
        reverse_output=args.reverse_output,
        use_scratchpad=args.use_scratchpad,
        literal_noise=args.literal_noise,
        literal_noise_prob=args.literal_noise_prob,
        max_seq_len=64,
        seed=999
    )
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    cfg = ModelConfig(
        vocab_size=VOCAB_SIZE,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
    )
    model = NanoArithTransformer(cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Grokking literature shows AdamW with weight decay accelerates generalization
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    
    # Cosine scheduler with warmup
    warmup_steps = 200
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, args.max_steps - warmup_steps))
        return 0.5 * (1.0 + torch.cos(torch.tensor(3.1415926535 * progress))).item()
        
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    step = 0
    t0 = time.time()
    best_val_em = 0.0
    history = []

    train_iter = iter(train_loader)
    while step < args.max_steps:
        step += 1
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
            
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        out = model(input_ids, labels=labels)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % args.eval_interval == 0 or step == args.max_steps:
            val_breakdown = evaluate_exact_match_by_digits(
                model, device,
                min_digits=args.min_digits,
                max_digits=args.max_digits,
                samples_per_digit=40,
                reverse_output=args.reverse_output,
                use_scratchpad=args.use_scratchpad,
                literal_noise=args.literal_noise,
                literal_noise_prob=args.literal_noise_prob
            )
            val_em = val_breakdown["overall"]
            breakdown_str = " ".join([f"{k}:{v*100:.0f}%" for k, v in val_breakdown.items() if k != "overall"])
            dt = time.time() - t0
            current_lr = scheduler.get_last_lr()[0]
            print(f"Step {step:4d}/{args.max_steps:4d} | Loss: {loss.item():.4f} | Val EM: {val_em * 100:.1f}% [{breakdown_str}] | LR: {current_lr:.2e} | Elapsed: {dt:.1f}s")
            
            history.append({
                "step": step,
                "loss": round(loss.item(), 4),
                "val_em": round(val_em, 4),
                "breakdown": val_breakdown,
                "lr": current_lr
            })
            
            if val_em > best_val_em:
                best_val_em = val_em
                torch.save({
                    "step": step,
                    "model_state": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "val_em": val_em,
                }, os.path.join(args.save_dir, "best.pt"))

    # Save final log
    with open(os.path.join(args.save_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
        
    print(f"\nTraining completed! Best Validation EM: {best_val_em * 100:.2f}%")
    
    # Showcase inference on fresh examples
    print("\n--- Model Inference Showcase ---")
    test_cases = [
        "45+78=", "123+456=", "999+1=", "854+679=", "7+8=",
        "45加上78等于几？", "123加456得", "９９９＋１＝", "问:7加上8等于多少？"
    ]
    model.eval()
    for prompt in test_cases:
        p_ids = torch.tensor([encode_str(prompt)], dtype=torch.long, device=device)
        gen = model.generate(p_ids, max_new_tokens=16, eos_id=EOS_ID)
        pred_raw = decode_ids(gen[0, len(prompt):].tolist())
        
        if args.use_scratchpad:
            from dataset import parse_scratchpad_to_number
            final_answer = parse_scratchpad_to_number(pred_raw)
        else:
            final_answer = pred_raw[::-1] if args.reverse_output else pred_raw
            
        print(f"Input: {prompt:<20} -> Raw: {pred_raw:<8} -> Answer: {final_answer}")

if __name__ == "__main__":
    train()
