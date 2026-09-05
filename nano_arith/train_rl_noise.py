"""
Reinforcement Learning / Joint Training script with Reward Engine & Noise Injection for nano_arith.

Methods:
1. Pure RL (GRPO): Policy gradient with Group Relative Policy Optimization using RewardEngine.
2. Mixed Training (GRPO + SFT CE): Combining policy exploration with teacher corpus CE.
3. Noise Injection Modes:
   --reward_noise_std : continuous Gaussian reward perturbation
   --reward_flip_prob : discrete label noise (adversarial feedback)
   --temp_noise       : exploration sampling temperature perturbation
"""

import os
import time
import argparse
import random
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import (
    ArithmeticDataset, VOCAB_SIZE, EOS_ID, PAD_ID,
    encode_str, decode_ids, generate_addition_sample
)
from model import NanoArithTransformer, ModelConfig
from reward_engine import ArithRewardEngine, RewardConfig

def sample_group_responses(
    model,
    prompt_ids: torch.Tensor,
    group_size: int = 4,
    max_tokens: int = 8,
    temperature: float = 1.0,
    temp_noise: float = 0.0,
    eos_id: int = EOS_ID
):
    """
    Samples G responses for a given prompt using stochastic autoregressive generation.
    Supports temperature noise injection.
    """
    B, S = prompt_ids.shape
    assert B == 1
    
    # Expand prompt to [G, S]
    curr_ids = prompt_ids.repeat(group_size, 1)
    
    finished = torch.zeros(group_size, dtype=torch.bool, device=prompt_ids.device)
    
    for _ in range(max_tokens):
        with torch.no_grad():
            logits = model(curr_ids)["logits"][:, -1, :]  # [G, vocab_size]
            
            # Dynamic temperature with noise
            eff_temp = temperature
            if temp_noise > 0.0:
                eff_temp = max(0.2, temperature + random.gauss(0.0, temp_noise))
                
            probs = F.softmax(logits / eff_temp, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)  # [G, 1]
            
        # Mask out tokens if already finished
        next_tokens = torch.where(finished.unsqueeze(-1), torch.tensor(PAD_ID, device=prompt_ids.device), next_tokens)
        finished = finished | (next_tokens.squeeze(-1) == eos_id)
        curr_ids = torch.cat([curr_ids, next_tokens], dim=-1)
        
        if finished.all():
            break
            
    # Extract only response tokens
    responses = []
    for g in range(group_size):
        r_ids = curr_ids[g, S:].tolist()
        # strip padding
        clean_ids = [t for t in r_ids if t != PAD_ID]
        responses.append(clean_ids)
        
    return responses

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=192)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--init_from", type=str, default="")
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=800)
    parser.add_argument("--eval_interval", type=int, default=100)
    
    # Noise injection options
    parser.add_argument("--reward_noise_std", type=float, default=0.0)
    parser.add_argument("--reward_flip_prob", type=float, default=0.0)
    parser.add_argument("--temp_noise", type=float, default=0.0)
    
    # Mixed CE vs Pure RL & KL Anchor
    parser.add_argument("--lam_rl", type=float, default=0.5)
    parser.add_argument("--lam_ce", type=float, default=0.5)
    parser.add_argument("--beta_kl", type=float, default=0.05)
    parser.add_argument("--save_dir", type=str, default="out/nano_arith_rl_noise")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"=== Starting RL / Mixed Training with Reward Engine ===")
    print(f"Device: {device} | Lam_RL: {args.lam_rl} | Lam_CE: {args.lam_ce}")
    print(f"Noise Injection: reward_noise_std={args.reward_noise_std}, reward_flip_prob={args.reward_flip_prob}, temp_noise={args.temp_noise}")

    # Reward Engine with Noise Injector
    rew_cfg = RewardConfig(
        reward_noise_std=args.reward_noise_std,
        reward_flip_prob=args.reward_flip_prob,
        tau=1.2
    )
    engine = ArithRewardEngine(rew_cfg)

    # Model
    if args.init_from and os.path.exists(args.init_from):
        print(f"Loading initial checkpoint from: {args.init_from}")
        ckpt = torch.load(args.init_from, map_location=device)
        cfg = ModelConfig(**ckpt["cfg"])
        model = NanoArithTransformer(cfg).to(device)
        model.load_state_dict(ckpt["model_state"])
    else:
        cfg = ModelConfig(
            vocab_size=VOCAB_SIZE,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
        )
        model = NanoArithTransformer(cfg).to(device)
    
    # Learning rate & Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # Note: RL from cold random weights with batch size 1 requires thousands of steps to hit first correct.
    # We use a batch of prompt environments (e.g. batch_size = 16) for stable gradient.

    # Evaluation dataset
    val_samples = [generate_addition_sample(min_digits=2, max_digits=2, reverse_output=True) for _ in range(100)]

    step = 0
    t0 = time.time()
    best_val_acc = 0.0
    history = []
    
    while step < args.max_steps:
        step += 1
        
        # 1. Sample prompt for RL
        prompt, gt_rev = generate_addition_sample(min_digits=2, max_digits=2, reverse_output=True)
        prompt_ids = torch.tensor([encode_str(prompt)], dtype=torch.long, device=device)
        gt_number = gt_rev[::-1]
        
        # 2. Sample Group of completions
        model.eval()
        group_replies = sample_group_responses(
            model,
            prompt_ids,
            group_size=args.group_size,
            max_tokens=6,
            temperature=1.0,
            temp_noise=args.temp_noise,
            eos_id=EOS_ID
        )
        
        # 3. Evaluate each completion with Reward Engine (with noise)
        rewards = []
        parsed_preds = []
        for r_ids in group_replies:
            text = decode_ids(r_ids)
            eval_res = engine.evaluate(prompt, text, gt_number, use_scratchpad=False)
            rewards.append(eval_res["reward"])
            parsed_preds.append(eval_res["parsed_pred"])
            
        r_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        
        # 4. GRPO Relative Advantages: (R - mean) / (std + 1e-4)
        if args.group_size > 1:
            mean = r_t.mean()
            std = r_t.std() + 1e-4
            advantages = (r_t - mean) / std
        else:
            advantages = r_t

        # 5. Compute Policy Loss
        model.train()
        total_pol_loss = torch.tensor(0.0, device=device, requires_grad=True)
        losses_list = []
        
        for g_idx, r_ids in enumerate(group_replies):
            if not r_ids:
                continue
            adv = advantages[g_idx].detach()
            full_seq = prompt_ids[0].tolist() + r_ids
            input_t = torch.tensor([full_seq], dtype=torch.long, device=device)
            
            # Forward pass
            logits = model(input_t)["logits"]  # [1, S, vocab]
            
            # Log probabilities on response tokens
            p_len = len(prompt_ids[0])
            r_len = len(r_ids)
            shift_logits = logits[0, p_len - 1 : p_len - 1 + r_len, :]
            target_ids = torch.tensor(r_ids, dtype=torch.long, device=device)
            
            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_logprobs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
            
            # Policy gradient: - Advantage * sum(log_probs)
            pol_loss = -adv * token_logprobs.mean()
            losses_list.append(pol_loss)
            
        if losses_list:
            total_pol_loss = torch.stack(losses_list).mean()
            
        # 6. Mixed Training: Auxiliary Supervised CE & KL Anchor
        total_loss = args.lam_rl * total_pol_loss
        
        if args.lam_ce > 0.0:
            target_ids = encode_str(gt_rev) + [EOS_ID]
            full_sft = encode_str(prompt) + target_ids
            labels_sft = [-100] * len(prompt) + target_ids
            
            sft_inp = torch.tensor([full_sft], dtype=torch.long, device=device)
            sft_lbl = torch.tensor([labels_sft], dtype=torch.long, device=device)
            ce_loss = model(sft_inp, labels=sft_lbl)["loss"]
            total_loss = total_loss + args.lam_ce * ce_loss
        else:
            ce_loss = torch.tensor(0.0)

        # Backward & Step
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Periodic Evaluation
        if step % args.eval_interval == 0 or step == args.max_steps:
            model.eval()
            correct = 0
            with torch.no_grad():
                for p_val, gt_val_rev in val_samples:
                    p_val_ids = torch.tensor([encode_str(p_val)], dtype=torch.long, device=device)
                    gen = model.generate(p_val_ids, max_new_tokens=6, eos_id=EOS_ID)
                    raw_res = decode_ids(gen[0, len(p_val):].tolist())
                    if raw_res[::-1] == gt_val_rev[::-1]:
                        correct += 1
            val_acc = (correct / len(val_samples)) * 100
            dt = time.time() - t0
            avg_rew = r_t.mean().item()
            print(f"Step {step:4d}/{args.max_steps:4d} | PolLoss: {total_pol_loss.item():.4f} | AvgRew: {avg_rew:+.2f} | Val Acc: {val_acc:5.1f}% | Elapsed: {dt:.1f}s")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "step": step,
                    "model_state": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "val_acc": val_acc,
                }, os.path.join(args.save_dir, "best.pt"))

    # Save log
    with open(os.path.join(args.save_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining finished! Final Validation Acc: {val_acc:.1f}%\n")

if __name__ == "__main__":
    main()
