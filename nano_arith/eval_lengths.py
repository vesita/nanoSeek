"""
Zero-Shot Length Generalization Tester for nano_arith.

Evaluates how well a model trained on 1-4 digits performs on:
- Unseen longer digits: 5-digit, 6-digit, 7-digit, 8-digit addition
- Asymmetric additions: e.g. 2-digit + 5-digit
"""

import argparse
import torch
from dataset import encode_str, decode_ids, EOS_ID, generate_addition_sample
from model import NanoArithTransformer, ModelConfig

def test_length_generalization(ckpt_path: str, max_test_digits: int = 10, samples_per_digit: int = 50, use_scratchpad: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {ckpt_path} on {device}")
    
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ModelConfig(**ckpt["cfg"])
    # Allow larger sequence length for evaluation
    cfg.max_seq_len = 96
    model = NanoArithTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    from dataset import generate_scratchpad_addition_sample, parse_scratchpad_to_number

    print(f"\n================ Length Generalization Matrix (1 to {max_test_digits} Digits) [Scratchpad: {use_scratchpad}] ================")
    print(f"{'Digits':<8} | {'Exact Match Acc':<16} | {'Sample Prediction':<35} | {'Ground Truth'}")
    print("-" * 75)

    for d in range(1, max_test_digits + 1):
        correct = 0
        last_pred = ""
        last_gt = ""
        last_prompt = ""
        for _ in range(samples_per_digit):
            if use_scratchpad:
                prompt, target = generate_scratchpad_addition_sample(min_digits=d, max_digits=d)
            else:
                prompt, target = generate_addition_sample(min_digits=d, max_digits=d, reverse_output=True)
                
            p_ids = torch.tensor([encode_str(prompt)], dtype=torch.long, device=device)
            gen = model.generate(p_ids, max_new_tokens=len(target) + 4, eos_id=EOS_ID)
            pred_raw = decode_ids(gen[0, len(prompt):].tolist())
            
            if use_scratchpad:
                pred_ans = parse_scratchpad_to_number(pred_raw)
                gt_ans = parse_scratchpad_to_number(target)
            else:
                pred_ans = pred_raw[::-1]
                gt_ans = target[::-1]
                
            if pred_ans == gt_ans:
                correct += 1
            last_prompt = prompt
            last_pred = pred_ans
            last_gt = gt_ans

        acc = (correct / samples_per_digit) * 100
        sample_info = f"{last_prompt}{last_pred}"
        print(f"{d}-digit  | {acc:6.1f}%          | {sample_info:<35} | {last_gt}")

    print("\n================ Asymmetric Length Test ================")
    asymmetric_cases = [
        "12+3456=",
        "9999+1=",
        "5+12345=",
        "88888+22=",
        "1234+5678=",
        "12345+67890=",
    ]
    for prompt in asymmetric_cases:
        p_ids = torch.tensor([encode_str(prompt)], dtype=torch.long, device=device)
        gen = model.generate(p_ids, max_new_tokens=16, eos_id=EOS_ID)
        pred_raw = decode_ids(gen[0, len(prompt):].tolist())
        if use_scratchpad:
            pred_ans = parse_scratchpad_to_number(pred_raw)
        else:
            pred_ans = pred_raw[::-1]
        
        parts = prompt[:-1].split("+")
        gt = str(int(parts[0]) + int(parts[1]))
        status = "✅" if pred_ans == gt else f"❌ (GT: {gt})"
        print(f"Test: {prompt:<14} -> Raw: {pred_raw:<12} -> Ans: {pred_ans:<10} {status}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="out/nano_arith_1to4digit/best.pt")
    parser.add_argument("--max_digits", type=int, default=8)
    parser.add_argument("--use_scratchpad", action="store_true", default=False)
    args = parser.parse_args()
    
    test_length_generalization(args.ckpt, args.max_digits, use_scratchpad=args.use_scratchpad)
