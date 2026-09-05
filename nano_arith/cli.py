"""
CLI Inference and Playground for nano_arith.

Allows users to input arbitrary arithmetic expressions:
e.g.
  python nano_arith/cli.py "123+456="
  python nano_arith/cli.py "9999+1="
  python nano_arith/cli.py "45+8="
Or interactive mode:
  python nano_arith/cli.py
"""

import sys
import argparse
import torch
from dataset import encode_str, decode_ids, EOS_ID
from model import NanoArithTransformer, ModelConfig

def solve(model, prompt: str, device, reverse_output: bool = True, use_scratchpad: bool = False):
    if not prompt.endswith("="):
        prompt += "="
    
    prompt_ids = torch.tensor([encode_str(prompt)], dtype=torch.long, device=device)
    # Generate up to 24 tokens
    gen = model.generate(prompt_ids, max_new_tokens=24, eos_id=EOS_ID)
    raw_pred = decode_ids(gen[0, len(prompt):].tolist())
    
    if use_scratchpad:
        from dataset import parse_scratchpad_to_number
        ans = parse_scratchpad_to_number(raw_pred)
    else:
        ans = raw_pred[::-1] if reverse_output else raw_pred
    
    # Calculate ground truth if it's standard addition or subtraction
    gt_str = None
    try:
        if "+" in prompt:
            a_s, b_s = prompt[:-1].split("+")
            gt_str = str(int(a_s) + int(b_s))
        elif "-" in prompt:
            a_s, b_s = prompt[:-1].split("-")
            gt_str = str(int(a_s) - int(b_s))
    except Exception:
        pass
        
    return ans, raw_pred, gt_str

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("expr", nargs="?", default=None, help="Expression to solve, e.g. '1234+567='")
    parser.add_argument("--ckpt", type=str, default="out/nano_arith_scratchpad/best.pt")
    parser.add_argument("--no_scratchpad", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ModelConfig(**ckpt["cfg"])
    cfg.max_seq_len = 96
    model = NanoArithTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    use_scratchpad = not args.no_scratchpad

    if args.expr:
        ans, raw, gt = solve(model, args.expr, device, use_scratchpad=use_scratchpad)
        status = f"✅ (Match)" if gt is not None and ans == gt else (f"❌ (GT: {gt})" if gt is not None else "")
        print(f"Expression    : {args.expr}")
        print(f"Predicted     : {ans} {status}")
        print(f"Raw Scratchpad: {raw}")
        return

    print(f"=== nano_arith Interactive CLI (Loaded {args.ckpt}) ===")
    print("Type expressions like '12+89=' or '12345+67890=', or 'exit'/'quit' to quit.\n")
    while True:
        try:
            line = input("nano_arith> ").strip()
            if not line:
                continue
            if line.lower() in ("exit", "quit", "q"):
                break
            ans, raw, gt = solve(model, line, device, use_scratchpad=use_scratchpad)
            status = f"✅" if gt is not None and ans == gt else (f"❌ (GT: {gt})" if gt is not None else "")
            print(f"  Result: {ans}  {status}  (raw scratchpad: {raw})")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

if __name__ == "__main__":
    main()
