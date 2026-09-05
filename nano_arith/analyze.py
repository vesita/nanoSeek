"""
Neuron and Attention Circuit Analysis for nano_arith.

Investigates:
1. Attention patterns: How does the model attend to corresponding digits (ones -> ones, tens -> tens)?
2. Carry Neuron Probing: Can we find individual MLP neurons that fire specifically when a carry occurs (a + b >= 10)?
"""

import os
import argparse
import torch
import numpy as np
from dataset import encode_str, decode_ids, EOS_ID
from model import NanoArithTransformer, ModelConfig

def analyze_attention(model, prompt: str, device):
    print(f"\n================ Attention Analysis: {prompt} ================")
    model.eval()
    tokens = encode_str(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    
    with torch.no_grad():
        model(input_ids)
        
    chars = list(prompt)
    for l_idx, layer in enumerate(model.layers):
        attn = layer.attn.last_attn_weights[0]  # [n_heads, S, S]
        print(f"\n--- Layer {l_idx} Attention (Averaged across heads) ---")
        avg_attn = attn.mean(dim=0).cpu().numpy()  # [S, S]
        
        header = "      " + "  ".join([f"{c:>2}" for c in chars])
        print(header)
        for i, row in enumerate(avg_attn):
            row_str = f"{chars[i]:>2} | " + " ".join([f"{val*100:2.0f}" for val in row])
            print(row_str)

def probe_carry_neurons(model, device, num_samples: int = 500):
    """
    Feed pairs of 2-digit additions: a1 a0 + b1 b0 =
    Classify whether ones-place has carry: (a0 + b0 >= 10)
    For every neuron in every layer's MLP intermediate layer, compute correlation / t-statistic with carry.
    """
    print("\n================ Carry Neuron Probing (2-Digit Addition) ================")
    import random
    
    carry_labels = []
    # Dict to collect activations: layer -> list of [num_samples, d_ff]
    activations = {l: [] for l in range(len(model.layers))}
    
    model.eval()
    with torch.no_grad():
        for _ in range(num_samples):
            a = random.randint(10, 99)
            b = random.randint(10, 99)
            prompt = f"{a}+{b}="
            has_carry_ones = 1 if ((a % 10) + (b % 10) >= 10) else 0
            carry_labels.append(has_carry_ones)
            
            tokens = encode_str(prompt)
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            
            model(input_ids)
            # Inspect activations at the '=' token (last token before generating answer)
            for l_idx, layer in enumerate(model.layers):
                act = layer.ffn.last_intermediate[0, -1, :].cpu().numpy()  # [d_ff]
                activations[l_idx].append(act)
                
    carry_labels = np.array(carry_labels)
    carry_idx = np.where(carry_labels == 1)[0]
    no_carry_idx = np.where(carry_labels == 0)[0]
    
    print(f"Sampled {num_samples} problems (Carries: {len(carry_idx)}, No-Carries: {len(no_carry_idx)})")
    
    best_overall = []
    
    for l_idx in range(len(model.layers)):
        acts = np.array(activations[l_idx])  # [num_samples, d_ff]
        carry_mean = acts[carry_idx].mean(axis=0)
        no_carry_mean = acts[no_carry_idx].mean(axis=0)
        
        diff = carry_mean - no_carry_mean
        
        # Calculate point biserial correlation for each neuron
        stds = acts.std(axis=0) + 1e-8
        corr = (diff * np.sqrt(len(carry_idx) * len(no_carry_idx)) / (num_samples * stds))
        
        top_neurons = np.argsort(np.abs(corr))[::-1][:3]
        print(f"\nLayer {l_idx} Top 3 Carry Neurons:")
        for rank, n_idx in enumerate(top_neurons):
            c_val = corr[n_idx]
            print(f"  #{rank+1}: Neuron {n_idx:3d} | Correlation: {c_val:+.4f} | Carry Mean: {carry_mean[n_idx]:.3f} vs Non-Carry: {no_carry_mean[n_idx]:.3f}")
            best_overall.append((abs(c_val), l_idx, n_idx, c_val))
            
    best_overall.sort(key=lambda x: x[0], reverse=True)
    top_c = best_overall[0]
    print(f"\n🏆 Strongest Carry Detector: Layer {top_c[1]}, Neuron #{top_c[2]} with correlation {top_c[3]:+.4f}!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="out/nano_arith_2digit/best.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {args.ckpt} on {device}")
    
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ModelConfig(**ckpt["cfg"])
    model = NanoArithTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    
    # 1. Probing carry neurons
    probe_carry_neurons(model, device)
    
    # 2. Attention map analysis on a sample
    analyze_attention(model, "45+78=", device)

if __name__ == "__main__":
    main()
