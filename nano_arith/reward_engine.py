"""
Reward Engine and Noise Injection framework for nano_arith.

Features:
1. Multi-dimensional Reward Engine for arithmetic:
   - r_correct: +2.0 if final answer matches ground truth, -1.0 if wrong.
   - r_partial: digit-level IoU / prefix credit [0.0 to +1.0] to give gradient signals before grokking.
   - r_format: +0.2 for valid tokens, no repeating junk, clean EOS termination.
   - r_brevity: penalty for generating unnecessary runaway tokens.
   - Exponential Shaping: non-linear reward expansion to strongly pull top candidates.

2. Noise Injector (Robustness & Anti-Overfitting):
   - Reward Noise: Gaussian noise N(0, sigma) added to raw reward scores.
   - Label Flipping / Adversarial Reward: with probability p_noise, invert or corrupt reward
     (e.g., falsely rewarding wrong answer or penalizing correct one), testing model's resilience.
   - Prompt / Operator Noise: injecting minor formatting perturbations (e.g. spaces, alternate ops).
   - Temperature Exploration Noise: sampling with dynamic stochastic temperature.
"""

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import torch

@dataclass
class RewardConfig:
    w_correct: float = 2.0
    w_partial: float = 1.0
    w_format: float = 0.5
    w_brevity: float = 0.3
    tau: float = 1.2
    # Noise injection parameters
    reward_noise_std: float = 0.0     # Gaussian noise on score: N(0, std)
    reward_flip_prob: float = 0.0     # Prob of flipping binary correct reward
    input_noise_prob: float = 0.0     # Prob of adding random noise tokens to prompt

class ArithRewardEngine:
    def __init__(self, cfg: Optional[RewardConfig] = None):
        self.cfg = cfg or RewardConfig()

    def exponential_shaping(self, score: float) -> float:
        sign = 1.0 if score >= 0 else -1.0
        return sign * (math.exp(abs(score) / self.cfg.tau) - 1.0)

    def evaluate(
        self,
        prompt: str,
        predicted_tokens: str,
        ground_truth: str,
        use_scratchpad: bool = False,
        rng: Optional[random.Random] = None
    ) -> Dict[str, float]:
        """
        Evaluates generated output and returns dimensional breakdown + shaped total reward.
        """
        r = rng if rng is not None else random
        
        # 1. Parse predicted tokens to final number
        if use_scratchpad:
            digits = [ch for ch in predicted_tokens if ch.isdigit()]
            parsed_pred = "".join(digits)[::-1]
        else:
            parsed_pred = predicted_tokens[::-1]  # reverse digits
            
        is_correct = (parsed_pred == ground_truth) and len(ground_truth) > 0
        
        # --- Noise Injection: Label Flip Noise ---
        if self.cfg.reward_flip_prob > 0.0:
            if r.random() < self.cfg.reward_flip_prob:
                is_correct = not is_correct

        # Dimensional Rewards
        # 1. Correctness
        r_correct = 1.0 if is_correct else -0.8
        
        # 2. Partial Credit (Digit-by-digit matching from ones place)
        # Even if 1234+5678 produces wrong 6911 (expected 6912), the higher digits are right!
        # This provides a smooth reward ramp instead of zero-gradient cliff.
        rev_pred = parsed_pred[::-1]
        rev_gt = ground_truth[::-1]
        matching_prefix = 0
        for p_ch, g_ch in zip(rev_pred, rev_gt):
            if p_ch == g_ch:
                matching_prefix += 1
            else:
                break
        r_partial = (matching_prefix / max(len(ground_truth), 1))
        
        # 3. Format & Syntactic Validity
        r_format = 0.0
        if predicted_tokens:
            valid_chars = set("0123456789c") if use_scratchpad else set("0123456789")
            all_valid = all(ch in valid_chars for ch in predicted_tokens)
            if all_valid:
                r_format += 0.5
            # Penalize absurd length
            expected_len = len(ground_truth) * (2 if use_scratchpad else 1) + 2
            if len(predicted_tokens) <= expected_len:
                r_format += 0.5
            else:
                r_format -= 0.5
                
        # 4. Brevity
        r_brevity = -0.05 * max(0, len(predicted_tokens) - len(ground_truth))

        raw_total = (
            self.cfg.w_correct * r_correct
            + self.cfg.w_partial * r_partial
            + self.cfg.w_format * r_format
            + self.cfg.w_brevity * r_brevity
        )
        
        # --- Noise Injection: Gaussian Reward Noise ---
        if self.cfg.reward_noise_std > 0.0:
            raw_total += r.gauss(0.0, self.cfg.reward_noise_std)
            
        shaped_reward = self.exponential_shaping(raw_total)
        
        return {
            "r_correct": r_correct,
            "r_partial": r_partial,
            "r_format": r_format,
            "r_brevity": r_brevity,
            "raw_total": raw_total,
            "reward": shaped_reward,
            "is_correct": is_correct,
            "parsed_pred": parsed_pred,
        }

if __name__ == "__main__":
    engine = ArithRewardEngine(RewardConfig(reward_noise_std=0.2, reward_flip_prob=0.1))
    print("Evaluating clean match:")
    print(engine.evaluate("12+34=", "64", "46"))
    print("\nEvaluating partial match:")
    print(engine.evaluate("123+456=", "974", "579"))
