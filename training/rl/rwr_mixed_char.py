#!/usr/bin/env python3
"""奖励引擎 × 词表 混合训练: 语料 CE 与奖励目标同一优化步 (用户提案)。

用户提案 (2026-09-04): "奖励引擎可以和词表混合着学" —— 让模型**同时**学
"词表/语言结构" (语料 next-token CE) 和 "奖励引擎认可的输出" (奖励加权信号),
而不是先 SFT 基座、再 RL 后训两阶段。

实现: loss = λ_ce·CE(语料, assistant-mask) + λ_rl·RL项 + β_kl·KL(→基座) + λ_q·EOS静默

RL 项两种模式 (v4 关键教训, dev-notes/69):
  --rl_mode grpo (推荐): 组相对优势 (可正可负, **主动压制**坏输出) —— 与
     rl_coh_dialog 同机制, 只是额外混入语料 CE。稳定性有保障。
  --rl_mode rwr  : softmax 加权模仿 (权重恒正, 只学正样本不压制负样本)。
     2.7M 小模型实测三次全塌 (v1/v2/v3): 模型采样出的"次优壳"被模仿后自我强化,
     均长→0。保留仅作对照, 不推荐。

对照口径 (同起点同步数): cont_v1_1epoch (SFT基座) → 300 步
  rl_coh_dialog = 纯 GRPO (无语料 CE)
  rl_mixed_v4   = GRPO + 语料 CE (混合)   ← 本脚本 --rl_mode grpo
  rl_mixed_v3   = RWR (无压制, 已塌)

用法 (冒烟):
  HSA_ENABLE_SDMA=0 HSA_OVERRIDE_GFX_VERSION=10.3.0 TMPDIR=/home/vesita/AI/scratch \
  PYTHONUNBUFFERED=1 .venv/bin/python training/rl/rwr_mixed_char.py \
    --ckpt out/cont_v1_1epoch/best.pt --out out/rl_mixed_v4_smoke --steps 30 \
    --rl_mode grpo --coherence_w 0.5

正式 (300 步):
  ... training/rl/rwr_mixed_char.py --ckpt out/cont_v1_1epoch/best.pt \
    --out out/rl_mixed_v4 --steps 300 --rl_mode grpo --coherence_w 0.5
"""
import argparse
import math
import os
import random
import re
import sys
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn.functional as F

from inference.scripts.sample_py import build_model_from_checkpoint, load_tokenizer
from training.rl.sampler import (
    sample_candidates_batch, get_token_logprobs_and_hidden,
    compute_quiet_loss_from_hidden, reward_shaping,
)
from training.rl.reward import RewardEngine, _ngram_set
from training.masking import build_assistant_mask
# 复用 grpo_char 的提示词池 / 语感门控常量
from training.rl.grpo_char import (
    IDENTITY_PROMPTS, TRAVEL_DESIRE_PROMPTS, MOOD_DAILY_PROMPTS,
    HEURISTIC_OPEN_PROMPTS, FACT_PROMPTS, ARITH_PROMPTS,
    COH_THETA, COH_SIGMA, COH_GATE_SIG, mean_ref_reply_logprob,
)
from training.rl.updater import GrpoUpdater

DATA_BIN = "data/chinese/train_char.bin"


def main():
    ap = argparse.ArgumentParser(description="奖励引擎×词表混合训练 (GRPO/RWR + 语料CE)")
    ap.add_argument("--ckpt", default="out/cont_v1_1epoch/best.pt", help="起始基座")
    ap.add_argument("--out", default="out/rl_mixed_v4", help="输出目录")
    ap.add_argument("--ref_base", default=None, help="KL 锚点模型目录 (默认=基座同目录)")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--group_size", type=int, default=4, help="每 prompt 采样候选数 G")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--rl_mode", default="grpo", choices=["grpo", "rwr"],
                    help="grpo=组相对优势(压制坏输出,推荐); rwr=softmax加权模仿(已证塌,对照)")
    ap.add_argument("--lam_ce", type=float, default=0.5, help="语料 CE loss 权重 (混合强度)")
    ap.add_argument("--lam_rl", type=float, default=1.0, help="RL 项权重")
    ap.add_argument("--beta_kl", type=float, default=0.6, help="到基座锚点的 KL 权重")
    ap.add_argument("--lambda_quiet", type=float, default=0.02, help="EOS 静默损失权重")
    ap.add_argument("--tau", type=float, default=1.5, help="指数奖励塑形温度 (grpo 用)")
    ap.add_argument("--winsorize", type=float, default=3.0,
                    help="优势 winsorize (median±k*MAD, 0=关闭)")
    ap.add_argument("--div_weight", type=float, default=1.5, help="组内多样性惩罚 (0=关闭)")
    ap.add_argument("--tau_rw", type=float, default=2.0, help="RWR softmax 温度 (仅 rwr 模式)")
    ap.add_argument("--corpus_bs", type=int, default=8, help="语料 CE batch 大小")
    ap.add_argument("--min_valid", type=int, default=32,
                    help="语料窗口最少有效(assistant)token; 低于重采样 (防 MTP 0/0=NaN)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--repeat_penalty", type=float, default=1.4)
    ap.add_argument("--coherence_w", type=float, default=0.0,
                    help="r_coherence 正向奖励权重 (与 grpo_char 同语义, 0=仅门控)")
    ap.add_argument("--arith_ratio", type=float, default=0.0, help="数学提示词比例")
    ap.add_argument("--corpus_bin", default=DATA_BIN, help="字级语料 bin 路径")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"▶ 设备: {device} | 基座: {args.ckpt} | 输出: {args.out} | RL模式: {args.rl_mode}")

    model, ckpt = build_model_from_checkpoint(os.path.dirname(args.ckpt))
    model.to(device)
    model.train()
    block_size = model.config.block_size

    ref_model = None
    if args.beta_kl > 0:
        ref_dir = os.path.dirname(args.ref_base) if args.ref_base else os.path.dirname(args.ckpt)
        ref_model, _ = build_model_from_checkpoint(ref_dir)
        ref_model.to(device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        print(f"  🔗 KL 锚点: {ref_dir}")

    tok = load_tokenizer(ckpt)
    eos_id = tok.token_to_id("<eos>")
    cont_id = tok.token_to_id("<cont>")
    print(f"  词表: 字级 WordLevel ({tok.get_vocab_size()} 词) | EOS={eos_id} CONT={cont_id}")

    reward_engine = RewardEngine()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    pool = []
    for p, kw in IDENTITY_PROMPTS: pool.append((p, "identity", kw))
    for p, kw in TRAVEL_DESIRE_PROMPTS: pool.append((p, "travel", kw))
    for p, kw in MOOD_DAILY_PROMPTS: pool.append((p, "mood", kw))
    for p, kw in HEURISTIC_OPEN_PROMPTS: pool.append((p, "heuristic", kw))
    for p, kw in FACT_PROMPTS: pool.append((p, "fact", kw))
    arith_pool = [(p, "arith", kw) for p, kw in ARITH_PROMPTS]
    print(f"  提示词池: 常规 {len(pool)} + 数学 {len(arith_pool)} (arith_ratio={args.arith_ratio})")

    corpus = np.memmap(args.corpus_bin, dtype=np.uint16, mode="r")
    n_tokens = len(corpus)
    print(f"  语料: {args.corpus_bin} ({n_tokens:,} token, block={block_size}, assistant-mask CE)")

    def pick_prompt():
        if arith_pool and random.random() < args.arith_ratio:
            return random.choice(arith_pool)
        return random.choice(pool)

    def corpus_batch(bs):
        """assistant-mask 有效 token >= min_valid 才收 (防 MTP 全 -100 目标 0/0=NaN)。"""
        xs, ys = [], []
        for _ in range(bs):
            while True:
                i = int(torch.randint(0, n_tokens - block_size - 1, (1,)))
                y = torch.from_numpy(corpus[i + 1:i + 1 + block_size].astype(np.int64))
                m = build_assistant_mask(y.unsqueeze(0), [eos_id, cont_id], [0])
                if m.sum().item() >= args.min_valid:
                    break
            x = torch.from_numpy(corpus[i:i + block_size].astype(np.int64))
            xs.append(x)
            ys.append(y)
        return (torch.stack(xs).to(device), torch.stack(ys).to(device))

    # 动态 tau EMA (grpo 模式, 与 grpo_char 同)
    ema_abs_scale = 1.0
    dyn_tau_ema = 0.9

    for step in range(1, args.steps + 1):
        # ── A. 语料 CE (词表/语言结构, assistant-mask 与基座 SFT 同目标) ──
        cx, cy = corpus_batch(args.corpus_bs)
        cy_masked = cy.clone()
        cy_masked[~build_assistant_mask(cy, [eos_id, cont_id], [0])] = -100
        _, ce_loss = model(cx, targets=cy_masked)

        # ── B. 采样候选 + 奖励引擎打分 ───────────────────────────────
        prompt_text, kind, keywords = pick_prompt()
        if random.random() >= 0.27:
            prompt_text = f"用户：{prompt_text}\n模型："
        prompt_ids = tok.encode(prompt_text).ids
        prompt_len = len(prompt_ids)

        model.eval()
        batch_reply_ids = sample_candidates_batch(
            model, tok, prompt_ids, eos_id, group_size=args.group_size, max_new_tokens=55,
            temperature=args.temperature, top_k=200, repeat_penalty=args.repeat_penalty,
            device=device, cont_id=cont_id,
        )
        candidates, raw_scores, mlp_vals = [], [], []
        for reply_ids in batch_reply_ids:
            reply_text = tok.decode(reply_ids).replace("<eos>", "").replace("<cont>", "").strip()
            mlp = mean_ref_reply_logprob(ref_model if ref_model is not None else model,
                                         prompt_ids, reply_ids, device) if reply_ids else None
            mlp_vals.append(mlp)
            if args.coherence_w > 0 and mlp is not None:
                z = (mlp - COH_THETA) / COH_SIGMA
                coherence_eff = z * (args.coherence_w / 2.0)
            else:
                coherence_eff = None
            _, score, _ = reward_engine.evaluate_reply(
                prompt_text, reply_text, reply_ids, eos_id, kind=kind,
                keywords=keywords, coherence=coherence_eff, cont_id=cont_id,
            )
            candidates.append((reply_ids, reply_text))
            raw_scores.append(score)

        # ── C. RL 权重: grpo=组相对优势 (可负, 压制) / rwr=softmax (恒正, 模仿) ──
        if args.rl_mode == "grpo":
            if args.dyn_tau if False else True:
                batch_abs = float(np.mean([abs(s) for s in raw_scores])) if raw_scores else 1.0
                ema_abs_scale = dyn_tau_ema * ema_abs_scale + (1.0 - dyn_tau_ema) * max(batch_abs, 0.2)
                tau_eff = args.tau * max(0.7, min(1.4, ema_abs_scale / 1.0))
            else:
                tau_eff = args.tau
            exp_rewards = []
            if args.div_weight > 0:
                ngrams = [_ngram_set(t.strip()) for _, t in candidates]
                for i in range(len(candidates)):
                    a = ngrams[i]
                    best_overlap = 0.0
                    for j in range(len(candidates)):
                        if j == i:
                            continue
                        b = ngrams[j]
                        if not a or not b:
                            continue
                        inter = len(a & b)
                        best_overlap = max(best_overlap, 2.0 * inter / (len(a) + len(b)))
                    penalty = args.div_weight * best_overlap
                    exp_rewards.append(reward_shaping(raw_scores[i] - penalty, shape="exp",
                                                      tau=tau_eff, c=4.0))
            else:
                exp_rewards = [reward_shaping(s, shape="exp", tau=tau_eff, c=4.0)
                               for s in raw_scores]
            advantages = GrpoUpdater.normalize_advantages(exp_rewards).to(device)
            if args.winsorize > 0 and advantages.numel() >= 3:
                med = torch.median(advantages)
                mad = (advantages - med).abs().median() + 1e-6
                advantages = advantages.clamp(med - args.winsorize * mad,
                                              med + args.winsorize * mad)
            # Anti-Dwarf: 空壳优势压到组底 (绝不被强化)
            _adv_bottom = float(advantages.min().item()) - 1.0
            for i, (rid, rt) in enumerate(candidates):
                _t = rt.strip()
                if not _t or (len(_t) <= 2 and not bool(re.fullmatch(r"\d{1,3}", _t))):
                    advantages[i] = _adv_bottom
            # 退化门控 (与 grpo_char 同口径: 用组最佳候选, 非组内最短)
            max_raw = max(raw_scores)
            best_mlp = max((m for m in mlp_vals if m is not None), default=None)
            salad_degen = best_mlp is not None and best_mlp < (COH_THETA - COH_GATE_SIG * COH_SIGMA)
            _best_ix = int(torch.argmax(torch.tensor(exp_rewards)).item())
            _best_len = len(candidates[_best_ix][1].strip())
            _best_is_digit_only = bool(re.fullmatch(r"\d{1,3}", candidates[_best_ix][1].strip()))
            _group_avg_len = float(sum(len(t) for _, t in candidates)) / max(len(candidates), 1)
            content_collapse = (max_raw < 0 and (_best_len <= 1 or _group_avg_len <= 4)
                                and not _best_is_digit_only)
            degenerate = (max_raw < 0 and salad_degen) or content_collapse
            if degenerate:
                advantages = torch.zeros_like(advantages)
                print(f"  ⚠ 组退化 (best_len={_best_len}, 均长={_group_avg_len:.1f}) → 优势归零, 仅语料CE+KL拉回")
        else:
            # rwr (对照): softmax 权重 + Anti-Dwarf 清零 + winsorize
            raw_t = torch.tensor(raw_scores, dtype=torch.float32)
            if args.winsorize > 0 and raw_t.numel() >= 3:
                med = torch.median(raw_t)
                mad = (raw_t - med).abs().median() + 1e-6
                raw_t = raw_t.clamp(med - args.winsorize * mad, med + args.winsorize * mad)
            w = torch.softmax(raw_t / args.tau_rw, dim=0)
            for i, (rid, rt) in enumerate(candidates):
                _t = rt.strip()
                if not _t or (len(_t) <= 2 and not bool(re.fullmatch(r"\d{1,3}", _t))):
                    w[i] = 0.0
            max_raw = max(raw_scores)
            best_mlp = max((m for m in mlp_vals if m is not None), default=None)
            salad_degen = best_mlp is not None and best_mlp < (COH_THETA - COH_GATE_SIG * COH_SIGMA)
            _best_ix = int(torch.argmax(raw_t).item())
            _best_len = len(candidates[_best_ix][1].strip())
            _best_is_digit_only = bool(re.fullmatch(r"\d{1,3}", candidates[_best_ix][1].strip()))
            _group_avg_len = float(sum(len(t) for _, t in candidates)) / max(len(candidates), 1)
            content_collapse = (max_raw < 0 and (_best_len <= 1 or _group_avg_len <= 4)
                                and not _best_is_digit_only)
            degenerate = (max_raw < 0 and salad_degen) or content_collapse
            if degenerate:
                w = torch.zeros_like(w)
                print(f"  ⚠ 组退化 (best_len={_best_len}, 均长={_group_avg_len:.1f}) → RWR 权重清零")
            advantages = w  # rwr 模式: 权重替代优势

        # ── D. RL loss + KL + EOS 静默 (KL/静默全程在场) ──────────────
        model.train()
        G = len(candidates)
        max_rl = max((len(r) for r, _ in candidates if r), default=1)
        new_lp = torch.zeros(G, max_rl, device=device)
        ref_lp = torch.zeros(G, max_rl, device=device)
        mask = torch.zeros(G, max_rl, dtype=torch.bool, device=device)
        quiet_losses = []
        for i, (reply_ids, _) in enumerate(candidates):
            if not reply_ids:
                quiet_losses.append(torch.tensor(0.0, device=device))
                continue
            reply_len = len(reply_ids)
            full_ids = prompt_ids + reply_ids
            curr_logprobs, h_f = get_token_logprobs_and_hidden(model, full_ids, device)
            with torch.no_grad():
                ref_logprobs, _ = get_token_logprobs_and_hidden(ref_model, full_ids, device)
            new_lp[i, :reply_len] = curr_logprobs[prompt_len - 1:prompt_len - 1 + reply_len]
            ref_lp[i, :reply_len] = ref_logprobs[prompt_len - 1:prompt_len - 1 + reply_len]
            mask[i, :reply_len] = True
            term_idx = -1
            for _sid in (eos_id, cont_id):
                if _sid in reply_ids:
                    term_idx = reply_ids.index(_sid)
                    break
            quiet_losses.append(compute_quiet_loss_from_hidden(h_f, term_idx, prompt_len))

        adv = advantages
        adv_exp = adv.unsqueeze(-1)
        pol_loss = -(adv_exp * new_lp * mask.float()).sum() / mask.sum().clamp(min=1)
        kl_tensor = F.kl_div(new_lp, ref_lp, log_target=True, reduction="none")
        kl_loss = (kl_tensor * mask.float()).sum() / mask.sum().clamp(min=1)
        qr = [q for q in quiet_losses]
        quiet = torch.stack([q.to(device) for q in qr]).mean() if qr else torch.tensor(0.0, device=device)

        total = (args.lam_ce * ce_loss + args.lam_rl * pol_loss
                 + args.beta_kl * kl_loss + args.lambda_quiet * quiet)
        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # ── 日志 ──────────────────────────────────────────────────────
        if step % 10 == 0 or step == 1:
            best_idx = int(torch.argmax(torch.tensor(raw_scores)).item())
            best_reply = candidates[best_idx][1].strip()
            n_eos = sum(1 for rid, _ in candidates if eos_id in rid)
            n_cont = sum(1 for rid, _ in candidates if cont_id in rid)
            avg_len = float(sum(len(t) for _, t in candidates)) / max(len(candidates), 1)
            degen_flag = " ⚠退化" if degenerate else ""
            print(f"Step [{step:3d}/{args.steps}] | CE(语料) {ce_loss.item():7.4f} | RL {pol_loss.item():7.4f}"
                  f" | KL {kl_loss.item():6.3f} | Quiet {quiet.item():6.3f} | Total {total.item():7.4f}{degen_flag}")
            print(f"  Term: eos={n_eos}/{G} cont={n_cont}/{G} | 均长 {avg_len:.1f}"
                  f" | 组原始分 {min(raw_scores):.1f}~{max(raw_scores):.1f}")
            print(f"  Q ({kind}): {prompt_text.strip().replace(chr(10), ' ')}")
            print(f"  A (raw_s={raw_scores[best_idx]:.2f}): {best_reply[:60]}")
            print("─" * 65)

    ckpt_out = os.path.join(args.out, "best.pt")
    save_dict = deepcopy(ckpt)
    save_dict["model"] = model.state_dict()
    torch.save(save_dict, ckpt_out)
    print(f"\n🎉 混合训练完成: {ckpt_out}")


if __name__ == "__main__":
    main()
