#!/usr/bin/env python3
"""GRPO（Group Relative Policy Optimization）训练骨架——DeepSeek-R1 风格规则奖励 RL。

设计（dev-notes/28）：
  1. 固定基座 checkpoint（默认当前最优 chinese-reb）
  2. 每个 prompt 采样 G 个回复（镜像 sample_py 解码：温度→重复惩罚→top-k→softmax→multinomial）
  3. 规则奖励（全脚本化，无人工）：
       +1.0  命中 <eos>（自终止）
       +0.5  rep3 < 0.02（不重复）
       +0.3  长度在 [20, 100] token（不啰嗦不敷衍）
       +0.5  出现第二轮（续轮）
       -0.5  中文字符占比 < 0.5（英文乱码惩罚）
  4. 组内归一化优势 adv = (r - mean(r_group)) / std(r_group)
  5. 策略梯度目标 + KL 惩罚回基座（β 防崩），只对生成的 token 位置算
  6. 预算 ≤500 步（远低于 1500 上限），lr 复用 SFT 末段

本次为框架骨架：可运行、可单步前向、可保存 checkpoint；不做效果调优。
用法（项目根目录）：
    .venv/bin/python training/rl/grpo.py --ckpt out/chinese-reb/best.pt --steps 10
"""
import argparse
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
from collections import Counter
from tokenizers import Tokenizer

from model import GPTConfig, GPT

EOS_ID = 3
PROMPTS = [
    "用户：你好\n模型：",
    "用户：最近好累，感觉撑不下去了\n模型：",
    "用户：周末去爬山还是看电影？\n模型：",
    "用户：你会做什么菜？\n模型：",
    "用户：我想把接下来的方向理一理。\n模型：",
]


def load_model(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = dict(ck["model_args"])
    if "use_csa_fused_qkv" not in args:   # 旧 checkpoint → 独立 QKV 布局
        args["use_csa_fused_qkv"] = False
    model = GPT(GPTConfig(**args))
    state = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
             for k, v in ck["model"].items()}
    model.load_state_dict(state)
    return model, ck


def sample_response(model, tok, prompt_ids, max_new=120, temperature=0.9,
                    top_k=200, repeat_penalty=1.2, seed=0):
    """单条采样：返回 (回复 token ids, 是否命中 EOS)。"""
    g = torch.Generator().manual_seed(seed)
    seen = list(prompt_ids)
    eos_hit = False
    for _ in range(max_new):
        idx = torch.tensor([seen[-model.config.block_size:]], dtype=torch.long)
        logits, _ = model(idx)
        v = logits[0, -1, :].clone() / temperature
        for t in seen:
            v[t] = v[t] / repeat_penalty if v[t] >= 0 else v[t] * repeat_penalty
        topv, _ = torch.topk(v, min(top_k, v.size(-1)))
        v[v < topv[-1]] = -float("Inf")
        nxt = int(torch.multinomial(torch.softmax(v, -1), 1, generator=g).item())
        if nxt == EOS_ID:
            eos_hit = True
            break
        seen.append(nxt)
    return seen[len(prompt_ids):], eos_hit


def rule_reward(reply_ids, tok):
    """规则奖励：返回标量。reply_ids 不含 prompt。"""
    r = 0.0
    text = tok.decode(reply_ids)
    # EOS 命中（采样时已记录，这里用文本长度辅助判断不重复计）
    # rep3
    seq = text.replace(" ", "").replace("\n", "")
    if len(seq) >= 6:
        grams = [seq[i:i + 3] for i in range(len(seq) - 2)]
        c = Counter(grams)
        rep3 = sum(1 for x in grams if c[x] > 1) / len(grams)
        if rep3 < 0.02:
            r += 0.5
        else:
            r -= 0.3
    # 长度
    n_tok = len(reply_ids)
    if 20 <= n_tok <= 100:
        r += 0.3
    elif n_tok < 5:
        r -= 0.5
    # 中文字符占比
    han = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    if text and han / max(len(text), 1) < 0.5:
        r -= 0.5
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/chinese-reb/best.pt")
    ap.add_argument("--steps", type=int, default=10, help="框架冒烟步数")
    ap.add_argument("--g", type=int, default=4, help="每组采样数 G")
    ap.add_argument("--kl-beta", type=float, default=0.05, help="KL 惩罚回基座系数")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="/tmp/grpo_smoke")
    a = ap.parse_args()

    model, ck = load_model(a.ckpt)
    ref = GPT(GPTConfig(**{**ck["model_args"], "use_csa_fused_qkv": False})) if "use_csa_fused_qkv" not in ck["model_args"] else None
    # 基座冻结副本（KL 参考）：直接克隆权重
    import copy
    ref = copy.deepcopy(model)
    for p in ref.parameters():
        p.requires_grad = False
    ref.eval()
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)

    tok = Tokenizer.from_file("data/chinese/tokenizer.json")
    print(f"GRPO 骨架：ckpt={a.ckpt} G={a.g} steps={a.steps} kl_beta={a.kl_beta}")

    for step in range(a.steps):
        prompt = random.choice(PROMPTS)
        pid = tok.encode(prompt).ids
        # 1) 采样 G 个回复
        samples = [sample_response(model, tok, pid, seed=step * a.g + gi) for gi in range(a.g)]
        # 2) 规则奖励
        rewards = [rule_reward(r, tok) + (1.0 if eos else 0.0) for r, eos in samples]
        # 3) 组内归一化优势
        mean_r = sum(rewards) / len(rewards)
        std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5 + 1e-6
        advs = [(r - mean_r) / std_r for r in rewards]

        # 4) 策略梯度 + KL 惩罚（只算生成区）
        total_loss = torch.tensor(0.0)
        for (reply_ids, _), adv in zip(samples, advs):
            if not reply_ids:
                continue
            seq = pid + reply_ids
            x = torch.tensor([seq[:-1]], dtype=torch.long)
            y = torch.tensor([seq[1:]], dtype=torch.long)
            # 传 targets 让 forward 计算完整序列 logits（targets=None 时只算最后位置）
            logits, _ = model(x, y)
            logp = F.log_softmax(logits, dim=-1)
            gen_logp = logp[0, len(pid) - 1:, :].gather(1, y[0, len(pid) - 1:].unsqueeze(-1)).squeeze(-1)
            with torch.no_grad():
                ref_logits, _ = ref(x, y)
                ref_logp = F.log_softmax(ref_logits, dim=-1)
                ref_gen_logp = ref_logp[0, len(pid) - 1:, :].gather(1, y[0, len(pid) - 1:].unsqueeze(-1)).squeeze(-1)
            # policy gradient: -adv * logp + kl * (logp_ref - logp)
            loss = (-adv * gen_logp + a.kl_beta * (ref_gen_logp - gen_logp)).mean()
            total_loss = total_loss + loss
        total_loss = total_loss / max(len(samples), 1)
        opt.zero_grad()
        total_loss.backward()
        opt.step()
        if step % 5 == 0 or step == a.steps - 1:
            print(f"step {step}: loss={total_loss.item():.4f} "
                  f"reward_mean={mean_r:.3f} eos_hits={sum(1 for _, e in samples if e)}/{len(samples)}")

    os.makedirs(a.out, exist_ok=True)
    torch.save({"model": model.state_dict(), "model_args": ck["model_args"]},
               os.path.join(a.out, "best.pt"))
    print(f"GRPO 骨架完成 ✅ checkpoint → {a.out}/best.pt")


if __name__ == "__main__":
    main()
