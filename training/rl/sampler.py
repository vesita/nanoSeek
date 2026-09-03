#!/usr/bin/env python3
"""组采样器 (Sampler): 从 grpo_char.py 抽出的全并行向量化双停止符采样。

职责单一: 给定 Policy 模型, 对单个 prompt 并行采样 G 个候选回复
(在 <eos> 收尾 / <cont> 递回 两处停止), 与奖励评估 / GRPO 优势合成完全解耦。
"""
import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def sample_candidates_batch(model, tok, prompt_ids, eos_id, group_size=4,
                            max_new_tokens=55, temperature=1.0, top_k=200,
                            repeat_penalty=1.4, device='cuda', cont_id=None):
    """组内 G 个候选回复全并行 GPU 批处理采样 (速度提升 3~5 倍)。

    在 <eos>(收尾) 与 <cont>(说完请继续/递回) 两处停止 (dev-notes/61 §4)；
    cont_id 为 None 时只认 <eos> (向后兼容)。
    """
    stop_ids = {eos_id}
    if cont_id is not None:
        stop_ids.add(cont_id)
    prompt_t = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    prompt_len = len(prompt_ids)

    idx = prompt_t.unsqueeze(0).expand(group_size, -1).clone()
    finished = torch.zeros(group_size, dtype=torch.bool, device=device)
    # 记录每条序列的终止符(收尾<eos>或递回<cont>); 未终止时保留 <eos> 作填充
    term_tok = torch.full((group_size,), eos_id, dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        v = logits[:, -1, :].clone() / temperature

        if repeat_penalty > 1.0:
            seen_mask = torch.zeros_like(v, dtype=torch.bool)
            seen_mask.scatter_(1, idx, True)
            v = torch.where(seen_mask, torch.where(v >= 0, v / repeat_penalty, v * repeat_penalty), v)

        if top_k is not None:
            topv, _ = torch.topk(v, top_k, dim=-1)
            v[v < topv[:, -1:]] = -float("Inf")

        probs = F.softmax(v, dim=-1)
        nxt = torch.multinomial(probs, 1)  # GPU 原生并发采样，零 CPU 同步

        # 填充用"各自序列的终止符", 避免 <cont> 收尾被后续填充写成 <eos>
        nxt = torch.where(finished.unsqueeze(1), term_tok.unsqueeze(1), nxt)
        idx = torch.cat((idx, nxt), dim=1)

        just_stopped = (~finished) & (nxt.squeeze(1) == eos_id)
        if cont_id is not None:
            just_stopped = just_stopped | ((~finished) & (nxt.squeeze(1) == cont_id))
        term_tok = torch.where(just_stopped, nxt.squeeze(1), term_tok)
        finished = finished | just_stopped
        if finished.all():
            break

    results = []
    for b in range(group_size):
        gen = idx[b, prompt_len:].tolist()
        # 在首个停止符处截断 (eos 或 cont), 只保留该序列自己的终止符
        first_stop = min((gen.index(s) for s in stop_ids if s in gen), default=None)
        if first_stop is not None:
            gen = gen[:first_stop + 1]
        results.append(gen)
    return results


def get_token_logprobs_and_hidden(model, full_ids, device):
    """计算全序列 token 的 log-probabilities 并捕获最终归一化隐藏状态 (供 quiet loss)"""
    x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)

    captured_h = []
    def hook_fn(module, inp, out):
        captured_h.append(out)

    handle = model.transformer.ln_f.register_forward_hook(hook_fn)
    logits, _ = model(x, targets=y)
    handle.remove()

    log_probs = F.log_softmax(logits, dim=-1)
    target_logprobs = log_probs.gather(2, y.unsqueeze(-1)).squeeze(-1).squeeze(0)
    h_f = captured_h[0] if captured_h else None
    return target_logprobs, h_f


def compute_quiet_loss_from_hidden(h_f, eos_idx_in_reply, prompt_len):
    """从捕获的隐藏状态中提取 EOS 位置的 L1 能量范数"""
    if h_f is None or eos_idx_in_reply < 0:
        return torch.tensor(0.0)
    eos_pos = prompt_len - 1 + eos_idx_in_reply
    if eos_pos < h_f.size(1):
        eos_vec = h_f[0, eos_pos, :]
        return torch.mean(torch.abs(eos_vec))
    return torch.tensor(0.0)


# -----------------------------------------------------------------------------
# 奖励塑形 (统一入口, 从 grpo_char.py 抽出)
# -----------------------------------------------------------------------------
def exponential_reward_shaping(score, tau=1.5, max_neg=-20.0):
    """指数分布奖励塑形：R = sign(s) * (exp(|s| / tau) - 1)。

    NOTE(反坍缩平衡): 指数塑形对负尾无界放大 —— 一次 -10.6 的降分可被放大到
    exp_R≈-3271, 让回退次的一两步被绝对主导, 反而把模型推向"短回复 / 空回复"
    这类低绝对分但大量叠加罚分的坍缩解。这里对负值施加 `max_neg` 幅值上限,
    仅截负尾、不碰正尾: abs(R) = min(exp(|s|/tau)-1, |max_neg|)。
    """
    sign = 1.0 if score >= 0 else -1.0
    mag = math.exp(abs(score) / tau) - 1.0
    if max_neg is not None and sign < 0:
        mag = min(mag, abs(max_neg))
    return sign * mag


def log_reward_shaping(score, c=4.0):
    """对数压缩奖励塑形：R = sign(s) * log1p(|s| / c)"""
    sign = 1.0 if score >= 0 else -1.0
    return sign * math.log1p(abs(score) / c)


def reward_shaping(score, shape="exp", tau=1.5, c=4.0, max_neg=-20.0):
    """按 --shape 分派奖励塑形形态 (统一入口)。

    max_neg: exp 塑形的负尾幅值上限 (仅截负值尾部, 见 exponential_reward_shaping)。
    """
    if shape == "log":
        return log_reward_shaping(score, c=c)
    elif shape == "tanh":
        return math.tanh(score / c) * c  # 有界到 ±c
    else:
        return exponential_reward_shaping(score, tau=tau, max_neg=max_neg)


__all__ = [
    "sample_candidates_batch", "get_token_logprobs_and_hidden",
    "compute_quiet_loss_from_hidden",
    "exponential_reward_shaping", "log_reward_shaping", "reward_shaping",
]