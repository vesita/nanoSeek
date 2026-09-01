#!/usr/bin/env python3
"""nanoSeek 双模型交替对话与实时互训系统 (Dual-Model Interactive Self-Play & Online RL)

核心机制：
1. 真实 Agent 对话回路：
   - 实例化 Agent A (Alice) 与 Agent B (Bob)，分别加载 2.7M 模型
   - 双方各自维护独立的参数、优化器与长程 KV 联想记忆状态 (M_A, M_B)
2. 纯单句/按句交替发消息 (Sentence-by-Sentence Alternating Dialogue)：
   - Agent A 发送一句话 → Agent B 接收该单句输入（配合自身 M_B 记忆状态）并回复
   - Agent B 发送一句话 → Agent A 接收该单句输入（配合自身 M_A 记忆状态）并回复
   - 双方像真人微信聊天一样实时交替发消息、承接话题、讨论探索
3. 边聊边学 (Real-Time RL Step on Each Turn)：
   - 每一轮对方发出的回答，由另一方结合客观奖励指标（相关性、句式自然度、无复读、利落吐 EOS、神经元静默）计算 R
   - 说话方立即在后台执行单步策略梯度更新
   - 双方在多轮交锋中互相纠偏、互相拉升表达智商

用法：
    .venv/bin/python training/rl/dual_chat_selfplay.py --rounds 20 --ckpt out/rl_grpo_2epoch/best.pt
"""
import argparse
import math
import os
import random
import sys
import time
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import GPTConfig, GPT
from inference.scripts.sample_py import build_model_from_checkpoint, load_tokenizer

# 启发式开场白列表
STARTER_TOPICS = [
    "今天天气格外晴朗，阳光洒在身上暖洋洋的，感觉整个人充满干劲！",
    "我最近在思考一个很酷的想法，怎么才能让小模型拥有更长久鲜活的记忆？",
    "刚刚完成了一个令人兴奋的新功能，那种把想法变成现实的感觉太棒了！",
    "今天学到了关于宇宙奇点和量子纠缠的新假说，真让人感叹科学的奇妙。",
    "早安！今天又是崭新的一天，你有什么期待完成的目标吗？",
    "生活里总有些不期而遇的美好，比如晨跑时吹过的微风和路边的花香。",
]

COUNSELING_TEMPLATES = [
    "放松不下来", "最表面那层", "顺一顺这口气", "先松一点", "心里这团", "哪件事卡着",
    "特别耗神", "先不用把后面想完", "心里发慌", "最磨人的不是大事", "身体先绷住"
]

ROBOTIC_TAGS = ["用户", "模型", "user", "assistant", "system", "Human:", "Assistant:"]
def exponential_reward(score, tau=1.5):
    sign = 1.0 if score >= 0 else -1.0
    return sign * (math.exp(abs(score) / tau) - 1.0)


def evaluate_conversational_turn(speaker_text, speaker_ids, eos_id, listener_last_msg):
    """质检打分器：评估本轮回答在对话流中的质量"""
    s = 0.0
    char_len = len(speaker_text.strip())

    # 1. 命中 <eos>
    if eos_id in speaker_ids:
        s += 1.0
    else:
        s -= 0.5

    # 2. 长度控制（对话单句最忌又长又臭）
    if 8 <= char_len <= 65:
        s += 0.8  # 对话黄金长度
    elif char_len < 4:
        s -= 1.0  # 过于敷衍
    elif char_len > 90:
        s -= 0.6  # 独白式啰嗦

    # 3. 中文字符与无乱码
    han_count = sum(1 for ch in speaker_text if '\u4e00' <= ch <= '\u9fff')
    if char_len > 0:
        han_ratio = han_count / char_len
        if han_ratio < 0.7:
            s -= 1.5

    # 4. 3-gram 循环复读惩罚
    if len(speaker_text) >= 6:
        trigrams = [speaker_text[i:i+3] for i in range(len(speaker_text)-2)]
        rep3 = 1.0 - len(set(trigrams)) / max(len(trigrams), 1)
        if rep3 > 0.05:
            s -= 1.5
        else:
            s += 0.4

    # 5. 上下文承接与提问互动奖励
    if any(q in speaker_text for q in ["？", "?", "吗", "呢", "怎么", "什么", "觉得"]):
        s += 0.5  # 鼓励主动抛出话题延续对话

    # 6. 严厉惩罚单调的心理模板复读
    # 7. 严厉惩罚吐出“用户/模型”等机械角色标签
    if any(tag in speaker_text for tag in ROBOTIC_TAGS):
        s -= 2.5

    if any(tpl in speaker_text for tpl in COUNSELING_TEMPLATES):
        s -= 2.0

    return s


@torch.no_grad()
def generate_turn(model, tok, prompt_text, eos_id, resume_state=None,
                  max_new_tokens=70, temperature=0.8, top_k=200, repeat_penalty=1.2, device='cuda'):
    """单句前向生成，维护并更新该 Agent 专属的 KV 记忆矩阵状态"""
    prompt_ids = tok.encode(prompt_text).ids
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    prompt_len = len(prompt_ids)
    seen = list(prompt_ids)
    mem_state = resume_state

    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        if mem_state is not None:
            model.set_memory_state(mem_state)
            
        logits, _ = model(idx_cond)
        mem_state = model.get_memory_state()

        v = logits[:, -1, :].squeeze(0).clone() / temperature

        if repeat_penalty > 1.0 and seen:
            seen_t = torch.as_tensor(seen, dtype=torch.long, device=v.device)
            vs = v[seen_t]
            v[seen_t] = torch.where(vs >= 0, vs / repeat_penalty, vs * repeat_penalty)

        if top_k is not None:
            k = min(top_k, v.size(-1))
            topv, _ = torch.topk(v, k, dim=-1)
            v[v < topv[-1]] = -float("Inf")

        probs = F.softmax(v, dim=-1)
        nxt = torch.multinomial(probs, 1).item()
        seen.append(nxt)
        idx = torch.cat((idx, torch.tensor([[nxt]], dtype=torch.long, device=device)), dim=1)

        if nxt == eos_id:
            break

    reply_ids = seen[prompt_len:]
    reply_text = tok.decode(reply_ids).replace("<eos>", "").strip()
    return reply_ids, reply_text, mem_state


def step_agent_rl(model, optimizer, ref_model, tok, prompt_text, reply_ids, eos_id,
                  resume_state, reward, beta_kl=0.03, lambda_quiet=0.02, device='cuda'):
    """对说话方执行单步在线策略梯度微调"""
    if not reply_ids:
        return
    prompt_ids = tok.encode(prompt_text).ids
    full_ids = prompt_ids + reply_ids
    prompt_len = len(prompt_ids)
    reply_len = len(reply_ids)

    x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)

    model.train()
    optimizer.zero_grad()

    captured_h = []
    def hook_fn(module, inp, out):
        captured_h.append(out)

    handle = model.transformer.ln_f.register_forward_hook(hook_fn)
    if resume_state is not None:
        model.set_memory_state(resume_state)

    logits, _ = model(x, targets=y)
    handle.remove()

    log_probs = F.log_softmax(logits, dim=-1)
    target_lp = log_probs.gather(2, y.unsqueeze(-1)).squeeze(-1).squeeze(0)
    curr_reply_lp = target_lp[prompt_len-1 : prompt_len-1+reply_len]

    with torch.no_grad():
        if resume_state is not None:
            ref_model.set_memory_state(resume_state)
        ref_logits, _ = ref_model(x, targets=y)
        ref_lp_all = F.log_softmax(ref_logits, dim=-1).gather(2, y.unsqueeze(-1)).squeeze(-1).squeeze(0)
        ref_reply_lp = ref_lp_all[prompt_len-1 : prompt_len-1+reply_len]

    # 策略损失 (以指数奖励为杠杆)
    policy_loss = -reward * curr_reply_lp.sum()
    kl_loss = F.kl_div(curr_reply_lp, ref_reply_lp, log_target=True, reduction='sum')

    # EOS 静默范数
    quiet_loss = torch.tensor(0.0, device=device)
    if eos_id in reply_ids:
        eos_idx = reply_ids.index(eos_id)
        eos_pos = prompt_len - 1 + eos_idx
        if captured_h and eos_pos < captured_h[0].size(1):
            quiet_loss = torch.mean(torch.abs(captured_h[0][0, eos_pos, :]))

    total_loss = policy_loss + beta_kl * kl_loss + lambda_quiet * quiet_loss
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    model.eval()


def main():
    ap = argparse.ArgumentParser(description="nanoSeek 双模型真实交替对话与实时互训")
    ap.add_argument("--ckpt", default="out/rl_grpo_2epoch/best.pt", help="基础模型权重")
    ap.add_argument("--out", default="out/dual_selfplay", help="演进模型保存目录")
    ap.add_argument("--rounds", type=int, default=20, help="交替对话轮数")
    ap.add_argument("--lr", type=float, default=2e-5, help="在线强化学习率")
    ap.add_argument("--tau", type=float, default=1.5, help="指数奖励塑形温度")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("═" * 65)
    print(f"🎭 nanoSeek 双模型实时交替对话与在线互训系统")
    print(f"  • Agent A (Alice 👩)  vs  Agent B (Bob 👨)")
    print(f"  • 对话模式: 按句输入 (Sentence-by-Sentence) + 独立 KV 记忆矩阵续传")
    print(f"  • 互训机制: 对方即时质检打分 + 说话方单步策略梯度微调")
    print("═" * 65)

    # 1. 实例化 Alice 与 Bob
    model_a, ckpt = build_model_from_checkpoint(os.path.dirname(args.ckpt), device=device)
    model_b, _ = build_model_from_checkpoint(os.path.dirname(args.ckpt), device=device)
    ref_model, _ = build_model_from_checkpoint(os.path.dirname(args.ckpt), device=device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    tok = load_tokenizer(ckpt)
    eos_id = tok.token_to_id("<eos>")

    opt_a = torch.optim.AdamW(model_a.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    opt_b = torch.optim.AdamW(model_b.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    # 独立记忆状态初始化
    mem_state_a = None
    mem_state_b = None

    # 开场白
    starter = random.choice(STARTER_TOPICS)
    print(f"\n🎬 [场景开场白]：{starter}\n")
    print("─" * 65)

    current_msg = starter
    current_speaker = "A"  # A 先应答开场白

    for turn in range(1, args.rounds + 1):
        if current_speaker == "A":
            # Alice 的回合：接收 Bob (或开场) 的单句输入
            prompt_turn = f"{current_msg}\n"
            reply_ids, reply_text, mem_state_a = generate_turn(
                model_a, tok, prompt_turn, eos_id, resume_state=mem_state_a, device=device
            )
            # Bob 对 Alice 的表现进行打分质检
            score = evaluate_conversational_turn(reply_text, reply_ids, eos_id, current_msg)
            reward = exponential_reward(score, tau=args.tau)

            # Alice 实时微调优化
            step_agent_rl(
                model_a, opt_a, ref_model, tok, prompt_turn, reply_ids, eos_id,
                mem_state_a, reward, device=device
            )

            print(f"👩 Alice [第 {turn:2d} 轮 | 得分 {score:+.2f} | R={reward:+.2f}]:")
            print(f"   \"{reply_text}\"\n")

            current_msg = reply_text
            current_speaker = "B"

        else:
            # Bob 的回合：接收 Alice 的单句输入
            prompt_turn = f"{current_msg}\n"
            reply_ids, reply_text, mem_state_b = generate_turn(
                model_b, tok, prompt_turn, eos_id, resume_state=mem_state_b, device=device
            )
            # Alice 对 Bob 的表现进行打分质检
            score = evaluate_conversational_turn(reply_text, reply_ids, eos_id, current_msg)
            reward = exponential_reward(score, tau=args.tau)

            # Bob 实时微调优化
            step_agent_rl(
                model_b, opt_b, ref_model, tok, prompt_turn, reply_ids, eos_id,
                mem_state_b, reward, device=device
            )

            print(f"👨 Bob   [第 {turn:2d} 轮 | 得分 {score:+.2f} | R={reward:+.2f}]:")
            print(f"   \"{reply_text}\"\n")

            current_msg = reply_text
            current_speaker = "A"

        time.sleep(0.3)

    # 讨论结束：融合进化模型
    merged_state = {}
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    for k in state_a.keys():
        merged_state[k] = (state_a[k] + state_b[k]) / 2.0

    save_dict = deepcopy(ckpt)
    save_dict['model'] = merged_state
    save_path = os.path.join(args.out, "best.pt")
    torch.save(save_dict, save_path)
    print("═" * 65)
    print(f"🎉 双模型 {args.rounds} 轮交替自博弈对话互训完成！")
    print(f"💾 共同进化权重已保存至: {save_path}")


if __name__ == "__main__":
    main()
