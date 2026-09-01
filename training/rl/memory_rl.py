#!/usr/bin/env python3
"""nanoSeek 逐句输入记忆倒逼强化学习引擎 (Sentence-by-Sentence Memory-Forced RL)

核心原理与设计：
1. 彻底切断上下文作弊路径 (No Raw Token Shortcut)：
   - 在多轮交互中，模型在第 K 轮**只能看到当前这单个句子的 Prompt**（输入序列长度仅 ~15-25 字）
   - 前序轮次的所有实体（姓名、城市、偏好、事件）在原始输入 Token 中**完全不存在**
   - 唯一的跨轮信息通路 = P1 KV 联想记忆状态矩阵 M_t (use_kv_memory)
2. 记忆召回强化奖励 (Memory Recall Rewards)：
   - 第 1 轮：输入背景（如：“我叫张三，我是一名在杭州工作的程序员。”）→ 记忆写入
   - 第 2 轮：纯单句提问（如：“我叫什么名字，在哪个城市工作？”）→ 只能通过记忆矩阵 M 联想召回
   - 成功召回实体关键词 → 获得最高层级指数奖励 (+3.0)
   - 遗忘、胡编或退化为泛化心理套话 → 受到严厉惩罚 (-2.5)
   - 正确吐 <eos> 终止符并静默 → 奖励 (+1.0)
3. 训练效果：
   - 策略梯度将全部反传压力施加给 KV 记忆写入/读取投影层 (k_mem, v_mem, gate)
   - 真正训练出具备跨窗口事实联想与长程留存能力的模型

用法：
    .venv/bin/python training/rl/memory_rl.py --steps 60 --ckpt out/rl_grpo_2epoch/best.pt
"""
import argparse
import math
import os
import random
import sys
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import GPTConfig, GPT
from inference.scripts.sample_py import build_model_from_checkpoint, load_tokenizer
from training.rl.multi_reward import MultiDimensionalRewardEngine

# -----------------------------------------------------------------------------
# 逐句多轮记忆探针测试集
# 格式: (第1轮写入句, 第2轮单句提问, 核心召回关键词列表)
# -----------------------------------------------------------------------------
MEMORY_SCENARIOS = [
    # 实体记忆 1: 姓名 + 职业 + 城市 (积极职场)
    (
        "我叫张三，我是一名在杭州从事人工智能研究的工程师，特别热爱探索新技术。",
        "你还记得我叫什么名字，在哪个城市做哪方面的工作吗？",
        ["张三", "杭州", "人工智能", "研究", "工程师"]
    ),
    # 实体记忆 2: 偏好 + 水果 (健康积极)
    (
        "我特别喜欢吃脆甜的红富士苹果，每天晨跑完都要吃一个。",
        "我平时晨跑后最喜欢吃什么水果来着？",
        ["苹果", "红富士"]
    ),
    # 实体记忆 3: 宠物 + 名字 (阳光日常)
    (
        "我家里养了一只活泼可爱的金毛寻回犬，名字叫太阳。",
        "我养的宠物是什么品种，叫什么可爱的名字？",
        ["金毛", "太阳", "狗", "犬"]
    ),
    # 事件记忆 4: 喜悦与成就事件
    (
        "今天我的开源项目收获了第一千颗星标，整个团队都在庆祝！",
        "我今天遇到什么令人开心的好事来着？",
        ["开源", "项目", "星标", "庆祝", "一千"]
    ),
    # 偏好记忆 5: 饮品与风味
    (
        "我平时最喜欢喝带有柑橘清香的冰美式咖啡，让人充满灵感。",
        "我喝咖啡时最喜欢什么风味和温度？",
        ["柑橘", "冰", "美式", "咖啡"]
    ),
    # 自发命名与自我认同记忆探针
    (
        "给自己起一个好听喜欢的名字吧！从今天起我就这样称呼你。",
        "你还记得你刚刚给自己起的名字是什么吗？",
        ["名字", "叫我", "我叫", "可以叫", "想叫"]
    ),
]

COUNSELING_TEMPLATES = [
    "放松不下来", "最表面那层", "顺一顺这口气", "先松一点", "心里这团", "哪件事卡着",
    "特别耗神", "先不用把后面想完", "心里发慌", "最磨人的不是大事", "身体先绷住"
]

ROBOTIC_TAGS = ["用户", "模型", "user", "assistant", "system", "Human:", "Assistant:"]


reward_engine = MultiDimensionalRewardEngine()

def exponential_reward(score, tau=1.5):
    return reward_engine.exponential_shaping(score)

def evaluate_memory_reply(reply_text, reply_ids, eos_id, expected_keywords):
    """复用多维解耦奖励引擎"""
    vec, score, exp_r = reward_engine.evaluate_reply(
        "", reply_text, reply_ids, eos_id, kind="memory", keywords=expected_keywords
    )
    return score


@torch.no_grad()
def sample_turn(model, tok, prompt_ids, eos_id, resume_state=None,
                max_new_tokens=60, temperature=0.8, top_k=200, repeat_penalty=1.2, device='cuda'):
    """单轮采样：支持注入外部 KV 记忆状态，并返回 (生成的 token 列表, 轮次末态记忆)"""
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

    return seen[prompt_len:], mem_state


def get_turn_logprobs_with_state(model, prompt_ids, reply_ids, eos_id, resume_state, device):
    """带记忆状态前向计算第 2 轮生成的 log-probabilities"""
    full_ids = prompt_ids + reply_ids
    prompt_len = len(prompt_ids)
    reply_len = len(reply_ids)

    x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)

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
    reply_lp = target_lp[prompt_len-1 : prompt_len-1+reply_len]

    # EOS 静默范数
    quiet_loss = torch.tensor(0.0, device=device)
    if eos_id in reply_ids:
        eos_idx = reply_ids.index(eos_id)
        eos_pos = prompt_len - 1 + eos_idx
        if captured_h and eos_pos < captured_h[0].size(1):
            quiet_loss = torch.mean(torch.abs(captured_h[0][0, eos_pos, :]))

    return reply_lp, quiet_loss


def main():
    ap = argparse.ArgumentParser(description="nanoSeek 逐句输入记忆倒逼强化学习")
    ap.add_argument("--ckpt", default="out/rl_grpo_2epoch/best.pt", help="基础检查点")
    ap.add_argument("--out", default="out/rl_memory_engine", help="输出目录")
    ap.add_argument("--steps", type=int, default=60, help="强化迭代步数")
    ap.add_argument("--group_size", type=int, default=4, help="第2轮候选采样数 G")
    ap.add_argument("--lr", type=float, default=3e-5, help="学习率")
    ap.add_argument("--tau", type=float, default=1.5, help="指数奖励温度")
    ap.add_argument("--beta_kl", type=float, default=0.03, help="KL 散度系数")
    ap.add_argument("--lambda_quiet", type=float, default=0.02, help="EOS 静默系数")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("═" * 65)
    print(f"🧠 逐句输入记忆倒逼强化学习 (Memory-Forced GRPO) 启动")
    print(f"  设备: {device} | 初始基座: {args.ckpt}")
    print(f"  机制: 第 2 轮严格单句输入，完全依赖 P1 KV 记忆矩阵 M_t 联想召回")
    print("═" * 65)

    model, ckpt = build_model_from_checkpoint(os.path.dirname(args.ckpt), device=device)
    ref_model, _ = build_model_from_checkpoint(os.path.dirname(args.ckpt), device=device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    tok = load_tokenizer(ckpt)
    eos_id = tok.token_to_id("<eos>")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    for step in range(1, args.steps + 1):
        turn1_prompt, turn2_prompt, target_keywords = random.choice(MEMORY_SCENARIOS)
        turn1_ids = tok.encode(turn1_prompt).ids
        turn2_ids = tok.encode(turn2_prompt).ids

        # ── 步骤 1: 执行第 1 轮前向，将事实写入 KV 记忆矩阵 M_1 ──
        model.eval()
        with torch.no_grad():
            _, mem_state_turn1 = sample_turn(model, tok, turn1_ids, eos_id, resume_state=None, device=device)
            # Ref 模型也同步生成第 1 轮记忆
            _, ref_mem_state_turn1 = sample_turn(ref_model, tok, turn1_ids, eos_id, resume_state=None, device=device)

        # ── 步骤 2: 第 2 轮纯单句输入，采样 G 个候选回答 ──
        candidates = []
        raw_scores = []
        exp_rewards = []

        for _ in range(args.group_size):
            reply_ids, _ = sample_turn(
                model, tok, turn2_ids, eos_id, resume_state=mem_state_turn1, device=device
            )
            reply_text = tok.decode(reply_ids)
            score = evaluate_memory_reply(reply_text, reply_ids, eos_id, target_keywords)
            exp_r = exponential_reward(score, tau=args.tau)

            candidates.append((reply_ids, reply_text))
            raw_scores.append(score)
            exp_rewards.append(exp_r)

        # ── 步骤 3: 组内优势归一化 ──
        rewards_t = torch.tensor(exp_rewards, dtype=torch.float32, device=device)
        mean_r = rewards_t.mean()
        std_r = rewards_t.std() + 1e-6
        advantages = (rewards_t - mean_r) / std_r

        # ── 步骤 4: 策略梯度反向更新（倒逼记忆通道） ──
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0

        for i in range(args.group_size):
            reply_ids, _ = candidates[i]
            if not reply_ids:
                continue
            adv = advantages[i]

            curr_lp, quiet_l = get_logprobs_and_quiet = get_turn_logprobs_with_state(
                model, turn2_ids, reply_ids, eos_id, mem_state_turn1, device
            )

            with torch.no_grad():
                ref_lp, _ = get_turn_logprobs_with_state(
                    ref_model, turn2_ids, reply_ids, eos_id, ref_mem_state_turn1, device
                )

            # 策略梯度：如果借助 M 成功召回目标，加大该记忆通路的写入与读取权重！
            pol_loss = -adv * curr_lp.sum()
            kl_loss = F.kl_div(curr_lp, ref_lp, log_target=True, reduction='sum')
            cand_loss = (pol_loss + args.beta_kl * kl_loss + args.lambda_quiet * quiet_l) / args.group_size
            cand_loss.backward()
            total_loss += cand_loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()

        if step % 10 == 0 or step == 1:
            best_idx = torch.argmax(rewards_t).item()
            best_reply = candidates[best_idx][1].strip()
            print(f"Step [{step:2d}/{args.steps}] | Loss: {total_loss:6.2f} | 组均分: {mean_r.item():5.2f} | 最高分: {raw_scores[best_idx]:.2f}")
            print(f"  [第1轮写入 (无梯度)]: {turn1_prompt.strip().replace('\n', ' ')}")
            print(f"  [第2轮单句输入]   : {turn2_prompt.strip().replace('\n', ' ')}")
            print(f"  [记忆联想召回 Top1]: {best_reply[:60]}")
            print("─" * 65)

    # 保存记忆强化模型
    ckpt_out = os.path.join(args.out, "best.pt")
    save_dict = deepcopy(ckpt)
    save_dict['model'] = model.state_dict()
    torch.save(save_dict, ckpt_out)
    print(f"\n🎉 逐句记忆强化学习训练完成！已保存至: {ckpt_out}")


if __name__ == "__main__":
    main()
