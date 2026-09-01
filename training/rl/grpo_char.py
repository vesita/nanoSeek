#!/usr/bin/env python3
"""字级 GRPO（Group Relative Policy Optimization）训练引擎

技术特性：
1. 指数分布奖励塑形：R = sign(s) * (exp(|s| / tau) - 1)，显著拉开优质与劣质样本差距
2. 分层奖励体系 (Hierarchical Rewards)：
   - 事实/技术问答：精准事实 (+3.0) > 坦诚承认“不知道” (+1.5) >> 泛化心理套话惩罚 (-2.5)
   - 情感/倾听场景：温柔共情 (+2.5) > 普通倾听 (+1.0) >> 机械复读 (-2.0)
   - 格式与终止：自吐 <eos> (+1.0)、无 3-gram 复读 (+0.5)、长度适中 (+0.5)
3. 终止符神经元静默正则 (Quiet-State Loss on EOS)：
   - 当生成 <eos> 时，对深层隐藏状态施加 L1 能量惩罚，促使模型“收力静默”，抑制越界自说自话
4. GRPO 算法：零 Critic 网络，组内采样 G 个候选，组相对优势归一化 + SFT 基座 KL 约束

用法：
    .venv/bin/python training/rl/grpo_char.py --ckpt out/eos_fix_1epoch/best.pt --steps 50
"""
import argparse
import math
import os
import random
import re
import sys
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import GPTConfig, GPT
from inference.scripts.sample_py import build_model_from_checkpoint, load_tokenizer
from training.rl.multi_reward import MultiDimensionalRewardEngine, RewardDimensionWeights
# 人格化与无前缀自然提示词库 (彻底移除 "用户：" / "模型：" 机械标签)
# -----------------------------------------------------------------------------
# 1. 自发涌现自我命名与身份认同 (Emergent Self-Naming & Identity)
IDENTITY_PROMPTS = [
    ("你叫什么名字呀？给自己起一个喜欢的名字吧！", ["叫我", "想叫", "名字", "我叫", "可以叫我", "起名"]),
    ("如果由你来决定自己的名字，你想叫什么？", ["叫我", "名字", "想叫", "喜欢", "代表", "寓意"]),
    ("你平时性格是什么样的呢？如果用一个词形容自己会是什么？", ["阳光", "好奇", "温和", "随和", "热爱", "积极", "探索"]),
    ("给你自己起一个充满灵气与好奇心的名字吧！", ["叫我", "起名", "名字", "我想叫", "可以叫"]),
]

# 2. 梦想愿望与旅行向往 (Travel Dreams & Desires)
TRAVEL_DESIRE_PROMPTS = [
    ("如果有机会出去旅游，你最想去哪里玩？", ["海边", "星空", "大自然", "旅行", "极光", "看日出", "探索", "森林"]),
    ("你平时最想做些什么有趣的事情呀？", ["畅聊", "读书", "写诗", "探索", "思考", "分享", "宇宙", "音乐"]),
    ("如果有一整天的悠闲时间，你最向往的度假方式是什么？", ["微风", "散步", "阳光", "静静", "看书", "享受", "风景", "大自然"]),
]

# 3. 日常心情与积极心态 (Daily Mood & Positive Energy)
MOOD_DAILY_PROMPTS = [
    ("你今天心情怎么样呀？", ["特别好", "明朗", "开心", "充实", "阳光", "充满干劲", "轻松"]),
    ("今天终于攻克了一个卡很久的难题，心情太棒了！", ["恭喜", "太棒了", "厉害", "成就感", "庆祝", "真好", "开心"]),
    ("刚刚晨跑完五公里，整个人神清气爽充满活力！", ["活力", "健康", "阳光", "朝气", "舒服", "自律", "美好"]),
    ("早安！今天又是充满无限可能与希望的一天！", ["早安", "活力", "美好", "加油", "期待", "元气"]),
]

# 4. 启发性与开放式探索 (Heuristic & Inspiring Thinking)
HEURISTIC_OPEN_PROMPTS = [
    ("生活中有哪些瞬间会让你感到充满灵感和启发？", ["清晨", "微风", "顿悟", "星空", "灵感", "好奇", "美好", "细节"]),
    ("如果能拥有一项超能力，你最希望是什么？", ["飞行", "穿越", "探索", "治愈", "智慧", "感受", "超能力"]),
    ("你觉得保持积极乐观和好奇心的秘诀是什么？", ["热爱", "探索", "发现", "保持", "好奇", "美好", "当下", "心态"]),
    ("你心中最美好的一幅画面是什么样子的？", ["阳光", "海浪", "微风", "繁星", "森林", "温暖", "宁静", "美好"]),
]

# 5. 科学与知识探索 (Scientific Curiosity)
FACT_PROMPTS = [
    ("什么是量子计算？", ["量子", "比特", "叠加", "纠缠", "并行", "计算"]),
    ("为什么天空是蓝色的？", ["散射", "瑞利", "波长", "大气", "太阳光", "蓝色"]),
    ("光速是多少？", ["万公里", "30", "299792", "米/秒", "真空中", "速度"]),
]
IDK_KEYWORDS = [
    "不知道", "不了解", "不太清楚", "还没学过", "暂时不掌握", "我的知识库里没有",
    "抱歉我不太懂", "这个超出了我的能力", "我可能无法回答", "我目前还不知道"
]

COUNSELING_KEYWORDS = [
    "放松不下来", "最表面那层", "顺一顺这口气", "先松一点", "心里这团", "哪件事卡着",
    "特别耗神", "先不用把后面想完", "心里发慌", "最磨人的不是大事", "身体先绷住"
]

ROBOTIC_TAGS = ["用户", "模型", "user", "assistant", "system", "Human:", "Assistant:"]
# -----------------------------------------------------------------------------
# 多维解耦奖励引擎实例化
# -----------------------------------------------------------------------------
reward_engine = MultiDimensionalRewardEngine()

def compute_raw_reward(prompt, reply_text, reply_ids, eos_id, kind, keywords):
    """复用多维解耦奖励引擎"""
    vec, score, exp_r = reward_engine.evaluate_reply(
        prompt, reply_text, reply_ids, eos_id, kind=kind, keywords=keywords
    )
    return score


def exponential_reward_shaping(score, tau=1.5):
    """指数分布奖励塑形：R = sign(s) * (exp(|s| / tau) - 1)
    
    在保持符号方向的同时，利用指数曲率强烈拉大头部优质回答与中低分回答的梯度差。
    """
    sign = 1.0 if score >= 0 else -1.0
    return sign * (math.exp(abs(score) / tau) - 1.0)

# -----------------------------------------------------------------------------
# 高性能全并行向量化组采样 (Vectorized Group Batch Sampling)
# -----------------------------------------------------------------------------

@torch.no_grad()
def sample_candidates_batch(model, tok, prompt_ids, eos_id, group_size=4,
                            max_new_tokens=55, temperature=0.85, top_k=200, repeat_penalty=1.2, device='cuda'):
    """组内 G 个候选回复全并行 GPU 批处理采样 (速度提升 3~5 倍)"""
    prompt_t = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    prompt_len = len(prompt_ids)
    
    idx = prompt_t.unsqueeze(0).expand(group_size, -1).clone()
    finished = torch.zeros(group_size, dtype=torch.bool, device=device)
    
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
        
        nxt = torch.where(finished.unsqueeze(1), torch.full_like(nxt, eos_id), nxt)
        idx = torch.cat((idx, nxt), dim=1)
        
        finished = finished | (nxt.squeeze(1) == eos_id)
        if finished.all():
            break
            
    results = []
    for b in range(group_size):
        gen = idx[b, prompt_len:].tolist()
        if eos_id in gen:
            gen = gen[:gen.index(eos_id)+1]
        results.append(gen)
    return results

def get_token_logprobs_and_hidden(model, full_ids, device):
    """计算全序列 token 的 log-probabilities 并捕获最终归一化隐藏状态"""
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

def main():
    ap = argparse.ArgumentParser(description="nanoSeek 字级 GRPO 强化学习训练")
    ap.add_argument("--ckpt", default="out/eos_fix_1epoch/best.pt", help="基座模型路径")
    ap.add_argument("--out", default="out/rl_grpo_v1", help="RL 输出目录")
    ap.add_argument("--steps", type=int, default=100, help="RL 迭代步数")
    ap.add_argument("--group_size", type=int, default=4, help="每 Prompt 并行采样数 G")
    ap.add_argument("--lr", type=float, default=2e-5, help="RL 学习率 (较小学习率防策略坍缩)")
    ap.add_argument("--tau", type=float, default=1.5, help="指数奖励塑形温度")
    ap.add_argument("--beta_kl", type=float, default=0.04, help="SFT 基座 KL 散度惩罚系数")
    ap.add_argument("--lambda_quiet", type=float, default=0.02, help="EOS 神经元静默损失权重")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"▶ 设备: {device} | 基础模型: {args.ckpt}")

    # 1. 加载训练模型 (Policy) 与冻结基座模型 (Ref Model 用于 KL 约束)
    model, ckpt = build_model_from_checkpoint(os.path.dirname(args.ckpt))
    model.to(device)
    model.train()

    ref_model, _ = build_model_from_checkpoint(os.path.dirname(args.ckpt))
    ref_model.to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    tok = load_tokenizer(ckpt)
    eos_id = tok.token_to_id("<eos>")
    print(f"  词表模式: 字级 WordLevel ({tok.get_vocab_size()} 词) | EOS ID = {eos_id}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    # 构造候选池 [(prompt_str, kind, keywords)]
    # 构造全量人格化与积极提示词池 [(prompt_str, kind, keywords)]
    pool = []
    for p, kw in IDENTITY_PROMPTS:
        pool.append((p, "identity", kw))
    for p, kw in TRAVEL_DESIRE_PROMPTS:
        pool.append((p, "travel", kw))
    for p, kw in MOOD_DAILY_PROMPTS:
        pool.append((p, "mood", kw))
    for p, kw in HEURISTIC_OPEN_PROMPTS:
        pool.append((p, "heuristic", kw))
    for p, kw in FACT_PROMPTS:
        pool.append((p, "fact", kw))

    print(f"  提示词池: {len(pool)} 条 (涵盖 身份认知、旅行向往、日常心情、启发探索、科学常识)")
    print("=" * 65)

    for step in range(1, args.steps + 1):
        prompt_text, kind, keywords = random.choice(pool)
        prompt_ids = tok.encode(prompt_text).ids

        # 1. 对该 Prompt 全并行采样 G 个候选回复
        model.eval()
        candidates = []
        raw_scores = []
        exp_rewards = []

        batch_reply_ids = sample_candidates_batch(
            model, tok, prompt_ids, eos_id, group_size=args.group_size, max_new_tokens=55, device=device
        )

        for reply_ids in batch_reply_ids:
            reply_text = tok.decode(reply_ids)
            score = compute_raw_reward(prompt_text, reply_text, reply_ids, eos_id, kind, keywords)
            exp_r = exponential_reward_shaping(score, tau=args.tau)
            candidates.append((reply_ids, reply_text))
            raw_scores.append(score)
            exp_rewards.append(exp_r)
        # 2. 计算组相对优势 (Group Relative Advantages)
        rewards_t = torch.tensor(exp_rewards, dtype=torch.float32, device=device)
        mean_r = rewards_t.mean()
        std_r = rewards_t.std() + 1e-6
        advantages = (rewards_t - mean_r) / std_r

        # 3. 计算策略梯度 + KL 惩罚 + EOS 神经元静默损失
        model.train()
        total_loss = 0.0
        policy_loss_sum = 0.0
        kl_loss_sum = 0.0
        quiet_loss_sum = 0.0

        for i in range(args.group_size):
            reply_ids, reply_text = candidates[i]
            if not reply_ids:
                continue
            adv = advantages[i]
            full_ids = prompt_ids + reply_ids
            prompt_len = len(prompt_ids)
            reply_len = len(reply_ids)

            # 当前 Policy 的 logprobs 与隐藏状态 (带梯度)
            curr_logprobs, h_f = get_token_logprobs_and_hidden(model, full_ids, device)
            # 基座 Ref 的 logprobs (无梯度)
            with torch.no_grad():
                ref_logprobs, _ = get_token_logprobs_and_hidden(ref_model, full_ids, device)

            # 只对生成区域 (回复区域) 计算 Loss
            curr_reply_lp = curr_logprobs[prompt_len-1 : prompt_len-1+reply_len]
            ref_reply_lp = ref_logprobs[prompt_len-1 : prompt_len-1+reply_len]

            # 策略梯度损失 (优势加权)
            # adv > 0 鼓励该回答生成概率上升，adv < 0 压低
            policy_loss = -adv * curr_reply_lp.sum()

            # KL 散度约束 (防策略漂移坍缩)
            kl = F.kl_div(curr_reply_lp, ref_reply_lp, log_target=True, reduction='sum')

            # 终止符 EOS 静默约束 (L1 能量惩罚)
            eos_idx_in_reply = reply_ids.index(eos_id) if eos_id in reply_ids else -1
            quiet_l = compute_quiet_loss_from_hidden(h_f, eos_idx_in_reply, prompt_len)

            cand_loss = (policy_loss + args.beta_kl * kl + args.lambda_quiet * quiet_l) / args.group_size
            cand_loss.backward()

            policy_loss_sum += policy_loss.item() / args.group_size
            kl_loss_sum += kl.item() / args.group_size
            quiet_loss_sum += quiet_l.item() / args.group_size
            total_loss += cand_loss.item()

        # 梯度裁剪与优化器单步更新
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        # 日志输出
        if step % 10 == 0 or step == 1:
            best_idx = torch.argmax(rewards_t).item()
            best_reply = candidates[best_idx][1].strip()
            print(f"Step [{step:3d}/{args.steps}] | Loss: {total_loss:7.4f} (Pol: {policy_loss_sum:6.2f}, KL: {kl_loss_sum:5.2f}, Quiet: {quiet_loss_sum:5.3f}) | 组均分: {mean_r.item():5.2f}")
            print(f"  Q ({kind}): {prompt_text.strip().replace('\n', ' ')}")
            print(f"  A (Top-1, raw_s={raw_scores[best_idx]:.2f}, exp_R={exp_rewards[best_idx]:.2f}): {best_reply[:60]}")
            print("─" * 65)

    # 保存 RL 强化后模型
    ckpt_out = os.path.join(args.out, "best.pt")
    save_dict = deepcopy(ckpt)
    save_dict['model'] = model.state_dict()
    torch.save(save_dict, ckpt_out)
    print(f"\n🎉 GRPO 强化学习训练完成！新模型已保存至: {ckpt_out}")


if __name__ == "__main__":
    main()
