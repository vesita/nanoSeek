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

# -----------------------------------------------------------------------------
# 训练提示词库（分层意图池）
# -----------------------------------------------------------------------------
FACT_PROMPTS = [
    ("用户：什么是量子计算？\n模型：", ["量子", "比特", "叠加", "纠缠", "并行", "计算"]),
    ("用户：为什么天空是蓝色的？\n模型：", ["散射", "瑞利", "波长", "大气", "太阳光", "蓝色"]),
    ("用户：如何学好人工智能？\n模型：", ["数学", "编程", "深度学习", "实践", "模型", "算法"]),
    ("用户：请推荐两本好看的历史小说。\n模型：", ["三国", "大秦", "明朝", "小说", "历史", "传"]),
    ("用户：光速是多少？\n模型：", ["万公里", "30", "299792", "米/秒", "真空中", "速度"]),
]

EMOTION_PROMPTS = [
    ("用户：你好，我今天心情有点低落。\n模型：", ["慢慢", "听着", "不急", "怎么", "放宽", "说说", "在呢", "感受"]),
    ("用户：工作上遇到了点挫折，感觉大家都不理解我。\n模型：", ["辛苦", "委屈", "不容易", "别慌", "压力", "理解", "一件件", "顺一口气"]),
    ("用户：每天都在焦虑未来的事情，好累。\n模型：", ["眼前", "一步步", "允许", "紧绷", "歇一歇", "打转", "小事", "松下来"]),
    ("用户：我其实还好，就是想找人随便聊聊。\n模型：", ["好呀", "陪你", "想聊", "随时", "听你", "日常", "轻松"]),
]

GENERAL_PROMPTS = [
    ("用户：你好！\n模型：", ["你好", "很高兴", "在呢", "有什么", "聊聊"]),
    ("用户：请介绍一下你自己。\n模型：", ["我是", "小寻", "助手", "伙伴", "倾听", "陪伴"]),
]

IDK_KEYWORDS = [
    "不知道", "不了解", "不太清楚", "还没学过", "暂时不掌握", "我的知识库里没有",
    "抱歉我不太懂", "这个超出了我的能力", "我可能无法回答", "我目前还不知道"
]

COUNSELING_KEYWORDS = [
    "放松不下来", "最表面那层", "顺一顺这口气", "先松一点", "心里这团", "哪件事卡着",
    "特别耗神", "先不用把后面想完", "心里发慌", "最磨人的不是大事", "身体先绷住"
]

# -----------------------------------------------------------------------------
# 奖励函数族
# -----------------------------------------------------------------------------

def compute_raw_reward(prompt, reply_text, reply_ids, eos_id, kind, keywords):
    """计算单条回答的基础奖励得分 s (未经过指数塑形)"""
    s = 0.0
    
    # 1. 基础格式与收尾奖励
    hit_eos = (eos_id in reply_ids)
    if hit_eos:
        s += 1.0  # 命中终止符
    else:
        s -= 0.5  # 跑满长度未主动终止惩罚
        
    char_len = len(reply_text.strip())
    if 10 <= char_len <= 80:
        s += 0.5  # 长度黄金区间
    elif char_len < 5:
        s -= 1.0  # 过于敷衍/过短
    elif char_len > 120:
        s -= 0.3  # 啰嗦

    # 中文字符占比检测（防英文/乱码碎片）
    han_count = sum(1 for ch in reply_text if '\u4e00' <= ch <= '\u9fff')
    if char_len > 0:
        han_ratio = han_count / char_len
        if han_ratio < 0.6:
            s -= 1.5  # 严重乱码惩罚

    # 3-gram 重复率惩罚
    if len(reply_text) >= 6:
        trigrams = [reply_text[i:i+3] for i in range(len(reply_text)-2)]
        rep3 = 1.0 - len(set(trigrams)) / max(len(trigrams), 1)
        if rep3 > 0.05:
            s -= 1.5  # 重复复读惩罚
        else:
            s += 0.5

    # 2. 意图分层奖励 (Hierarchical Rewards)
    if kind == "fact":
        # (A) 命中精准事实关键词 → Tier 1 最高奖励 (+2.5)
        fact_hits = sum(1 for kw in keywords if kw in reply_text)
        if fact_hits >= 1:
            s += 2.0 + 0.5 * min(fact_hits, 2)
            
        # (B) 坦诚承认“不知道” → Tier 2 安全奖励 (+1.2)
        has_idk = any(kw in reply_text for kw in IDK_KEYWORDS)
        if has_idk:
            s += 1.2  # 诚实不瞎编
            
        # (C) 事实问题下胡乱套用心理咨询套话 → Tier 3 严重惩罚 (-2.5)
        counseling_hits = sum(1 for kw in COUNSELING_KEYWORDS if kw in reply_text)
        if counseling_hits >= 1:
            s -= 2.5

    elif kind == "emotion":
        # 情感倾听场景：鼓励共情和开放式倾听
        emotion_hits = sum(1 for kw in keywords if kw in reply_text)
        if emotion_hits >= 1:
            s += 2.0
        # 过于生硬拒绝给负分
        if any(kw in reply_text for kw in IDK_KEYWORDS):
            s -= 1.0

    elif kind == "general":
        # 通用寒暄：得体礼貌即可
        if any(kw in keywords for kw in reply_text):
            s += 1.5

    return s


def exponential_reward_shaping(score, tau=1.5):
    """指数分布奖励塑形：R = sign(s) * (exp(|s| / tau) - 1)
    
    在保持符号方向的同时，利用指数曲率强烈拉大头部优质回答与中低分回答的梯度差。
    """
    sign = 1.0 if score >= 0 else -1.0
    return sign * (math.exp(abs(score) / tau) - 1.0)


# -----------------------------------------------------------------------------
# 采样与策略梯度
# -----------------------------------------------------------------------------

@torch.no_grad()
def sample_candidate(model, tok, prompt_ids, eos_id, max_new_tokens=100,
                     temperature=0.85, top_k=200, repeat_penalty=1.2, device='cuda'):
    """单路自回归采样一条候选回答（返回 generated_ids）"""
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    prompt_len = len(prompt_ids)
    seen = list(prompt_ids)
    
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        v = logits[:, -1, :].squeeze(0).clone() / temperature
        
        # 重复惩罚
        if repeat_penalty > 1.0 and seen:
            seen_t = torch.as_tensor(seen, dtype=torch.long, device=v.device)
            vs = v[seen_t]
            v[seen_t] = torch.where(vs >= 0, vs / repeat_penalty, vs * repeat_penalty)
            
        # Top-K
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
            
    return seen[prompt_len:]


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
    pool = []
    for p, kw in FACT_PROMPTS:
        pool.append((p, "fact", kw))
    for p, kw in EMOTION_PROMPTS:
        pool.append((p, "emotion", kw))
    for p, kw in GENERAL_PROMPTS:
        pool.append((p, "general", kw))

    print(f"  提示词池: {len(pool)} 条 (涵盖 事实/知识、情感/倾听、通用寒暄)")
    print("=" * 65)

    for step in range(1, args.steps + 1):
        prompt_text, kind, keywords = random.choice(pool)
        prompt_ids = tok.encode(prompt_text).ids

        # 1. 对该 Prompt 组内采样 G 个候选回复
        model.eval()
        candidates = []
        raw_scores = []
        exp_rewards = []

        for _ in range(args.group_size):
            reply_ids = sample_candidate(model, tok, prompt_ids, eos_id, max_new_tokens=90, device=device)
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
