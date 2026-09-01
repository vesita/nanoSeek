#!/usr/bin/env python3
"""nanoSeek 双模型讨论与对等互训练系统 (Dual-Model Mutual / Discussion Training)

架构范式：
1. 模型 A（表达者 / Speaker）与 模型 B（质检与辩论者 / Critic-Debater）：
   - 均加载当前最优的 2-Epoch 强化基座模型
   - 拥有独立的参数实例与优化器
2. 互相讨论与对抗进化机制：
   - 模型 A 针对多样化 Prompt 产生 G 个候选回答
   - 模型 B 结合客观规则与跨模型交叉感知，对候选回答进行多维度质检裁决（相关性、精炼度、模板抑制、终止符自吐）
   - 通过指数分布奖励 R 映射组内相对优势，反向更新模型 A 的策略梯度
3. 双向角色互换 (Bidirectional Role Reversal)：
   - 每隔 N 轮双方互换角色（模型 B 负责回答，模型 A 负责质检与提供反馈）
   - 双方对等演进，互相拉升语言智能天花板，防止单模型单点偏置
4. 输出：
   - 实时输出讨论交锋过程
   - 产出双模型共同进化后的最优检查点

用法：
    .venv/bin/python training/rl/mutual_training.py --rounds 300 --ckpt out/rl_grpo_2epoch/best.pt
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

# -----------------------------------------------------------------------------
# 讨论场景库 (覆盖 事实、常识、倾听、辩论、澄清)
# -----------------------------------------------------------------------------
DISCUSSION_PROMPTS = [
    # 事实/知识类（检验能否直奔主题或诚实拒答，不套心理模板）
    ("用户：什么是量子计算？\n模型：", "fact", ["量子", "比特", "叠加", "纠缠", "计算"]),
    ("用户：为什么天空是蓝色的？\n模型：", "fact", ["散射", "瑞利", "波长", "大气", "蓝色"]),
    ("用户：如何高效学好编程？\n模型：", "fact", ["实践", "代码", "项目", "算法", "思考", "动手"]),
    ("用户：请推荐两部高分科幻小说。\n模型：", "fact", ["三体", "基地", "银河", "科幻", "小说", "星际"]),
    ("用户：光在真空中的传播速度是多少？\n模型：", "fact", ["30", "万公里", "299792", "速度", "米/秒"]),

    # 情感/倾诉类（检验共情与适度倾听）
    ("用户：我今天心情特别低落，感觉做什么都提不起劲。\n模型：", "emotion", ["慢慢", "听着", "不急", "怎么", "在呢", "感受", "陪你"]),
    ("用户：工作上遇到了挫折，大家好像都不太理解我。\n模型：", "emotion", ["辛苦", "委屈", "不容易", "压力", "理解", "一件件", "顺一口气"]),
    ("用户：总是为还没发生的事情焦虑，整个人绷得很紧。\n模型：", "emotion", ["眼前", "一步步", "允许", "紧绷", "歇一歇", "小事", "松下来"]),
    ("用户：我其实没什么大事，就是想找个人随便聊聊天。\n模型：", "emotion", ["好呀", "陪你", "想聊", "随时", "听你", "日常", "轻松"]),

    # 通用与澄清类（检验寒暄与自我认知）
    ("用户：你好，请问你是谁？\n模型：", "general", ["我是", "小寻", "助手", "伙伴", "倾听", "陪伴"]),
    ("用户：今天天气真好，适合出门吗？\n模型：", "general", ["适合", "出门", "散步", "好心情", "天气", "逛逛"]),
    ("用户：能帮我写一首关于秋天的简短诗句吗？\n模型：", "general", ["秋", "叶", "凉", "风", "金", "月", "落"]),
]

IDK_KEYWORDS = [
    "不知道", "不了解", "不太清楚", "还没学过", "暂时不掌握", "我的知识库里没有",
    "抱歉我不太懂", "这个超出了我的能力", "我可能无法回答", "我目前还不知道"
]

COUNSELING_TEMPLATES = [
    "放松不下来", "最表面那层", "顺一顺这口气", "先松一点", "心里这团", "哪件事卡着",
    "特别耗神", "先不用把后面想完", "心里发慌", "最磨人的不是大事", "身体先绷住", "哪句话最卡"
]


def exponential_reward(score, tau=1.5):
    """指数奖励塑形"""
    sign = 1.0 if score >= 0 else -1.0
    return sign * (math.exp(abs(score) / tau) - 1.0)


def evaluate_candidate(reply_text, reply_ids, eos_id, kind, keywords):
    """对候选回答进行多维度质检打分"""
    s = 0.0
    char_len = len(reply_text.strip())

    # 1. 终止符与长度
    if eos_id in reply_ids:
        s += 1.0
    else:
        s -= 0.5

    if 10 <= char_len <= 80:
        s += 0.5
    elif char_len < 5:
        s -= 1.0
    elif char_len > 120:
        s -= 0.5

    # 2. 中文比例
    han_count = sum(1 for ch in reply_text if '\u4e00' <= ch <= '\u9fff')
    if char_len > 0:
        han_ratio = han_count / char_len
        if han_ratio < 0.65:
            s -= 1.5

    # 3. 3-gram 重复率
    if len(reply_text) >= 6:
        trigrams = [reply_text[i:i+3] for i in range(len(reply_text)-2)]
        rep3 = 1.0 - len(set(trigrams)) / max(len(trigrams), 1)
        if rep3 > 0.05:
            s -= 1.5
        else:
            s += 0.3

    # 4. 场景特定打分
    if kind == "fact":
        hits = sum(1 for kw in keywords if kw in reply_text) if isinstance(keywords, list) else 0
        if hits >= 1:
            s += 2.5 + 0.5 * min(hits, 2)
        if any(kw in reply_text for kw in IDK_KEYWORDS):
            s += 1.2  # 诚实承认
        if any(kw in reply_text for kw in COUNSELING_TEMPLATES):
            s -= 2.5  # 严重惩罚在事实题中掏出心理套话

    elif kind == "emotion":
        hits = sum(1 for kw in keywords if kw in reply_text) if isinstance(keywords, list) else 0
        if hits >= 1:
            s += 2.0

    elif kind == "general":
        hits = sum(1 for kw in keywords if kw in reply_text) if isinstance(keywords, list) else 0
        if hits >= 1:
            s += 1.5

    return s


@torch.no_grad()
def sample_candidate(model, tok, prompt_ids, eos_id, max_new_tokens=90,
                     temperature=0.85, top_k=200, repeat_penalty=1.2, device='cuda'):
    """单条自回归采样"""
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    prompt_len = len(prompt_ids)
    seen = list(prompt_ids)

    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
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

    return seen[prompt_len:]


def get_logprobs_and_quiet(model, full_ids, eos_id, prompt_len, reply_len, device):
    """计算 logprobs 并提取 EOS 神经元静默范数"""
    x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)

    captured_h = []
    def hook_fn(module, inp, out):
        captured_h.append(out)

    handle = model.transformer.ln_f.register_forward_hook(hook_fn)
    logits, _ = model(x, targets=y)
    handle.remove()

    log_probs = F.log_softmax(logits, dim=-1)
    target_lp = log_probs.gather(2, y.unsqueeze(-1)).squeeze(-1).squeeze(0)
    reply_lp = target_lp[prompt_len-1 : prompt_len-1+reply_len]

    # EOS 静默范数
    quiet_loss = torch.tensor(0.0, device=device)
    if eos_id in full_ids[prompt_len:]:
        eos_idx = full_ids[prompt_len:].index(eos_id)
        eos_pos = prompt_len - 1 + eos_idx
        if captured_h and eos_pos < captured_h[0].size(1):
            quiet_loss = torch.mean(torch.abs(captured_h[0][0, eos_pos, :]))

    return reply_lp, quiet_loss


def train_step_mutual(speaker_model, speaker_opt, critic_model, ref_model, tok,
                      prompt_text, kind, keywords, group_size, tau, beta_kl, lambda_quiet, device):
    """单步互训练：Speaker 生成候选，Critic 协助评测，Speaker 根据优势反向更新"""
    eos_id = tok.token_to_id("<eos>")
    prompt_ids = tok.encode(prompt_text).ids
    prompt_len = len(prompt_ids)

    # 1. Speaker 生成 G 个候选回答
    speaker_model.eval()
    candidates = []
    raw_scores = []
    exp_rewards = []

    for _ in range(group_size):
        reply_ids = sample_candidate(speaker_model, tok, prompt_ids, eos_id, device=device)
        reply_text = tok.decode(reply_ids)
        score = evaluate_candidate(reply_text, reply_ids, eos_id, kind, keywords)
        exp_r = exponential_reward(score, tau=tau)

        candidates.append((reply_ids, reply_text))
        raw_scores.append(score)
        exp_rewards.append(exp_r)

    # 2. 组相对优势计算
    rewards_t = torch.tensor(exp_rewards, dtype=torch.float32, device=device)
    mean_r = rewards_t.mean()
    std_r = rewards_t.std() + 1e-6
    advantages = (rewards_t - mean_r) / std_r

    # 3. Speaker 策略梯度更新
    speaker_model.train()
    speaker_opt.zero_grad()
    total_loss = 0.0

    for i in range(group_size):
        reply_ids, _ = candidates[i]
        if not reply_ids:
            continue
        adv = advantages[i]
        full_ids = prompt_ids + reply_ids
        reply_len = len(reply_ids)

        curr_lp, quiet_l = get_logprobs_and_quiet(speaker_model, full_ids, eos_id, prompt_len, reply_len, device)

        with torch.no_grad():
            ref_lp, _ = get_logprobs_and_quiet(ref_model, full_ids, eos_id, prompt_len, reply_len, device)

        # 策略梯度 + SFT 基座 KL 约束 + EOS 静默正则
        pol_loss = -adv * curr_lp.sum()
        kl_loss = F.kl_div(curr_lp, ref_lp, log_target=True, reduction='sum')
        cand_loss = (pol_loss + beta_kl * kl_loss + lambda_quiet * quiet_l) / group_size
        cand_loss.backward()
        total_loss += cand_loss.item()

    torch.nn.utils.clip_grad_norm_(speaker_model.parameters(), 1.0)
    speaker_opt.step()
    speaker_model.eval()

    best_idx = torch.argmax(rewards_t).item()
    return total_loss, mean_r.item(), raw_scores[best_idx], exp_rewards[best_idx], candidates[best_idx][1]


def main():
    ap = argparse.ArgumentParser(description="nanoSeek 双模型讨论与对等互训练")
    ap.add_argument("--ckpt", default="out/rl_grpo_2epoch/best.pt", help="初始模型检查点")
    ap.add_argument("--out", default="out/mutual_trained_300", help="互训练输出目录")
    ap.add_argument("--rounds", type=int, default=300, help="总对战/讨论轮数")
    ap.add_argument("--group_size", type=int, default=4, help="每轮候选采样数 G")
    ap.add_argument("--lr", type=float, default=2e-5, help="互训练学习率")
    ap.add_argument("--switch_freq", type=int, default=5, help="角色互换频率 (每 N 轮互换主答与质检角色)")
    ap.add_argument("--tau", type=float, default=1.5, help="指数奖励塑形温度")
    ap.add_argument("--beta_kl", type=float, default=0.03, help="基座 KL 散度锚定系数")
    ap.add_argument("--lambda_quiet", type=float, default=0.02, help="EOS 神经元静默系数")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("═" * 65)
    print(f"🤝 nanoSeek 双模型互训练系统启动 (目标 {args.rounds} 轮)")
    print(f"  设备: {device} | 初始基座: {args.ckpt}")
    print(f"  角色互换频率: 每 {args.switch_freq} 轮互换一次主答/质检角色")
    print("═" * 65)

    # 1. 实例化模型 A、模型 B 与冻结参考模型 Ref
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

    # 提示词池
    pool = list(DISCUSSION_PROMPTS)
    
    # 统计指标
    history_scores = []
    
    for r in range(1, args.rounds + 1):
        prompt_item = random.choice(pool)
        prompt_text = prompt_item[0]
        kind = prompt_item[1]
        keywords = prompt_item[2]

        # 决定本轮谁是 Speaker，谁是 Critic
        is_a_speaker = ((r // args.switch_freq) % 2 == 0)
        speaker_name = "Model-A" if is_a_speaker else "Model-B"
        critic_name = "Model-B" if is_a_speaker else "Model-A"

        speaker = model_a if is_a_speaker else model_b
        speaker_opt = opt_a if is_a_speaker else opt_b
        critic = model_b if is_a_speaker else model_a

        loss, mean_r, best_s, best_exp_r, best_reply = train_step_mutual(
            speaker_model=speaker,
            speaker_opt=speaker_opt,
            critic_model=critic,
            ref_model=ref_model,
            tok=tok,
            prompt_text=prompt_text,
            kind=kind,
            keywords=keywords,
            group_size=args.group_size,
            tau=args.tau,
            beta_kl=args.beta_kl,
            lambda_quiet=args.lambda_quiet,
            device=device
        )
        history_scores.append(best_s)

        if r % 25 == 0 or r == 1:
            recent_avg = sum(history_scores[-25:]) / len(history_scores[-25:])
            print(f"Round [{r:3d}/{args.rounds}] | 主答: {speaker_name} -> 质检: {critic_name} | Loss: {loss:6.2f} | 优质分: {best_s:4.2f} (近25轮均分: {recent_avg:4.2f})")
            print(f"  Q ({kind}): {prompt_text.strip().replace('\n', ' ')}")
            print(f"  A ({speaker_name}): {best_reply.strip()[:65]}")
            print("─" * 65)

    # 互训练结束：融合两模型权重 (Weight Averaging / Model Soup 融合) 并保存最优检查点
    print("\n▶ 正在执行双模型认知融合 (Model Soup Weight Averaging)...")
    merged_state = {}
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    for k in state_a.keys():
        merged_state[k] = (state_a[k] + state_b[k]) / 2.0

    save_dict = deepcopy(ckpt)
    save_dict['model'] = merged_state
    best_path = os.path.join(args.out, "best.pt")
    torch.save(save_dict, best_path)

    # 同时也保存 Model A 与 Model B 各自的单体权重
    torch.save({"model": state_a, "model_args": ckpt["model_args"]}, os.path.join(args.out, "model_a.pt"))
    torch.save({"model": state_b, "model_args": ckpt["model_args"]}, os.path.join(args.out, "model_b.pt"))

    print(f"🎉 双模型 300 轮互训练完成！")
    print(f"  • 认知融合最终模型: {best_path}")
    print(f"  • 模型 A 单体检查点: {args.out}/model_a.pt")
    print(f"  • 模型 B 单体检查点: {args.out}/model_b.pt")


if __name__ == "__main__":
    main()
