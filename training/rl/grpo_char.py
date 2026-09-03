#!/usr/bin/env python3
"""字级 GRPO（Group Relative Policy Optimization）训练引擎

技术特性：
1. 指数分布奖励塑形：R = sign(s) * (exp(|s| / tau) - 1)，显著拉开优质与劣质样本差距
2. 分层奖励体系 (Hierarchical Rewards)：
   - 事实/技术问答：精准事实 (+3.0) > 坦诚承认“不知道” (+1.5) >> 泛化心理套话惩罚 (-2.5)
   - 情感/倾听场景：温柔共情 (+2.5) > 普通倾听 (+1.0) >> 机械复读 (-2.0)
   - 格式与终止：自吐 <eos> (+1.0)、无 3-gram 复读 (+0.5)、长度适中 (+0.5)
   - 数学：100 以内加减乘除，空格切分首个数字 = 答案 (+2.5)；多余数字扣分；
     顺序列举 (+1 连续 ≥3) 豁免多余数字并给少许奖励 (全局规则)
3. 终止符神经元静默正则 (Quiet-State Loss on EOS)：
   - 当生成 <eos> 时，对深层隐藏状态施加 L1 能量惩罚，促使模型“收力静默”，抑制越界自说自话
4. GRPO 算法：零 Critic 网络，组内采样 G 个候选，组相对优势归一化 + SFT 基座 KL 约束

架构 (dev-notes/63, 奖励引擎解耦为三模块)：
  - Sampler   : training.rl.sampler  —— 组采样 + logprob/hidden + 静默损失 + 塑形
  - RewardEngine : training.rl.reward —— 插件式奖励维度注册 + 加权合成
  - Updater   : training.rl.updater  —— 组相对优势 + winsorize + PPO-clip/KL/quiet 更新

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
import torch.nn.functional as F
from model import GPTConfig, GPT
from inference.scripts.sample_py import build_model_from_checkpoint, load_tokenizer
from training.rl.sampler import (
    sample_candidates_batch, get_token_logprobs_and_hidden,
    compute_quiet_loss_from_hidden, reward_shaping,
)
from training.rl.updater import GrpoUpdater
from training.rl.reward import RewardEngine, _ngram_set
from training.rl.reward import arith_gen  # 配置驱动数学题 (arith_rules.toml)
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

# 6. 数学题 (100 以内加减乘除, warm-start)。奖励引擎按算式自动取答案 (全局规则 R1/R2)。
#    提示词不再硬编码: 由 arith_rules.toml 基础规则 + 代码格式化噪音生成
#    (运算符写法 +/＋/加/加上…、问句模板随机组合), 奖励识别与生成共用同一份配置。
ARITH_PROMPTS = arith_gen.gen_prompts(count=30, seed=20260903)
# -----------------------------------------------------------------------------
# 多维解耦奖励引擎实例化 (插件式)
# -----------------------------------------------------------------------------
reward_engine = RewardEngine()

def compute_raw_reward(prompt, reply_text, reply_ids, eos_id, kind, keywords, coherence=None, cont_id=None):
    """复用多维解耦奖励引擎 (coherence: 基座似然度 z 值, None 表示不启用)"""
    vec, score, exp_r = reward_engine.evaluate_reply(
        prompt, reply_text, reply_ids, eos_id, kind=kind, keywords=keywords,
        coherence=coherence, cont_id=cont_id,
    )
    return score


# -----------------------------------------------------------------------------
# r_anti_salad 门控校准常量 (来自基座 SFT 模型实测分布, .probe2.py):
#   基座自采样 mean-logp: mean=-7.8 std=1.87; v6/v7 乱码沙拉: mean=-13.9
# 结论: 基座似然度只能区分"乱码沙拉 vs 模型语感", 无法奖励流畅文本 (基座把
# 流畅中文也判为低似然, 训练语料本身不通顺)。因此它只做**门控**: 组内最佳候选
# 低于阈值 → 整组降级为退化, 跳过策略梯度 (只 KL 拉回基座), 防止"全组乱码
# 仍被组内相对优势当正样本强化"的自增强循环 (v6/v7 实测坍缩路径)。不作为奖励
# 项, 避免反向惩罚流畅表达。
# -----------------------------------------------------------------------------
COH_THETA = -7.8      # 基座自采样 mean-logp 中心
COH_SIGMA = 2.0       # 分布宽度 (放宽到 2.0, 避免个体噪声误伤)
COH_GATE_SIG = 2.2    # 阈值 = THETA - 2.2*SIGMA ≈ -12.2 (沙拉典型区间, 基座样本几乎不触)


def mean_ref_reply_logprob(ref_model, prompt_ids, reply_ids, device):
    """基座模型对"回复区域"的平均 token log-prob (无梯度, 语感一致性度量)。

    乱码/口水串在这项上显著偏低 (汉字随机拼接的条件熵极高); 基座自采样回复
    落在 theta±sigma 内。GRPO 用组内相对优势, 绝对刻度不重要, 排序对即可。
    """
    if not reply_ids:
        return None
    full = prompt_ids + reply_ids
    x = torch.tensor([full[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([full[1:]], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _ = ref_model(x, targets=y)
    lp = F.log_softmax(logits, dim=-1).gather(2, y.unsqueeze(-1)).squeeze(-1).squeeze(0)
    pl = len(prompt_ids)
    reply_lp = lp[pl:pl + len(reply_ids)]
    return float(reply_lp.mean().item())


def main():
    ap = argparse.ArgumentParser(description="nanoSeek 字级 GRPO 强化学习训练")
    ap.add_argument("--ckpt", default="out/eos_fix_1epoch/best.pt", help="基座模型路径")
    ap.add_argument("--out", default="out/rl_grpo_v1", help="RL 输出目录")
    ap.add_argument("--ref_base", default=None,
                    help="语感一致性/KL 锚点模型(基座 SFT)路径; 默认 = --ckpt 同目录 best.pt。"
                         "课程串跑时必须固定传原始基座, 否则锚点随阶段漂移, 乱码会被渐强")
    ap.add_argument("--steps", type=int, default=100, help="RL 迭代步数")
    ap.add_argument("--group_size", type=int, default=4, help="每 Prompt 并行采样数 G")
    ap.add_argument("--lr", type=float, default=2e-5, help="RL 学习率 (较小学习率防策略坍缩)")
    ap.add_argument("--tau", type=float, default=1.5, help="指数奖励塑形温度")
    ap.add_argument("--beta_kl", type=float, default=0.6, help="SFT 基座 KL 散度惩罚系数 (需足够强, 防策略漂移坍缩)")
    ap.add_argument("--lambda_quiet", type=float, default=0.02, help="EOS 神经元静默损失权重")
    ap.add_argument("--temperature", type=float, default=1.0, help="采样温度 (防模式坍缩需偏高)")
    ap.add_argument("--repeat_penalty", type=float, default=1.4, help="repeat penalty (压固定短语回环)")
    ap.add_argument("--div_weight", type=float, default=1.5,
                    help="组内多样性惩罚权重: 候选与组内其他候选 n-gram 重叠越高扣分越多, "
                         "打破'单一短语滚雪球'式模式坍缩 (0=关闭)")
    ap.add_argument("--winsorize", type=float, default=3.0,
                    help="组内优势 winsorize 裁剪: 以 median±k*MAD 收窄极端异常样本(乱码/超大负分), "
                         "防止单候选炸裂 std 归一化 (0=关闭)")
    ap.add_argument("--dyn_tau", action="store_true",
                    help="动态 tau: 用组内 exp 奖励的尺度 EMA 自适应缩放指数塑形温度, 防止尺度漂移")
    ap.add_argument("--shape", default="exp", choices=["exp", "log", "tanh"],
                    help="奖励塑形形态: exp(默认,指数拉大头部分差) / tanh(有界防-300爆炸) / log(对数压缩)。"
                         "注意: tanh 单点(100步)指标好, 但完整12阶段课程会累积压制对裸prompt乱码/空回复的"
                         "惩罚分辨率, 致裸prompt空率恶化(v7实测20-47%), 故默认用 exp + winsorize 防爆炸")
    ap.add_argument("--arith_ratio", type=float, default=0.35,
                    help="数学题提示词在候选池中的出现比例 (warm-start 用, 其余配比由池大小决定)")
    ap.add_argument("--coherence_w", type=float, default=0.0,
                    help="语感一致性作为正向奖励的权重开关 (0=沿用旧行为 coherence 仅门控)。"
                         ">0 时把候选回复的基座 mean-logp 转成 z-score 传入 r_coherence, "
                         "低似然(乱码沙拉)被惩罚、正常中文≈0 (dev-notes/66: 需 z-score 化, "
                         "原始 mean-logp ~-8 直接传会把所有回复拉成 -5)。")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"▶ 设备: {device} | 基础模型: {args.ckpt}")

    # 1. 加载训练模型 (Policy) 与冻结基座模型 (Ref Model 用于 KL 约束 + r_coherence)
    model, ckpt = build_model_from_checkpoint(os.path.dirname(args.ckpt))
    model.to(device)
    model.train()

    ref_dir = os.path.dirname(args.ref_base) if args.ref_base else os.path.dirname(args.ckpt)
    ref_model, _ = build_model_from_checkpoint(ref_dir)
    ref_model.to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    print(f"  🔗 锚点(ref) 模型: {ref_dir} (KL 约束 + r_coherence 语感)")

    tok = load_tokenizer(ckpt)
    eos_id = tok.token_to_id("<eos>")
    cont_id = tok.token_to_id("<cont>")
    print(f"  词表模式: 字级 WordLevel ({tok.get_vocab_size()} 词) | EOS ID = {eos_id} | CONT ID = {cont_id}"
          f"  (双停止符 + 完整 r_control 自控奖励)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    # 构造候选池 [(prompt_str, kind, keywords)]。数学题独立成池, 按 --arith_ratio
    # 控制出现频率 (配置驱动的提示词池可达 30 条, 若混在一起会失衡到 ~62%)。
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
    arith_pool = [(p, "arith", kw) for p, kw in ARITH_PROMPTS]

    print(f"  提示词池: 常规 {len(pool)} 条 + 数学 {len(arith_pool)} 条 (arith_ratio={args.arith_ratio:.2f})")
    print("=" * 65)

    # 动态 tau: EMA 跟踪组内 raw_score 绝对尺度, 缩放指数塑形温度, 防尺度漂移
    ema_abs_scale = 1.0
    dyn_tau_ema = 0.9

    updater = GrpoUpdater(
        model, eos_id, cont_id=cont_id,
        beta_kl=args.beta_kl, clip_eps=0.2, quiet_scale=args.lambda_quiet,
    )

    def pick_prompt():
        """按 arith_ratio 从常规池 / 数学池抽样 (数学题频率受 --arith_ratio 控制)。"""
        if arith_pool and random.random() < args.arith_ratio:
            return random.choice(arith_pool)
        return random.choice(pool)

    for step in range(1, args.steps + 1):
        prompt_text, kind, keywords = pick_prompt()
        if random.random() >= 0.27:
            prompt_text = f"用户：{prompt_text}\n模型："
        prompt_ids = tok.encode(prompt_text).ids
        prompt_len = len(prompt_ids)

        # 1. 采样 + 原始奖励 + 基座似然门控
        model.eval()
        batch_reply_ids = sample_candidates_batch(
            model, tok, prompt_ids, eos_id, group_size=args.group_size, max_new_tokens=55,
            temperature=args.temperature, top_k=200, repeat_penalty=args.repeat_penalty, device=device,
            cont_id=cont_id,
        )

        candidates = []
        raw_scores = []
        mlp_vals = []
        for reply_ids in batch_reply_ids:
            reply_text = tok.decode(reply_ids).replace("<eos>", "").replace("<cont>", "").strip()
            mlp = mean_ref_reply_logprob(ref_model, prompt_ids, reply_ids, device)
            mlp_vals.append(mlp)
            # 语感一致性: 仅当 --coherence_w>0 时把 mean-logp 转 z-score 传入奖励维度
            # (r_coherence 内部按 z 给分, 乱码 z≪0 → 负, 正常中文 z≈0)。原始 logp 不能
            # 直接传 (dev-notes/66: ~-8 会把所有回复拉成 -5, 抹平信号)。权重 w_coherence=2,
            # 缩放系数使最终贡献 ≈ coherence_w * clamp(z)。
            if args.coherence_w > 0 and mlp is not None:
                z = (mlp - COH_THETA) / COH_SIGMA
                coherence_eff = z * (args.coherence_w / 2.0)
            else:
                coherence_eff = None
            score = compute_raw_reward(prompt_text, reply_text, reply_ids, eos_id, kind, keywords,
                                       coherence=coherence_eff, cont_id=cont_id)
            candidates.append((reply_ids, reply_text))
            raw_scores.append(score)

        # 2. 奖励塑形 + 组内多样性惩罚 → exp 奖励
        if args.dyn_tau:
            import numpy as _np
            batch_abs = float(_np.mean([abs(s) for s in raw_scores])) if raw_scores else 1.0
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
                exp_rewards.append(reward_shaping(raw_scores[i] - penalty, shape=args.shape, tau=tau_eff, c=4.0))
        else:
            exp_rewards = [reward_shaping(s, shape=args.shape, tau=tau_eff, c=4.0) for s in raw_scores]

        # 3. 优势归一化 + winsorize (GrpoUpdater)
        advantages = GrpoUpdater.normalize_advantages(exp_rewards).to(device)
        if args.winsorize > 0 and advantages.numel() >= 3:
            med = torch.median(advantages)
            mad = (advantages - med).abs().median() + 1e-6
            advantages = advantages.clamp(med - args.winsorize * mad, med + args.winsorize * mad)

        # 3.5 逐候选"短空壳"硬压制 (Anti-Dwarf): 混合组里单个 `"` 空壳不能在组内
        #    归一化下拿到不偏/正优势 —— 否则它与稍微更烂的候选互相掩护、被逐步强化
        #    (rl_cont_v5 20 步健康→150 步 90% 单引号的根源)。把"单字/空/二字非数字"
        #    的候选优势直接压到该组最负, 让它永远只能被压制、绝不被强化。
        _adv_bottom = float(advantages.min().item()) - 1.0
        for i, (rid, rt) in enumerate(candidates):
            _t = rt.strip()
            if not _t:                                              # 空回复
                advantages[i] = _adv_bottom
            elif len(_t) <= 2 and not bool(re.fullmatch(r"\d{1,3}", _t)):  # `"` / 二字残壳
                advantages[i] = _adv_bottom
            # 注意: 单数字回复 (如 "7") 保留正常优势, 仍算数学努力。

        # 4. 整组退化保护 (Anti-Collapse Gate): 只在"全组都无可训练候选"时才归零优势。
        #    NOTE(Fix): 旧版用 max_raw < 0 或 best_mlp 过低 → 一旦整组碰巧全负就归零,
        #    导致模型一坍缩就彻底失去策略梯度 (仅剩 KL 拉回 → 坍缩自我强化, 实测根因)。
        #    现在改为:
        #      - 只要组里有**任一**非退化候选 (raw >= 0, 即至少一个可强化对象),
        #        就保留优势 (负项交给 winsorize 钳制, 不让极端负值主导)。
        #      - 仅当"全部 raw < 0 **且** 组最佳基座似然也过低(乱码)"才归零。
        max_raw = max(raw_scores)
        best_mlp = max((m for m in mlp_vals if m is not None), default=None)
        salad_degen = best_mlp is not None and best_mlp < (COH_THETA - COH_GATE_SIG * COH_SIGMA)

        # ── 内容式坍缩检测 (Anti-Collapse Gate 补强) ─────────────────────────
        # 旧门只认"基座似然过低(乱码)" + 组最佳 len<=1, 对 len 2~4 的"短空壳"
        # (如 `可以10"` / `不我,"`) 完全失明: 它们长度>1 逃过旧门, 又被组内相对
        # 归一化当成"最不烂"给正优势强化 → 组均长被一步步拖向 1 (rl_cont_v4 实测
        # 路径: 均长 16→4→2→1)。补两点:
        #   (a) 组平均长度坍缩: 整组均长 < 门限 → 无可强化对象, 归零优势。
        #   (b) 组最佳极短空壳 (len<=2 非单数字) 依然判死。
        _group_avg_len = float(sum(len(t) for _, t in candidates)) / max(len(candidates), 1)
        threshold_ix = int(torch.argmax(torch.tensor(exp_rewards)).item())
        best_text = candidates[threshold_ix][1].strip()
        _best_len = len(best_text)
        _best_is_digit_only = bool(re.fullmatch(r"\d{1,3}", best_text))
        content_collapse = (
            max_raw < 0
            and (
                _best_len <= 1                        # 组最佳单字空壳
                or _group_avg_len <= 4                # 整组被拖成短空壳 (均长≤4)
            )
            and not _best_is_digit_only  # 单数字回复仍可算数学努力, 不按坍缩判死
        )
        degenerate = (max_raw < 0 and salad_degen) or content_collapse
        if degenerate:
            reason = ("组最佳基座似然过低/乱码" if salad_degen else
                       f"组最佳候选为空壳 len={_best_len} {best_text!r}")
            print(f"  ⚠ 组退化 ({reason}) → 归零优势, 仅 KL/EOS 静默拉回")
            advantages = torch.zeros_like(advantages)
        elif max_raw < 0:
            # 全组偶然全负但并非乱码/空壳: 不归零, 用 winsorize 已钳制的优势继续学 (明辨相对高低)。
            print(f"  · 组最高分 {max_raw:6.2f} < 0 (全负但非空壳) → 保留优势, winsorize 已钳极端负值")

        # 5. 收集各候选 logprob + quiet (带梯度), 交给 Updater 批量更新
        model.train()
        G = len(candidates)
        max_rl = max((len(r) for r, _ in candidates if r), default=1)
        new_lp = torch.zeros(G, max_rl, device=device)
        ref_lp = torch.zeros(G, max_rl, device=device)
        mask = torch.zeros(G, max_rl, dtype=torch.bool, device=device)
        quiet_losses = []

        for i in range(G):
            reply_ids, _ = candidates[i]
            if not reply_ids:
                quiet_losses.append(torch.tensor(0.0, device=device))
                continue
            reply_len = len(reply_ids)
            full_ids = prompt_ids + reply_ids
            # 当前 Policy logprobs + hidden (带梯度)
            curr_logprobs, h_f = get_token_logprobs_and_hidden(model, full_ids, device)
            with torch.no_grad():
                ref_logprobs, _ = get_token_logprobs_and_hidden(ref_model, full_ids, device)
            # logprobs 数组索引 i 对应 full_ids[i+1] 的预测概率, 回复位在 [prompt_len, prompt_len+reply_len)
            new_lp[i, :reply_len] = curr_logprobs[prompt_len - 1:prompt_len - 1 + reply_len]
            ref_lp[i, :reply_len] = ref_logprobs[prompt_len - 1:prompt_len - 1 + reply_len]
            mask[i, :reply_len] = True
            # 终止符 (eos 或 cont) 位置的静默能量
            term_idx = -1
            for _sid in (eos_id, cont_id):
                if _sid in reply_ids:
                    term_idx = reply_ids.index(_sid)
                    break
            quiet_losses.append(compute_quiet_loss_from_hidden(h_f, term_idx, prompt_len))

        # 6. 单步更新 (GrpoUpdater 负责 优势加权/PPO-clip/KL/静默 合成)
        #    优先权: 退化时 advantages 已全 0 → 策略项≈0, 仅 KL/静默拉回基座。
        pol_loss, kl_loss, quiet_l = updater.update_step(
            advantages, new_lp, ref_lp, mask, quiet_losses, optimizer,
        )
        total_loss = pol_loss + args.beta_kl * kl_loss + args.lambda_quiet * quiet_l

        # 7. 日志
        if step % 10 == 0 or step == 1:
            rewards_t = torch.tensor(exp_rewards, device=device)
            best_idx = torch.argmax(rewards_t).item()
            best_reply = candidates[best_idx][1].strip()
            n_eos = sum(1 for rid, _ in candidates if eos_id in rid)
            n_cont = sum(1 for rid, _ in candidates if cont_id in rid)
            n_term = n_eos + n_cont
            avg_len = float(sum(len(t) for _, t in candidates)) / max(len(candidates), 1)
            # 数学正确率: 该步若是 arith 题, 统计组内"答案数字命中"比例
            # (extract_answer_number 与奖励引擎同逻辑: 优先取等于/答案标记后的数字)
            arith_acc = "-"
            if kind == "arith":
                hits = 0
                ar = arith_gen.extract_arith(prompt_text)  # 与奖励引擎同一配置
                for _, t in candidates:
                    if ar and arith_gen.extract_answer_number(t.strip()) == ar[3]:
                        hits += 1
                arith_acc = f"{hits}/{G}"
            print(f"Step [{step:3d}/{args.steps}] | Loss: {total_loss:7.4f} (Pol: {pol_loss:6.2f}, KL: {kl_loss:5.2f}, Quiet: {quiet_l:5.3f}) | 组均分: {exp_rewards[0]:5.2f}")
            print(f"  Term: eos={n_eos}/{G} cont={n_cont}/{G} 未终止={G-n_term}/{G} | 均长 {avg_len:.1f} | 数学命中(arith): {arith_acc}")
            print(f"  Q ({kind}): {prompt_text.strip().replace(chr(10), ' ')}")
            print(f"  A (Top-1, raw_s={raw_scores[best_idx]:.2f}, exp_R={rewards_t[best_idx].item():.2f}): {best_reply[:60]}")
            print("─" * 65)

    # 保存 RL 强化后模型
    ckpt_out = os.path.join(args.out, "best.pt")
    os.makedirs(os.path.dirname(os.path.abspath(ckpt_out)), exist_ok=True)
    save_dict = deepcopy(ckpt)
    save_dict['model'] = model.state_dict()
    torch.save(save_dict, ckpt_out)
    print(f"\n🎉 GRPO 强化学习训练完成！新模型已保存至: {ckpt_out}")


if __name__ == "__main__":
    main()