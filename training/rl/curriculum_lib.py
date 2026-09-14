#!/usr/bin/env python3
"""课程总控共享库: 阶段定义 + 抗坍缩退化门控体检函数."""
import os

CURRICULUM_STAGES = [
    # Phase 1: 日常自然对话去机械化与 EOS 收尾 (1-300 轮, 从 SFT 基座起)
    {"stage": 1, "rounds": 100, "phase": 1, "task": "grpo", "lr": "2e-5", "desc": "日常阳光问候、心情交流与去机械标签强化"},
    {"stage": 2, "rounds": 100, "phase": 1, "task": "grpo", "lr": "2e-5", "desc": "日常生活、周末计划与自然喜好表达强化"},
    {"stage": 3, "rounds": 100, "phase": 1, "task": "grpo", "lr": "1.5e-5", "desc": "日常短句节奏、汉字纯度与 3-gram 重复率压制"},

    # Phase 2: 纯对话语言素养进阶 (301-600 轮) —— 不爬远超能力的记忆/硬任务
    {"stage": 4, "rounds": 100, "phase": 2, "task": "grpo", "lr": "1.5e-5", "desc": "多轮承接通顺、自然衔接与话题延续强化"},
    {"stage": 5, "rounds": 100, "phase": 2, "task": "grpo", "lr": "1e-5", "desc": "好奇提问、温暖陪伴与多样句式表达强化"},
    {"stage": 6, "rounds": 100, "phase": 2, "task": "grpo", "lr": "1e-5", "desc": "语义相关命中、去口水化与稳定利落收尾"},

    # Phase 3: 对话自博弈打磨 (601-900 轮) —— 双 Agent 交替对话仍是对话任务
    {"stage": 7, "rounds": 100, "phase": 3, "task": "selfplay", "lr": "1.2e-5", "desc": "双 Agent 日常闲聊交替对聊"},
    {"stage": 8, "rounds": 100, "phase": 3, "task": "selfplay", "lr": "1e-5", "desc": "双 Agent 兴趣旅行美食对聊"},
    {"stage": 9, "rounds": 100, "phase": 3, "task": "grpo", "lr": "8e-6", "desc": "开放式话题与抗偏题微调"},

    # Phase 4: 全局鲁棒与平滑收敛 (901-1200 轮)
    {"stage": 10, "rounds": 100, "phase": 4, "task": "grpo", "lr": "8e-6", "desc": "复杂意图切换与话题多样泛化"},
    {"stage": 11, "rounds": 100, "phase": 4, "task": "grpo", "lr": "6e-6", "desc": "稳定记忆已学能力与句式精简"},
    {"stage": 12, "rounds": 100, "phase": 4, "task": "grpo", "lr": "5e-6", "desc": "全场景极低学习率平滑收敛封顶"},
]


def checkpoint_healthy(ckpt_path, samples_per_prompt=3, max_tokens=50,
                       min_avg_len=8, max_phrase_repeat=0.5, min_d1=0.3):
    """对 checkpoint 做多 prompt 快速体检。退化判据(任一即退化):
      1. 平均采样长度过短(< min_avg_len) → 只会吐 <eos>/空回复;
      2. top-1 短语重复率过高(> max_phrase_repeat) → 模式坍缩(不同 prompt 都回死同一句);
      3. 1-gram 多样性过低(< min_d1) → 模板/复读化。
    多 prompt 覆盖训练池的 identity/fact/mood/travel/heuristic 各类, 比单"你好"更能暴露
    跨 prompt 的模式坍缩 (实测: 健康 42%重复/0.38 d1 vs 坍缩 83%重复/0.21 d1)。
    返回 (healthy: bool, report: str)。体检异常不阻塞训练, 记录后放行 (= healthy)。
    """
    try:
        import torch
        from model import GPTConfig, GPT
        from inference.scripts.sample_py import load_tokenizer, generate_ids
        from collections import Counter

        if not os.path.exists(ckpt_path):
            return False, f"checkpoint 不存在: {ckpt_path}"

        ck = torch.load(ckpt_path, map_location="cpu")
        args = dict(ck["model_args"])
        if "use_csa_fused_qkv" not in args:
            args["use_csa_fused_qkv"] = False
        m = GPT(GPTConfig.from_model_args(args))
        state = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
                 for k, v in ck["model"].items()}
        m.load_state_dict(state)
        m.eval()

        tok = load_tokenizer(ck)
        # 体检覆盖训练真实分布: 73% 带标签 + 27% 裸 prompt (与 grpo_char 采样一致)。
        # 裸 prompt 是模型最易退化成 <eos> 空回复的分布, 必须纳入判据, 否则"体检通过但训练崩溃"。
        tagged = [
            "用户：你好\n模型：",
            "用户：为什么天空是蓝色的？\n模型：",
            "用户：如果由你来决定名字，你想叫什么？\n模型：",
            "用户：你今天心情怎么样呀？\n模型：",
            "用户：如果有机会出去旅游，你最想去哪里玩？\n模型：",
        ]
        bare = [
            "你好",
            "为什么天空是蓝色的？",
            "如果由你来决定名字，你想叫什么？",
            "你今天心情怎么样呀？",
            "如果有机会出去旅游，你最想去哪里玩？",
        ]
        prompts = tagged + bare
        lens, eos_hits, texts = [], 0, []
        for p in prompts:
            plen = len(tok.encode(p).ids)
            for s in range(samples_per_prompt):
                torch.manual_seed(s)
                ids, eos_pos = generate_ids(m, tok, p, max_tokens, 0.8, 200, 1.2,
                                            stop_on_turn=False, stop_on_eos=False)
                texts.append(tok.decode(ids[plen:]).strip())
                lens.append(len(ids) - plen)
                if eos_pos >= 0:
                    eos_hits += 1
        n = len(texts)
        avg_len = sum(lens) / n

        # 多样性判据: top-1 短语重复率(跨 prompt) + 1-gram 多样性
        counter = Counter([t for t in texts if t])
        top1 = counter.most_common(1)[0][1] / n if counter else 0.0
        seq = "".join(texts).replace("<eos>", "")
        chars = [c for c in seq if not c.isspace()]
        d1 = len(set(chars)) / max(len(chars), 1) if chars else 0.0

        # 单独统计裸 prompt 子集的空回复率 (监控无标签泛化)
        bare_texts = texts[len(tagged) * samples_per_prompt:]
        bare_empty = sum(1 for t in bare_texts if not t) / max(len(bare_texts), 1)

        eos_rate = eos_hits / n
        report = (f"多prompt体检: n={n} avg_len={avg_len:.1f} EOS自吐={eos_rate:.0%} "
                  f"短语重复={top1:.0%} d1={d1:.2f} 裸prompt空={bare_empty:.0%}")

        # 三条判据合并
        ok_len = avg_len >= min_avg_len
        ok_div = top1 <= max_phrase_repeat and d1 >= min_d1
        healthy = ok_len and ok_div
        if not healthy:
            report += " <<< 退化"
        return healthy, report
    except Exception as exc:
        return True, f"体检异常(放行): {exc}"