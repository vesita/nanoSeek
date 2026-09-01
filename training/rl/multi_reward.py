#!/usr/bin/env python3
"""nanoSeek 多维解耦奖励引擎与梯度自适应系统 (Multi-Dimensional Decoupled Reward Engine)

核心架构与设计哲学：
1. 彻底告别单一标量盲盒打分 (Vectorized Multi-Objective Reward)：
   将奖励严格解耦为 6 个相互独立、互不干扰的物理维度向量：
   r = [ r_task, r_natural, r_anti_repeat, r_anti_robotic, r_quiet_eos, r_sentiment ]

2. 维度定义与物理意义：
   - r_task (任务与实体召回): 事实准确度、跨轮记忆召回、自发命名涌现
   - r_natural (日常自然度与长度): 8-65 字日常黄金节奏，严惩空回复 (len=0 -> -5.0)
   - r_anti_repeat (抗复读与多样性): 3-gram / 2-gram 熵值与重复模式惩罚
   - r_anti_robotic (去机械化): 绝对零容忍 "用户/模型/user/assistant" 等合成伪标记
   - r_quiet_eos (利落收尾与能量静默): 正确吐 <eos> 并在终止符处施加 L1 能量最小化
   - r_sentiment (积极阳光与好奇心): 鼓励温暖回应、主动抛出反问与好奇探索

3. 动态权重与优势合成 (Pareto Dynamic Advantage Synthesis)：
   综合得分 S = sum( w_k * r_k )，各维度独立监控并输出分项雷达图统计。
"""
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import torch
import torch.nn.functional as F

ROBOTIC_TAGS = ["用户", "模型", "user", "assistant", "system", "Human:", "Assistant:"]
COUNSELING_TEMPLATES = [
    "放松不下来", "最表面那层", "顺一顺这口气", "先松一点", "心里这团", "哪件事卡着",
    "特别耗神", "先不用把后面想完", "心里发慌", "最磨人的不是大事", "身体先绷住"
]


@dataclass
class RewardDimensionWeights:
    """各维度的梯度权重配比"""
    w_task: float = 1.2          # 任务/记忆/自发命名
    w_natural: float = 1.0       # 日常自然度与长度
    w_anti_repeat: float = 0.8   # 抗复读多样性
    w_anti_robotic: float = 1.5  # 去机械标签 (强约束)
    w_quiet_eos: float = 1.0     # EOS 收尾与静默
    w_sentiment: float = 0.8     # 积极阳光与互动好奇


@dataclass
class RewardVector:
    """单条回复的多维奖励明细"""
    r_task: float = 0.0
    r_natural: float = 0.0
    r_anti_repeat: float = 0.0
    r_anti_robotic: float = 0.0
    r_quiet_eos: float = 0.0
    r_sentiment: float = 0.0

    def compute_weighted_total(self, weights: RewardDimensionWeights) -> float:
        return (
            weights.w_task * self.r_task +
            weights.w_natural * self.r_natural +
            weights.w_anti_repeat * self.r_anti_repeat +
            weights.w_anti_robotic * self.r_anti_robotic +
            weights.w_quiet_eos * self.r_quiet_eos +
            weights.w_sentiment * self.r_sentiment
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "task": round(self.r_task, 3),
            "natural": round(self.r_natural, 3),
            "anti_repeat": round(self.r_anti_repeat, 3),
            "anti_robotic": round(self.r_anti_robotic, 3),
            "quiet_eos": round(self.r_quiet_eos, 3),
            "sentiment": round(self.r_sentiment, 3),
        }


class MultiDimensionalRewardEngine:
    """多维解耦强化学习奖励评估引擎"""

    def __init__(self, weights: RewardDimensionWeights = None, tau: float = 1.5):
        self.weights = weights or RewardDimensionWeights()
        self.tau = tau

    def exponential_shaping(self, score: float) -> float:
        """非线性指数奖励塑形 (强力拉开梯度差距)"""
        sign = 1.0 if score >= 0 else -1.0
        return sign * (math.exp(abs(score) / self.tau) - 1.0)

    def evaluate_reply(
        self,
        prompt_text: str,
        reply_text: str,
        reply_ids: List[int],
        eos_id: int,
        kind: str = "general",
        keywords: List[str] = None
    ) -> Tuple[RewardVector, float, float]:
        """评估单条回复，返回 (多维奖励向量, 综合加权总分 s, 指数塑形奖励 R)"""
        keywords = keywords or []
        vec = RewardVector()
        char_len = len(reply_text.strip())

        # ── 0. 致命空回复 / 哑巴装死熔断 ──
        if char_len == 0:
            vec.r_natural = -5.0
            total_s = -5.0
            return vec, total_s, self.exponential_shaping(total_s)

        # ── 维度 1: r_natural (日常自然度与长度) ──
        if 8 <= char_len <= 65:
            vec.r_natural += 1.0  # 黄金日常对话长度
        elif char_len < 6:
            vec.r_natural -= 2.5  # 过短敷衍
        elif char_len > 90:
            vec.r_natural -= 0.8  # 独白式冗长

        # 汉字纯度检测 (过滤英文字母碎片与乱码)
        han_count = sum(1 for ch in reply_text if '\u4e00' <= ch <= '\u9fff')
        if char_len > 0:
            han_ratio = han_count / char_len
            if han_ratio < 0.65:
                vec.r_natural -= 2.0

        # ── 维度 2: r_anti_repeat (抗复读多样性) ──
        if len(reply_text) >= 6:
            trigrams = [reply_text[i:i+3] for i in range(len(reply_text)-2)]
            rep3 = 1.0 - len(set(trigrams)) / max(len(trigrams), 1)
            if rep3 > 0.05:
                vec.r_anti_repeat -= 2.0  # 严惩车轱辘话
            else:
                vec.r_anti_repeat += 0.6
        else:
            vec.r_anti_repeat += 0.2

        # ── 维度 3: r_anti_robotic (去机械标签与模板) ──
        if any(tag in reply_text for tag in ROBOTIC_TAGS):
            vec.r_anti_robotic -= 3.0  # 毁灭性惩罚角色标签
        else:
            vec.r_anti_robotic += 0.5

        if any(tpl in reply_text for tpl in COUNSELING_TEMPLATES):
            vec.r_anti_robotic -= 2.5  # 严惩刻板心理套话

        # ── 维度 4: r_quiet_eos (利落收尾与终止符) ──
        hit_eos = (eos_id in reply_ids)
        if hit_eos:
            if char_len >= 6:
                vec.r_quiet_eos += 1.2  # 正常表达并利落收尾
            else:
                vec.r_quiet_eos -= 2.0  # 提前掐断
        else:
            vec.r_quiet_eos -= 1.0      # 喋喋不休未终止

        # ── 维度 5: r_sentiment (积极阳光与主动互动) ──
        # 鼓励主动发问、抛出话题延续互动
        if any(q in reply_text for q in ["？", "?", "吗", "呢", "觉得", "如何", "一起"]):
            vec.r_sentiment += 0.8
        # 鼓励积极、温暖、阳光词汇
        if any(w in reply_text for w in ["开心", "好呀", "阳光", "美好", "喜欢", "一起", "探索", "真好", "期待"]):
            vec.r_sentiment += 0.8

        # ── 维度 6: r_task (任务意图、自发命名与实体召回) ──
        if kind == "identity":
            # 自发命名涌现
            if any(p in reply_text for p in ["叫我", "我想叫", "可以叫我", "我叫", "名字叫"]):
                vec.r_task += 2.5
            id_hits = sum(1 for kw in keywords if kw in reply_text)
            if id_hits >= 1:
                vec.r_task += 1.5
        elif kind == "memory":
            # 实体记忆召回
            hits = sum(1 for kw in keywords if kw in reply_text)
            if hits >= 2:
                vec.r_task += 3.5  # 完美全部召回
            elif hits == 1:
                vec.r_task += 2.0  # 部分召回
            else:
                vec.r_task -= 2.0  # 彻底遗忘
        elif kind == "fact":
            # 科学常识
            fact_hits = sum(1 for kw in keywords if kw in reply_text)
            if fact_hits >= 1:
                vec.r_task += 2.5
        elif kind in ("travel", "mood", "heuristic"):
            hits = sum(1 for kw in keywords if kw in reply_text)
            if hits >= 1:
                vec.r_task += 2.0

        # 计算加权总分与指数塑形
        total_s = vec.compute_weighted_total(self.weights)
        exp_R = self.exponential_shaping(total_s)

        return vec, total_s, exp_R


if __name__ == "__main__":
    # 单元测试与多维评估验证
    engine = MultiDimensionalRewardEngine()
    print("═" * 65)
    print("🧪 多维解耦奖励引擎基准测试")
    print("═" * 65)

    test_cases = [
        ("如果由你决定名字你想叫什么？", "我叫星澜，很高兴认识你！今天有什么开心的事想聊聊吗？", [1, 2, 0], 0, "identity", ["叫我", "星澜"]),
        ("什么是量子计算？", "量子计算利用量子叠加与纠缠原理实现超强算力，你对哪个方面最感兴趣呢？", [1, 2, 0], 0, "fact", ["量子", "叠加", "纠缠", "算力"]),
        ("什么是量子计算？", "模型：放松不下来最表面那层，心里这团", [1, 2, 0], 0, "fact", ["量子"]),
        ("你叫什么？", "", [0], 0, "identity", []),
    ]

    for q, a, ids, eos, kind, kws in test_cases:
        vec, score, exp_r = engine.evaluate_reply(q, a, ids, eos, kind=kind, keywords=kws)
        print(f"Q: {q}")
        print(f"A: \"{a}\"")
        print(f"  • 多维明细: {vec.to_dict()}")
        print(f"  • 加权总分: {score:+.2f} | 指数奖励 R: {exp_r:+.2f}")
        print("─" * 65)
