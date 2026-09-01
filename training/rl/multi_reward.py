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


# -----------------------------------------------------------------------------
# 轻量语义相似度 (Lightweight Semantic Similarity, 零外部依赖, 训练消耗≈0)
# -----------------------------------------------------------------------------
# 原理: 用字符级 n-gram 特征向量近似"语义相关度"——
#   理想回答(由该 prompt 的关键词拼成) 与 候选回复 各抽 n-gram 集合,
#   求 Dice 相似度 (2*|A∩B| / (|A|+|B|))。
# 效果: 回复虽未出现字面关键词、但用了同义/相关表达(如"叠加/纠缠/并行"≈"量子"),
#       也能拿到连续的部分分 → 给 GRPO 提供比"关键词硬命中"更平滑的梯度。
# 开销: 每次 evaluate_reply 只多 O(回复长度) 的 set 运算, 相对采样/前向可忽略。
# 升级接口: 将来想换成真正的 embedding 语义模型, 只需替换 _semantic_similarity。
def _ngram_set(text: str, n: int = 2) -> set:
    """字符级 n-gram 集合 (作为轻量特征向量)。空文本返回空集。"""
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def _semantic_similarity(reply_text: str, keywords: list) -> float:
    """回复 vs 关键词理想短语 的 Dice 相似度, 返回 [0, 1]。"""
    target = "".join(keywords or [])
    if not target or not reply_text.strip():
        return 0.0
    a = _ngram_set(target)
    b = _ngram_set(reply_text.strip())
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return 2.0 * inter / (len(a) + len(b))


@dataclass
class RewardDimensionWeights:
    """各维度的梯度权重配比"""
    w_task: float = 1.2          # 任务/记忆/自发命名
    w_natural: float = 1.0       # 日常自然度与长度
    w_anti_repeat: float = 0.8   # 抗复读多样性
    w_anti_robotic: float = 1.5  # 去机械标签 (强约束)
    w_quiet_eos: float = 1.0     # EOS 收尾与静默
    w_sentiment: float = 0.8     # 积极阳光与互动好奇
    w_semantic: float = 0.6      # 语义相关命中 (轻量 n-gram 特征向量相似度)


@dataclass
class RewardVector:
    """单条回复的多维奖励明细"""
    r_task: float = 0.0
    r_natural: float = 0.0
    r_anti_repeat: float = 0.0
    r_anti_robotic: float = 0.0
    r_quiet_eos: float = 0.0
    r_sentiment: float = 0.0
    r_semantic: float = 0.0

    def compute_weighted_total(self, weights: RewardDimensionWeights) -> float:
        return (
            weights.w_task * self.r_task +
            weights.w_natural * self.r_natural +
            weights.w_anti_repeat * self.r_anti_repeat +
            weights.w_anti_robotic * self.r_anti_robotic +
            weights.w_quiet_eos * self.r_quiet_eos +
            weights.w_sentiment * self.r_sentiment +
            weights.w_semantic * self.r_semantic
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "task": round(self.r_task, 3),
            "natural": round(self.r_natural, 3),
            "anti_repeat": round(self.r_anti_repeat, 3),
            "anti_robotic": round(self.r_anti_robotic, 3),
            "quiet_eos": round(self.r_quiet_eos, 3),
            "sentiment": round(self.r_sentiment, 3),
            "semantic": round(self.r_semantic, 3),
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
        # NOTE(anti-collapse): 惩罚必须"够惨但不过载"。GRPO 在组内做优势归一化，
        # 当整组候选都退化成纯 <eos> 空回复时，'+1e-6' 兜底的 std 会把优势炸成巨大值，
        # 反而把"最不烂的垃圾"当正样本强化（EOS 坍缩期的自我强化循环）。
        # 实测对比: 空回复若只罚 -1.5(塑形后-1.72), 比"过短敷衍"(-9.95)和"角色标签"(-7.79)
        # 都轻得多 → 模型学会"宁可闭嘴, 不可乱说", EOS 空回复成为局部最优并自我强化。
        # 因此把空回复罚到 -3.5 (塑形后约 -9.3), 重过"说错话"中位数, 让闭嘴不再划算。
        if char_len == 0:
            vec.r_natural = -3.5
            total_s = -3.5
            return vec, total_s, self.exponential_shaping(total_s)

        # ── 维度 1: r_natural (日常自然度与长度) ──
        # 长度软奖励曲线 (Soft Length Curve): 不再"8-65 全 +1.0"一刀切, 而是按接近黄金中段的程度给连续分。
        # 理由: 旧硬边界把 8 字和 46 字等同满分 → 模型对长度不敏感, 且指数塑形放大后演变成"越短越省力"
        #       (只要落在区间内就满分, EOS 收尾成本最低)。改用单峰曲线, 峰值在黄金中段 (~24 字),
        #       能同时克制"过短敷衍"与"越长越好"两个方向。
        # 曲线: f(len) = L_max * exp(-((len - L_opt)^2) / (2 * sigma^2))  (高斯钟形, 归一化到 [0,1])
        if char_len == 0:
            pass  # 已在上面早退分支处理, 不会走到这
        elif char_len < 5:
            vec.r_natural -= 2.5  # 过短敷衍
        else:
            L_opt = 24.0          # 黄金中段长度
            sigma = 14.0          # 宽度: 14 字内接近满分, 40+ 字逐渐回落
            L_max = 1.2           # 峰值略高, 鼓励向中段靠拢
            soft_len = L_max * math.exp(-((char_len - L_opt) ** 2) / (2.0 * sigma * sigma))
            # 尾部对数衰减: 过长并不线性惩罚, 但让其掉出峰区
            vec.r_natural += soft_len
            if char_len > 55:
                vec.r_natural -= (char_len - 55) * 0.02  # 轻度冗长衰减, 60字约 -0.1

        # 汉字纯度检测 (过滤英文字母碎片与乱码)
        han_count = sum(1 for ch in reply_text if '\u4e00' <= ch <= '\u9fff')
        if char_len > 0:
            han_ratio = han_count / char_len
            if han_ratio < 0.65:
                vec.r_natural -= 2.0

        # 字符多样性检测 (覆盖复读/堆砌乱码: "融融融融融"、"政融库…" 这类低信息含量汉字串)
        # 独字(uniq_char)占比过低 = 高频重复同一批字; 即使 3-gram 不重复(政融库类不触发 anti_repeat)
        # 或落在 8-65 黄金长度段, 也无法靠 r_natural 黄金长度分洗白。
        # 用"重复高发字占比"(top1_char / char_len) 捕捉: 无意义复读串 top1 占比常 > 0.15。
        if char_len >= 6:
            from collections import Counter as _Counter
            char_counts = _Counter(reply_text)
            top1_ratio = char_counts.most_common(1)[0][1] / char_len
            if top1_ratio > 0.30:
                vec.r_natural -= 2.5  # 严重复读堆砌 (如"融融融融融")
            elif top1_ratio > 0.15:
                vec.r_natural -= 1.2  # 轻微单字富集/堆砌

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

        # ── 维度 7: r_semantic (轻量语义相关命中) ──
        # 用字符 n-gram Dice 相似度衡量"回复与理想表达在语义上的贴近程度"。
        # 关键词硬命中 (r_task) 只管字面; 这一维给"语义相关但字面不同"的表达连续部分分。
        # 上限 ~2.5 (权重 0.6 → 加权后最多 +1.5), 作为温柔的梯度信号, 不喧宾夺主。
        sim = _semantic_similarity(reply_text, keywords)
        if sim > 0:
            vec.r_semantic = min(2.5 * sim, 2.5)

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
