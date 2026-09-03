#!/usr/bin/env python3
"""奖励引擎基础数据类型与维度注册机制 (Reward Engine Foundations)

本模块是"维度插件式"奖励体系的核心：
- RewardVector         单条回复的多维奖励明细 (每个维度一个浮点字段)
- RewardDimensionWeights  各维度梯度权重配比
- EvalContext          一次评估的**全部只读输入**快照 (打包成冻结 dataclass 传给纯函数维度)
- DimensionFn          维度纯函数签名: fn(ctx) -> RewardVector (只填自己负责的字段)
- 维度注册 / 合成        引擎通过 register_duration / evaluate_reply 组合各维度

设计动机 (dev-notes/63):
  旧 multi_reward.py 把全部维度写死在单个 evaluate_reply 方法里, 每次加维度都要
  改一个几百行的巨方法, 且维度之间通过共享局部变量 (char_len/body/...) 隐式耦合。
  这里让每个维度 = 一个独立纯函数, 只读 EvalContext, 只给自己负责的字段赋值 →
  新增数学/枚举/anti-collapse 维度只需写一个函数并注册, 不碰其他维度。
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

# 一个 RewardVector 上可用的全部维度字段名 (同 order 用于雷达图展示)
DIMENSION_FIELDS: Tuple[str, ...] = (
    "task",            # 任务意图 / 实体召回 / 自发命名
    "natural",         # 日常自然度与长度
    "anti_repeat",     # 抗复读多样性
    "anti_robotic",    # 去机械化 (角色标签等)
    "quiet_eos",       # 终止符自控 (合法 <eos>/<cont> 决策) — r_control
    "sentiment",       # 积极阳光与主动互动
    "semantic",        # 轻量语义相关命中
    "coherence",       # 语感一致性 (基座似然度, 门控/反向饱和)
    "arith",           # 数学题解答 + 顺序列举豁免 (全局规则)
    "anti_collapse",   # 反坍缩 (惩罚 <cont> 滥刷 / 空递回)
)


@dataclass
class RewardVector:
    """单条回复的多维奖励明细 (新增维度时: 在此加字段, 并把名字加入 DIMENSION_FIELDS)。"""
    r_task: float = 0.0
    r_natural: float = 0.0
    r_anti_repeat: float = 0.0
    r_anti_robotic: float = 0.0
    r_quiet_eos: float = 0.0
    r_sentiment: float = 0.0
    r_semantic: float = 0.0
    r_coherence: float = 0.0
    r_arith: float = 0.0
    r_anti_collapse: float = 0.0

    def compute_weighted_total(self, weights: "RewardDimensionWeights") -> float:
        return (
            weights.w_task * self.r_task
            + weights.w_natural * self.r_natural
            + weights.w_anti_repeat * self.r_anti_repeat
            + weights.w_anti_robotic * self.r_anti_robotic
            + weights.w_quiet_eos * self.r_quiet_eos
            + weights.w_sentiment * self.r_sentiment
            + weights.w_semantic * self.r_semantic
            + weights.w_coherence * self.r_coherence
            + weights.w_arith * self.r_arith
            + weights.w_anti_collapse * self.r_anti_collapse
        )

    def to_dict(self) -> Dict[str, float]:
        # 字段展示名比内部名短, 便于日志
        names = {
            "r_task": "task", "r_natural": "natural", "r_anti_repeat": "anti_repeat",
            "r_anti_robotic": "anti_robotic", "r_quiet_eos": "quiet_eos",
            "r_sentiment": "sentiment", "r_semantic": "semantic",
            "r_coherence": "coherence", "r_arith": "arith", "r_anti_collapse": "anti_collapse",
        }
        return {names[k]: round(getattr(self, k), 3) for k in names}

    def __add__(self, o: "RewardVector") -> "RewardVector":
        n = RewardVector()
        for f in DIMENSION_FIELDS:
            setattr(n, "r_" + f, getattr(self, "r_" + f) + getattr(o, "r_" + f))
        return n

    def __iadd__(self, o: "RewardVector") -> "RewardVector":
        for f in DIMENSION_FIELDS:
            setattr(self, "r_" + f, getattr(self, "r_" + f) + getattr(o, "r_" + f))
        return self


@dataclass
class RewardDimensionWeights:
    """各维度的梯度权重配比 (新增维度: 加字段并在 RewardVector.__add__/compute 对齐)。"""
    w_task: float = 1.2
    w_natural: float = 1.0
    w_anti_repeat: float = 0.8
    w_anti_robotic: float = 1.5
    w_quiet_eos: float = 1.0
    w_sentiment: float = 0.8
    w_semantic: float = 0.6
    w_coherence: float = 2.0
    w_arith: float = 1.5       # 数学解答 + 顺序列举豁免 (全局)
    w_anti_collapse: float = 1.5   # 反坍缩


@dataclass(frozen=True)
class EvalContext:
    """一次 evaluate_reply 的全部**只读**输入快照。

    维度纯函数只读这个对象, 不访问外部状态 → 完全可单测、可随意增删维度。
    预计算字段 (char_len / body / arith / first_digit / ordered_run) 由引擎在
    分发前算好并塞进来, 避免每个维度重复解析回复。
    """
    prompt_text: str
    reply_text: str
    reply_ids: List[int]
    eos_id: int
    cont_id: int = None          # None = 旧版强制 <eos> (无 <cont> 机制)
    kind: str = "general"
    keywords: List[str] = field(default_factory=list)
    coherence: float = None      # 基座似然度 z 值; None = 本维度不启用
    tau: float = 1.5

    # ---- 预计算 (由引擎填充, 维度只读) ----
    char_len: int = 0            # len(reply_text.strip())
    body: str = ""               # reply 去 <eos>/<cont> 后的正文
    han_ratio: float = 1.0       # 汉字占比
    term_kind: str = "none"      # "eos" / "cont" / "none" (是否吐了终止符、哪个)
    arith: Tuple = None          # (a, op, b, result) 当 prompt 含算式; 否则 None
    arith_hit: bool = False      # prompt 是否含有效算式 (已 solve 成功)
    first_digit: int = None      # 回复里空格切分后的第一个数字 token; 无则 None
    arith_correct: bool = False  # 数学题: 宽松首数字 (夹标点/汉字) == 正确结果
    arith_attempted: bool = False  # 数学题: 回复里有"答案标记后的数字"作答动作(答对或答错都给)
    extra_digits: List[int] = field(default_factory=list)  # 首数字之后的其余数字 token
    enum_run_len: int = 1        # 回复里最长"+1 连续递增"数字序列长度 (1 = 无顺序列举)

# 维度纯函数签名: fn(ctx: EvalContext) -> RewardVector
DimensionFn = Callable[[EvalContext], RewardVector]