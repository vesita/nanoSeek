"""多维解耦奖励引擎 —— 兼容重导出垫片 (Compatibility Shim)

本模块保持旧版 public ABI 不变，内部改为委托给 `training.rl.reward`（插件式
奖励引擎包）。旧调用方（dual_chat_selfplay.py / memory_rl.py / grpo_char.py）
通过这里的旧名字继续工作：

- MultiDimensionalRewardEngine
- RewardVector
- RewardDimensionWeights
- _ngram_set
- exponential_shaping / evaluate_reply

新的完整实现、插件维度注册、数学/枚举全局规则见 `training.rl.reward`。
"""

from training.rl.reward import (  # noqa: F401
    RewardEngine,
    RewardVector,
    RewardDimensionWeights,
    EvalContext,
    DIMENSION_FIELDS,
    rules,
    _ngram_set as _ngram_set,
)

# 旧类名 → 新引擎别名（保持构造签名兼容：可无参实例化）
MultiDimensionalRewardEngine = RewardEngine


# 显式暴露旧函数，方便 `from training.rl.multi_reward import ...`
def exponential_shaping(score, tau=1.5):
    """指数奖励整形 sign(s)*(exp(|s|/tau)-1)，与旧实现签名字段一致。"""
    import math
    if score >= 0:
        return math.exp(score / tau) - 1.0
    return -(math.exp(-score / tau) - 1.0)


__all__ = [
    "MultiDimensionalRewardEngine",
    "RewardVector",
    "RewardDimensionWeights",
    "EvalContext",
    "DIMENSION_FIELDS",
    "_ngram_set",
    "exponential_shaping",
    "rules",
]