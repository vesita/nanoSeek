#!/usr/bin/env python3
"""reward 奖励引擎包 (插件式, 与采样/GRPO 解耦)。

导出:
  - RewardEngine       : 插件式奖励引擎 (评估入口)。
  - RewardVector / RewardDimensionWeights / EvalContext : 数据结构。
  - MultiDimensionalRewardEngine : 旧名兼容别名 (委托给 RewardEngine),
      供 dual_chat_selfplay.py / memory_rl.py 无改动使用。
  - _ngram_set         : 兼容导出 (grpo_char.py:378 直接 import)。
"""
from .base import RewardVector, RewardDimensionWeights, EvalContext, DIMENSION_FIELDS
from .engine import RewardEngine
from . import rules, dimensions

# 旧名兼容别名: 委托给新 RewardEngine。
MultiDimensionalRewardEngine = RewardEngine

# 兼容导出: _ngram_set 之前在 training.rl.multi_reward。
_ngram_set = rules._ngram_set

__all__ = [
    "RewardEngine", "RewardVector", "RewardDimensionWeights", "EvalContext",
    "DIMENSION_FIELDS", "MultiDimensionalRewardEngine", "_ngram_set",
    "rules", "dimensions",
]