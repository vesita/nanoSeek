#!/usr/bin/env python3
"""插件式奖励引擎 (Plugin RewardEngine): 评估入口 + 维度注册/合成。

职责单一: 输入 (prompt, reply 文本/ids, 终止符, kind, keywords, coherence)
→ 产出 RewardVector + 加权总分 s + 指数塑形奖励 R。**不关心**采样或 GRPO。

核心分工 (dev-notes/63):
  - EvalContext 预计算 (char_len/body/han_ratio/arith/digits/enum) 在这里做一次,
    维度函数不再各自重复解析回复。
  - 依次执行 REGISTRY 里的维度纯函数, 把返回的部分 RewardVector 逐项求和。
  - 与采样 (Sampler) 和 GRPO 优势合成 (Updater) 完全解耦。

兼容性: 保留 evaluate_reply 的旧签名与返回 (vec, total_s, exp_R), 且支持
exponential_shaping —— dual_chat_selfplay.py / memory_rl.py 可无改动继续用。
"""
import math
from typing import Dict, List, Optional, Tuple

from .base import EvalContext, RewardVector, RewardDimensionWeights, DIMENSION_FIELDS
from . import dimensions as dims
from . import rules
from . import arith_gen  # 配置驱动算式识别 + 中文数字 (arith_rules.toml)


class RewardEngine:
    def __init__(self, weights: RewardDimensionWeights = None, tau: float = 1.5,
                 enabled: Optional[List[str]] = None, extra_dimensions: Optional[Dict[str, object]] = None):
        """enabled: 只跑这些维度的名字 (None = 全跑)。extra_dimensions 覆盖同名字注册。"""
        self.weights = weights or RewardDimensionWeights()
        self.tau = tau
        self._dims = dict(dims.REGISTRY)
        if extra_dimensions:
            self._dims.update(extra_dimensions)
        if enabled is not None:
            self._dims = {k: v for k, v in self._dims.items() if k in enabled}

    # ------------------------------------------------------------------ shaping
    def exponential_shaping(self, score: float) -> float:
        """非线性指数奖励塑形 (强力拉开梯度差距)。"""
        sign = 1.0 if score >= 0 else -1.0
        return sign * (math.exp(abs(score) / self.tau) - 1.0)

    # ------------------------------------------------------------------ context
    def build_context(self, prompt_text, reply_text, reply_ids, eos_id,
                      kind="general", keywords=None, coherence=None, cont_id=None) -> EvalContext:
        """预计算 EvalContext 快照 (维度只读)。"""
        keywords = keywords or []
        body = (reply_text or "").replace("<eos>", "").replace("<cont>", "").strip()
        char_len = len(body)
        dom = body or reply_text or ""
        han_ratio = sum(1 for ch in reply_text if '\u4e00' <= ch <= '\u9fff') / char_len if char_len else 1.0
        # 终止符类别
        hit_eos = eos_id in reply_ids
        hit_cont = (cont_id is not None and cont_id in reply_ids)
        term_kind = "eos" if hit_eos else ("cont" if hit_cont else "none")
        # 数学题与数字抽取 (R1/R2)。算式识别用**配置驱动**版 (arith_rules.toml):
        # 提示词由同一份配置生成 (含 +/＋/加/加上 等格式化噪音), 识别与生成共用规则,
        # 保证"提示词用的每一种写法, 奖励一定认得出"。
        arith = arith_gen.extract_arith(prompt_text)
        arith_hit = arith is not None
        first_digit, extra = rules.extract_first_digit(body)
        # arith_correct: 提取回复里的"答案数字" == 正确结果 (阿拉伯或中文都认)。
        # 用 extract_answer_number 而非裸首数字 —— 复述题目("3加4等于7")时
        # 首数字是操作数 3, 必须优先取 "等于/答案/得/=" 标记之后的数字 (dev-notes/65)。
        ans_num = arith_gen.extract_answer_number(body)
        arith_correct = bool(arith_hit and ans_num is not None and ans_num == arith[3])
        # arith_attempted: 数学题上是否出现"明确作答"动作 (对错都算)。
        # 用 has_answer_intent(答案标记/算式尾/纯数字) 而非 extract_answer_number 的
        # 兜底(任意首个数字)——否则废话流里零散的"一"(一起/那个)会被误判成作答,
        # 逃过拒答重罚 (dev-notes/66 实测废话流长 +6.24 避难所残留根因)。
        arith_attempted = bool(arith_hit and arith_gen.has_answer_intent(body))
        # 顺序列举长度 = 用**回复全量数字 token**(首数字 + 多余数字) 判 +1 连续递增,
        # 而不仅看"多余数字", 否则 "13 14 15" 因首数字 13 被丢弃而少算一个 (漏奖)。
        all_digits = ([first_digit] if first_digit is not None else []) + extra
        enum_len = rules.ordered_enum_run_len(all_digits)
        return EvalContext(
            prompt_text=prompt_text, reply_text=reply_text, reply_ids=list(reply_ids),
            eos_id=eos_id, cont_id=cont_id, kind=kind, keywords=keywords,
            coherence=coherence, tau=self.tau,
            char_len=char_len, body=body, han_ratio=han_ratio, term_kind=term_kind,
            arith=arith, arith_hit=arith_hit,
            first_digit=first_digit, arith_correct=arith_correct, arith_attempted=arith_attempted,
            extra_digits=extra, enum_run_len=enum_len,
        )

    # ------------------------------------------------------------------ eval
    def evaluate_reply(self, prompt_text, reply_text, reply_ids, eos_id,
                       kind="general", keywords=None, coherence=None,
                       cont_id=None) -> Tuple[RewardVector, float, float]:
        """评估单条回复, 返回 (多维奖励向量, 综合加权总分 s, 指数塑形奖励 R)。

        签名与旧 MultiDimensionalRewardEngine.evaluate_reply 完全兼容。
        """
        ctx = self.build_context(prompt_text, reply_text, reply_ids, eos_id,
                                 kind=kind, keywords=keywords, coherence=coherence, cont_id=cont_id)
        total = RewardVector()
        for fn in self._dims.values():
            part = fn(ctx)
            total += part
        total_s = total.compute_weighted_total(self.weights)
        exp_R = self.exponential_shaping(total_s)
        return total, total_s, exp_R


# 兼容别名: 旧模块名/类名可直接沿用 (dual_chat_selfplay / memory_rl 无改动)
__all__ = ["RewardEngine", "RewardVector", "RewardDimensionWeights",
           "EvalContext", "DIMENSION_FIELDS", "rules", "dims"]