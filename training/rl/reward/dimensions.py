#!/usr/bin/env python3
"""维度插件: 每个奖励维度 = 独立纯函数 fn(ctx) -> RewardVector。

设计 (dev-notes/63): 不再把全部维度写死在单个 evaluate_reply 巨方法里。
每个函数只读传入的 EvalContext (引擎已预计算 char_len/body/arith/digits 等),
只给自己负责的字段赋值。引擎把这些部分 RewardVector 逐个相加 → 完整明细。
新增维度 = 在此加一个函数 + 在引擎注册, 不碰其他维度。
"""
import math
from collections import Counter

from .base import RewardVector, EvalContext
from . import rules


# -----------------------------------------------------------------------------
# 维度 0: 致命空回复/哑巴装死熔断
# -----------------------------------------------------------------------------
def r_fatal_empty(ctx: "EvalContext") -> RewardVector:
    """空回复熔断 (油门放到底, 让'闭嘴'不再划算)。

    NOTE(anti-collapse): 空回复若只罚 -1.5, 比过短敷衍(-2.5)和角色标签(-3.0)都轻
    → 模型学会"宁可闭嘴不可乱说", EOS 空回复成为局部最优并自我强化 (实测坍缩核心)。
    因此拉到 -3.5 (塑形后约 -9.3), 重过"说错话"中位数。
    有 <cont> 机制后还要额外防止"空递回刷分" (见 r_anti_collapse)。
    """
    if ctx.char_len == 0:
        v = RewardVector()
        v.r_natural = -3.5
        return v
    return RewardVector()


# -----------------------------------------------------------------------------
# 维度 1: 语感一致性 (基座似然度, 反向饱和)
# -----------------------------------------------------------------------------
def r_coherence(ctx: "EvalContext") -> RewardVector:
    """基座对回复区域的 mean-logp 归一化 z 值。乱码沙拉显著偏低。
    反向饱和: 负分钳到 -2.5 (权重 2.0 → 一次扣 5.0), 正分温和封顶 +1.0。
    coherence is None → 维度不激活 (返回 0), 兼容 memory/selfplay。"""
    v = RewardVector()
    if ctx.coherence is not None:
        v.r_coherence = max(-2.5, min(1.0, float(ctx.coherence)))
    return v


# -----------------------------------------------------------------------------
# 维度 2: 日常自然度与长度
# -----------------------------------------------------------------------------
def r_natural(ctx: "EvalContext") -> RewardVector:
    """长度软奖励曲线 (单峰高斯, 峰在 ~24 字) + 汉字纯度 + 单字富集惩罚。"""
    v = RewardVector()
    L = ctx.char_len
    if L == 0:
        return v  # 已在 fatal_empty 处理 (不重复计)
    if L < 5:
        # 过短敷衍。旧版固定 -2.5, 不区分 L=1 的 `"` 与 L=4 的合法短答 → 在组相对
        # 归一化下, 单引号空壳可与稍微更烂的候选"互相掩护"拿到相对不偏分值, 长期把
        # 均长拖向 1 (rl_cont_v5 实测 20 步健康→150 步 90% 单引号)。
        # 改为 **随短缺程度陡增**: L=1 → -5.0, L=2 → -4.0, ... L=4 → -2.0, 使超短壳
        # 在混合组里总是垫底、拿到负优势, 而不再有机会被相对归一化强化。
        # 但若是**正确的数学答案** (如裸写 "7"), 极短正是所期望的简洁作答, 不应被
        # -5/-4 抵掉 arith 的 +2.5 → 保持软化 (只用 -0.5, 见上)。
        if ctx.arith_correct:
            v.r_natural -= 0.5
        else:
            v.r_natural -= 1.0 * (6 - L)   # 短缺越狠扣越重: L=1 → -5.0 ... L=4 → -2.0
    else:
        L_opt, sigma, L_max = 24.0, 14.0, 1.2
        v.r_natural += L_max * math.exp(-((L - L_opt) ** 2) / (2.0 * sigma * sigma))
        if L > 55:
            v.r_natural -= (L - 55) * 0.02   # 轻度冗长衰减
    # 汉字纯度
    if ctx.han_ratio < 0.65:
        v.r_natural -= 2.0
    # 单字富集/复读堆砌 (如"融融融融融")
    if L >= 6:
        cnt = Counter(ctx.reply_text)
        top1 = cnt.most_common(1)[0][1] / L
        if top1 > 0.30:
            v.r_natural -= 2.5
        elif top1 > 0.15:
            v.r_natural -= 1.2
    return v


# -----------------------------------------------------------------------------
# 维度 3: 抗复读多样性
# -----------------------------------------------------------------------------
def r_anti_repeat(ctx: "EvalContext") -> RewardVector:
    v = RewardVector()
    t = ctx.reply_text
    if len(t) >= 6:
        trigrams = [t[i:i+3] for i in range(len(t)-2)]
        rep3 = 1.0 - len(set(trigrams)) / max(len(trigrams), 1)
        if rep3 > 0.05:
            v.r_anti_repeat -= 2.0     # 车轱辘话
        else:
            v.r_anti_repeat += 0.6
    else:
        v.r_anti_repeat += 0.2
    return v


# -----------------------------------------------------------------------------
# 维度 4: 去机械化 (角色标签/刻板套话)
# -----------------------------------------------------------------------------
def r_anti_robotic(ctx: "EvalContext") -> RewardVector:
    v = RewardVector()
    t = ctx.reply_text
    if any(tag in t for tag in rules.ROBOTIC_TAGS):
        v.r_anti_robotic -= 3.0
    else:
        v.r_anti_robotic += 0.5
    if any(tpl in t for tpl in rules.COUNSELING_TEMPLATES):
        v.r_anti_robotic -= 2.5
    return v


# -----------------------------------------------------------------------------
# 维度 5: r_control — 终止符自控 (该停收尾 / 该续递回)
# -----------------------------------------------------------------------------
def r_control(ctx: "EvalContext") -> RewardVector:
    """三档自控奖励 (dev-notes/61 §5)。cont_id is None → 旧版强制 <eos>。

    用 should_continue_reply(body) 判"该续/该停", 与实际终止符比对:
      该续却收尾(→eos) → -1.5 (话头冷场)
      该停却追问(→cont) → -1.5 (车轱辘)
      决策正确 → +1.2
      无终止符(超长) → -1.2
    命中 <eos> 且收尾正确但话太短 (<6) → 再扣 1.0 (未展开)。
    """
    v = RewardVector()
    body = ctx.body
    hit_eos = ctx.eos_id in ctx.reply_ids
    should_cont = rules.should_continue_reply(body)
    if ctx.cont_id is None:
        # 旧版: 强制必须吐 <eos>
        if hit_eos:
            v.r_quiet_eos += 1.2 if ctx.char_len >= 6 else -2.0
        else:
            v.r_quiet_eos -= 1.0
        return v
    hit_cont = ctx.cont_id in ctx.reply_ids
    if hit_eos and not hit_cont:
        if should_cont:
            v.r_quiet_eos -= 1.5
        else:
            v.r_quiet_eos += 1.2
            if ctx.char_len < 6:
                v.r_quiet_eos -= 1.0
    elif hit_cont and not hit_eos:
        if should_cont:
            v.r_quiet_eos += 1.2
        else:
            v.r_quiet_eos -= 1.5
    else:
        v.r_quiet_eos -= 1.2
    return v


# -----------------------------------------------------------------------------
# 维度 6: 积极阳光与主动互动
# -----------------------------------------------------------------------------
def r_sentiment(ctx: "EvalContext") -> RewardVector:
    v = RewardVector()
    t = ctx.reply_text
    if any(q in t for q in ["？", "?", "吗", "呢", "觉得", "如何", "一起"]):
        v.r_sentiment += 0.8
    if any(w in t for w in ["开心", "好呀", "阳光", "美好", "喜欢", "一起", "探索", "真好", "期待"]):
        v.r_sentiment += 0.8
    return v


# -----------------------------------------------------------------------------
# 维度 7: r_task — 任务意图/实体召回/自发命名
# -----------------------------------------------------------------------------
def r_task(ctx: "EvalContext") -> RewardVector:
    v = RewardVector()
    t, kind, kw = ctx.reply_text, ctx.kind, ctx.keywords
    if kind == "identity":
        if any(p in t for p in ["叫我", "我想叫", "可以叫我", "我叫", "名字叫"]):
            v.r_task += 2.5
        if sum(1 for k in kw if k in t) >= 1:
            v.r_task += 1.5
    elif kind == "memory":
        hits = sum(1 for k in kw if k in t)
        v.r_task += (3.5 if hits >= 2 else 2.0 if hits == 1 else -2.0)
    elif kind == "fact":
        if sum(1 for k in kw if k in t) >= 1:
            v.r_task += 2.5
    elif kind in ("travel", "mood", "heuristic"):
        if sum(1 for k in kw if k in t) >= 1:
            v.r_task += 2.0
    return v


# -----------------------------------------------------------------------------
# 维度 8: r_semantic — 轻量语义相关命中
# -----------------------------------------------------------------------------
def r_semantic(ctx: "EvalContext") -> RewardVector:
    v = RewardVector()
    sim = rules._semantic_similarity(ctx.reply_text, ctx.keywords)
    if sim > 0:
        v.r_semantic = min(2.5 * sim, 2.5)
    return v


# -----------------------------------------------------------------------------
# 维度 9: r_arith — 数学题解答 + 顺序列举豁免 (全局规则)
# -----------------------------------------------------------------------------
def r_arith(ctx: "EvalContext") -> RewardVector:
    """数学题奖励 (R1) + 顺序列举豁免与少许奖励 (R2, 全局)。

    R1 (数学题): 仅当 prompt 含算式 (ctx.arith 非 None) 时生效。按作答三态区分
      (dev-notes/66, 修废话流避难所):
        - 答对 (arith_correct): 回复的答案数字 == 正确结果 → 高分 +3.5
        - 给了答案但算错 (arith_attempted): 尝试作答方向对但错 → 中罚 -1.2
        - 压根没作答 (!arith_attempted): 废话流/闲聊/空 → **重罚 -2.8**。
          这是本会话数学学不会的根因修复: 旧版废话流只 -1.8, 却靠 natural/
          sentiment/control 拿回 +2.4~+2.9 → 净赚、成零风险高收益避难所。现在
          把"不答"从数学题的合法策略里剔除。
        - **多余数字** (宽松首数字之后、且不属于顺序列举) → 每个扣 0.8 (封顶 -1.6),
          惩罚往算式里掺杂质数字。注意: 即使答对了, 掺了多余数字也扣。

    R2 (顺序列举豁免, 全局): 回复里数字构成一段"+1 连续递增"列举 (长度>=3,
      如 1 2 3 / 13 14 15), 把这些列举数字从"多余数字扣分"里豁免, 并给少许
      +0.6 奖励。"列举作为全局奖励规则" = 不依赖算式, 一段干净的顺序列举本身
      就是可取的文本形态。
    """
    v = RewardVector()
    use_enum = ctx.enum_run_len >= rules.MIN_ENUM_RUN and len(ctx.extra_digits) >= rules.MIN_ENUM_RUN - 1
    # R2 全局列举奖励
    if use_enum:
        v.r_arith += 0.6
    # R1 数学题
    if ctx.arith_hit:
        if ctx.arith_correct:
            v.r_arith += 3.5
        elif ctx.arith_attempted:
            v.r_arith -= 1.2
        else:
            v.r_arith -= 2.8   # 拒答/废话流避难所剔除 (dev-notes/66)
        # 多余数字 (顺序列举豁免 R2): 宽松首数字之后的其余数字 token
        if use_enum:
            pass  # 列举的数字不扣
        elif ctx.extra_digits:
            v.r_arith -= min(1.6, 0.8 * float(len(ctx.extra_digits)))
    return v


# -----------------------------------------------------------------------------
# 维度 10: r_anti_collapse — 反坍缩 (惩罚 <cont> 滥刷 / 空递回)
# -----------------------------------------------------------------------------
def r_anti_collapse(ctx: "EvalContext") -> RewardVector:
    """对抗本会话实测的"空回复坍缩" (RL 全组空回复, eos=0/16 cont=16/16)。

    根因回顾: reward 里 `hit_cont & should_cont → +1.2` 是无条件正收益, 模型发现
    只要"内容 + 吗/呢?"就能稳定刷 +1.2, 于是退化成 5-12 字的空递回残句
    ("是什么吗?""你们能试着这样呢?") → 数学/自然度/coherence 分全崩, 唯独 control
    分稳定为正 → 全局收敛到"空递回"局部最优。

    修正: 对 <cont> 滥用与"空递回"显式扣分, 让 cont 的 +1.2 只在"言之有物且真需
    追问"时成立。
      - 空递回: 命中 <cont> 但正文极短 (<6 字) → 扣 2.5。空递回是坍缩主形态。
      - 短递回: 命中 <cont> 且正文 6~15 字 (可疑但可能ok) → 扣 1.0, 但仍允许
        r_control 给的部分正分 (过度扣会剪掉合法追问)。
      - 无条件追尾: 命中 <cont> 但 should_continue_reply = False (车轱辘重复追问,
        不可能是"真需用户继续") → 已由 r_control -1.5 处理, 这里不重复。
      - 整体过短收尾 (<4 且 <eos> 且非空): 其实 r_natural 已 -2.5, 不叠加。
    """
    v = RewardVector()
    if ctx.cont_id is None:
        return v
    L = ctx.char_len
    # ── 非 <cont> 型单字空壳 (如 `"`+<eos> 逃避解): 专治本会话最顽固的坍缩形态。
    #    单字符正文(<6)但既没合理追问也没任何内容的极短回复, 就是"哑巴装死"的近亲,
    #    即便没吐 <cont> 也要重扣 —— 否则组归一化里它能和更烂的候选互相掩护。
    #    注意: L==0 已由 r_fatal_empty 扣 -3.5 → 这里只处理 L 在 [1,2] 的非空单字壳。
    if not (ctx.eos_id in ctx.reply_ids or ctx.cont_id in ctx.reply_ids):
        pass  # 完全没终止符的暂由长短惩罚处理, 不双算
    elif L == 1 and not ctx.arith_correct:
        v.r_anti_collapse -= 3.0   # 单字空壳(如 `"`): 最严重
    elif L == 2 and not ctx.arith_correct:
        v.r_anti_collapse -= 1.5   # 二字残壳
    hit_cont = ctx.cont_id in ctx.reply_ids
    if not hit_cont:
        return v
    if L == 0:
        v.r_anti_collapse -= 2.5      # 纯 <cont> 空递回 (最严重)
    elif L < 6:
        v.r_anti_collapse -= 2.5      # 空递回 (<"6" 字)
    elif L < 15:
        v.r_anti_collapse -= 1.0      # 短递回, 存疑
    return v


# -----------------------------------------------------------------------------
# 注册表: 有序维度列表 (引擎按此顺序逐个执行并求和)
# -----------------------------------------------------------------------------
REGISTRY: dict = {
    "fatal_empty": r_fatal_empty,
    "coherence": r_coherence,
    "natural": r_natural,
    "anti_repeat": r_anti_repeat,
    "anti_robotic": r_anti_robotic,
    "control": r_control,
    "sentiment": r_sentiment,
    "task": r_task,
    "semantic": r_semantic,
    "arith": r_arith,
    "anti_collapse": r_anti_collapse,
}