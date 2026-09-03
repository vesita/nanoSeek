#!/usr/bin/env python3
"""全局奖励规则 (纯函数库): 数学题解答 + 顺序列举豁免 + 语义/自控辅助。

本模块是"全局奖励规则"的家: 与维度插件正交, 是所有维度可共用的判定原语。
关键全局规则 (dev-notes/63):
  R1  数学题: 若 prompt 含「100 以内加减乘除」算式, 以回复里
       用**空格切分**后的**第一个数字**作为奖励点;
       之后的"多余数字"会扣分 (否则等于往算式里掺杂质数字)。
  R2  顺序列举豁免 (全局, 不只在数学题触发): 若回复的数字 token 构成一段
       "+1 连续递增"的列举 (如 1 2 3 / 3 4 5), 则把这些列举数字从"多余数字"
       扣分中豁免, 并给少许奖励。"列举作为全局奖励规则" = 即使没有算式,
       一段干净的顺序列举也是可取的文本形态。

其余: _ngram_set / _semantic_similarity (轻量语义) 与 should_continue_reply /
CONTINUE_PHRASE_R (r_control 自控) 也从这里提供, 供维度函数使用。
"""
import math
import re
from typing import List, Optional, Tuple

ROBOTIC_TAGS = ["用户", "模型", "user", "assistant", "system", "Human:", "Assistant:"]
COUNSELING_TEMPLATES = [
    "放松不下来", "最表面那层", "顺一顺这口气", "先松一点", "心里这团", "哪件事卡着",
    "特别耗神", "先不用把后面想完", "心里发慌", "最磨人的不是大事", "身体先绷住"
]

# -----------------------------------------------------------------------------
# 算术题判定与求解 (纯字符串层, 与词表无关)
# -----------------------------------------------------------------------------
# 支持: 加法 + 连加「X加Y(得)」减法 乘法 整除除法; 数值 0~100, 结果须为 0~无穷整数。
# 算子集合: ASCII 符号、中文单字、全角符号。
ARITH_OP = r"[+\-×xX*/＋－×÷]"
ARITH_PATTERN = re.compile(
    r"(?<!\d)(\d{1,3})\s*" + ARITH_OP + r"\s*(\d{1,3})(?!\d)"
)
# 中文算式: N加N / N加上N / N减N / N乘N / N除以N / N等于?  (字间允许零宽/空白)
CN_ARITH = re.compile(
    r"(?<!\d)(\d{1,3})\s*(加|加上|减|乘|乘以|除以|除)\s*(\d{1,3})(?!\d)"
)

_OPS = {
    "+": lambda a, b: a + b, "-": lambda a, b: a - b,
    "×": lambda a, b: a * b, "*": lambda a, b: a * b, "x": lambda a, b: a * b,
    "X": lambda a, b: a * b, "＊": lambda a, b: a * b, "÷": lambda a, b: a / b,
    "/": lambda a, b: a / b,
    "加": lambda a, b: a + b, "加上": lambda a, b: a + b,
    "减": lambda a, b: a - b, "乘": lambda a, b: a * b, "乘以": lambda a, b: a * b,
    "除": lambda a, b: a / b, "除以": lambda a, b: a / b,
}


def solve_arith(a: int, op: str, b: int) -> Optional[int]:
    """算出「a op b」的整数结果; 无法整除或除数为零返回 None。"""
    fn = _OPS.get(op)
    if fn is None:
        return None
    val = fn(a, b)
    # 除法要整除才算"100 以内四则运算"里可判的题; 其余任意实数都收
    if op in ("÷", "/", "除", "除以"):
        if b == 0 or float(val) != int(val):
            return None
        return int(val)
    return int(val)


def extract_arith(prompt: str) -> Optional[Tuple[int, str, int, int]]:
    """若 prompt 含一个有效算式则返回 (a, op, b, result), 否则 None。

    只认第一个命中; 结果必须是可求值的整数 (整除除法), 且 a,b 都在 0~100。
    返回的 op 保持原始符号, 供日志展示。
    """
    if not prompt:
        return None
    for pat in (ARITH_PATTERN, CN_ARITH):
        m = pat.search(prompt)
        if not m:
            continue
        # 从整段匹配里抠出中间算子: 去掉首尾数字 (允许算子前有空白/全角空格)
        whole = m.group(0)
        op = ""
        for ch in whole:
            if ch in _OPS:
                op += ch
        if not op:
            op = "+"
        a = int(m.group(1))
        try:
            b = int(m.group(3))
        except IndexError:
            b = int(m.group(2))
        r = solve_arith(a, op, b)
        if r is not None and 0 <= a <= 100 and 0 <= b <= 100:
            return (a, op, b, r)
        return None
    return None


# -----------------------------------------------------------------------------
# 回复首数字提取 (空格切分) 与 多余数字 / 顺序列举判定
# -----------------------------------------------------------------------------
def extract_first_digit(reply: str) -> Tuple[Optional[int], List[int]]:
    """返回 (第一个数字, 其余数字) — 依「空格切分」语义 (兼容旧签名与调用方)。

    规则: 把回复按空白切成 token, 第一个能整体 int() 的词即"答案数字";
    它后面的所有纯数字 token 都算"多余数字" (R1)。

    NOTE(实测): 基座从不会吐空格分隔的裸数字 "7 ", 而是 "7," "答案是7，..." 这类
    数位夹在标点/汉字里的形态。若只看空格 token, 正确数字会被漏判 → arith -1.2,
    即使答对也拿不到分 → 数学维度完全失效。因此**额外**用正则抓"游离的阿拉伯数字串"
    作为宽松首数字 (see first_digit_loose)。随机非枚举数字命中同样的启发式, 由调用方
    用 arith_correct (答案精确匹配) 收敛, 避免误报。
    """
    tokens = reply.strip().split()
    first = None
    rest: List[int] = []
    seen_first = False
    for t in tokens:
        if t.isdigit():
            v = int(t)
            if not seen_first:
                first = v
                seen_first = True
            else:
                rest.append(v)
    return first, rest


_FIRST_DIGIT_RE = re.compile(r"(\d{1,3})")   # 抓回复里第一个阿拉伯数字串 (任意位置)


def first_digit_loose(reply: str) -> Optional[int]:
    """宽松首数字: 回复里**第一个**阿拉伯数字串 (可夹标点/汉字), 无则 None。

    用于 R1 数学题: 判断回复是否出现正确结果 (即便写作 "答案是7，" / "7," )。
    只认 1~3 位、首字符非 0 前缀 (防把 "0.6" 之类当首数字的歧义 —— 本项目结果均为
    非零整数, 直接用 int() 语义安全)。
    """
    m = _FIRST_DIGIT_RE.search(reply or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def ordered_enum_run_len(digits: List[int]) -> int:
    """回复中数字 token 能拼出的最长『+1 连续递增』序列长度。

    用于 R2 顺序列举豁免: 长度 >= MIN_ENUM_RUN 视为"顺序列举", 其数字从
    '多余数字扣分' 中豁免并给少许奖励。仅看数值相邻 +1, 不要求连续存在于
    token 位置 (因为回复里可能夹杂文字), 但要求出现在同一回复里。
    """
    if not digits:
        return 1
    uniq = sorted(set(digits))
    best = 1
    cur = 1
    for i in range(1, len(uniq)):
        if uniq[i] == uniq[i - 1] + 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


MIN_ENUM_RUN = 3   # 连续 +1 至少 3 个才算是"列举"(1 2 3 / 4 5 6)

# -----------------------------------------------------------------------------
# 轻量语义相似度 (Lightweight Semantic Similarity)
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# r_control 自控判定 (待续/递回 vs 收尾) — 镜像 data/chinese/prepare.py
# -----------------------------------------------------------------------------
CONTINUE_PHRASE_R: tuple = (
    "你觉", "你呢", "怎么样", "是不是", "想不想", "要不要",
    "你说呢", "对不对", "如何", "可以吗", "好吗", "吗?", "吗？"
)


def should_continue_reply(reply: str) -> bool:
    """回复是否属"待续/递回(期待用户继续)"类型。启发式: 含疑问标点或追问/递回短语。"""
    r = reply.strip()
    if not r:
        return False
    for q in ("？", "?"):
        if q in r:
            return True
    for p in CONTINUE_PHRASE_R:
        if p in r:
            return True
    return False