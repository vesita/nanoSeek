#!/usr/bin/env python3
"""配置驱动的数学题生成与解析（格式化噪音）。

设计（dev-notes 待续）:
  - 基础规则在 `arith_rules.toml`：数字范围 / 每种运算的表面写法 / 问句模板。
  - **生成**（提示词池）：代码从配置展开，随机挑运算符写法（`+`/`＋`/`加`/`加上`…）
    与问句模板，产生大量"同语义、不同表面"的数学题 —— 即格式化噪音。
  - **解析**（奖励闭环）：用同一份配置构建识别正则，`extract_arith(prompt)` 返回
    `(a, op, b, result)`（签名与 `rules.extract_arith` 一致，奖励引擎可无缝替换），
    保证"提示词用到的每一种写法，奖励一定认得出"。

范围约束：操作数 a、b ∈ [min_num, max_num]（默认 0~100，含端点）；
`require_nonneg_result=true` 时结果须非负；除法要求整除且除数非零。
"""
from __future__ import annotations

import random
import re
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_RULES_PATH = Path(__file__).parent / "arith_rules.toml"

# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------

_FN = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,  # 整除校验在外层做
}


@lru_cache(maxsize=1)
def load_rules() -> dict:
    """读取 arith_rules.toml（进程内缓存一次）。"""
    with open(_RULES_PATH, "rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------------------
# 表面写法 → 求值 映射 + 识别正则（与提示词生成共用同一份配置）
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _spelling_map() -> Dict[str, Tuple[str, str]]:
    """{ 表面写法 : (op_name, fn_name) }。"""
    rules = load_rules()
    m: Dict[str, Tuple[str, str]] = {}
    for op in rules["op"]:
        for s in op.get("symbols", []):
            m[s] = (op["name"], op["fn"])
        for w in op.get("words", []):
            m[w] = (op["name"], op["fn"])
    return m


@lru_cache(maxsize=1)
def _extractor() -> re.Pattern:
    r"""算式识别正则：`(?<!\d)(\d{1,3})\s*(OP)\s*(\d{1,3})(?!\d)`。

    OP 交替 = 全部表面写法，按长度降序排列（长词优先，防 `加` 吞 `加上` 前缀）。
    """
    spellings = sorted(_spelling_map().keys(), key=len, reverse=True)
    op_alt = "|".join(re.escape(s) for s in spellings)
    return re.compile(rf"(?<!\d)(\d{{1,3}})\s*({op_alt})\s*(\d{{1,3}})(?!\d)")


def extract_arith(prompt: str) -> Optional[Tuple[int, str, int, int]]:
    """若 prompt 含一个有效算式则返回 `(a, op, b, result)`，否则 None。

    覆盖配置里全部运算符写法；除法要求整除且除数非零；
    `require_nonneg_result` 时结果须非负。
    """
    if not prompt:
        return None
    m = _extractor().search(prompt)
    if not m:
        return None
    a, spelling, b = int(m.group(1)), m.group(2), int(m.group(3))
    _op_name, fn_name = _spelling_map()[spelling]
    cfg = load_rules()["arith"]
    if not (cfg["min_num"] <= a <= cfg["max_num"] and cfg["min_num"] <= b <= cfg["max_num"]):
        return None
    fn = _FN[fn_name]
    if fn_name == "div":
        if b == 0:
            return None
        val = fn(a, b)
        if float(val) != int(val):  # 不整除 → 不可判题
            return None
        result = int(val)
    else:
        result = int(fn(a, b))
    if cfg.get("require_nonneg_result", False) and result < 0:
        return None
    return (a, spelling, b, result)


# ---------------------------------------------------------------------------
# 生成：带格式化噪音的提示词池
# ---------------------------------------------------------------------------

def _gen_valid_pair(op_def: dict, rng: random.Random, lo: int, hi: int, nonneg: bool, result_max: int,
                    exclude_trivial: bool = True):
    """生成满足约束的 (a, b, result)；不满足则重试。

    约束: a、b ∈ [lo, hi]；`nonneg` 时 result ≥ 0；**result ≤ result_max**
    （"100 以内"指操作数**和结果**都不超界，乘法 66×46=3036 这类要排除）。
    `exclude_trivial` 时排除平凡题：操作数含 0，或 乘/除 出现 1（a×1/1×a/a÷1/a÷a）——
    平凡题让模型靠回声/模式匹配蒙答案，学不到真算术。
    """
    fn_name = op_def["fn"]
    for _ in range(1024):
        a = rng.randint(lo, hi)
        b = rng.randint(lo, hi)
        if fn_name == "div":
            if b == 0 or a % b != 0:
                continue
            result = a // b
        else:
            result = int(_FN[fn_name](a, b))
        if nonneg and result < 0:
            continue
        if result > result_max:
            continue
        if exclude_trivial:
            if a == 0 or b == 0:
                continue
            if fn_name in ("mul", "div") and (a == 1 or b == 1):
                continue
            if fn_name == "div" and a == b:   # a÷a=1 也是平凡题
                continue
        return a, b, result
    # 极端兜底（0 加法恒满足；平凡约束下几乎到不了这里）
    return 0, 0, 0


def gen_prompts(count: int = 24, seed: Optional[int] = None) -> List[Tuple[str, List[str]]]:
    """生成 `count` 条带格式化噪音的数学题提示词。

    返回 `[(prompt, keywords)]`——keywords 是**完整作答短语**（"答案是7"/"等于7"/
    "7"… 及中文数字版），驱动 `r_semantic` 相似度维度，让模型往"写出答案"的方向靠。
    结构上与 `grpo_char.ARITH_PROMPTS` 池元素一致。
    """
    cfg = load_rules()
    arith = cfg["arith"]
    ops: List[dict] = cfg["op"]
    questions: List[dict] = cfg["question"]
    rng = random.Random(seed)

    out: List[Tuple[str, List[str]]] = []
    for _ in range(count):
        op = rng.choice(ops)
        a, b, result = _gen_valid_pair(
            op, rng, arith["min_num"], arith["max_num"],
            arith.get("require_nonneg_result", True), arith["max_num"],
            arith.get("exclude_trivial", True),
        )
        spellings = list(op.get("symbols", [])) + list(op.get("words", []))
        spelling = rng.choice(spellings) if spellings else op["name"]
        q = rng.choice(questions)["template"]
        prompt = q.replace("{a}", str(a)).replace("{op}", spelling).replace("{b}", str(b))

        # 期望作答短语：覆盖阿拉伯/中文数字的常见表达 (r_semantic 用)
        ans_arab = str(result)
        ans_cn = number_to_chinese(result)
        kw: List[str] = [ans_arab]
        for phrase in ("等于", "答案是", "等于", "就是", "应该是"):
            kw.append(f"{phrase}{ans_arab}")
        kw.append(f"{a}{spelling}{b}等于{ans_arab}")
        if ans_cn != ans_arab:
            kw.append(ans_cn)
            for phrase in ("等于", "答案是"):
                kw.append(f"{phrase}{ans_cn}")
        out.append((prompt, kw))
    return out


# ---------------------------------------------------------------------------
# 中文数字 0~100 转换（奖励端识别"七"这类答案）
# ---------------------------------------------------------------------------

_DIGIT = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3,
          "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def chinese_to_int(s: str) -> Optional[int]:
    """解析 0~100 中文数字（零/一…/十/百），失败返回 None。

    支持写法: 零 七 十 十五 二十 二十一 一百 一十二 …
    """
    if not s:
        return None
    if s in ("百", "一百"):
        return 100
    if s in ("零", "〇"):
        return 0
    if s == "十":
        return 10
    # 单字（个位数）
    if len(s) == 1:
        return _DIGIT.get(s)
    # 十X / X十[Y]
    if s.startswith("十"):
        y = s[1]
        return 10 + _DIGIT.get(y, -10) if len(s) == 2 and y in _DIGIT else None
    if "十" in s:
        parts = s.split("十")
        if len(parts) != 2:
            return None
        tens_s, ones_s = parts
        tens = 1 if tens_s == "" else _DIGIT.get(tens_s)
        ones = 0 if ones_s == "" else _DIGIT.get(ones_s)
        if tens is None or ones is None:
            return None
        return tens * 10 + ones
    return None


def number_to_chinese(n: int) -> str:
    """0~100 → 中文数字。"""
    if not 0 <= n <= 100:
        return str(n)
    if n == 100:
        return "一百"
    if n == 0:
        return "零"
    digits = "零一二三四五六七八九"
    if n < 10:
        return digits[n]
    if n < 20:
        return "十" + (digits[n % 10] if n % 10 else "")
    tens, ones = divmod(n, 10)
    s = digits[tens] + "十"
    if ones:
        s += digits[ones]
    return s


_CN_RUN_RE = re.compile(r"[零〇一二两三四五六七八九十百]+")

# "答案标记"：优先在这些标记**之后**找数字, 避免把复述题目里的操作数当成答案
# (如 "3加4等于7" 的首数字是 3, 但答案 7 在 "等于" 之后)。
# 注意: 不把裸 "是" 当标记 —— "我算出来是3加4等于7" 里 "是" 后紧跟操作数,
# 裸 "是" 会误抓 3; 而 "答案是7" 这类有 "答案/等于" 前缀或走兜底首数字即可。
_ANSWER_MARKER_RE = re.compile(
    r"(?:等于|答案|結果|结果|得|为|就是|应该是|:|：|=)\s*([0-9零〇一二两三四五六七八九十百]+)"
)
# 若上述标记后没抓到数字, 退回去找"位于算式之后"的数字: 算式形态 "N op N ... N"
_ARITH_TAIL_RE = re.compile(
    r"(?<!\d)(\d{1,3})\s*[+\-×xX*/＋－×÷加加上减减去减掉乘乘以乘上除以除]\s*(\d{1,3})[^\d零〇一二两三四五六七八九十百]*"
    r"([0-9零〇一二两三四五六七八九十百]+)"
)


def extract_answer_number(reply: str) -> Optional[int]:
    """从回复中提取"答案数字"（阿拉伯或中文），None 表示没找到。

    优先级:
      1. "等于/答案/是/得/=..." 标记之后的数字 (最可靠);
      2. 算式（操作数 运算符 操作数）之后的数字;
      3. 兜底: 全回复里第一个数字 (兼容 "7" 这类裸答案)。
    """
    if not reply:
        return None
    m = _ANSWER_MARKER_RE.search(reply)
    if m:
        s = m.group(1)
        if s.isdigit():
            return int(s)
        cn = chinese_to_int(s)
        if cn is not None:
            return cn
    m = _ARITH_TAIL_RE.search(reply)
    if m:
        s = m.group(3)
        if s.isdigit():
            return int(s)
        cn = chinese_to_int(s)
        if cn is not None:
            return cn
    # 兜底: 宽松首数字 / 中文数字
    fd = re.search(r"(\d{1,3})", reply)
    if fd:
        return int(fd.group(1))
    cm = _CN_RUN_RE.search(reply)
    if cm:
        return chinese_to_int(cm.group(0))
    return None


def loose_chinese_digit(reply: str) -> Optional[int]:
    """宽松中文数字：回复里**第一个**中文数字串（可夹标点/汉字），解析为 0~100。

    与 `rules.first_digit_loose` 的阿拉伯数字版对应，识别"答案是七"这类中文答案。
    """
    m = _CN_RUN_RE.search(reply or "")
    if not m:
        return None
    return chinese_to_int(m.group(0))


def has_answer_intent(reply: str) -> bool:
    """数学题上是否出现"明确作答动作"（对错都算）。

    用于 r_arith 区分"答错"与"压根没作答(废话流/闲聊)":
      - 答错(attempted): 回复里出现阿拉伯数字作答信号 → True
      - 拒答(not attempted): 只是像话的废话/闲聊 → False

    为何用"阿拉伯数字"做主信号 (dev-notes/66):
      - 模型要为数学题作答, 输出的几乎必是阿拉伯数字 ("等于67"/"答案是35"/裸"67").
        废话流里**几乎不可能碰巧带阿拉伯数字** → 干净可靠。
      - 反之汉字数字("一起/那个一")在废话里常见, 不能当作答信号 → 用纯中文数字作答
        (整段就是 "二十四") 单独识别, 不用 fallback 在正文里捞零散汉字数字。
      因此绝不调用 extract_answer_number 的兜底(首个任意数字)。
    """
    if not reply:
        return False
    if re.search(r"\d", reply):                  # 任何阿拉伯数字 = 作答信号
        return True
    if _ANSWER_MARKER_RE.search(reply):          # 答案标记 (中文数字场景兜一层)
        return True
    stripped = (reply or "").strip().strip("。，,.、:： ;；\n")
    if _CN_RUN_RE.fullmatch(stripped):           # 纯中文数字作答 "二十四"
        return chinese_to_int(stripped) is not None
    return False


# ---------------------------------------------------------------------------
# 简易自检（python training/rl/reward/arith_gen.py）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    prompts = gen_prompts(10, seed=42)
    print(f"=== 配置驱动数学题 {len(prompts)} 条 (seed=42) ===")
    for p, kw in prompts:
        ar = extract_arith(p)
        print(f"  {p!r:30} -> {ar}  kw={kw}")

    print("\n=== extract_arith 变体覆盖 ===")
    for s in ["3加4等于几？", "3加上4等于多少？", "15减7等于？", "8乘3等于多少？",
              "24除以6等于几？", "20减去13等于？", "12×9等于几？", "7＋8等于多少？",
              "6乘上7等于几？", "100除以10等于多少？"]:
        print(f"  {s!r:26} -> {extract_arith(s)}")

    print("\n=== 中文数字转换 ===")
    for n in [0, 7, 10, 12, 20, 21, 100]:
        cn = number_to_chinese(n)
        print(f"  {n} -> {cn!r} -> {chinese_to_int(cn)}")
