#!/usr/bin/env python
"""WildChat 中文子集的 PII 检测 / 脱敏工具（默认 dry-run，`--apply` 才写数据）。

## 为什么要它

`data/chinese/clean_v3/wildchat_zh.txt` 是**真实用户对话**（allenai/WildChat-1M 中文子集，
经 HF `benchang1110/WildChat-Chinese` 转换），和四大名著 / c4_zh 那种"没有隐私主体"的
语料不同：里面夹着真人手机号、邮箱、快递收件人姓名+地址块。上游只筛过 toxicity，
**没筛过 PII**。

## 口径（先看数据再写规则，见 `analysis/wildchat_pii.md` 侦察一节）

规则**只覆盖"可正则判定 + 有校验位可交叉验证"的类别**。凡是需要语义理解才能判定的
（人名、住址、单位名、缩写化联系方式），本工具**不碰** —— 宁可漏，不可误删。
误删的代价是：把正常中文里的 4~6 位数字（年份 / 价格 / 编号）改掉，模型学到
`<PHONE>` 这种噪声 token。所以每条规则都要有"负对照"。

★ **例外：名单块整块删除。** 姓名 + 手机 + 完整住址的"名单"形态不是正则能清的 ——
把手机号换成 `<PHONE>` 之后，姓名 + 门牌号仍然可定位到具体的人。所以对满足
"不同手机号 >= 3 且其中 >= 3 个的 ±30 字内有 >= 3 个地址字"的块，**整块删掉**
（判据与实测间隔见 `find_roster_blocks` 的注释）。这是本工具唯一会**删内容**的地方，
用 `--no-block-delete` 可以关掉。

## 用法

    # 1) 自检（正对照 / 负对照 / 幂等 / 故意失败注入点）
    .venv/bin/python data/chinese/pii_scrub.py --selftest

    # 2) dry-run：只统计 + 抽样，不写任何文件
    .venv/bin/python data/chinese/pii_scrub.py

    # 3) 真写（`--out` 必须显式给，且不能等于 `--in`）
    .venv/bin/python data/chinese/pii_scrub.py \
        --out data/chinese/clean_v3/wildchat_zh.scrubbed.txt --apply

## 纪律

* 默认 dry-run，与 `data/chinese/clean_corpus.py` / `scripts/cleanup_out.py` 一致。
* **绝不修改 `--in`**：`--apply` 只往 `--out` 写；`out == in` 直接报错。
* 脱敏**保形**：每类替换成可读占位符（`13812345678` → `<PHONE>`），
  不用 `***`，因为 `***` 会和 markdown 正文混淆、也看不出类别。
* **幂等**：占位符里不含任何规则能再次命中的字符；`--selftest` 与 dry-run 都会
  实跑第二遍并断言 `scrub(scrub(x)) == scrub(x)`。
* 规则命中**必须过校验器**才算（18 位身份证的 ISO 7064 校验位、银行卡 Luhn），
  没过的候选会记进 `rejected` 统计 —— 这样"候选多但真 PII 少"是看得见的。
* 确定性：同样的输入永远同样的输出（无随机、无时间戳、无字典序依赖）。

## `--inject-bug`（自检的"故意失败"注入点）

`--selftest --inject-bug <rule>` 会把该条规则的正则在**运行时**改成一个永不匹配的模式，
自检**必须**变红并非零退出。这是为了证明"自检不是永远绿的"——
一个从来不会失败的测试等于没有测试。
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import os
import re
import sys
from dataclasses import dataclass, field

DEFAULT_IN = 'data/chinese/clean_v3/wildchat_zh.txt'
RULE_ORDER_NOTE = '按列表顺序定优先级（越前越高）；重叠时高优先级胜出'

# ==========================================================================
# 校验器（纯函数，便于单测/自检）
# ==========================================================================
# 18 位身份证：ISO 7064:1983 MOD 11-2
ID_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
ID_CHECK = '10X98765432'
# 省级行政区前两位（沿用 GB/T 2260 的合法取值；不校验具体到市）
ID_PROVINCE = {
    '11', '12', '13', '14', '15', '21', '22', '23', '31', '32', '33', '34',
    '35', '36', '37', '41', '42', '43', '44', '45', '46', '50', '51', '52',
    '53', '54', '61', '62', '63', '64', '65', '71', '81', '82', '91',
}


def id18_valid(num: str, start: int = 0, end: int = 0, text: str = '') -> bool:
    """18 位身份证：省级码 + 合法生日 + 校验位。"""
    if not re.fullmatch(r'\d{17}[\dXx]', num):
        return False
    if num[:2] not in ID_PROVINCE:
        return False
    body = num[:17]
    try:
        datetime.date(int(body[6:10]), int(body[10:12]), int(body[12:14]))
    except ValueError:
        return False
    s = sum(int(a) * w for a, w in zip(body, ID_WEIGHTS))
    return ID_CHECK[s % 11] == num[17].upper()


def luhn_valid(num: str, start: int = 0, end: int = 0, text: str = '') -> bool:
    """银行卡 Luhn 校验（16~19 位纯数字）。"""
    digits = re.sub(r'\D', '', num)
    if not 16 <= len(digits) <= 19:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# 卡号附近的"这就是银行卡"关键词（20 字窗口内）
CARD_KEYWORDS = ('银行卡', '銀行卡', '信用卡', '借记卡', '借記卡', '储蓄卡', '儲蓄卡',
                 '银联', '銀聯', '卡号', '卡號', '卡号是', '我的卡')


def bank_card_valid(num: str, start: int = 0, end: int = 0, text: str = '') -> bool:
    """真卡号判据：非前导 0 + Luhn + 前后 20 字内有银行卡关键词。

    为什么排除前导 0：内核 oops 日志里的 `0000000025487003` 之类补零地址
    **有 1/10 的概率蒙对 Luhn**（实测本语料 14 个 Luhn-valid 候选里 9 个是它）。
    真卡号不会以 0 开头。

    为什么**连分组写法也要求关键词**（这是实测改出来的）：
    "4-4-4-4 分组 + Luhn" 单独用会误删正常文本 —— 本语料实测两条：
      * `2023 2024 2025 2026`（连续年份）→ 拼成 2023202420252026，Luhn 蒙对
      * `[2000 2200 2400 2600 7200 7400 7600 7800]`（MATLAB 频率数组）→
        `7200 7400 7600 7800` 拼成 7200740076007800，Luhn 也蒙对
    4 连 4 位数字只要够多，蒙对 Luhn 是**必然事件**（约 1/10 的概率 × 候选数）。
    代价：**裸写且无上下文的真卡号会漏检**（本语料 0 条真卡）—— 已写进报告盲区。
    """
    digits = re.sub(r'\D', '', num)
    if (not digits) or digits[0] == '0':
        return False
    if not luhn_valid(digits):
        return False
    win = text[max(0, start - 20):min(len(text), end + 20)]
    return any(k in win for k in CARD_KEYWORDS)


# ==========================================================================
# 规则表
# ==========================================================================
@dataclass
class Rule:
    name: str
    pattern: str
    placeholder: str
    note: str
    validator: object = None          # callable(str) -> bool，None = 不过校验
    reject_note: str = ''
    flags: int = 0
    regex: re.Pattern = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.regex = re.compile(self.pattern, self.flags)


# --------------------------------------------------------------------------
# 前置边界：**必须把"字面转义序列"当成边界**
#
# 主 AI 复核时抓到的最小复现：`'彭培泉\\n\\n18959938568\\n\\n福建省'` 命中 0。
# 根因：WildChat 里有一段 C# 题面，把 5 人名单写成**字符串字面量**，
# 换行是**字面 `\n`**（反斜杠 + n 两个字符，码位 0x5c 0x6e）。号码前一个字符
# 因此是字母 `n`，被 `(?<![\dA-Za-z])` 挡掉 —— 实测漏掉 2 个真手机号。
#
# `(?<=\\[A-Za-z])`（宽度 2，定长）覆盖 `\n` / `\r` / `\t` / `\f` / `\v` 等
# 所有"反斜杠 + 单字母"转义；与定长 lookbehind 的择一在 Python `re` 里合法。
# 注意它不会误放行 `\alpha123...` 这种：那要求紧邻的两个字符是 `\a`。
# --------------------------------------------------------------------------
LB_ALNUM = r'(?:(?<![\dA-Za-z])|(?<=\\[A-Za-z]))'
LB_CJK = r'(?:(?<![A-Za-z0-9\u4e00-\u9fff])|(?<=\\[A-Za-z]))'


def lb_alnum(*extra: str) -> str:
    """前置边界 + 额外要排除的字符（如 `.` `/`）。转义豁免始终保留。"""
    cls = '\\dA-Za-z' + ''.join(re.escape(c) for c in extra)
    return r'(?:(?<![' + cls + r'])|(?<=\\[A-Za-z]))'


# ⚠ 顺序即优先级。理由：更"具体"（更长、更难伪装）的类别放前面，
# 避免一个 16 位卡号候选把里面的 11 位手机候选吃掉后又被 Luhn 否掉。
RULES: list[Rule] = [
    Rule(
        name='email',
        # `(?<!\\)`：防止字面 `\nzhangsan@qq.com` 里的 `n` 被吞进 local-part
        # （否则会匹配成 `nzhangsan@qq.com`，留下一个孤立的 `\`）
        pattern=r'(?<!\\)[A-Za-z0-9._%+\-]+@[A-Za-z0-9](?:[A-Za-z0-9.\-]*[A-Za-z0-9])?\.[A-Za-z]{2,}',
        placeholder='<EMAIL>',
        note='标准邮箱。本语料 31 命中里多数是 `example.com` 文档占位地址，'
             '只有 4 个看着像真的 —— 但占位地址也是 PII 形状，工具无法分辨真假，一律脱敏。',
    ),
    Rule(
        name='url_cred',
        pattern=LB_ALNUM + r'(?<!\\)[A-Za-z][A-Za-z0-9+.\-]*://[^\s/@:]{1,64}:[^\s/@]{1,64}@[^\s/?#\s]{1,253}',
        placeholder='<URL_CRED>',
        note='URL userinfo（`scheme://user:pass@host`）。整条替换，因为 userinfo 本身'
             '就是凭据，且 scheme/host 一起删掉才不会留下"哪台机器"的线索。',
    ),
    Rule(
        name='id18',
        pattern=LB_ALNUM + r'\d{17}[\dXx](?![\dA-Za-z])',
        placeholder='<ID>',
        note='18 位身份证。正则只做形状，**必须**再过 ISO 7064 MOD 11-2 校验位 + '
             '合法生日 + 省级码；否则 `1010處為0.000951056516295151，1011...` 这种'
             '小数连排会大面积误伤。',
        validator=id18_valid,
        reject_note='形状像 18 位身份证但校验位/生日/省码不合法（本语料全部是这种，'
                    '即 0 个真身份证）',
    ),
    Rule(
        name='bank_card',
        pattern=lb_alnum('.', '/') + r'(?:\d{4}[ \-]\d{4}[ \-]\d{4}[ \-]\d{4}(?:[ \-]\d{1,3})?|\d{16,19})(?![\dA-Za-z])',
        placeholder='<BANKCARD>',
        note='16~19 位银行卡。过 Luhn + 非前导 0，且要求"4-4-4-4 分组写法"或'
             '"前后 20 字内出现 银行卡/信用卡/卡号/卡號/借记卡/儲蓄卡/銀聯"。'
             '**不要求**的话，`優惠id:2102023122853299`（优惠券 id，19 位）和'
             '`/post/7320037969980637225`（掘金帖子 id）会蒙对 Luhn 被误删。',
        validator=bank_card_valid,
        reject_note='16~19 位数字但 Luhn 不过 / 前导 0 / 裸数字附近无银行卡关键词'
                    '（本语料里是内核地址、浮点尾数、论文 id、优惠券 id）',
    ),
    Rule(
        name='mobile_cn',
        pattern=LB_ALNUM + r'(?:\+?86[ \-]?)?1[3-9]\d(?:[ \-]?\d{4}){2}(?![\dA-Za-z])',
        placeholder='<PHONE>',
        note='中国大陆手机号，支持 `13812345678` / `+86138...` / `138 1234 5678` / '
             '`138-1234-5678`，**且支持前置字面转义 `\\n` / `\\r\\n`**（见 LB_ALNUM 注释：'
             '这是主 AI 复核抓到的漏检，实测 +2 个真号）。**已知假阳性**：11 位纯数字的'
             '算术结果（`264503*69258=18304210874`）和俄文统计表里的 `14000000000` 蒙对形状，'
             '本语料各占 2 条和 5 条，工具无法从形状上区分。',
    ),
    Rule(
        name='landline',
        pattern=LB_ALNUM + r'0[1-9]\d{1,2}[ \-]\d{7,8}(?:[-转]\d{1,5})?(?![\dA-Za-z])',
        placeholder='<TEL>',
        note='座机，**强制要求区号与号码之间有分隔符**（`0774-3939888`）。'
             '不强制分隔符时，hex 串 `0123456789abcdef`、小数 `0.00100000005` 会命中 —— '
             '实测无分隔符版本 64 命中里真座机只有 2 条。',
    ),
    Rule(
        name='qq',
        pattern=r'(?i)(?:我(?:的)?\s*)?(?:QQ|扣扣|企鹅号|企鵝號|腾讯QQ|騰訊QQ)\s*(?:号|號|号码|號碼|群)?\s*'
                r'(?:是|为|為|：|:|＝|=|加|联系|聯絡)\s*\d{5,12}',
        placeholder='QQ号：<QQ>',
        note='**显式自述**的 QQ 号（`我的QQ是12345678` / `QQ号：12345678`）。'
             '必须带"是/为/：/="这类自述连接词，否则 `QQ音乐2024` 之类会误伤。'
             '裸 5~12 位数字不碰（和编号无法区分）。',
    ),
    Rule(
        name='wechat',
        pattern=r'(?i)(?:微信|微訊|VX|V信|威信|WeChat|weixin)\s*(?:号|號|号码|號碼|ID|id)?\s*'
                r'(?:是|为|為|：|:|＝|=|加|联系|聯絡)\s*[A-Za-z][A-Za-z0-9_\-]{5,19}',
        placeholder='微信：<WECHAT>',
        note='**显式自述**的微信号。同样要求自述连接词，避开 `微信小程序` / '
             '`微信公众平台` / `微信支付` 这类专有名词。',
    ),
    Rule(
        name='ipv4',
        pattern=r'(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.){3}'
                r'(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)(?![\d.])',
        placeholder='<IP>',
        note='IPv4。本语料 298 命中里绝大多数是文档/示例地址（`127.0.0.1` 83 次、'
             '`192.168.x` 一大片），真公网 IP 只有 `106.75.13.27`(18)、'
             '`140.255.148.208`(2)、`178.128.61.189`(1) 等少数。'
             '无法区分"教程里的 IP"和"我的服务器 IP"，一律脱敏并在此声明。',
    ),
    Rule(
        name='plate',
        pattern=LB_CJK +
                r'[京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新]'
                r'[A-Z][A-Z0-9]{4,5}[A-Z0-9挂学警港澳]'
                r'(?![A-Za-z0-9\u4e00-\u9fff])',
        placeholder='<PLATE>',
        note='中国车牌（含新能源 6 位）。省份字前的 `(?<![\\u4e00-\\u9fff])` 是关键：'
             '没有它，`更新HEADER` 会命中（`新H`+`EADE`+`R`）—— 这正是初版正则的假阳性。'
             '`LB_CJK` 额外给出"前置字面转义"豁免，与其它规则保持同一口径。',
    ),
]

RULES_BY_NAME = {r.name: r for r in RULES}


# ==========================================================================
# 核心：脱敏
# ==========================================================================
@dataclass
class RuleStat:
    hits: int = 0
    rejected: int = 0
    samples: list = field(default_factory=list)   # [(before_ctx, after_ctx)]


def find_candidates(text: str, rules: list[Rule]) -> list[tuple[int, int, int, Rule]]:
    """收集所有候选 span：(start, end, 优先级, rule)。校验器不过的直接丢弃并计数。"""
    out: list[tuple[int, int, int, Rule]] = []
    for prio, rule in enumerate(rules):
        for m in rule.regex.finditer(text):
            val = m.group(0)
            if rule.validator is not None and not rule.validator(val, m.start(), m.end(), text):
                continue
            out.append((m.start(), m.end(), prio, rule))
    return out


def resolve(cands: list[tuple[int, int, int, Rule]]) -> list[tuple[int, int, int, Rule]]:
    """区间调度：按 (start, 优先级) 排序后贪心取不相交者（高优先级在前的规则先占位）。"""
    cands = sorted(cands, key=lambda c: (c[0], c[2], -(c[1] - c[0])))
    accepted: list[tuple[int, int, int, Rule]] = []
    for c in cands:
        if accepted and c[0] < accepted[-1][1]:
            continue
        accepted.append(c)
    return accepted


# ==========================================================================
# 删块：姓名 + 手机 + 住址的"名单块"
# ==========================================================================
# 为什么不能只靠脱敏：主 AI 复核指出，把手机号换成 `<PHONE>` 之后，
# **姓名 + 门牌号仍然可定位到具体的人**。名单形态（多个人 + 手机 + 完整住址）
# 不是正则能清的 —— 正确做法是把整块丢掉。
#
# 判据（两条同时满足，实测在本语料上只命中 1 块，且与次名有巨大间隔）：
#   A. 块内**不同**的 11 位手机号 >= 3
#   B. 至少 3 个手机号的 ±30 字窗口里有 >= 3 个地址指示字（表 + 门牌后缀）
#   C. 块内地址指示字总数 >= 12（兜底）
# 为什么加 B/C：主 AI 给的"裸 11 位手机 + >=6 地址字"参考判据会命中 10 块，
# 其中 9 块是假阳性（商家查电话、日志、SMS 教程、俄文教育统计报表里那 5 个
# `1x000000000` 形状巧合）。加 B（**地址必须紧挨着手机**）后：
#   命中 1 块（#317：不同手机=5，地址窗口=16）
#   次名 #2989（俄文报表）：不同手机=5，地址窗口=**0** ⇒ 不删
#   #90 / #1567 / #2842：不同手机=1 ⇒ 不删
ADDR_CHARS = frozenset('省市区县镇乡村路街号栋室楼单元小区縣鎮鄉號棟樓單元區')
ROSTER_PHONE = re.compile(LB_ALNUM + r'1[3-9]\d{9}(?![\dA-Za-z])')
ROSTER_MIN_DISTINCT = 3      # 判据 A
ROSTER_MIN_WINDOW = 3        # 判据 B
ROSTER_WINDOW = 30           # B 的窗口半径（字）
ROSTER_MIN_ADDR = 12         # 判据 C


@dataclass
class BlockHit:
    """一个被判定为"名单块"、整块删除的 span。"""
    start: int
    end: int
    n_phones: int          # 手机形状出现次数
    n_distinct: int        # 不同手机号个数
    n_addr: int            # 地址指示字总数
    n_window: int          # ±30 字窗口内地址字 >=3 的手机个数
    head: str              # 该块前 60 字（审计用）

    @property
    def reason(self) -> str:
        return (f'名单块：不同手机={self.n_distinct}(>={ROSTER_MIN_DISTINCT})、'
                f'地址窗口手机={self.n_window}(>={ROSTER_MIN_WINDOW})、'
                f'地址字={self.n_addr}(>={ROSTER_MIN_ADDR})、手机出现={self.n_phones}')


def iter_block_spans(text: str) -> list[tuple[int, int]]:
    """按 `\\n\\n` 切块，返回 [(start, end), ...]（不含分隔符本身）。"""
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in re.finditer(r'\n\n', text):
        spans.append((pos, m.start()))
        pos = m.end()
    spans.append((pos, len(text)))
    return spans


def find_roster_blocks(text: str) -> list[BlockHit]:
    """找出需要**整块删除**的"姓名+手机+住址名单块"（判据见上方注释）。"""
    hits: list[BlockHit] = []
    for s, e in iter_block_spans(text):
        if e - s < 10:
            continue
        block = text[s:e]
        phones = [m.group(0) for m in ROSTER_PHONE.finditer(block)]
        distinct = set(phones)
        if len(distinct) < ROSTER_MIN_DISTINCT:
            continue
        n_addr = sum(1 for ch in block if ch in ADDR_CHARS)
        if n_addr < ROSTER_MIN_ADDR:
            continue
        n_window = 0
        for m in ROSTER_PHONE.finditer(block):
            win = block[max(0, m.start() - ROSTER_WINDOW):m.end() + ROSTER_WINDOW]
            if sum(1 for ch in win if ch in ADDR_CHARS) >= 3:
                n_window += 1
        if n_window < ROSTER_MIN_WINDOW:
            continue
        hits.append(BlockHit(s, e, len(phones), len(distinct), n_addr, n_window,
                             block[:60].replace('\n', '⏎')))
    return hits


def scrub(text: str, rules: list[Rule] | None = None, max_samples: int = 3,
          ctx: int = 45, delete_rosters: bool = True
          ) -> tuple[str, dict[str, RuleStat], list[BlockHit]]:
    """返回 (脱敏后文本, 逐规则统计, 被整块删除的名单块)。纯函数，无副作用。"""
    rules = rules if rules is not None else RULES
    stats = {r.name: RuleStat() for r in rules}

    # 被校验器否掉的候选也要计数
    for rule in rules:
        for m in rule.regex.finditer(text):
            if rule.validator is not None and not rule.validator(m.group(0), m.start(), m.end(), text):
                stats[rule.name].rejected += 1

    rost = find_roster_blocks(text) if delete_rosters else []
    accepted = resolve(find_candidates(text, rules))
    # 落在删块里的规则候选不再单独计数（整块都要没了）
    accepted = [c for c in accepted
                if not any(b.start <= c[0] and c[1] <= b.end for b in rost)]

    ops: list[tuple[int, int, str, str, object]] = []
    for b in rost:
        ops.append((b.start, b.end, '', 'roster', b))
    for c in accepted:
        ops.append((c[0], c[1], c[3].placeholder, 'rule', c[3]))
    ops.sort(key=lambda o: (o[0], o[1]))

    pieces: list[str] = []
    pos = 0
    for start, end, rep, kind, obj in ops:
        if kind == 'roster':
            # 连同**左侧**一个 `\n\n` 分隔符一起删，保证剩下块之间仍有一个空行
            if start >= 2 and text[start - 2:start] == '\n\n':
                start -= 2
            elif text[end:end + 2] == '\n\n':
                end += 2
            if start < pos:
                continue
        pieces.append(text[pos:start])
        pieces.append(rep)
        pos = end
        if kind == 'rule':
            st = stats[obj.name]
            st.hits += 1
            if len(st.samples) < max_samples:
                a = max(0, start - ctx)
                b = min(len(text), end + ctx)
                st.samples.append((text[a:b].replace('\n', '⏎'),
                                   (text[a:start] + rep + text[end:b]).replace('\n', '⏎')))
    pieces.append(text[pos:])
    return ''.join(pieces), stats, rost


# ==========================================================================
# 旁证指标
# ==========================================================================
SHORT_NUM = re.compile(r'(?<![\dA-Za-z])\d{4,6}(?![\dA-Za-z])')


def short_num_count(text: str) -> int:
    """4~6 位纯数字 token 数（年份/价格/编号）。脱敏前后必须相等 = 0 误伤。"""
    return len(SHORT_NUM.findall(text))


def number_audit(text: str, rules: list[Rule] | None = None, max_rows: int = 30,
                 delete_rosters: bool = True) -> list[tuple]:
    """误伤审计：列出**落在被删块或被脱敏 span 内部**的 4~6 位数字 token。

    这是"误伤"唯一可能发生的地方（span 之外一个字符都不动）。逐条看这些 token 的
    上下文，就能判断规则有没有把正常文本（年份列表、MATLAB 数组、价格）当成 PII。
    返回 [(token, 来源, 上下文), ...]，最多 max_rows 条。
    """
    rules = rules if rules is not None else RULES
    spans = [(s, e, r.name) for s, e, _p, r in resolve(find_candidates(text, rules))]
    if delete_rosters:
        spans += [(b.start, b.end, 'ROSTER-DELETE') for b in find_roster_blocks(text)]
    rows: list[tuple] = []
    for m in SHORT_NUM.finditer(text):
        for s, e, name in spans:
            if s <= m.start() and m.end() <= e:
                # 上下文截断：被删的名单块可以长到 4000+ 字，不能整块打出来
                rows.append((m.group(0), name,
                             text[max(0, s - 30):min(e + 20, s + 150)].replace('\n', '⏎')))
                break
        if len(rows) >= max_rows:
            break
    return rows


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


# ==========================================================================
# 自检
# ==========================================================================
# 正对照：每类至少一条**已知答案**的 PII（银行卡用 Luhn 通过的测试卡号，
# 身份证用校验位正确的公开测试号）。
# ★ 每个"转义形态"用例（`\\n` 在 Python 源码里是**字面反斜杠+n 两个字符**）
#   都是主 AI 复核抓到的漏检类别，不是凭空加的。
POS_CASES: list[tuple[str, str]] = [
    ('email', '联系我：zhangsan_2024@qq.com，谢谢'),
    ('url_cred', '数据库地址 mongodb://admin:hunter2@db.internal:27017/app'),
    ('id18', '身份证号 11010519491231002X 已提交'),
    ('bank_card', '银行卡号：6222 0212 3456 7894 请勿外传'),
    ('bank_card', '我的银行卡 6222021234567894 转账'),
    ('mobile_cn', '打 13812345678 找我'),
    ('mobile_cn', '手机 +8613812345678 同微信'),
    ('mobile_cn', '座机手机 138-1234-5678'),
    ('landline', '公司电话 010-62345678 转 808'),
    ('landline', '前台 0755-12345678'),
    ('qq', '我的QQ是12345678'),
    ('qq', 'QQ号：87654321'),
    ('wechat', '微信号：zhangsan_2024'),
    ('wechat', '加我微信：abc_12345'),
    ('ipv4', '服务器 203.0.113.7 挂了'),
    ('plate', '车牌 京A12345 被拍'),
    ('plate', '新能源 沪AD12345'),
    # ---- 转义形态（字面 \n / \r\n / \t）—— 主 AI 复核的最小复现同款 ----
    ('mobile_cn', r'彭培泉\n\n18959938568\n\n福建省泉州市'),      # 字面 \n
    ('mobile_cn', r'駱生\r\n\r\n13902212028\r\n\r\n廣東省'),      # 字面 \r\n
    ('mobile_cn', r'梁治傲\t18257673871\t浙江省'),                # 字面 \t
    ('landline', r'電話\n0774-3939888\n梧州學院'),                # 字面 \n
    ('id18', r'身份證\n11010519491231002X\n已提交'),              # 字面 \n
    ('bank_card', r'銀行卡號\n6222 0212 3456 7894\n請勿外傳'),     # 字面 \n
    ('email', r'郵箱\nzhangsan_2024@qq.com\n謝謝'),               # 字面 \n
    ('plate', r'車牌\n京A12345\n被拍'),                           # 字面 \n
]

# 删块对照：正对照（该删的名单块）与负对照（不该删的近邻块）
ROSTER_POS = (
    '用户：帮我把下面的名单整理成表格\n'
    '張三 13800000001 北京市海淀區中關村大街1號3棟502室\n'
    '李四 13800000002 上海市浦東新區世紀大道100號8號樓1201室\n'
    '王五 13800000003 廣東省深圳市南山區科技園南路9號A棟801室\n'
    '模型：好的，以下是整理結果……'
)
ROSTER_NEG = [
    # 近邻 1：单手机 + 地址，但只有 1 个人 ⇒ 不该整块删（对应主 AI 说的 #90 类）
    ('单个商家电话 + 地址（不该删）',
     '用户：中山市宏達戶外照明燈具廠電話\n模型：很抱歉，我無法提供即時電話資訊。'
     '建議您在網路搜尋引擎中搜索"中山市宏達戶外照明燈具廠"的官方網站或聯絡方式。'
     '也可以撥打 0760-88888888 諮詢。'),
    # 近邻 2：多个 11 位数字但**不与地址相邻**（对应俄文统计报表 #2989 类）⇒ 不该删
    ('多个 1x000000000 形状 + 零散地址字（不该删）',
     '用户：Информация для расчета 14000000000 - Белгородская область '
     '15000000000 - Брянская область 17000000000 - Владимирская область\n'
     '模型：Спасибо, это статистика по регионам России. 该指标反映了各省市的 '
     '教育投入占比，不涉及个人信息。'),
    # 近邻 3：俄文报表 + 大量地址字但手机不挨着地址 ⇒ 不该删
    ('5 个形状巧合 + 94 个地址字，但地址不与手机相邻（不该删）',
     '用户：Информация 14000000000 15000000000 17000000000 18000000000 19000000000 '
     'млн. ₽ 北京市 上海市 廣東省 浙江省 江蘇省 山東省 河南省 四川省 湖北省 湖南省 '
     '福建省 安徽省 河北省 陝西省 遼寧省 吉林省 黑龍江省 江西省 雲南省 貴州省\n'
     '模型：以上是各省市的統計數據彙總。'),
]

# 数字负对照：全是"像 PII 但不是 PII"的 4~6 位数字（年份/价格/编号/型号）
NUM_NEG = (
    '2024年的价格为1280元，订单编号12345，型号A380，共3件。'
    '1998年出生，2023年毕业，宿舍编号608，邮编100084，电话分机8001。'
    'GDP增长5.2%，第123456号文件，公交302路，票价2元，面积130平米。'
) * 5

NEG_SOURCE = 'data/chinese/clean_v3/红楼梦.txt'


def _load_neg_text() -> tuple[str, str]:
    """负对照文本：红楼梦前 20000 字。返回 (path, text)。"""
    with open(NEG_SOURCE, encoding='utf-8') as f:
        return NEG_SOURCE, f.read()[:20000]


def run_selftest(rules: list[Rule], inject_bug: str | None = None,
                 verbose: bool = True) -> int:
    """返回退出码：0 = 全绿，1 = 有失败。"""
    out = print if verbose else (lambda *a, **k: None)
    bad: list[str] = []
    out('=' * 72)
    out('PII 脱敏工具自检')
    out('=' * 72)
    if inject_bug:
        out(f'!! 故意失败注入点已开启：--inject-bug {inject_bug} （该规则被改成永不匹配）')
        for r in rules:
            if r.name == inject_bug:
                # 运行时破坏：换成正则上永不可能匹配的模式（编译通过但匹配不到）
                r.pattern = r'(?!x)x'
                r.regex = re.compile(r.pattern)
                r.validator = None

    # ---------- 1) 正对照 ----------
    out('\n[正对照] 往干净中文里注入已知 PII，要求逐类 100% 命中')
    base = '今天天气不错，我们去公园散步吧。' * 3
    injected = base
    expect: dict[str, int] = {}
    for name, payload in POS_CASES:
        injected = injected + '\n' + payload
        expect[name] = expect.get(name, 0) + 1
    # 规则级正对照**关掉删块**：这一步只验证 10 条规则本身，
    # 删块有自己独立的 [删块对照] 一节（两者混在一起会互相干扰计数）。
    _, stats, _rost = scrub(injected, rules=rules, max_samples=0, delete_rosters=False)
    out(f'  注入总数 {len(POS_CASES)} 条，覆盖 {len(expect)} 类')
    out(f'  {"类别":<10} {"注入":>4} {"命中":>4}  结果')
    for name in sorted(expect):
        hit = stats[name].hits
        ok = hit == expect[name]
        if not ok:
            bad.append(f'正对照 {name}: 命中 {hit}/{expect[name]}')
        out(f'  {name:<10} {expect[name]:>4} {hit:>4}  {"OK" if ok else "MISS ✗"}')

    # ---------- 2) 负对照（字节级） ----------
    out('\n[负对照] 干净中文 + 数字密集干净中文，要求一个字符都不改')
    neg_path, neg_text = _load_neg_text()
    negatives = [(f'{neg_path}（前 {len(neg_text)} 字符）', neg_text),
                 ('内置数字密集负对照（年份/价格/编号，共 %d 字）' % len(NUM_NEG), NUM_NEG)]
    neg_afters = []
    for label, neg in negatives:
        neg_after, neg_stats, neg_rost = scrub(neg, rules=rules, max_samples=0)
        neg_afters.append(neg_after)
        h_before, h_after = sha256_text(neg), sha256_text(neg_after)
        ident = neg == neg_after
        if not ident:
            bad.append(f'负对照被改动: {label}')
        out(f'  来源     : {label}')
        out(f'  sha256 前: {h_before}')
        out(f'  sha256 后: {h_after}')
        out(f'  字符数   : {len(neg)} -> {len(neg_after)}')
        out(f'  结果     : {"字节级完全一致 OK" if ident else "被改动 ✗"}')
        nz = {k: v.hits for k, v in neg_stats.items() if v.hits}
        out(f'  各规则命中: {nz if nz else "全部 0"}')

    # ---------- 3) 幂等 ----------
    out('\n[幂等] scrub(scrub(x)) == scrub(x)')
    once, _st, _r1 = scrub(injected, rules=rules, max_samples=0, delete_rosters=False)
    twice, stats2, _r2 = scrub(once, rules=rules, max_samples=0, delete_rosters=False)
    if once == twice:
        out('  第二遍无任何变化 OK')
    else:
        bad.append('幂等失败')
        out(f'  第二遍仍在改文本 ✗  第二遍命中={ {k: v.hits for k, v in stats2.items() if v.hits} }')

    # ---------- 3.5) 删块对照 ----------
    out('\n[删块对照] 名单块（姓名+手机+住址）必须整块删除；近邻块必须不动')
    pos_after, pos_stats, pos_rost = scrub(ROSTER_POS, rules=rules, max_samples=0)
    ok_pos = len(pos_rost) == 1 and pos_after == ''
    if not ok_pos:
        bad.append('删块正对照失败')
    out(f'  正对照: 命中删块={len(pos_rost)} 期望=1；剩余字符={len(pos_after)} 期望=0 '
        f'{"OK" if ok_pos else "✗"}')
    if pos_rost:
        b = pos_rost[0]
        out(f'    span=[{b.start},{b.end}) {b.reason}')
    for label, neg_block in ROSTER_NEG:
        nb_after, nb_stats, nb_rost = scrub(neg_block, rules=rules, max_samples=0)
        kept = len(nb_rost) == 0
        if not kept:
            bad.append(f'删块负对照被误删: {label}')
        out(f'  负对照: {label} → 删块={len(nb_rost)} 期望=0 {"OK" if kept else "✗"} '
            f'(规则脱敏 {nb_stats["mobile_cn"].hits} 处手机、'
            f'{nb_stats["landline"].hits} 处座机，只脱敏不删块)')

    # ---------- 4) 误伤旁证 ----------
    out('\n[误伤旁证] 4~6 位纯数字 token（年份/价格/编号）在**干净文本**上必须 0 改动')
    out('  （注入文本里的数字本来就是要脱敏的 PII，不计入误伤）')
    for (label, neg), neg_after in zip(negatives, neg_afters):
        n0, n1 = short_num_count(neg), short_num_count(neg_after)
        ok = n0 == n1
        if not ok:
            bad.append(f'4~6 位数字被误伤: {label}')
        out(f'  {label}: {n0} -> {n1} {"OK" if ok else "✗"}')
    i0, i1 = short_num_count(injected), short_num_count(once)
    out(f'  （参考）注入文本: {i0} -> {i1}，减少的 {i0 - i1} 个正是被脱敏的卡号分组/'
        f'座机号里的数字，不是误伤')
    for label, neg_block in ROSTER_NEG:
        nb_after, _, _ = scrub(neg_block, rules=rules, max_samples=0)
        n0, n1 = short_num_count(neg_block), short_num_count(nb_after)
        out(f'  （删块负对照数字）{label}: {n0} -> {n1}')

    out('\n' + '=' * 72)
    if bad:
        out('自检结果：FAIL ✗')
        for b in bad:
            out('  - ' + b)
        out('=' * 72)
        return 1
    out('自检结果：PASS ✓（正对照逐类 100%、负对照字节级不变、幂等、删块正/负对照、0 数字误伤）')
    out('=' * 72)
    return 0


# ==========================================================================
# CLI
# ==========================================================================
def fmt_table(stats: dict[str, RuleStat], rules: list[Rule]) -> str:
    lines = [f'{"类别":<10} {"命中":>6} {"校验器否决":>10}  占位符']
    for r in rules:
        st = stats[r.name]
        lines.append(f'{r.name:<10} {st.hits:>6} {st.rejected:>10}  {r.placeholder}')
    tot = sum(stats[r.name].hits for r in rules)
    rej = sum(stats[r.name].rejected for r in rules)
    lines.append(f'{"合计":<10} {tot:>6} {rej:>10}')
    return '\n'.join(lines)


def do_scan(args, rules: list[Rule]) -> int:
    with open(args.input, encoding='utf-8') as f:
        text = f.read()
    orig_len = len(text)
    if args.limit:
        text = text[:args.limit]
    print(f'输入   : {args.input}')
    print(f'字符数 : {len(text)}' + (f'（--limit 截断，原 {orig_len}）' if args.limit else ''))
    print(f'sha256 : {sha256_text(text)}')
    print()
    scrubbed, stats, rosters = scrub(text, rules=rules, max_samples=args.max_samples,
                                     delete_rosters=not args.no_block_delete)
    print(fmt_table(stats, rules))
    print()
    if rosters:
        del_chars = sum(b.end - b.start for b in rosters)
        print(f'整块删除（名单块）: {len(rosters)} 块，{del_chars} 字符')
        for b in rosters:
            print(f'  off={b.start:>9d} len={b.end - b.start:>6d} | {b.reason}')
            print(f'    前60字: {b.head}')
    else:
        print('整块删除（名单块）: 0 块')
    print()
    print(f'字符数变化: {len(text)} -> {len(scrubbed)}  '
          f'Δ={len(scrubbed) - len(text):+d}  '
          f'({(len(scrubbed) - len(text)) / max(1, len(text)) * 100:+.4f}%)')
    n0, n1 = short_num_count(text), short_num_count(scrubbed)
    print(f'4~6 位数字 token: {n0} -> {n1}  '
          + ('（0 误伤 OK）' if n0 == n1
             else f'（{n0 - n1} 个数字落在删块/脱敏 span 内；span 之外构造上不可能改动。'
                  '逐条审计如下，需人工确认这些 span 确实是 PII）'))
    rows = number_audit(text, rules=rules, max_rows=args.max_samples * 20,
                        delete_rosters=not args.no_block_delete)
    if rows:
        print(f'  误伤审计（落在删块/脱敏 span 内的 4~6 位数字，最多 {len(rows)} 条首现）：')
        for tok, rn, c in rows:
            print(f'    [{rn}] {tok}  ...{c}...')
    second, st2, _r = scrub(scrubbed, rules=rules, max_samples=0,
                            delete_rosters=not args.no_block_delete)
    print(f'幂等第二遍: {"无变化 OK" if second == scrubbed else "仍有改动 ✗ " + str({k: v.hits for k, v in st2.items() if v.hits})}')
    if args.samples:
        print()
        print('=' * 72)
        print(f'抽样 before/after（每类最多 {args.max_samples} 条）')
        print('=' * 72)
        for r in rules:
            for i, (b, a) in enumerate(stats[r.name].samples, 1):
                print(f'--- {r.name} #{i}')
                print(f'  before: {b}')
                print(f'  after : {a}')
    if not args.apply:
        print()
        print('(dry-run：未写任何文件。要真写请加 --apply 并显式给 --out)')
        return 0

    # ---- apply ----
    if not args.out:
        print('错误：--apply 必须显式给 --out', file=sys.stderr)
        return 2
    if os.path.realpath(args.out) == os.path.realpath(args.input):
        print(f'错误：拒绝原地覆盖（--out 不能等于 --in：{args.input}）', file=sys.stderr)
        return 2
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(scrubbed)
    print()
    print(f'已写出: {args.out}  ({len(scrubbed)} 字符)')
    print(f'输出 sha256: {sha256_text(scrubbed)}')
    print(f'输入 sha256: {sha256_text(open(args.input, encoding="utf-8").read())}（未被修改）')
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='WildChat 中文 PII 检测/脱敏（默认 dry-run）')
    ap.add_argument('--in', dest='input', default=DEFAULT_IN, help=f'输入文件（默认 {DEFAULT_IN}）')
    ap.add_argument('--out', default=None, help='输出文件；--apply 时必填，禁止等于 --in')
    ap.add_argument('--apply', action='store_true', help='真写 --out（默认只报告）')
    ap.add_argument('--selftest', action='store_true', help='跑正/负对照 + 幂等自检')
    ap.add_argument('--inject-bug', default=None, metavar='RULE',
                    help='自检的故意失败注入点：把该规则改成永不匹配，自检必须变红')
    ap.add_argument('--limit', type=int, default=0, help='只处理前 N 字符（调试用）')
    ap.add_argument('--max-samples', type=int, default=3, help='每类抽样条数（默认 3）')
    ap.add_argument('--samples', action='store_true', default=True,
                    help='打印抽样 before/after（默认开）')
    ap.add_argument('--no-samples', dest='samples', action='store_false')
    ap.add_argument('--no-block-delete', action='store_true',
                    help='关掉"名单块整块删除"（只做正则脱敏；调试/对照用）')
    ap.add_argument('--quiet', action='store_true', help='自检只输出结论')
    args = ap.parse_args(argv)

    if args.inject_bug and not args.selftest:
        print('错误：--inject-bug 只能和 --selftest 一起用', file=sys.stderr)
        return 2
    if args.inject_bug and args.inject_bug not in RULES_BY_NAME:
        print(f'错误：未知规则 {args.inject_bug}；可选 {sorted(RULES_BY_NAME)}', file=sys.stderr)
        return 2

    if args.selftest:
        # 自检要在**副本**上做 --inject-bug 的破坏，避免污染后续（同一进程内无后续，但干净）
        import copy
        rules = copy.deepcopy(RULES)
        return run_selftest(rules, inject_bug=args.inject_bug, verbose=not args.quiet)
    return do_scan(args, RULES)


if __name__ == '__main__':
    sys.exit(main())
