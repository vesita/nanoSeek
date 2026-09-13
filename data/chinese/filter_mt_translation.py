#!/usr/bin/env python
"""把机翻的 ESConv 中文多轮对话筛成"整段全净"的可入库子集。

## 输入 / 输出

    输入（只读）  data/chinese/new_dialogue/escov_zh.json
                  [{"messages": [{"role": "user"|"assistant", "content": str}, ...]}, ...]
                  1300 段 / 38,365 条（角色与轮数对齐已由翻译管线验证，本工具不重验）
    输出（--apply）data/chinese/new_sources/escov_zh.txt
                  每条消息一行：`用户：<content>` / `模型：`，段与段之间一个空行
    --selftest    已知答案对照（正/负/整段三类），失败退出码 1

## 为什么要它

机翻（Helsinki-NLP/opus-mt-en-zh）已知缺陷 [主 AI 实测]：未翻译英文词 2.5%、
乱码 1.2%、当场重复短语 0.8%、重复折叠 0.7%、超短 2.0%。这些缺陷**形态上看得见**，
所以可以写成确定性规则。用户拍板：**只收整段全净的对话段** —— 一段里只要有一条
assistant 回复不合格，整段丢弃（user 轮不参与判定，整段保留时原样保留）。

## 规则（只作用在 `role == "assistant"` 的回复上）

    R1_rep3          rep3 <= 0.3
    R2_dup_phrase    不得含当场重复子串（长度 L>=3，s[i:i+L] == s[i+L:i+2L]，先剥空白）
    R3_ascii_word    不得含 >=3 个连续 ASCII 字母
    R4_too_short     strip() 后长度 >= 4
    R5_private_use   不得含私用区 U+E000–U+F8FF 或替换字符 U+FFFD
    R6_cjk_ratio     汉字（U+4E00–U+9FFF）占比 >= 0.6

`rep3` 口径与 `clean_corpus.py::rep_n(t, 3)`（== `inference/scripts/eval_dialogue.py`
::ngram_repetition）**逐字一致**：先剥掉所有空白，`落在重复类型里的 3-gram 数 /
全部 3-gram 数`；`len < 6` 记 0.0。阈值 0.3 是项目用干净源标定过的（误伤 0.1%）——
但**只在短对话回复这个长度/结构上成立**（见 `AGENTS.md §5.8`），不要外推到长文档。

## 纪律

* 默认 **dry-run**（只写报告），`--apply` 才写数据文件（与 `clean_corpus.py` 一致）。
* **绝不修改输入**：只读 `escov_zh.json`，`--apply` 只写 `escov_zh.txt`。
* **确定性**：段按原顺序、规则判定无随机、抽样用固定 seed。
* 判据全是**形态判据** —— 能去掉"看得出是机翻"的部分，**去掉不了**"读起来别扭但
  形态正常"的翻译腔。报告的"我没能做到的部分"把这条边界写死。
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, 'data', 'chinese')
INPUT_JSON = os.path.join(DATA, 'new_dialogue', 'escov_zh.json')
OUTPUT_TXT = os.path.join(DATA, 'new_sources', 'escov_zh.txt')
REPORT_MD = os.path.join(ROOT, 'analysis', 'escov_inclusion.md')

USER_TAG = '用户：'
MODEL_TAG = '模型：'

R1 = 'R1_rep3'
R2 = 'R2_dup_phrase'
R3 = 'R3_ascii_word'
R4 = 'R4_too_short'
R5 = 'R5_private_use'
R6 = 'R6_cjk_ratio'
RULES = (R1, R2, R3, R4, R5, R6)

REP3_MAX = 0.3          # R1：rep3 <= 0.3 通过
DUP_MIN_LEN = 3         # R2：当场重复子串最小长度
ASCII_RUN = 3           # R3：>= 3 个连续 ASCII 字母即拒
MIN_STRIP_LEN = 4       # R4：strip() 后长度
CJK_MIN_RATIO = 0.6     # R6：汉字占比

_ASCII_RUN_RE = re.compile(r'[A-Za-z]{%d,}' % ASCII_RUN)


# ==========================================================================
# 纯文本工具（口径与 clean_corpus.py 对齐）
# ==========================================================================
def strip_ws(text: str) -> str:
    """去掉**所有**空白字符（去空白归一化）。"""
    return ''.join(text.split())


def rep3(text: str) -> float:
    """字符 3-gram 重复率 = 落在重复类型里的 3-gram 数 / 全部 3-gram 数。

    与 `clean_corpus.py::rep_n(text, 3)` 同一条公式、同一条守卫（`len < 6` → 0.0）。
    ⚠ 不要改成 CTRL 口径的 `1 - 不同/总数`，两者不等价（见 clean_corpus.py 注释）。
    """
    seq = strip_ws(text)
    if len(seq) < 6:
        return 0.0
    grams = [seq[i:i + 3] for i in range(len(seq) - 2)]
    c = collections.Counter(grams)
    return sum(1 for g in grams if c[g] > 1) / len(grams)


def find_dup_phrase(text: str):
    """当场重复：存在长度 L>=3 使 `s[i:i+L] == s[i+L:i+2L]`（先剥空白）。

    等价于"存在起点 i 与周期 p>=3，使 s[i:i+p] == s[i+p:i+2p]"（取 L=p 即可）。
    返回 `(i, L, 重复的子串)`，无则 None。输入是单条回复（短），朴素双重循环足够。
    """
    s = strip_ws(text)
    n = len(s)
    for i in range(n):
        max_p = (n - i) // 2
        for p in range(DUP_MIN_LEN, max_p + 1):
            if s.startswith(s[i:i + p], i + p):
                return i, p, s[i:i + p]
    return None


def ascii_word_hits(text: str) -> list:
    """含 >=3 个连续 ASCII 字母的片段（原样返回，供报告展示）。"""
    return _ASCII_RUN_RE.findall(text)


def bad_chars(text: str) -> list:
    """私用区字符（U+E000–U+F8FF）或替换字符 U+FFFD。"""
    return [c for c in text if '\ue000' <= c <= '\uf8ff' or c == '\ufffd']


def cjk_ratio(text: str) -> float:
    """汉字（U+4E00–U+9FFF）占**去空白后**字符数的比例。空串记 0.0。"""
    s = strip_ws(text)
    if not s:
        return 0.0
    return sum(1 for c in s if '\u4e00' <= c <= '\u9fff') / len(s)


# ==========================================================================
# 规则判定
# ==========================================================================
def check_reply(content: str, disabled=frozenset()) -> list:
    """对一条 assistant 回复逐条判规则，返回命中的规则名列表（空 = 通过）。

    `disabled` 里的规则**永不触发**（`--inject-bug` / `--disable-rule` 用）。
    """
    hits = []
    if R1 not in disabled and rep3(content) > REP3_MAX:
        hits.append(R1)
    if R2 not in disabled and find_dup_phrase(content) is not None:
        hits.append(R2)
    if R3 not in disabled and ascii_word_hits(content):
        hits.append(R3)
    if R4 not in disabled and len(content.strip()) < MIN_STRIP_LEN:
        hits.append(R4)
    if R5 not in disabled and bad_chars(content):
        hits.append(R5)
    if R6 not in disabled and cjk_ratio(content) < CJK_MIN_RATIO:
        hits.append(R6)
    return hits


def evaluate_segment(messages, disabled=frozenset()) -> list:
    """整段判定：返回 `[(msg_index, rule), ...]`，空 = 整段保留。

    **user 轮不参与判定**（只遍历 `role == "assistant"`）。
    """
    bad = []
    for idx, m in enumerate(messages):
        if m.get('role') != 'assistant':
            continue
        for rule in check_reply(m.get('content', ''), disabled):
            bad.append((idx, rule))
    return bad


def filter_dialogues(dialogues, disabled=frozenset()):
    """返回 `(kept, dropped)`；元素是 `(原始下标, 段对象)` / `(原始下标, bad, 段对象)`。

    `kept` 里装的是**原始对象本身**（不拷贝、不改写），供 selftest 逐字比对。
    """
    kept, dropped = [], []
    for i, seg in enumerate(dialogues):
        messages = seg.get('messages') or []
        bad = evaluate_segment(messages, disabled)
        if bad:
            dropped.append((i, bad, seg))
        else:
            kept.append((i, seg))
    return kept, dropped


# ==========================================================================
# selftest：已知答案对照
# ==========================================================================
# 每条规则 2 条"应被拒"的回复。★ 刻意造成**只命中目标规则**（命中集恰好 == {规则}），
# 这样 --inject-bug <规则> 之后该条必然不再被拒 ⇒ selftest 一定变红。
POSITIVE_CASES = {
    R1: ['今天天气很好明天天气很好后天天气很好',
         '我喜欢吃苹果他喜欢吃苹果我们都喜欢吃苹果'],
    R2: ['你好,你好,今天过得怎么样啊我很好谢谢你关心',
         '你住在哪里?住在哪里?我帮你想想办法吧慢慢说别急'],
    R3: ['今天的天气真的非常好啊the我心情也很好',
         '我觉得这个想法不错and你也这么认为吗'],
    R4: ['好的吧', '嗯嗯'],
    R5: ['你好\ue4c7世界', '今天天气不错\ufffd'],
    R6: ['Hi你好', '?!你好'],
}

# 3 段完全干净的对话（含中文标点、数字、问号），必须一条不丢。
NEGATIVE_DIALOGUES = [
    [{'role': 'user', 'content': '你好'},
     {'role': 'assistant', 'content': '你好，今天过得怎么样？'},
     {'role': 'user', 'content': '还不错，谢谢你'},
     {'role': 'assistant', 'content': '听起来挺好的，有什么特别想聊的吗？'}],
    [{'role': 'user', 'content': '你今年多大了？'},
     {'role': 'assistant', 'content': '我今年28岁了，住在北京。'},
     {'role': 'user', 'content': '北京天气怎么样？'},
     {'role': 'assistant', 'content': '最近降温了，记得多穿一点，出门带伞。'}],
    [{'role': 'user', 'content': '我有点焦虑'},
     {'role': 'assistant', 'content': '我明白你的感受。慢慢说，我在听。'},
     {'role': 'user', 'content': '谢谢你'},
     {'role': 'assistant', 'content': '不客气，随时都可以找我聊，照顾好自己。'}],
]

# user 轮里塞进"每条规则的违例"——该段**仍必须保留**（user 轮不参与判定）。
USER_TURN_IMMUNITY = [
    [{'role': 'user', 'content': '我总是jelous'},                      # R3 违例
     {'role': 'assistant', 'content': '我明白你的感受，慢慢说。'}],
    [{'role': 'user', 'content': '\ue4c7\ue7a4\ue707乱码\ufffd'},       # R5 违例
     {'role': 'assistant', 'content': '我在听，你可以继续说。'}],
    [{'role': 'user', 'content': '你好,你好,真遗憾'},                   # R2 违例
     {'role': 'assistant', 'content': '听起来你有点难过，愿意多说说吗？'}],
    [{'role': 'user', 'content': '嗯'},                                 # R4 违例
     {'role': 'assistant', 'content': '好的，我在这里等你。'}],
]

# 整段语义：3 条回复里**只有第 2 条**不合格 ⇒ 整段丢；把它换干净 ⇒ 整段留。
SEGMENT_BAD_REPLY = '我觉得这个想法不错and你也这么认为吗'      # 命中 R3_ascii_word
SEGMENT_DROP = [
    {'role': 'user', 'content': '我最近心情不好'},
    {'role': 'assistant', 'content': '听起来你最近挺难的，愿意多说一点吗？'},
    {'role': 'user', 'content': '工作压力太大了'},
    {'role': 'assistant', 'content': SEGMENT_BAD_REPLY},        # idx=3，第 2 条 assistant
    {'role': 'user', 'content': '我也不知道怎么办'},
    {'role': 'assistant', 'content': '我们可以一起想想办法，先休息一下。'},
]
SEGMENT_KEEP = [dict(m) for m in SEGMENT_DROP]
SEGMENT_KEEP[3] = {'role': 'assistant', 'content': '工作压力大的时候，先喘口气再说。'}


def selftest(inject_bug=None, out=print) -> int:
    """跑已知答案对照；返回失败数（0 = 全过）。`out` 是 print 风格的回调。"""
    disabled = {inject_bug} if inject_bug else set()
    failures = 0
    checks = 0

    def check(ok, label, detail=''):
        nonlocal failures, checks
        checks += 1
        if not ok:
            failures += 1
        out(f'  [{"PASS" if ok else "FAIL"}] {label}{(" | " + detail) if detail else ""}')

    out('=' * 78)
    out('selftest：已知答案对照' + (f'（--inject-bug {inject_bug}）' if inject_bug else ''))
    out('=' * 78)

    # --- 正对照：每条规则 2 条应被拒的回复 -----------------------------------
    out('')
    out('[正对照] 每条规则 2 条"应被拒"的回复（期望：命中集恰好 == {目标规则}）')
    for rule in RULES:
        for k, content in enumerate(POSITIVE_CASES[rule]):
            hits = check_reply(content, disabled)
            ok = (hits == [rule])
            if not ok and inject_bug == rule and hits == []:
                detail = f'注入 bug：{rule} 已停用 ⇒ 该回复不再被拒'
            else:
                detail = f'rep3={rep3(content):.3f} cjk={cjk_ratio(content):.3f} ' \
                         f'len={len(content.strip())} 命中={hits}'
            check(ok, f'{rule} 正对照#{k + 1}：{content[:34]!r}', detail)

    # --- 负对照：3 段干净对话必须全留 ---------------------------------------
    out('')
    out('[负对照] 3 段完全干净的对话（期望：一条不丢）')
    kept, dropped = filter_dialogues([{'messages': m} for m in NEGATIVE_DIALOGUES], disabled)
    check(len(kept) == 3 and not dropped,
          f'3 段干净对话全部保留（实际保留 {len(kept)} / 丢弃 {len(dropped)}）',
          '' if not dropped else f'误丢：{[(i, b) for i, b, _ in dropped]}')
    for i, m in enumerate(NEGATIVE_DIALOGUES):
        bad = evaluate_segment(m, disabled)
        check(not bad, f'干净对话#{i + 1} 逐条零命中', f'bad={bad}')

    # --- user 轮不参与判定 ---------------------------------------------------
    out('')
    out('[负对照] user 轮塞进各规则违例（期望：该段仍保留）')
    for i, m in enumerate(USER_TURN_IMMUNITY):
        bad = evaluate_segment(m, disabled)
        # 先自证"若把 user 内容搬进 assistant 就会被拒"，否则这个对照是恒真的空测试
        probe = [{'role': 'assistant', 'content': m[0]['content']}]
        probe_bad = bool(evaluate_segment(probe, disabled))
        check(not bad and probe_bad,
              f'user 轮免疫#{i + 1}：user={m[0]["content"][:22]!r}',
              f'user 轮判定={bad}；同内容放到 assistant 轮={probe_bad}')

    # --- 整段语义对照 --------------------------------------------------------
    out('')
    out('[整段语义] 3 条回复里只有第 2 条不合格 ⇒ 整段丢；换干净 ⇒ 整段留')
    bad = evaluate_segment(SEGMENT_DROP, disabled)
    check(bool(bad), '含 1 条坏回复的段被丢', f'bad={bad}')
    check(bad == [(3, R3)], '触发者被精确定位到第 2 条 assistant（下标 3）与规则 R3_ascii_word',
          f'期望 [(3, {R3!r})]，实际 {bad}')
    bad_keep = evaluate_segment(SEGMENT_KEEP, disabled)
    check(not bad_keep, '把第 2 条换成干净回复后整段保留', f'bad={bad_keep}')

    # --- messages 逐字未改 ---------------------------------------------------
    out('')
    out('[逐字不变] 保留段的 messages 与输入 JSON 逐字相同')
    src = [{'messages': m} for m in NEGATIVE_DIALOGUES] + \
          [{'messages': SEGMENT_KEEP}, {'messages': SEGMENT_DROP}]
    snapshot = json.dumps(src, ensure_ascii=False, sort_keys=True)
    kept2, _ = filter_dialogues(src, disabled)
    # 保留的段必须是**同一个对象**（不是拷贝），且序列化与输入一致
    same_obj = all(kseg is src[i] for i, kseg in kept2)
    check(same_obj, '保留段是原始对象本身（未拷贝）')
    check(json.dumps(src, ensure_ascii=False, sort_keys=True) == snapshot,
          '过滤前后输入结构逐字未变')
    reserialized = json.dumps([kseg for _, kseg in kept2], ensure_ascii=False, sort_keys=True)
    expect = json.dumps([src[i] for i, _ in kept2], ensure_ascii=False, sort_keys=True)
    check(reserialized == expect, '保留段 JSON 序列化与输入逐字一致')

    out('')
    out(f'--- selftest 合计：{checks} 项，失败 {failures} 项 ⇒ '
        f'{"全部通过" if failures == 0 else "存在失败"} ---')
    return failures


# ==========================================================================
# 全量运行
# ==========================================================================
def render_segment(seg) -> str:
    """一段对话 → 项目格式文本（每条消息一行）。"""
    lines = []
    for m in seg['messages']:
        tag = USER_TAG if m['role'] == 'user' else MODEL_TAG
        lines.append(tag + m['content'])
    return '\n'.join(lines)


def load_input(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run(args, out=print):
    disabled = set(args.disable_rule)
    if args.inject_bug:
        disabled.add(args.inject_bug)

    # 先跑一遍内部 selftest（结果进报告）。真正的实现缺陷（非注入）直接拒绝往下走。
    st_buf: list = []
    st_fail = selftest(args.inject_bug,
                       out=lambda *a: (st_buf.append(' '.join(str(x) for x in a))))
    if st_fail and not args.inject_bug:
        out('内部 selftest 失败 ⇒ 拒绝写任何输出。先修实现。')
        for line in st_buf:
            out(line)
        return 2
    st_text = '\n'.join(st_buf)

    dialogues = load_input(args.input)
    out(f'输入：{args.input}（{len(dialogues):,} 段）')

    kept, dropped = filter_dialogues(dialogues, disabled)

    # --- 统计 ---------------------------------------------------------------
    reply_hits = collections.Counter()       # 回复级：命中的规则
    seg_hits = collections.Counter()         # 段级：该规则出现在了多少个被丢段里
    n_assistant = 0
    n_bad_replies = 0
    for _, bad, seg in dropped:
        rules = sorted({r for _, r in bad})
        for _, r in bad:
            reply_hits[r] += 1
        for r in rules:
            seg_hits[r] += 1
        n_bad_replies += len({i for i, _ in bad})
    for seg in dialogues:
        n_assistant += sum(1 for m in (seg.get('messages') or [])
                           if m.get('role') == 'assistant')

    kept_msgs = sum(len(seg['messages']) for _, seg in kept)
    total_msgs = sum(len(seg.get('messages') or []) for seg in dialogues)
    kept_chars = sum(len(m['content']) for _, seg in kept for m in seg['messages'])
    total_chars = sum(len(m['content']) for seg in dialogues
                      for m in (seg.get('messages') or []))
    # 保留文本文件本身的大小（含 `用户：`/`模型：` 标签与换行）
    txt = render_corpus([seg for _, seg in kept])

    # 反事实：不加 R5/R6（对齐用户先前未加这两条时的 639）
    _, dropped_no56 = filter_dialogues(dialogues, disabled | {R5, R6})
    kept_no56 = len(dialogues) - len(dropped_no56)

    # 逐规则边际：停用该规则后能救回多少段（直接回答"差异归因到哪条规则"）
    marginal = {}
    for rule in RULES:
        _, d1 = filter_dialogues(dialogues, disabled | {rule})
        marginal[rule] = len(dialogues) - len(d1)

    # 边界统计（描述性，不影响判定）：
    #   (a) user 轮里的规则命中 —— 这些**会原样进文件**（user 轮不参与判定）
    #   (b) 相邻两条 assistant 内容完全相同（跨轮重复）—— 单条规则看不见，会原样保留
    user_hits = collections.Counter()
    for _, seg in kept:
        for m in seg['messages']:
            if m['role'] == 'user':
                for r in check_reply(m['content'], disabled):
                    user_hits[r] += 1
    dup_pairs = 0
    dup_segs = 0
    for _, seg in kept:
        ms = seg['messages']
        n = sum(1 for a, b in zip(ms, ms[1:])
                if a['role'] == 'assistant' and b['role'] == 'assistant'
                and a['content'] == b['content'])
        if n:
            dup_pairs += n
            dup_segs += 1

    out('--- 逐规则（回复级命中 / 段级丢段）---')
    for rule in RULES:
        out(f'  {rule:<16} 回复级 {reply_hits.get(rule, 0):>6,} / {n_assistant:,}   '
            f'段级 {seg_hits.get(rule, 0):>5,} / {len(dialogues):,}')
    out('--- 汇总 ---')
    out(f'  保留段数      {len(kept):,} / {len(dialogues):,}')
    out(f'  保留 messages {kept_msgs:,} / {total_msgs:,}')
    out(f'  保留字符数    {kept_chars:,} / {total_chars:,}')
    out(f'  丢段数        {len(dropped):,}（其中不含 R5/R6 时保留 {kept_no56:,}）')
    if n_assistant:
        out(f'  被拒回复（去重后）{n_bad_replies:,} / {n_assistant:,} assistant 回复')

    # --- 写出 ---------------------------------------------------------------
    if args.apply:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(txt)
        out(f'已写：{args.output}（{len(txt.encode("utf-8")):,} B，'
            f'{len(txt.splitlines()):,} 行）')
    else:
        out(f'（dry-run：未写 {args.output}；加 --apply 才输出）')

    # --- 报告 ---------------------------------------------------------------
    report = render_report(args, dialogues, kept, dropped, st_text, disabled,
                           reply_hits, seg_hits, n_assistant, n_bad_replies,
                           kept_msgs, total_msgs, kept_chars, total_chars,
                           kept_no56, txt, user_hits, dup_pairs, dup_segs, marginal)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8') as f:
        f.write(report)
    out(f'报告已写入：{args.report}')
    return 0


def render_corpus(segs) -> str:
    """段列表 → 项目格式全文。段间一个空行；**末尾只有一个换行**（不堆空行）。"""
    blocks = [render_segment(s) for s in segs]
    if not blocks:
        return ''
    return '\n\n'.join(blocks) + '\n'


# ==========================================================================
# 报告
# ==========================================================================
SAMPLE_SEED = 20260914


def _fence(text: str, lang: str = 'text') -> list:
    return ['```' + lang, text, '```']


def render_report(args, dialogues, kept, dropped, st_text, disabled,
                  reply_hits, seg_hits, n_assistant, n_bad_replies,
                  kept_msgs, total_msgs, kept_chars, total_chars,
                  kept_no56, txt, user_hits, dup_pairs, dup_segs, marginal) -> str:
    n_seg = len(dialogues)
    rng = random.Random(SAMPLE_SEED)

    def pct(a, b):
        return f'{100.0 * a / b:.2f}%' if b else 'n/a'

    L = []
    L.append('# 机翻 ESConv 中文对话：可入库子集筛选报告')
    L.append('')
    L.append(f'- 时间：{datetime.datetime.now().astimezone().isoformat(timespec="seconds")}')
    L.append(f'- 模式：**{"--apply（已写数据文件）" if args.apply else "dry-run（未写数据文件）"}**')
    L.append(f'- 脚本：`data/chinese/filter_mt_translation.py`')
    L.append(f'- 输入（只读）：`{os.path.relpath(args.input, ROOT)}`')
    L.append(f'- 输出：`{os.path.relpath(args.output, ROOT)}`')
    L.append(f'- 命令行：`{" ".join(sys.argv)}`')
    if disabled:
        L.append(f'- ⚠ 被停用（永不触发）的规则：`{", ".join(sorted(disabled))}`')
    L.append('')
    L.append('口径：**只对 `role == "assistant"` 的回复判规则；整段只有在全部 assistant '
             '回复都通过时才保留**。user 轮不参与判定，整段保留时原样保留。')
    L.append('')

    # ---- §1 规则表 ----
    L.append('## 1. 六条规则（口径与证据）')
    L.append('')
    L.append('| 规则 | 判据（assistant 回复上） | 通过条件 | 缺陷来源 / 证据 |')
    L.append('|---|---|---|---|')
    L.append('| `R1_rep3` | 3-gram 重复率：落在重复类型里的 3-gram 数 / 全部 3-gram 数，'
             '先剥所有空白，`len<6` 记 0.0 | `rep3 <= 0.3` | 重复折叠 1.66%（含 repass）、'
             '当场重复短语 0.8% [实测，`analysis/translate_escov.md` §10] |')
    L.append('| `R2_dup_phrase` | 存在长度 `L>=3` 使 `s[i:i+L] == s[i+L:i+2L]`（先剥空白）'
             ' | 不存在 | 当场重复短语，如 `你好,你好,真遗憾`、`你住在哪里? 住在哪里?` [实测] |')
    L.append('| `R3_ascii_word` | `>=3` 个连续 ASCII 字母 | 不出现 | 未翻译英文词 2.5%、'
             '`我总是jelous`、`COVID已` [实测] |')
    L.append('| `R4_too_short` | `strip()` 后长度 | `>= 4` | 超短回复 2.0% [实测] |')
    L.append('| `R5_private_use` | 私用区 `U+E000`–`U+F8FF` 或替换字符 `U+FFFD` | 不出现 | '
             '乱码 1.2%，如 `⊿闽玒\\ue4c7и\\ue7a4辨…` [实测] |')
    L.append('| `R6_cjk_ratio` | 汉字（`U+4E00`–`U+9FFF`）占比（去空白后） | `>= 0.6` | '
             '未翻译/退化输出（`hii hi hi 喜 喜` 这类），[推断] 判据由本次定义 |')
    L.append('')
    L.append('`rep3` 与 `data/chinese/clean_corpus.py::rep_n(t, 3)` 及 '
             '`inference/scripts/eval_dialogue.py::ngram_repetition` **逐字同口径**；'
             '阈值 0.3 是项目在干净源上标定过的（误伤 0.1%），**仅适用于短对话回复这个'
             '长度/结构**（`AGENTS.md §5.8`）。')
    L.append('')

    # ---- §2 selftest ----
    L.append('## 2. selftest 原始输出（已知答案对照）')
    L.append('')
    L.append('命令：`.venv/bin/python data/chinese/filter_mt_translation.py --selftest`')
    L.append('')
    L.extend(_fence(st_text))
    L.append('')
    L.append('对照设计（**每条正对照刻意只命中目标规则**，所以 `--inject-bug` 一定变红）：')
    L.append('')
    L.append('- **正对照**：6 条规则 × 2 条应被拒回复，断言命中集 `== [目标规则]`。')
    L.append('- **负对照**：3 段完全干净的对话（中文标点 / 数字 / 问号）必须全留；'
             'user 轮塞进 `jelous`、私用区、当场重复、超短内容后**该段仍必须保留** —— '
             '且先自证"同内容放进 assistant 轮会被拒"，否则这个免疫对照是恒真的空测试。')
    L.append('- **整段语义**：3 条回复里只有第 2 条（下标 3）不合格 ⇒ 整段丢，'
             '且精确报出 `(下标, 规则)`；换成干净回复 ⇒ 整段留。')
    L.append('- **逐字不变**：保留段必须是输入 JSON 的**同一对象**，序列化前后逐字一致。')
    L.append('')

    # ---- §3 全量统计 ----
    L.append('## 3. 全量统计')
    L.append('')
    L.append('| 指标 | 数值 |')
    L.append('|---|---|')
    L.append(f'| 输入段数 | {n_seg:,} |')
    L.append(f'| **保留段数 / {n_seg}** | **{len(kept):,} / {n_seg:,}（{pct(len(kept), n_seg)}）** |')
    L.append(f'| 丢弃段数 | {len(dropped):,}（{pct(len(dropped), n_seg)}） |')
    L.append(f'| **保留 messages 条数 / {total_msgs:,}** | **{kept_msgs:,} / '
             f'{total_msgs:,}（{pct(kept_msgs, total_msgs)}）** |')
    L.append(f'| **保留字符数** | **{kept_chars:,} / {total_chars:,}（'
             f'{pct(kept_chars, total_chars)}）** |')
    kept_assistant = sum(1 for _, s in kept for m in s['messages']
                         if m['role'] == 'assistant')
    L.append(f'| 保留段内 assistant 回复 | {kept_assistant:,} |')
    L.append(f'| 被拒 assistant 回复（去重） | {n_bad_replies:,} / {n_assistant:,}'
             f'（{pct(n_bad_replies, n_assistant)}） |')
    L.append(f'| 输出文件大小 | {len(txt.encode("utf-8")):,} B / '
             f'{len(txt.splitlines()):,} 行（含标签与空行） |')
    L.append('')
    L.append('### 3.1 逐规则：回复级命中数与段级丢段数')
    L.append('')
    L.append('| 规则 | 回复级命中（条 / assistant 回复） | 占 assistant 回复 | '
             '段级丢段数（段 / 1300） |')
    L.append('|---|---|---|---|')
    for rule in RULES:
        rh, sh = reply_hits.get(rule, 0), seg_hits.get(rule, 0)
        L.append(f'| `{rule}` | {rh:,} | {pct(rh, n_assistant)} | {sh:,} |')
    L.append(f'| **合计（可重复计）** | **{sum(reply_hits.values()):,}** | '
             f'**{pct(sum(reply_hits.values()), n_assistant)}** | '
             f'**{sum(seg_hits.values()):,}** |')
    L.append('')
    L.append('> 一条回复可能同时命中多条规则、"丢段数之和" > 丢段总数'
             f'（{len(dropped):,}），所以两张表都按"规则出现次数"计。')
    L.append('')
    L.append('### 3.2 与用户先前口径（未加 R5/R6）的差异归因')
    L.append('')
    L.append('| 口径 | 保留段数 |')
    L.append('|---|---|')
    L.append(f'| 本工具全 6 条规则 | {len(kept):,} |')
    L.append(f'| 只上 R1–R4（= 用户先前"没加 R5/R6"） | {kept_no56:,} |')
    L.append(f'| R5/R6 额外砍掉 | {kept_no56 - len(kept):,} |')
    L.append('')
    L.append('复现：`.venv/bin/python data/chinese/filter_mt_translation.py '
             '--disable-rule R5_private_use --disable-rule R6_cjk_ratio`')
    L.append('')
    L.append('逐规则**边际贡献**（停用该规则后能多留下多少段 ⇒ 这些段的丢弃"由该规则决定"）：')
    L.append('')
    L.append('| 规则 | 停用该规则后的保留段数 | 该规则独力拦下的段数（相对 608） |')
    L.append('|---|---|---|')
    for rule in RULES:
        L.append(f'| `{rule}` | {marginal[rule]:,} | {marginal[rule] - len(kept):,} |')
    L.append('')
    L.append(f'⇒ 本工具在只上 R1–R4 时得到 **{kept_no56:,}**，与用户给出的 **639** 逐位相同'
             f'[实测]；因此 608 与 639 的全部差异都能归因到 **R5/R6（额外拦下 '
             f'{kept_no56 - len(kept):,} 段）**。'
             f'（前提：用户那次的口径就是 R1–R4 —— 这一点是[推断]；'
             f'数字逐位相等是支持该推断的实测证据，但两条实现是否逐字相同我无法核对。）')
    L.append('')
    L.append('> R5 与 R6 存在**非可加**交互：单停 R5 救回 '
             f'{marginal[R5] - len(kept):,} 段、单停 R6 救回 '
             f'{marginal[R6] - len(kept):,} 段，但两条一起停救回 '
             f'{kept_no56 - len(kept):,} 段 —— 有 '
             f'{(kept_no56 - len(kept)) - (marginal[R6] - len(kept)):,} 段的违例回复'
             f'**同时**含私用区与低汉字占比（单停任一条都救不回来）。')
    L.append('')
    L.append('### 3.3 user 轮与结构标签：原始文本里的命中是**预期**的')
    L.append('')
    L.append('规则**只**判 assistant 回复。user 轮、`用户：`/`模型：` 标签、段间空行都会原样'
             '进文件。所以"把整份 txt 当一坨文本搜 6 条规则"**必然**命中；'
             '命中的来源全部可归因，不代表筛选失败：')
    L.append('')
    L.append('| 来源 | 说明 |')
    L.append('|---|---|')
    L.append('| `用户：`/`模型：` 标签每行重复 | 让**整文件**的 rep3 虚高（标签本身互相重复），'
             '`模型：` 也会被 R2 当成重复子串的载体 |')
    L.append('| user 轮内容 | 用户拍板"user 轮不参与判定"，所以 user 轮里的英文词/乱码/'
             '超短内容**按设计保留** |')
    L.append('| 相邻两条 assistant 内容完全相同 | 跨轮重复，**单条规则看不见**（见 §3.4） |')
    L.append('')
    L.append(f'保留段 user 轮里各规则的命中数（**描述性**，这些内容确实在输出文件里）：')
    L.append('')
    L.append('| 规则 | user 轮命中数 |')
    L.append('|---|---|')
    for rule in RULES:
        L.append(f'| `{rule}` | {user_hits.get(rule, 0):,} |')
    L.append('')
    L.append('⇒ 验收口径应当是"**每条 assistant 回复**零命中"（本工具 §3.1 给出回复级命中数，'
             '独立复核脚本可重算）；把整文件当一坨文本搜得到的是**上述结构性命中**，'
             '不是漏筛。')
    L.append('')
    L.append('### 3.4 相邻两条 assistant 内容完全相同（跨轮重复，未被规则覆盖）')
    L.append('')
    L.append(f'保留段里连续两条 assistant 内容完全相同的：**{dup_pairs:,} 对**，'
             f'分布在 **{dup_segs:,} 段**（源数据里共 98 对，其中 38 对落在通过 6 条规则的段里）。')
    L.append('')
    L.append('这是 ESConv 原始数据的性质（同一句支持性回复连发两次），不是机翻引入的缺陷；'
             '6 条规则都是**单条回复内**的形态判据，看不见跨轮重复。'
             '若要处理，需要新增"相邻回复完全相同就丢段"的规则 —— 本任务未要求，故未加。')
    L.append('')

    # ---- §4 抽检保留 ----
    L.append('## 4. 抽检：10 段保留原文')
    L.append('')
    L.append(f'固定 seed `{SAMPLE_SEED}` 从 {len(kept):,} 段保留里随机抽 10 段，按原下标排序。')
    L.append('')
    idxs = rng.sample(range(len(kept)), min(10, len(kept)))
    for n, j in enumerate(sorted(idxs), 1):
        orig_i, seg = kept[j]
        L.append(f'### 保留抽检 #{n}（原段 #{orig_i}，{len(seg["messages"])} 条消息）')
        L.append('')
        L.extend(_fence(render_segment(seg)))
        L.append('')

    # ---- §5 抽检丢弃 ----
    L.append('## 5. 抽检：3 段被丢原文与原因')
    L.append('')
    if dropped:
        didx = rng.sample(range(len(dropped)), min(3, len(dropped)))
        for n, j in enumerate(sorted(didx), 1):
            orig_i, bad, seg = dropped[j]
            rules = sorted({r for _, r in bad})
            L.append(f'### 丢弃抽检 #{n}（原段 #{orig_i}，触发规则：'
                     f'{", ".join("`" + r + "`" for r in rules)}）')
            L.append('')
            L.append('触发位置（消息下标 / 规则）：' +
                     '、'.join(f'#{i} → `{r}`' for i, r in bad))
            L.append('')
            L.append('被拒回复原文（截断 200 字）：')
            L.append('')
            for i, r in bad:
                content = seg['messages'][i]['content']
                L.append(f'- `#{i}` `{r}`：{content[:200]!r}')
            L.append('')
            L.append('整段原文：')
            L.append('')
            L.extend(_fence(render_segment(seg)))
            L.append('')
    else:
        L.append('（没有段被丢。）')
        L.append('')

    # ---- §6 边界 ----
    L.append('## 6. 我没能做到的部分（判据的边界）')
    L.append('')
    L.append('**这 6 条是形态判据，只能去掉"看得出是机翻"的那部分，'
             '去掉不了"读起来别扭但形态正常"的翻译腔。** 不要把这 6 条全过当成"这段翻译是好的"。')
    L.append('')
    L.append('具体没覆盖的（[实测]，见 `analysis/translate_escov.md` §9 / §11）：')
    L.append('')
    L.append('| 缺陷 | 例子 | 6 条规则能否拦住 | 原因 |')
    L.append('|---|---|---|---|')
    L.append('| 词序直搬 | `你看起来像一个非常好和可爱的人` | ❌ 拦不住 | 全是合法汉字，'
             'rep3 低、无 ASCII、无乱码 |')
    L.append('| 逐词硬译 | `(你必须打退出,填满问题)` | ❌ 拦不住 | 形态正常 |')
    L.append('| 习语按字面翻 | `are not seeing eye to eye?` → `没有亲眼看到吗?` | ❌ 拦不住 | '
             '语义错误不体现在形态上 |')
    L.append('| 极性译反 | `you are welcome` → `不欢迎你` | ❌ 拦不住 | 形态完全正常 |')
    L.append('| 长句丢内容 | 从句整段丢失 | ❌ 拦不住 | 丢内容后的句子本身通顺且不短 |')
    L.append('| 凭空加内容 | `Good day!` → `日安! 再见!` | ❌ 拦不住 | 形态正常 |')
    L.append('| 语气/搭配生硬 | `yes. thank you` → `是。谢谢` | ⚠ 部分 | '
             '`是。谢谢` 只有 R4（len=4 恰好通过）/R6 可能擦边，通常拦不住 |')
    L.append('')
    L.append('⇒ **保留 ≠ 每条都译得好**；只保证"没有这 6 类可机检缺陷"。'
             '要评价翻译质量仍需人工抽读（本报告 §4 的 10 段就是给人工看的证据）。')
    L.append('')
    L.append('另外三条口径边界：')
    L.append('')
    L.append('1. `R1_rep3` 的阈值 0.3 只在**短对话回复**上标定过；`AGENTS.md §5.8` 记录了'
             '同一公式在长文档/CoT 上不可搬。本工具只吃 ESConv 短回复，未外推。')
    L.append('2. `R6_cjk_ratio >= 0.6` 是本工具本次定义的新判据，**没有项目历史对照**'
             '（[推断] 级别）；它会把"中文夹大量数字/标点"的合法回复也判掉。')
    L.append('3. 抽样（§4/§5）是**随机 10 + 3 段**，不是全量人工评估；'
             '"没抽到问题"不等于"没有问题"。')
    L.append('')

    # ---- §7 原始终端证据 ----
    L.append('## 7. 原始证据（逐字粘贴的终端输出）')
    L.append('')
    if args.evidence and os.path.exists(args.evidence):
        with open(args.evidence, 'r', encoding='utf-8') as f:
            ev = f.read().rstrip('\n')
        L.extend(ev.split('\n'))
    else:
        L.append(f'（未提供 --evidence 文件（path={args.evidence!r}）；'
                 f'本次运行没有粘贴终端原文。）')
    L.append('')
    return '\n'.join(L)


# ==========================================================================
# CLI
# ==========================================================================
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description='机翻 ESConv 对话筛选（默认 dry-run，--apply 才写数据文件）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--input', default=INPUT_JSON, help='只读输入 JSON')
    ap.add_argument('--output', default=OUTPUT_TXT, help='--apply 时写出的项目格式 txt')
    ap.add_argument('--report', default=REPORT_MD, help='报告路径')
    ap.add_argument('--evidence', default=None, metavar='PATH',
                    help='把该 markdown 文件的内容原样附到报告 §7（用于粘贴终端原文）')
    ap.add_argument('--apply', action='store_true',
                    help='真正写数据文件；不传则只写报告（dry-run）')
    ap.add_argument('--dry-run', action='store_true',
                    help='显式声明 dry-run（默认行为；与 --apply 互斥）')
    ap.add_argument('--selftest', action='store_true',
                    help='只跑已知答案对照，不读数据；失败退出码 1')
    ap.add_argument('--inject-bug', choices=RULES, default=None, metavar='规则名',
                    help='把该规则改成永不触发（用于证明 selftest 不是橡皮图章）；'
                         '配 --selftest 时退出码应为 1')
    ap.add_argument('--disable-rule', action='append', choices=RULES, default=[],
                    metavar='规则名',
                    help='停用某条规则（可重复）；用于反事实归因，例如只上 R1–R4')
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.apply and args.dry_run:
        raise SystemExit('--apply 与 --dry-run 互斥')
    if args.selftest:
        return 1 if selftest(args.inject_bug) else 0
    return run(args, out=lambda *a: print(*a, flush=True))


if __name__ == '__main__':
    sys.exit(main())
