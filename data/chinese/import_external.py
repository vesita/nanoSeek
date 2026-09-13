"""把外部数据集转换成项目语料格式（`用户：`/`模型：` + 空行分块）。

为什么单独抽一个脚本：新数据会从多个渠道进来（ModelScope JSONL、HF parquet、
翻译产物），格式各不相同（question/answer、messages、instruction/output）。转换规则
必须**只有一份实现**，否则每进来一份数据就重写一次、口径各不相同。

用法：
    .venv/bin/python data/chinese/import_external.py \
        --format qa_jsonl --in <path.jsonl> --out data/chinese/new_sources/qa_knowledge.txt
    .venv/bin/python data/chinese/import_external.py --selftest

格式：
  qa_jsonl         每行 {"question": str, "answer": str}
  messages_json    list[{"messages":[{"role":"user"|"assistant","content":str}, ...]}]
  alpaca_json      list[{"instruction": str, "input": str, "output": str}]
  sharegpt_jsonl   ShareGPT 系 JSONL（两种结构，见 sharegpt_jsonl 文档）
  belle_json       Belle `multiturn_chat_0.8M.json`（990MB，**流式**，见 belle_json 文档）
  wildchat_parquet WildChat parquet（**按 row group** 读，见 wildchat_parquet 文档）

约定（与 data/chinese/multi_turn_dialogue.txt 同款）：
  * 一条样本 = 一个 block，block 内每行一轮，`用户：…` / `模型：…`
  * block 之间用空行分隔
  * 内容里的换行压成空格（否则一条样本会被拆成多个 block）
  * 少于 --min-turns 轮的样本默认丢弃（单轮 QA 设成 1 即可保留）

大文件纪律：`belle_json` / `wildchat_parquet` 都**不能**先 json.load 到内存
（990MB / 122k 行），所以这两个分支自带流式解析 + 质量过滤 + 按字符预算的
**确定性抽样**（固定 seed，两遍扫描：先量总量、再按 hash 阈值抽）。
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys

USER = '用户：'
MODEL = '模型：'


def clean_line(s):
    """压掉换行/多余空白：block 结构靠空行分隔，样本内不能有换行。"""
    return re.sub(r'\s+', ' ', str(s)).strip()


# ---------------------------------------------------------------------------
# 通用工具：流式 JSON 记录扫描 / 确定性抽样 / 中英文字符统计
# ---------------------------------------------------------------------------
_JSON_SPECIAL = re.compile(r'["{}\\]')


def iter_json_records(path, chunk_size=1 << 22, force_state_machine=False):
    """流式产出**顶层** JSON 对象（字符串）。兼容两种物理布局：

    (a) 整个文件是一个 JSON 大数组 `[{...},\\n{...}]`（`force_state_machine=True`
        或首字节是 `[` 时走通用状态机）；
    (b) NDJSON：一行一个对象（走 `readline` 快路径）。

    Belle 的 `multiturn_chat_0.8M.json` 实测是 (b)（831,036 行、每行 `}\\n` 结尾），
    但**不假设** —— 通用状态机在自证脚本里被强制跑满整文件，用来证明两种口径
    的条数与内容逐条一致。
    """
    with open(path, 'rb') as fb:
        head = fb.read(4096)
    use_sm = force_state_machine or head.lstrip()[:1] == b'['
    if not use_sm:
        with open(path, encoding='utf-8') as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    yield ln
        return
    # 通用状态机：按特殊字符（`"` `{` `}` `\\`）跳转，字符串外的普通字符整段切片，
    # 避免逐字符 Python 循环（990MB 逐字符要几十分钟）。
    depth = 0
    in_str = False
    esc = False
    buf = []
    with open(path, encoding='utf-8') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            i, n = 0, len(chunk)
            while i < n:
                if not in_str:
                    m = _JSON_SPECIAL.search(chunk, i)
                    end = m.start() if m else n
                    if depth > 0 and end > i:
                        buf.append(chunk[i:end])
                    if not m:
                        i = n
                        break
                    ch = m.group()
                    i = m.start() + 1
                    if ch == '"':
                        in_str = True
                        buf.append('"')
                    elif ch == '{':
                        depth += 1
                        buf.append('{')
                    elif ch == '}':
                        depth -= 1
                        buf.append('}')
                        if depth == 0:
                            yield ''.join(buf)
                            buf = []
                    elif depth > 0:
                        buf.append(ch)
                else:
                    if esc:
                        # 转义序列的第二个字符：不论是什么都原样吃掉（可能是普通字符，
                        # 也可能跨 chunk 边界 —— 所以 esc 必须在「吃掉下一个字符」时清，
                        # 而不是只在遇到特殊字符时清）
                        esc = False
                        buf.append(chunk[i])
                        i += 1
                        continue
                    m = _JSON_SPECIAL.search(chunk, i)
                    end = m.start() if m else n
                    buf.append(chunk[i:end + 1] if m else chunk[i:])
                    if not m:
                        i = n
                        break
                    ch = m.group()
                    i = end + 1
                    if ch == '\\':
                        esc = True
                    elif ch == '"':
                        in_str = False
    if depth != 0:
        raise ValueError(f'{path}: JSON 对象中途结束（depth={depth}），结构损坏')


def hash_frac(seed, idx):
    """`blake2b(seed:idx)` → [0,1) 的确定性伪随机数。

    用哈希而不是 `random` 是为了**跨 Python 版本/实现可复现**，也不依赖遍历顺序之外的
    任何状态。
    """
    h = hashlib.blake2b(f'{seed}:{idx}'.encode(), digest_size=8).digest()
    return int.from_bytes(h, 'big') / (1 << 64)


def trim_to_target(kept, target_chars):
    """`kept` 是 [(u, turns), …]（u = 抽样哈希键）。按 u 升序填充到 ≤ target_chars。

    为什么要截断：Bernoulli 抽样的总量方差随 block 字符数的**重尾**放大
    （WildChat 里单条可达 10 万字符），实测 15.0M 目标抽到 16.1M（+7%）。
    按 u 序确定性填充 ⇒ 既保住「与 seed 绑定、不取前缀」，又把总量焊死在目标以下。
    """
    if not target_chars:
        return [t for _, t in kept]
    out, acc = [], 0
    for _, turns in sorted(kept, key=lambda x: x[0]):
        n = block_chars(turns)
        if acc + n > target_chars:
            continue
        out.append(turns)
        acc += n
    return out


def block_chars(turns):
    """一个 block 渲染后的字符数（不含用于分隔 block 的空行）。"""
    return len('\n'.join(f'{r}{c}' for r, c in turns))


CJK_RE = re.compile(r'[\u4e00-\u9fff]')
LATIN_RE = re.compile(r'[A-Za-z]')


def count_scripts(text):
    """返回 (汉字数, 拉丁字母数)。"""
    return len(CJK_RE.findall(text)), len(LATIN_RE.findall(text))


def qa_jsonl(path):
    out = []
    with open(path, encoding='utf-8', errors='ignore') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                o = json.loads(ln)
            except json.JSONDecodeError:
                continue
            q, a = o.get('question'), o.get('answer')
            if q and a:
                out.append([(USER, clean_line(q)), (MODEL, clean_line(a))])
    return out


def turns_from_messages(msgs):
    """messages 列表 → (用户/模型) 轮次；跳过 system/tool 与空内容。

    这是**唯一**一份 role→标签 的映射实现：`messages_json`（JSON 文件）与
    `wildchat_parquet`（parquet，等价于先落 messages-jsonl）都调它。
    """
    turns = []
    for m in msgs or []:
        if not isinstance(m, dict):
            continue
        role = (m.get('role') or '').lower()
        content = clean_line(m.get('content', ''))
        if not content:
            continue
        if role in ('user', 'human'):
            turns.append((USER, content))
        elif role in ('assistant', 'gpt', 'model'):
            turns.append((MODEL, content))
        # 其它 role（system/tool）跳过：本项目语料没有对应的说话人标签
    return turns


def messages_json(path):
    data = json.load(open(path, encoding='utf-8'))
    if isinstance(data, dict):
        data = data.get('data') or data.get('messages') or []
    out = []
    for conv in data:
        msgs = conv.get('messages') if isinstance(conv, dict) else None
        if not msgs:
            continue
        turns = turns_from_messages(msgs)
        if turns:
            out.append(turns)
    return out


def sharegpt_jsonl(path):
    """ShareGPT 系 JSONL，兼容两种常见结构：

    (a) `{"conversation": [{"human": "...", "assistant": "..."}, ...]}` —— 每元素是一整个来回
        （`shareAI/ShareGPT-Chinese-English-90k` 的 `unknow_zh_38k.jsonl` 就是这种，
        2026-09-13 实测：一条 = 一个 dict 同时含 human 与 assistant 两个键）；
    (b) `{"conversations": [{"from": "human"/"gpt", "value": "..."}, ...]}` —— 标准 ShareGPT。

    跳过 system/工具轮；两种结构都按出现顺序展开成 (用户, 模型) 交替的轮次。
    """
    H = ('human', 'user')
    A = ('gpt', 'assistant', 'model')
    out = []
    with open(path, encoding='utf-8', errors='ignore') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                o = json.loads(ln)
            except json.JSONDecodeError:
                continue
            turns = []
            pairs = o.get('conversation')            # 结构 (a)
            if isinstance(pairs, list):
                for p in pairs:
                    if not isinstance(p, dict):
                        continue
                    for k in H:
                        if p.get(k):
                            turns.append((USER, clean_line(p[k])))
                            break
                    for k in A:
                        if p.get(k):
                            turns.append((MODEL, clean_line(p[k])))
                            break
            else:
                convs = o.get('conversations') or o.get('messages')   # 结构 (b)
                for m in convs or []:
                    if not isinstance(m, dict):
                        continue
                    who = str(m.get('from') or m.get('role') or '').lower()
                    val = clean_line(m.get('value') if m.get('value') is not None else m.get('content', ''))
                    if not val:
                        continue
                    if who in H:
                        turns.append((USER, val))
                    elif who in A:
                        turns.append((MODEL, val))
            if turns:
                out.append(turns)
    return out


def alpaca_json(path):
    data = json.load(open(path, encoding='utf-8'))
    if isinstance(data, dict):
        data = data.get('data') or []
    out = []
    for o in data:
        q = clean_line(o.get('instruction', ''))
        inp = clean_line(o.get('input', ''))
        a = clean_line(o.get('output', ''))
        if not (q and a):
            continue
        if inp:
            q = f'{q} {inp}'
        out.append([(USER, q), (MODEL, a)])
    return out


FORMATS = {'qa_jsonl': qa_jsonl, 'messages_json': messages_json, 'alpaca_json': alpaca_json,
           'sharegpt_jsonl': sharegpt_jsonl}


# ---------------------------------------------------------------------------
# Belle multiturn_chat_0.8M：多轮对话埋在 instruction 字符串里
# ---------------------------------------------------------------------------
BELLE_MARK = re.compile(r'(?:^|\n)[ \t]*(Human|Assistant)[ \t]*:')
BELLE_RESIDUAL = re.compile(r'(?:Human|Assistant)[ \t]*:')
BELLE_STAT_KEYS = ('raw', 'bad_json', 'no_marker', 'residual', 'empty_user', 'empty_model',
                   'english', 'short')


def parse_belle(instruction, output, en_ratio_max=0.5):
    """还原一条 Belle 记录的多轮对话。返回 `(turns, None)` 或 `(None, 丢弃原因)`。

    `instruction` 形如 `'Human:…\\nAssistant: …\\nHuman: …\\nAssistant:'`，
    **最后一个 `Assistant:` 后面通常是空的**，真正的尾轮内容在同一条记录的 `output` 里。
    丢掉 output ⇒ 末尾那个空 `Assistant:` 变成一个空轮，被丢弃 ⇒ 少一轮 ——
    这正是负向对照要证明的事。

    丢弃原因：no_marker（没有标记）/ residual（切分后段内仍含 `Human:`/`Assistant:`，
    说明标记切分不可信）/ empty_user / empty_model（空回复）/ english（纯英文占比过高）。
    """
    text = instruction or ''
    marks = list(BELLE_MARK.finditer(text))
    if not marks:
        return None, 'no_marker'
    segs = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        segs.append((m.group(1), text[m.end():end]))
    if any(BELLE_RESIDUAL.search(raw) for _, raw in segs):
        return None, 'residual'
    if segs[-1][0] == 'Assistant':
        segs[-1] = ('Assistant', segs[-1][1] + ' ' + (output or ''))
    elif (output or '').strip():
        segs.append(('Assistant', output))
    turns = [(USER if who == 'Human' else MODEL, clean_line(raw)) for who, raw in segs]
    # 末尾空模型轮（output 也为空）是结构性的，丢掉；中段空轮是坏样本，整条丢。
    while turns and turns[-1][0] == MODEL and not turns[-1][1]:
        turns.pop()
    if not turns:
        return None, 'no_marker'
    if any(r == USER and not c for r, c in turns):
        return None, 'empty_user'
    if any(r == MODEL and not c for r, c in turns):
        return None, 'empty_model'
    cjk, latin = count_scripts(''.join(c for _, c in turns))
    if en_ratio_max < 1.0 and (cjk == 0 or latin / (cjk + latin) > en_ratio_max):
        return None, 'english'
    return turns, None


def _belle_scan(path, min_turns, en_ratio_max, frac, seed, st, count_stats):
    total, kept = 0, []
    for idx, raw in enumerate(iter_json_records(path)):
        if count_stats:
            st['raw'] += 1
        try:
            o = json.loads(raw)
        except json.JSONDecodeError:
            if count_stats:
                st['bad_json'] += 1
            continue
        turns, why = parse_belle(o.get('instruction', ''), o.get('output', ''),
                                 en_ratio_max=en_ratio_max)
        if why:
            if count_stats:
                st[why] += 1
            continue
        if sum(1 for r, _ in turns if r == MODEL) < min_turns:
            if count_stats:
                st['short'] += 1
            continue
        n = block_chars(turns)
        total += n
        if frac >= 1.0:
            kept.append((0.0, turns))
        else:
            u = hash_frac(seed, idx)
            if u < frac:
                kept.append((u, turns))
    return total, kept


def belle_json(path, min_turns=4, target_chars=None, seed=42, en_ratio_max=0.5, stats=None):
    """Belle `multiturn_chat_0.8M.json` → 样本列表（流式 + 过滤 + 确定性抽样）。

    990MB 不能整文件 `json.load`；两遍扫描：第 1 遍量「通过过滤的 block 字符总量」，
    第 2 遍按 `target_chars / total` 的哈希阈值抽，再用 `trim_to_target` 按 hash 序
    截到 target 以下（固定 seed ⇒ 确定性、不取文件前缀）。
    """
    st = stats if stats is not None else {}
    for k in BELLE_STAT_KEYS + ('kept', 'kept_chars_all', 'kept_chars', 'sample_frac'):
        st.setdefault(k, 0)
    total, _ = _belle_scan(path, min_turns, en_ratio_max, 1.0, seed, st, True)
    st['kept_chars_all'] = total
    frac = 1.0 if not target_chars or total <= target_chars else target_chars / total
    st['sample_frac'] = frac
    _, kept = _belle_scan(path, min_turns, en_ratio_max, frac, seed, st, False)
    kept = trim_to_target(kept, target_chars if frac < 1.0 else 0)
    st['kept'] = len(kept)
    st['kept_chars'] = sum(block_chars(t) for t in kept)
    return kept


# ---------------------------------------------------------------------------
# WildChat parquet：按 row group 读，等价于先落 messages-jsonl 再走 messages_json
# ---------------------------------------------------------------------------
WILDCHAT_STAT_KEYS = ('raw', 'toxic', 'lang', 'short', 'short_after', 'kept_pre', 'kept')


def _wildchat_iter(path, min_turns, require_nontoxic, language, st):
    import pyarrow.parquet as pq
    files = (sorted(glob.glob(os.path.join(path, '*.parquet')))
             if os.path.isdir(path) else [path])
    cols = ['conversation', 'toxic', 'language', 'turn']
    for fp in files:
        pf = pq.ParquetFile(fp)
        for rg in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(rg, columns=cols)      # 只读需要的列 + 一个 row group
            convs = t.column('conversation')
            toxics = t.column('toxic').to_pylist()
            langs = t.column('language').to_pylist()
            turns_col = t.column('turn').to_pylist()
            for i in range(t.num_rows):
                st['raw'] += 1
                if require_nontoxic and toxics[i]:
                    st['toxic'] += 1
                    continue
                if language and langs[i] != language:
                    st['lang'] += 1
                    continue
                if turns_col[i] < min_turns:
                    st['short'] += 1
                    continue
                turns = turns_from_messages(convs[i].as_py())
                if sum(1 for r, _ in turns if r == MODEL) < min_turns:
                    st['short_after'] += 1
                    continue
                st['kept_pre'] += 1
                yield turns


def wildchat_parquet(path, min_turns=4, target_chars=None, seed=42,
                     require_nontoxic=True, language='Chinese', stats=None):
    """WildChat parquet → 样本列表（按 row group 流式 + 过滤 + 确定性抽样）。

    `path` 可以是单个 `.parquet` 或装着 `*.parquet` 的目录。
    过滤：`toxic == False` 且 `language == 'Chinese'` 且 `turn >= min_turns`
    （实测 `turn` == assistant 轮数 == user 轮数），转换后再按 `模型：` 轮数复核一次。
    抽样的两遍扫描与 Belle 同款（`belle_json`）。
    """
    st = stats if stats is not None else {}
    for k in WILDCHAT_STAT_KEYS + ('kept_chars_all', 'kept_chars', 'sample_frac'):
        st.setdefault(k, 0)

    def scan(frac, count_stats):
        total, kept = 0, []
        # 抽样的 idx 用「过滤器之后的序号」，与 Belle 的原始行号口径一致（都是流式位置）
        for idx, turns in enumerate(_wildchat_iter(
                path, min_turns, require_nontoxic, language,
                st if count_stats else {k: 0 for k in WILDCHAT_STAT_KEYS})):
            n = block_chars(turns)
            total += n
            if frac >= 1.0:
                kept.append((0.0, turns))
            else:
                u = hash_frac(seed, idx)
                if u < frac:
                    kept.append((u, turns))
        return total, kept

    total, _ = scan(1.0, True)
    st['kept_chars_all'] = total
    frac = 1.0 if not target_chars or total <= target_chars else target_chars / total
    st['sample_frac'] = frac
    _, kept = scan(frac, False)
    kept = trim_to_target(kept, target_chars if frac < 1.0 else 0)
    st['kept'] = len(kept)
    st['kept_chars'] = sum(block_chars(t) for t in kept)
    return kept


FORMATS.update({'belle_json': belle_json, 'wildchat_parquet': wildchat_parquet})


def to_text(samples, min_turns=1):
    """把样本列表渲染成项目 .txt 文本；返回 (text, kept, dropped_short, dropped_empty)。"""
    blocks, short = [], 0
    for turns in samples:
        n_model = sum(1 for r, _ in turns if r == MODEL)
        if n_model < min_turns:
            short += 1
            continue
        if not turns:
            continue
        blocks.append('\n'.join(f'{r}{c}' for r, c in turns))
    return '\n\n'.join(blocks) + ('\n\n' if blocks else ''), len(blocks), short, len(samples) - len(blocks) - short


def selftest():
    """已知答案自检：三种格式各造一条，断言转换结果逐字符合预期。"""
    import tempfile
    fails = []
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, 'qa.jsonl')
        open(p, 'w', encoding='utf-8').write('{"question":" 你好 ","answer":"你好呀\\n最近如何"}\n')
        got = to_text(qa_jsonl(p))[0]
        want = f'{USER}你好\n{MODEL}你好呀 最近如何\n\n'
        if got != want:
            fails.append(f'qa_jsonl 不符：\n got={got!r}\nwant={want!r}')

        p = os.path.join(tmp, 'msg.json')
        json.dump([{'messages': [{'role': 'user', 'content': 'A'},
                                 {'role': 'assistant', 'content': 'B'},
                                 {'role': 'system', 'content': 'X'}]}], open(p, 'w'))
        got = to_text(messages_json(p))[0]
        want = f'{USER}A\n{MODEL}B\n\n'
        if got != want:
            fails.append(f'messages_json 不符：{got!r}')

        p = os.path.join(tmp, 'alp.json')
        json.dump([{'instruction': '问', 'input': '补充', 'output': '答'}], open(p, 'w'))
        got = to_text(alpaca_json(p))[0]
        want = f'{USER}问 补充\n{MODEL}答\n\n'
        if got != want:
            fails.append(f'alpaca_json 不符：{got!r}')

        p = os.path.join(tmp, 'sg.jsonl')
        with open(p, 'w', encoding='utf-8') as f:
            f.write(json.dumps({'conversation': [{'human': '甲', 'assistant': '乙'},
                                                 {'human': '丙', 'assistant': '丁'}]}, ensure_ascii=False) + '\n')
            f.write(json.dumps({'conversations': [{'from': 'human', 'value': '戊'},
                                                  {'from': 'gpt', 'value': '己'},
                                                  {'from': 'system', 'value': 'X'}]}, ensure_ascii=False) + '\n')
        got = to_text(sharegpt_jsonl(p))[0]
        want = f'{USER}甲\n{MODEL}乙\n{USER}丙\n{MODEL}丁\n\n{USER}戊\n{MODEL}己\n\n'
        if got != want:
            fails.append(f'sharegpt_jsonl 不符：\n got={got!r}\nwant={want!r}')

        # 对照：min_turns=2 时单轮样本必须被丢掉（证明该参数真的在生效）
        p = os.path.join(tmp, 'qa.jsonl')
        got, kept, short, _ = to_text(qa_jsonl(p), min_turns=2)
        if kept != 0 or short != 1:
            fails.append(f'min_turns 过滤没生效：kept={kept} short={short}')
        # 对照：system role 必须真的被跳过（而不是变成某一方说的话）
        if 'X' in to_text(messages_json(os.path.join(tmp, 'msg.json')))[0]:
            fails.append('system 消息没有被跳过')

        # ---- belle_json：已知答案 + 尾轮负向对照 ----
        b = os.path.join(tmp, 'belle.jsonl')
        recs = [
            {'instruction': 'Human: 你好\nAssistant: 你好呀\nHuman: 今天天气如何？\nAssistant:',
             'input': '', 'output': ' 晴天，适合出门。'},
            {'instruction': 'Human: 第一问\nAssistant: 第一答\nHuman: 第二问\nAssistant: 第二答\n'
                            'Human: 第三问\nAssistant:', 'input': '', 'output': '第三答'},
        ]
        with open(b, 'w', encoding='utf-8') as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        got = to_text(belle_json(b, min_turns=1))[0]
        want = (f'{USER}你好\n{MODEL}你好呀\n{USER}今天天气如何？\n{MODEL}晴天，适合出门。\n\n'
                f'{USER}第一问\n{MODEL}第一答\n{USER}第二问\n{MODEL}第二答\n'
                f'{USER}第三问\n{MODEL}第三答\n\n')
        if got != want:
            fails.append(f'belle_json 不符：\n got={got!r}\nwant={want!r}')
        # 负向对照：去掉 output 尾轮 ⇒ **每条**的模型轮数必须少 1
        b2 = os.path.join(tmp, 'belle_noout.jsonl')
        with open(b2, 'w', encoding='utf-8') as f:
            for r in recs:
                f.write(json.dumps(dict(r, output=''), ensure_ascii=False) + '\n')
        full = belle_json(b, min_turns=1)
        noout = belle_json(b2, min_turns=1)
        diffs = [sum(1 for r, _ in x if r == MODEL) - sum(1 for r, _ in y if r == MODEL)
                 for x, y in zip(full, noout)]
        if diffs != [1, 1]:
            fails.append(f'belle 尾轮没接上：带/不带 output 的模型轮数差={diffs}（期望 [1, 1]）')
        # 残标必须整条丢掉（切分不可信）
        t, why = parse_belle('Human: 你认识 Assistant: 这个词吗\nAssistant: 认识', '')
        if why != 'residual' or t is not None:
            fails.append(f'belle 残标未拦截：turns={t} why={why}')
        # 纯英文条目必须被 english 规则丢掉
        t, why = parse_belle('Human: What is the capital of France?\nAssistant:',
                             ' Paris is the capital of France.')
        if why != 'english':
            fails.append(f'belle 纯英文未拦截：why={why}')
        # 流式扫描器（状态机）必须和逐行 json.loads 逐条一致
        sm = [json.loads(x) for x in iter_json_records(b, force_state_machine=True)]
        naive = [json.loads(x) for x in open(b, encoding='utf-8') if x.strip()]
        if sm != naive:
            fails.append('iter_json_records 状态机与逐行解析不一致')

        # ---- wildchat_parquet：必须与「先落 messages-jsonl 再走 messages_json」等价 ----
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            pa = None
        if pa is not None:
            convs = [
                [{'role': 'user', 'content': '问一'}, {'role': 'assistant', 'content': '答一'}],
                [{'role': 'system', 'content': 'S'}, {'role': 'user', 'content': '问二\n换行'},
                 {'role': 'assistant', 'content': '答二'}],
            ]
            wp = os.path.join(tmp, 'wc.parquet')
            pq.write_table(pa.table({'conversation': convs, 'toxic': [False, True],
                                     'language': ['Chinese', 'English'], 'turn': [1, 1]}), wp)
            nf = dict(min_turns=1, require_nontoxic=False, language=None)
            got = to_text(wildchat_parquet(wp, **nf))[0]
            mj = os.path.join(tmp, 'wc.json')
            json.dump([{'messages': c} for c in convs], open(mj, 'w'))
            want = to_text(messages_json(mj))[0]
            if got != want:
                fails.append(f'wildchat_parquet 与 messages_json 不等价：\n got={got!r}\nwant={want!r}')
            # 复用同一份 role 映射（而不是另写一套）
            if wildchat_parquet(wp, **nf)[1] != turns_from_messages(convs[1]):
                fails.append('wildchat_parquet 没有复用 turns_from_messages')
            # 过滤生效：toxic=True / 非中文 必须被丢掉
            got2 = to_text(wildchat_parquet(wp, min_turns=1))[0]
            if got2 != f'{USER}问一\n{MODEL}答一\n\n':
                fails.append(f'wildchat 过滤没生效：{got2!r}')

    print('===== import_external selftest ' + ('全部通过 ✅' if not fails else f'失败 ❌ ({len(fails)})') + ' =====')
    for x in fails:
        print('  ✗ ' + x)
    return 1 if fails else 0


BIG_FORMATS = ('belle_json', 'wildchat_parquet')


def build_samples(a, stats):
    """按格式调对应的解析器；大格式多传过滤/抽样参数。"""
    if a.format == 'belle_json':
        return belle_json(a.src, min_turns=a.min_turns, target_chars=a.target_chars,
                          seed=a.seed, en_ratio_max=(1.0 if a.no_filter else a.max_en_ratio),
                          stats=stats)
    if a.format == 'wildchat_parquet':
        return wildchat_parquet(a.src, min_turns=(1 if a.no_filter else a.min_turns),
                                target_chars=a.target_chars, seed=a.seed,
                                require_nontoxic=not a.no_filter,
                                language=None if a.no_filter else 'Chinese', stats=stats)
    return FORMATS[a.format](a.src)


def main():
    ap = argparse.ArgumentParser(description='外部数据集 → 项目语料格式')
    ap.add_argument('--format', choices=sorted(FORMATS))
    ap.add_argument('--in', dest='src', default=None)
    ap.add_argument('--out', dest='dst', default=None)
    ap.add_argument('--min-turns', type=int, default=1,
                    help='至少这么多条「模型：」回复才保留（单轮 QA 用 1；多轮数据用 4）')
    ap.add_argument('--target-chars', type=int, default=0,
                    help='大格式：按字符预算做确定性抽样（0 = 不限）')
    ap.add_argument('--seed', type=int, default=42, help='抽样 seed（确定性）')
    ap.add_argument('--max-en-ratio', type=float, default=0.5,
                    help='belle_json：拉丁字母/(汉字+拉丁字母) 超过它 → 丢（1.0 = 关）')
    ap.add_argument('--no-filter', action='store_true',
                    help='**调试/反向对照用**：关掉质量过滤（belle 的英文/空回复、'
                         'wildchat 的 toxic+language+轮数）')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(selftest())
    if not (a.format and a.src and a.dst):
        raise SystemExit('需要 --format / --in / --out（或 --selftest）')

    stats = {}
    samples = build_samples(a, stats)
    text, kept, short, other = to_text(samples, a.min_turns)
    os.makedirs(os.path.dirname(os.path.abspath(a.dst)), exist_ok=True)
    with open(a.dst, 'w', encoding='utf-8') as f:
        f.write(text)
    n_turns = sum(len(s) for s in samples[:kept]) if kept else 0
    print(f'格式 {a.format}：读入 {len(samples):,} 条样本')
    print(f'  保留 {kept:,} 条（轮数不足丢弃 {short:,}，其它 {other:,}）')
    print(f'  字符 {len(text):,} | 平均 {len(text)/max(1,kept):.0f} 字/条 | 前 {kept} 条共 {n_turns} 轮')
    if stats:
        print('  过滤统计：' + ' | '.join(f'{k}={v}' for k, v in stats.items()
                                          if not isinstance(v, float) or v < 1e-3))
    if a.format in BIG_FORMATS and a.target_chars:
        print(f'  抽样：目标 {a.target_chars:,} 字符 | 过滤后全量 {stats.get("kept_chars_all", 0):,}'
              f' | frac={stats.get("sample_frac", 1.0):.6f} | 实得 {stats.get("kept_chars", 0):,}')
    print(f'  → {a.dst}')


if __name__ == '__main__':
    main()
