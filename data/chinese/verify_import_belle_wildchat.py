#!/usr/bin/env python
"""Belle / WildChat 导入的**必做自证**（4 条）。产出可直接粘进报告的结论。

    .venv/bin/python data/chinese/verify_import_belle_wildchat.py          # 全量
    .venv/bin/python data/chinese/verify_import_belle_wildchat.py --quick  # 只跑前 200k 条

四条自证：
  1. **解析正确性**：Belle 的流式状态机扫描 vs 逐行 `json.loads` 逐条比（条数 + 前 5000 条内容）；
  2. **已知答案**：2 条手工对话走完整转换后逐字比对 + 负向对照（去掉 output 尾轮 ⇒ 每条少 1 轮）；
  3. **过滤生效**：过滤前后条数 + 产出文件结构复核（无 toxic / 无 <4 轮 / 无残标）
     + 反向对照（把过滤全关 ⇒ toxic=True 与 <4 轮的条目确实会出现）；
  4. **语言检查**：两份产出的汉字 vs 拉丁字母占比。

⚠️ 本脚本**只读**产出文件与原始数据，不改任何东西。
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import sys
import tempfile
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
NEW_SOURCES = os.path.join(ROOT, 'data', 'chinese', 'new_sources')
BELLE_PATH = '/home/vesita/datasets/NLP/_belle/multiturn_chat_0.8M.json'
WILDCHAT_DIR = '/home/vesita/datasets/NLP/_wildchat/data'

_spec = importlib.util.spec_from_file_location(
    'import_external', os.path.join(HERE, 'import_external.py'))
ie = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ie)

RESULTS: list[tuple[str, bool, str]] = []


def ok(name, cond, detail=''):
    RESULTS.append((name, bool(cond), detail))
    print(f'  {"PASS ✅" if cond else "FAIL ❌"} {name}' + (f' — {detail}' if detail else ''))


def read_blocks(path):
    with open(path, encoding='utf-8') as f:
        return [b for b in f.read().split('\n\n') if b.strip()]


def block_stats(blocks):
    turn_counts = [sum(1 for ln in b.split('\n') if ln.startswith(ie.MODEL)) for b in blocks]
    chars = [len(b) for b in blocks]
    cjk = sum(ie.count_scripts(b)[0] for b in blocks)
    latin = sum(ie.count_scripts(b)[1] for b in blocks)
    return {
        'blocks': len(blocks),
        'chars': sum(chars),
        'mean_chars': sum(chars) / max(1, len(blocks)),
        'mean_model_turns': sum(turn_counts) / max(1, len(blocks)),
        'model_turns': sum(turn_counts),
        'min_model_turns': min(turn_counts) if turn_counts else 0,
        'cjk': cjk,
        'latin': latin,
        'cjk_ratio': cjk / max(1, cjk + latin),
    }


# ---------------------------------------------------------------------------
# 自证 1：流式状态机 vs 逐行 json.loads
# ---------------------------------------------------------------------------
def check1_belle_stream(quick=False):
    limit = 200_000 if quick else None
    n_first = 5000
    t0 = time.time()
    naive, naive_count = [], 0
    with open(BELLE_PATH, encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            naive_count += 1
            if len(naive) < n_first:
                naive.append(json.loads(ln))
            if limit and naive_count >= limit:
                break
    print(f'  [逐行 json.loads] 条数={naive_count:,}（{time.time() - t0:.1f}s）')

    t0 = time.time()
    sm, sm_count = [], 0
    for raw in ie.iter_json_records(BELLE_PATH, force_state_machine=True):
        sm_count += 1
        if len(sm) < n_first:
            sm.append(json.loads(raw))
        if limit and sm_count >= limit:
            break
    print(f'  [流式状态机]   条数={sm_count:,}（{time.time() - t0:.1f}s）')
    ok('自证1a 条数一致', sm_count == naive_count, f'{sm_count:,} == {naive_count:,}')
    ok(f'自证1b 前 {n_first} 条内容逐条一致', sm == naive,
       f'比对 {len(sm)} 条 dict')
    return naive_count


# ---------------------------------------------------------------------------
# 自证 2：已知答案 + 负向对照
# ---------------------------------------------------------------------------
def check2_known_answers(tmp):
    recs = [
        {'instruction': 'Human: 你好\nAssistant: 你好呀\nHuman: 今天天气如何？\nAssistant:',
         'input': '', 'output': ' 晴天，适合出门。'},
        {'instruction': 'Human: 第一问\nAssistant: 第一答\nHuman: 第二问\nAssistant: 第二答\n'
                        'Human: 第三问\nAssistant:', 'input': '', 'output': '第三答'},
    ]
    p = os.path.join(tmp, 'known.jsonl')
    with open(p, 'w', encoding='utf-8') as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    text, kept, _, _ = ie.to_text(ie.belle_json(p, min_turns=1), min_turns=1)
    want = ('用户：你好\n模型：你好呀\n用户：今天天气如何？\n模型：晴天，适合出门。\n\n'
            '用户：第一问\n模型：第一答\n用户：第二问\n模型：第二答\n用户：第三问\n模型：第三答\n\n')
    ok('自证2a 已知答案逐字一致', text == want and kept == 2,
       f'kept={kept} chars={len(text)}')

    p2 = os.path.join(tmp, 'known_noout.jsonl')
    with open(p2, 'w', encoding='utf-8') as f:
        for r in recs:
            f.write(json.dumps(dict(r, output=''), ensure_ascii=False) + '\n')
    full = ie.belle_json(p, min_turns=1)
    noout = ie.belle_json(p2, min_turns=1)
    diffs = [sum(1 for r, _ in x if r == ie.MODEL) - sum(1 for r, _ in y if r == ie.MODEL)
             for x, y in zip(full, noout)]
    ok('自证2b 负向对照：去 output 每条少 1 轮', diffs == [1, 1],
       f'带/不带 output 的模型轮数差={diffs}')
    # 额外：残标必须被拦截（否则「过滤」是空转）
    t, why = ie.parse_belle('Human: 你知道 Assistant: 吗\nAssistant: 知道', '')
    ok('自证2c 残标（段内 Human:/Assistant:）被拦截', t is None and why == 'residual', f'why={why}')


# ---------------------------------------------------------------------------
# 自证 3a：Belle 过滤
# ---------------------------------------------------------------------------
def check3_belle(tmp, raw_count):
    st = {k: 0 for k in ie.BELLE_STAT_KEYS}
    ie._belle_scan(BELLE_PATH, 4, 0.5, 0.0, 42, st, True)     # frac=0：只统计、不存样本
    kept_all = st['raw'] - sum(st[k] for k in ie.BELLE_STAT_KEYS if k != 'raw')
    print(f'  过滤统计：{st}')
    ok('自证3a-1 过滤前条数 == 逐行解析条数', st['raw'] == raw_count,
       f"{st['raw']:,} == {raw_count:,}")
    ok('自证3a-2 过滤后条数 < 过滤前（确实丢了东西）', kept_all < st['raw'],
       f'{kept_all:,} / {st["raw"]:,} = {kept_all / st["raw"]:.1%}')

    blocks = read_blocks(os.path.join(NEW_SOURCES, 'belle_multiturn.txt'))
    min_turns = min(sum(1 for ln in b.split('\n') if ln.startswith(ie.MODEL)) for b in blocks)
    ok('自证3a-3 产出每个 block ≥4 轮', min_turns >= 4, f'最小模型轮数={min_turns}')
    resid = [b for b in blocks if ie.BELLE_RESIDUAL.search(b)]
    ok('自证3a-4 产出无 Human:/Assistant: 残留', not resid, f'命中 {len(resid)} 条')
    empty = [b for b in blocks if any(ln.strip() == ie.MODEL for ln in b.split('\n'))]
    ok('自证3a-5 产出无空回复行', not empty, f'命中 {len(empty)} 条')

    # 反向对照：把 en_ratio_max 关掉（=1.0）在同一批前 20000 条上必须多留下英文条目
    sub = os.path.join(tmp, 'belle_head20k.jsonl')
    with open(sub, 'w', encoding='utf-8') as f:
        for ln in itertools.islice(open(BELLE_PATH, encoding='utf-8'), 20000):
            f.write(ln)
    on = ie.belle_json(sub, min_turns=1, en_ratio_max=0.5)
    off = ie.belle_json(sub, min_turns=1, en_ratio_max=1.0)
    ok('自证3a-6 反向对照：关掉英文过滤后条数变多', len(off) > len(on),
       f'开={len(on)} 关={len(off)}（多 {len(off) - len(on)}）')
    return st, kept_all


# ---------------------------------------------------------------------------
# 自证 3b：WildChat 过滤
# ---------------------------------------------------------------------------
def _wildchat_row_iter(files):
    import pyarrow.parquet as pq
    for fp in files:
        pf = pq.ParquetFile(fp)
        for rg in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(rg, columns=['conversation', 'toxic', 'language', 'turn'])
            for i in range(t.num_rows):
                yield (ie.turns_from_messages(t.column('conversation')[i].as_py()),
                       t.column('toxic')[i].as_py(),
                       t.column('language')[i].as_py(),
                       t.column('turn')[i].as_py())


def check3_wildchat(quick=False):
    import glob
    files = sorted(glob.glob(os.path.join(WILDCHAT_DIR, '*.parquet')))
    if quick:
        files = files[:1]
    want = {b for b in read_blocks(os.path.join(NEW_SOURCES, 'wildchat_zh.txt'))}
    print(f'  产出 block 数={len(want):,}；回原始 parquet 逐行核对来源…')
    hit, missing, bad = 0, set(want), []
    n_rows = 0
    for turns, toxic, lang, turn in _wildchat_row_iter(files):
        n_rows += 1
        if not turns:
            continue
        text = '\n'.join(f'{r}{c}' for r, c in turns)
        if text in want:
            hit += 1
            missing.discard(text)
            if toxic or lang != 'Chinese' or turn < 4:
                bad.append((toxic, lang, turn))
    print(f'  扫描原始行 {n_rows:,} 条')
    ok('自证3b-1 产出每条都能回溯到原始行', not missing and hit > 0,
       f'命中 {hit:,} 行 / 产出 {len(want):,} block，未回溯 {len(missing)}')
    ok('自证3b-2 产出**不存在** toxic==True / 非中文 / <4 轮', not bad,
       f'违规 {len(bad)} 条')
    sample = sorted(want)[:1000]
    ok('自证3b-3 抽样 1000 条复核（结构与来源）',
       all(sum(1 for ln in b.split('\n') if ln.startswith(ie.MODEL)) >= 4 for b in sample),
       f'抽样 {len(sample)} 条全部 ≥4 轮')

    # 反向对照：过滤全关 ⇒ toxic=True 与 <4 轮的条目必须出现
    off = ie.wildchat_parquet(files[0], min_turns=1, target_chars=0, seed=42,
                              require_nontoxic=False, language=None)
    off_texts = {'\n'.join(f'{r}{c}' for r, c in t) for t in off}
    toxic_texts, short_texts = set(), set()
    for turns, toxic, lang, turn in _wildchat_row_iter(files[:1]):
        if not turns:
            continue
        text = '\n'.join(f'{r}{c}' for r, c in turns)
        if toxic:
            toxic_texts.add(text)
        if turn < 4:
            short_texts.add(text)
    ok('自证3b-4 反向对照：关过滤后 toxic=True 条目出现',
       len(off_texts & toxic_texts) > 0,
       f"关过滤得 {len(off)} 条；其中 {len(off_texts & toxic_texts)} 条来自 toxic=True 行"
       f'（原始 toxic 行共 {len(toxic_texts)}）')
    ok('自证3b-5 反向对照：关过滤后 <4 轮条目出现',
       len(off_texts & short_texts) > 0,
       f'其中 {len(off_texts & short_texts)} 条来自 <4 轮行（原始 <4 轮行共 {len(short_texts)}）')
    return off


# ---------------------------------------------------------------------------
# 自证 4：语言占比
# ---------------------------------------------------------------------------
def check4_language():
    for name in ('belle_multiturn.txt', 'wildchat_zh.txt'):
        blocks = read_blocks(os.path.join(NEW_SOURCES, name))
        s = block_stats(blocks)
        print(f'  {name}: block={s["blocks"]:,} 字符={s["chars"]:,} '
              f'均{s["mean_chars"]:.0f}字/块 均{s["mean_model_turns"]:.2f}轮/块 '
              f'汉字={s["cjk"]:,} 拉丁={s["latin"]:,} 汉字占比={s["cjk_ratio"]:.2%}')
    ok('自证4 两份产出汉字占比均 >50%（汉字是占比最高的文字）',
       all(block_stats(read_blocks(os.path.join(NEW_SOURCES, n)))['cjk_ratio'] > 0.5
           for n in ('belle_multiturn.txt', 'wildchat_zh.txt')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='只扫前 200k 条 / 只扫第一个 parquet')
    a = ap.parse_args()
    print('===== Belle / WildChat 导入自证 =====')
    raw_count = check1_belle_stream(quick=a.quick)
    with tempfile.TemporaryDirectory() as tmp:
        check2_known_answers(tmp)
        check3_belle(tmp, raw_count)
        check3_wildchat(quick=a.quick)
    check4_language()
    fails = [n for n, c, _ in RESULTS if not c]
    print('===== ' + (f'全部通过 ✅（{len(RESULTS)} 项）' if not fails else f'失败 ❌ {fails}'))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
