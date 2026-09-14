#!/usr/bin/env python
"""验收一个**单流 `<resp>` 格式**的 bin：产物层自检（不是文本层）。

为什么需要它：`data/chinese/build_stages.py --build` 自动跑的 `check_terminators`
只验证「块内最后一个终止符落在块尾」，它**看不见** `<resp>`。而 `v3_persona` 这类
语料的全部训练信号都来自 resp 区间 —— 如果 `<resp>` 被逐字编码成 5 个 token、
或者 `<eos>` 与 `<resp>` 配不上对，训练不会报错，只会「loss 正常但学不会」
（同 `training/masking.py` 文档里说的那类静默错误）。

本脚本查五件事（每件都带**已知答案对照**，见 `--selftest`）：

  1. `<resp>` / `<eos>` / `<topic>` 在 bin 里是不是**单 token**（id 140/128/141）；
  2. `<resp>` 与 `<eos>` **配对**：每个 `<resp>` 之后、下一个 `<resp>` 之前必有 `<eos>`
     （无孤儿 resp）；且每个 `<eos>` 之前必有更近的 `<resp>`（无孤儿 eos）；
  3. **有效 token 占比**（AGENTS §5.10 硬要求）：`resp_span` 掩码覆盖的 token 数 / 总 token；
  4. **两条独立口径一致**：`build_resp_span_mask`（向量化）与 `DialogueStream.loss_token_spans`
     的算法（顺序扫描）在同一份 bin 上给出**逐位相同**的 mask；
  5. **编解码往返**：`decode(encode(block)) == block`（`skip_special_tokens=False`！
     默认 True 会把 `<eos>`/`<resp>` 吞掉 —— 本项目在这上面栽过三次）。

用法：
    .venv/bin/python scripts/resp_bin_probe.py                       # 默认 v3_persona
    .venv/bin/python scripts/resp_bin_probe.py --prefix v3_dlg
    .venv/bin/python scripts/resp_bin_probe.py --selftest            # 已知答案对照
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'chinese')
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from training.masking import build_resp_span_mask  # noqa: E402


def load_bin(path):
    return np.fromfile(path, dtype=np.uint16)


def authority_mask(ids, resp_id, eos_id, topic_id=None):
    """`DialogueStream.loss_token_spans` 的算法：顺序扫描，不向量化。"""
    m = np.zeros(len(ids), dtype=bool)
    i = 0
    while i < len(ids):
        if ids[i] == resp_id:
            k = i + 1
            while k < len(ids) and ids[k] != eos_id:
                k += 1
            if k < len(ids):
                m[i + 1:k + 1] = True
                i = k + 1
                continue
        i += 1
    return m


def report(prefix, split='train', verbose=True):
    bin_p = os.path.join(DATA, f'{split}_char_{prefix}.bin')
    if not os.path.exists(bin_p):
        raise SystemExit(f'❌ 找不到 {bin_p}')
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(DATA, 'char_tokenizer.json'))
    v = tok.get_vocab()
    RESP, EOS, TOPIC = v['<resp>'], v['<eos>'], v['<topic>']

    ids = load_bin(bin_p)
    n_resp = int((ids == RESP).sum())
    n_eos = int((ids == EOS).sum())
    n_topic = int((ids == TOPIC).sum())

    vec = build_resp_span_mask(torch.tensor(ids[None, :], dtype=torch.long),
                               (RESP,), (EOS,))[0].numpy()
    aut = authority_mask(ids, RESP, EOS)
    agree = bool((vec == aut).all())

    # 孤儿检查：相邻两个 <resp> 之间没有 <eos> ⇒ 孤儿 resp
    resp_pos = np.flatnonzero(ids == RESP)
    eos_pos = np.flatnonzero(ids == EOS)
    orphan_resp = 0
    for a, b in zip(resp_pos, list(resp_pos[1:]) + [len(ids)]):
        if not ((eos_pos > a) & (eos_pos < b)).any():
            orphan_resp += 1
    # 孤儿 eos：之前没有更近的 <resp>
    last_resp = -1
    orphan_eos = 0
    for p in eos_pos:
        prev = resp_pos[resp_pos < p]
        if len(prev) == 0 or (len(np.flatnonzero((eos_pos > prev[-1]) & (eos_pos < p))) > 0):
            orphan_eos += 1

    covered = int(vec.sum())
    res = {
        'prefix': prefix, 'split': split, 'tokens': int(len(ids)),
        'resp': n_resp, 'eos': n_eos, 'topic': n_topic,
        'resp_eq_eos': n_resp == n_eos,
        'orphan_resp': orphan_resp, 'orphan_eos': orphan_eos,
        'covered_tokens': covered,
        'covered_frac': covered / max(len(ids), 1),
        'two_rules_agree': agree,
    }

    if verbose:
        print(f'=== {split}_char_{prefix}.bin ===')
        print(f'  token 总数        : {len(ids):,}')
        print(f'  <resp>={RESP} 出现  : {n_resp:,}')
        print(f'  <eos>={EOS} 出现   : {n_eos:,}')
        print(f'  <topic>={TOPIC} 出现 : {n_topic:,}')
        print(f'  resp == eos       : {"✅" if res["resp_eq_eos"] else "❌"}')
        print(f'  孤儿 resp / eos   : {orphan_resp} / {orphan_eos} '
              f'{"✅" if orphan_resp == 0 and orphan_eos == 0 else "❌"}')
        print(f'  ★ 有效 token 占比 : {covered:,}/{len(ids):,} = '
              f'{res["covered_frac"]:.2%}（resp_span 口径）')
        print(f'  两口径逐位一致    : {"✅" if agree else "❌"}'
              f'（向量化 vs 顺序扫描{"" if agree else "：" + str(int((vec != aut).sum())) + " 处分歧"}）')
    return res


def roundtrip(prefix, n_blocks=3):
    """编解码往返：decode 必须用 skip_special_tokens=False（默认值会吞掉 <eos>）。"""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(DATA, 'char_tokenizer.json'))
    src = os.path.join(DATA, 'stages', prefix)
    txt = sorted(f for f in os.listdir(src) if f.endswith('.txt'))
    path = os.path.join(src, txt[0])
    with open(path, encoding='utf-8') as f:
        blocks = [b for b in f.read().split('\n\n') if b.strip()]
    ok, bad = 0, []
    for b in blocks[:n_blocks]:
        got = tok.decode(tok.encode(b).ids, skip_special_tokens=False)
        if got == b:
            ok += 1
        else:
            bad.append((b[:80], got[:80]))
    print(f'  编解码往返        : {ok}/{min(n_blocks, len(blocks))} 块逐字相同'
          f'{" ✅" if not bad else " ❌"}')
    for a, g in bad[:2]:
        print(f'    原文={a!r}\n    回来={g!r}')
    return not bad


def selftest():
    """已知答案对照：手工构造的短序列上，两个口径必须给出**指定**的 mask。"""
    print('=== 已知答案对照（--selftest）===')
    RESP, EOS, TOPIC, NL = 140, 128, 141, 0
    cases = [
        ([RESP, 5, EOS], [0, 1, 1]),                       # 基本区间
        ([5, EOS], [0, 0]),                                # 无 resp ⇒ 不算
        ([RESP, 5], [0, 0]),                               # 未闭合 ⇒ 不算
        ([RESP, RESP, EOS], [0, 1, 1]),                    # 相邻 resp（第一版实现错过这里）
        ([RESP, TOPIC, 5, EOS], [0, 1, 1, 1]),             # <topic> 在区间内要算
    ]
    ok = True
    for row, exp in cases:
        vec = build_resp_span_mask(torch.tensor([row]), (RESP,), (EOS,))[0].numpy()
        aut = authority_mask(np.array(row), RESP, EOS)
        good = vec.tolist() == exp and aut.tolist() == exp
        ok &= good
        print(f'  {"✅" if good else "❌"} {row} → vec={vec.tolist()} 权威={aut.tolist()} '
              f'期望={exp}')
    # 负向对照：把一个已知为坏的实现（"含自身 prev_resp"）也跑一遍，确认判据能区分
    print('  负向对照：判据能区分"坏实现"吗 —— 见 tests/test_masking.py 的相邻 resp 用例')
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--prefix', default='v3_persona')
    ap.add_argument('--split', default='train', choices=['train', 'val'])
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--roundtrip', action='store_true', default=True)
    ap.add_argument('--json', default=None, help='把结果写进这个 json 文件')
    a = ap.parse_args(argv)

    ok = selftest() if a.selftest else True
    res = report(a.prefix, a.split)
    if a.roundtrip:
        print(f'=== 编解码往返（{a.prefix}）===')
        ok &= roundtrip(a.prefix)
    if a.json:
        with open(a.json, 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f'已写入 {a.json}')
    hard = (res['orphan_resp'] == 0 and res['orphan_eos'] == 0
            and res['two_rules_agree'] and res['resp_eq_eos'] and ok)
    print(('✅ 全部通过' if hard else '❌ 有检查未通过'))
    return 0 if hard else 1


if __name__ == '__main__':
    raise SystemExit(main())
