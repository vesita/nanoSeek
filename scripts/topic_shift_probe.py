"""话题切换点探针 —— 量「多轮语料里有多少个『新话题从这里开始』的点」。

> ★★★ **实测结论（2026-09-17）：这个方向在本语料上走不通，脚本保留作证据。**
> 1. **信号 A（字符 bigram Jaccard）判死**：全量 508,624 个候选点上，
>    中位数 **0.009**、30% 恰好为 0、52.5% < 0.01、97.2% < 0.10。
>    ⇒ **任何 ≤0.10 的阈值都会把 50%~97% 的轮判成"换话题"** —— 这不是判据，是恒真。
>    根因：这个语料相邻轮**本来就词面独立**（与"Δ 8-gram 承接 = 0.0000"一致）。
>    自身拷问：`--selftest` 里那条**真承接**样本也只有 0.051，我当时就该看出问题。
> 2. **★ 更根本：目标能力在语料里几乎不存在**。模型轮 798,828 个里，
>    开头带开话题线索的只有 **3,189（0.399%）**，而且**读原文大多是答内连接词**
>    （`…缓解压力，另外定期规律的作息…` = "furthermore"，不是换话题）；
>    「助手连续两段（自己开轮）」只有 **1,489**（<0.2%）。
>    ⇒ **`<topic>` 是"缺数据"，不是"缺标注"**（我 2026-09-17 早些时候写的"只差标注"**已作废**）。
> 3. 注入本身**确实零成本**（`<topic>` 直接写进 `.txt` 就编成单个 id 141，见 `dev-notes/85` §4.1），
>    所以本脚本的**否定结论**只否掉"从现有语料里检测"，**不否掉**将来"构造这类数据"的路线。

## 为什么曾需要它

`training/dialogue_stream.py` **早就有插 `<topic>` 的机制**（`new_topic=True`），
但**从来没人决定过"在哪插"** —— 于是 `v3_dlg` 的 bin 里 `<topic>`(141) 数量为 **0**，
「主动换话题 / 自开轮次」这一维**零监督信号**（`dev-notes/85` §4）。

## 两个**互相独立**的信号（不要取平均，见 AGENTS §5.11）

- **信号 A（词面重叠骤降）**：本轮 `用户：` 与**它之前的所有上下文**的
  字符 bigram Jaccard 相似度低 ⇒ 疑似换话题。
  ★ 已知缺陷：`用户：继续` 这种**承接**也不共享 bigram ⇒ A 会把它误判成换话题。
  这正是为什么需要信号 B 与**读原文**。
- **信号 B（显式线索）**：本轮 `用户：` 里有"换个话题/另外/顺便/还有一个问题"这类**显式**标记。

两个信号给出不同数时**不许取平均**，要写出为什么不同。

## 纪律

- `--selftest`：**已知答案对照**（§5.4）。一对"明显承接"与一对"明显换话题"必须被判开。
  判据分不开好/坏 ⇒ 先修判据，别拿它量语料。
- `--dump`：把落在阈值附近的样本**原文**打出来（§5.9：关键词判据只能粗筛，判读必须读原文）。
- **只读**：不写任何语料、不碰 `.bin`。

用法：
    .venv/bin/python scripts/topic_shift_probe.py --selftest
    .venv/bin/python scripts/topic_shift_probe.py --dump 12
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, Iterable, List, Tuple

SOURCES_DIR = 'data/chinese/stages/v3_dlg'

# 信号 B：显式话题切换线索（用户侧）
CUES = [
    '换个话题', '换一个话题', '说点别的', '聊点别的', '另外', '顺便',
    '还有一个问题', '再问一个', '问个别', '下一个问题', '接下来',
    '新问题', '对了', '话说回来', '不聊这个', '不说这个',
]

USER_PREFIX = '用户：'
BOT_PREFIX = '模型：'


def bigrams(s: str) -> set:
    s = re.sub(r'\s+', '', s)
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) >= 2 else set()


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def split_blocks(text: str) -> List[str]:
    return [b for b in text.split('\n\n') if b.strip()]


def turns(block: str) -> List[Tuple[str, str]]:
    """-> [(前缀, 正文)]，保留顺序。"""
    out = []
    for ln in block.split('\n'):
        if ln.startswith(USER_PREFIX):
            out.append((USER_PREFIX, ln[len(USER_PREFIX):]))
        elif ln.startswith(BOT_PREFIX):
            out.append((BOT_PREFIX, ln[len(BOT_PREFIX):]))
    return out


def candidates(block: str) -> Iterable[Tuple[int, str, float, List[str]]]:
    """产出 (user 轮序号从1算, 本轮正文, 与**此前全部上下文**的 bigram Jaccard, 命中的显式线索)。

    ★ 上下文 = 本轮**之前**的所有轮（用户+模型都算）—— 这才是"这个话题聊到哪了"。
    """
    ts = turns(block)
    ctx: set = set()
    ui = 0
    for pref, body in ts:
        if pref == USER_PREFIX:
            ui += 1
            if ui >= 2:                      # 第 1 轮没有"上下文"，没有切换可言
                hit = [c for c in CUES if c in body]
                yield ui, body, jaccard(ctx, bigrams(body)), hit
            ctx |= bigrams(body)
        else:
            ctx |= bigrams(body)


def scan(dirpath: str, limit: int | None = None) -> Dict[str, object]:
    stats = {'blocks': 0, 'multiturn': 0, 'cands': 0, 'cue_only': 0,
             'low_only': 0, 'both': 0, 'neither': 0}
    per_source: Dict[str, List[int]] = {}
    samples: List[Tuple[float, str, str, List[str]]] = []
    for name in sorted(os.listdir(dirpath)):
        if not name.endswith('.txt'):
            continue
        path = os.path.join(dirpath, name)
        n_c = 0
        buf: List[str] = []
        with open(path, encoding='utf-8', errors='replace') as f:
            for line in f:
                if line.strip():
                    buf.append(line)
                    continue
                blk = ''.join(buf)
                buf = []
                stats['blocks'] += 1
                ts = turns(blk)
                nu = sum(1 for p, _ in ts if p == USER_PREFIX)
                if nu >= 2:
                    stats['multiturn'] += 1
                    for _ui, body, j, hit in candidates(blk):
                        stats['cands'] += 1
                        n_c += 1
                        if hit and j < 0.10:
                            stats['both'] += 1
                        elif hit:
                            stats['cue_only'] += 1
                        elif j < 0.10:
                            stats['low_only'] += 1
                        else:
                            stats['neither'] += 1
                        samples.append((j, name, body[:110].replace('\n', ' '), hit))
                if limit and stats['blocks'] >= limit:
                    break
        per_source[name] = [n_c]
    return {'stats': stats, 'per_source': per_source, 'samples': samples}


# ------------------------------------------------------------------ 已知答案对照

_CONT_OK_CTX = '我们刚才在聊北京有哪些值得去的公园，颐和园和天坛都挺不错。'
_CONT_OK_TURN = '对，那颐和园哪个季节去最好？'
_CONT_SW_CTX = _CONT_OK_CTX
_CONT_SW_TURN = '换个话题，帮我写一段 Python 快排。'


def selftest() -> int:
    """★ 已知答案对照：承接的低、换话题的高（且被显式线索命中）。"""
    j_cont = jaccard(bigrams(_CONT_OK_CTX), bigrams(_CONT_OK_TURN))
    j_sw = jaccard(bigrams(_CONT_SW_CTX), bigrams(_CONT_SW_TURN))
    cue_sw = [c for c in CUES if c in _CONT_SW_TURN]
    print(f'  承接样本   Jaccard = {j_cont:.3f}   （应偏高）')
    print(f'  换话题样本 Jaccard = {j_sw:.3f}   （应偏低）  显式线索={cue_sw}')
    ok = True
    if not j_cont > j_sw:
        print('  ✗ 判据分不开承接/换话题 —— 先修判据'); ok = False
    if not cue_sw:
        print('  ✗ 显式线索没命中「换个话题」'); ok = False
    # 已知的反例：承接但不共享 bigram（"继续"）—— 必须承认它会掉进"低重叠"桶
    j_next = jaccard(bigrams(_CONT_OK_CTX), bigrams('继续'))
    print(f'  ★ 已知缺陷对照：用户只说「继续」（真承接，但不共享 bigram）Jaccard = {j_next:.3f}')
    if j_next >= 0.10:
        print('  ✗ 预期「继续」落在低重叠桶，判据行为变了 —— 请复核'); ok = False
    print('===== 已知答案对照：%s =====' % ('通过' if ok else '失败'))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--sources-dir', default=SOURCES_DIR)
    ap.add_argument('--limit', type=int, default=None, help='只扫前 N 个块（调试用）')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--dump', type=int, default=0, help='在阈值附近 dump 多少条原文')
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()

    r = scan(a.sources_dir, a.limit)
    s = r['stats']
    print(f"=== 信号 A（bigram Jaccard < 0.10 = 疑似换话题）与 B（显式线索）===")
    print(f"  块 {s['blocks']}  其中多轮块 {s['multiturn']}  候选切换点（第2轮起的用户轮）{s['cands']}")
    for k, label in (('both', 'A∩B 双信号都说是'), ('cue_only', '仅 B（显式线索）'),
                     ('low_only', '仅 A（词面重叠骤降）'), ('neither', '都不是')):
        n = s[k]
        print(f"    {label:22s} {n:7d}  ({100*n/max(s['cands'],1):5.1f}%)")
    print('\n  逐源候选点：')
    for name, (n,) in r['per_source'].items():
        print(f'    {name:28s} {n:7d}')

    if a.dump:
        print(f"\n=== 阈值附近原文（Jaccard 最低的 {a.dump} 条）===")
        for j, name, body, hit in sorted(r['samples'], key=lambda x: x[0])[:a.dump]:
            print(f'  [{j:.3f}] {name}  线索={hit}\n      {body}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
