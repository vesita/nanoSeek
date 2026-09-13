#!/usr/bin/env python
"""`out/` 全树清理：只删**可重建的派生物**（`.pt` 权重 / `.npz` 缓存表），
把全部日志/JSON/CSV/PNG 证据留下。

## 为什么用"只删派生物"而不是"整目录删"

2026-09-11 实测 `out/` 体积构成：
    .pt   212 个  74.06 GB  (94.2%)
    .npz    3 个   4.50 GB  ( 5.7%)
    其余全部（.log/.json/.csv/.png/.md）      35 MB  (0.04%)
⇒ 证据只占 0.04%。整目录删会连证据一起丢掉，而删派生物能**释放 98% 的空间、丢掉 0 条结论**。

## 保护名单（无论如何都不碰）

* `out/base_v2`      —— 当前主 run，训练正在写它
* `out/nanoseek_100m` —— v1 基座，`PROJECT_STATE §0.5.6` 的 v1/v2 配对比较要用它的 `last.pt`

## 用法

    .venv/bin/python scripts/cleanup_out.py              # dry-run，只打印计划
    .venv/bin/python scripts/cleanup_out.py --apply      # 真正删除，并写清单
    .venv/bin/python scripts/cleanup_out.py --apply --keep-representative
        # 额外为每个实验家族保留一个最新 .pt（约多占 10GB）

无论是否 `--apply`，都会先打印逐个目录的明细。
`--apply` 时会先生成 `out/CLEANUP_MANIFEST.md`（记录删了什么、释放多少），再动手。
"""
from __future__ import annotations

import argparse
import os
import shutil
import time

OUT = 'out'
# ★ 保护集：这些目录里的 .pt/.npz **绝不删**（可能正在训练 / 是回溯点 / 将来补跑）。
#   2026-09-13 加入 v3 阶段目录 —— 当时正要起 `out/base_v3_dlg`，而它不在保护集里，
#   一旦有人跑 `cleanup_out.py --apply`，正在训练的产物会被当临时目录清掉。
PROTECTED_DIRS = {'base_v2', 'nanoseek_100m', 'base_v3_dlg', 'base_v3_know'}
# 本次会话自己产生的扫描残留：整目录删（里面只有 tqdm 日志和结果 CSV，没有结论）
SESSION_JUNK_PREFIXES = ('_smoke', '_foc_', '_bench_', '_knob_')
# 参与"每家族留一个代表性 .pt"的家族前缀（按最长匹配优先）
FAMILY_PREFIXES = [
    'nanoseek_100m_neural_db', 'nanoseek_100m_ndb', 'nanoseek_100m_clean',
    'nanoseek_100m_db', 'nanoseek_100m_speedtest', 'nanoseek_100m',
    'neuron_db', 'residual_db', 'ndb_', 'db_', 'ngram_full', 'mem_store',
    'curriculum', 'pilot_', 'rl_', 'ab_char_', 'char_fact', 'eos_fix',
    'nano_arith', 'arith_sft', 'smoke', 'test_', 'cont_v1', 'base_probe', 'bench',
]


def human(n: float) -> str:
    return f'{n / 2**30:.2f} GB' if n >= 2**30 else f'{n / 2**20:.1f} MB'


def family_of(name: str) -> str:
    for p in FAMILY_PREFIXES:
        if name.startswith(p):
            return p.rstrip('_')
    return name


def scan():
    """返回 (逐目录明细, 汇总)。明细里每项记录将删除的文件及体积。"""
    detail, total = [], {'pt': 0, 'npz': 0, 'junk': 0, 'evidence': 0, 'kept_pt': 0}
    for d in sorted(os.listdir(OUT)):
        dp = os.path.join(OUT, d)
        if not os.path.isdir(dp):
            continue
        if d.startswith(SESSION_JUNK_PREFIXES):
            s = sum(os.path.getsize(os.path.join(r, f))
                    for r, _, fs in os.walk(dp) for f in fs)
            detail.append({'dir': d, 'kind': '会话残留目录（整目录删）', 'files': [], 'bytes': s})
            total['junk'] += s
            continue

        del_files, del_bytes, kept_pt_bytes, ev_bytes = [], 0, 0, 0
        for r, _, fs in os.walk(dp):
            for f in fs:
                p = os.path.join(r, f)
                e = os.path.splitext(f)[1].lower()
                try:
                    s = os.path.getsize(p)
                except OSError:
                    continue
                if e == '.pt':
                    if d in PROTECTED_DIRS:
                        kept_pt_bytes += s
                        total['kept_pt'] += s
                    else:
                        del_files.append((os.path.relpath(p, OUT), s))
                        del_bytes += s
                        total['pt'] += s
                elif e == '.npz':
                    del_files.append((os.path.relpath(p, OUT), s))
                    del_bytes += s
                    total['npz'] += s
                else:
                    ev_bytes += s
                    total['evidence'] += s
        if del_files:
            detail.append({'dir': d, 'kind': '删派生物，留证据', 'files': del_files,
                           'bytes': del_bytes, 'ev': ev_bytes, 'kept_pt': kept_pt_bytes})
    return detail, total


def apply_representative(detail):
    """为每个家族保留一个最新的 .pt（从待删列表里撤回）。"""
    best: dict[str, tuple[float, str]] = {}
    for item in detail:
        if item['kind'] != '删派生物，留证据':
            continue
        for rel, s in item['files']:
            if not rel.endswith('.pt'):
                continue
            fam = family_of(os.path.basename(os.path.dirname(os.path.relpath(rel, OUT))) or item['dir'])
            m = os.path.getmtime(os.path.join(OUT, rel))
            if fam not in best or m > best[fam][0]:
                best[fam] = (m, rel)
    keep = {rel for _, rel in best.values()}
    freed = 0
    for item in detail:
        if item['kind'] != '删派生物，留证据':
            continue
        new = [(rel, s) for rel, s in item['files'] if rel not in keep]
        freed += sum(s for rel, s in item['files'] if rel in keep)
        item['files'] = new
        item['kept_repr'] = sorted(rel for rel, _ in item['files'] if rel in keep) + \
                            sorted(rel for rel in keep if rel.startswith(item['dir'] + os.sep))
    return freed, sorted(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true', help='真正删除（默认只 dry-run）')
    ap.add_argument('--keep-representative', action='store_true',
                    help='每个实验家族额外保留一个最新 .pt')
    args = ap.parse_args()

    detail, total = scan()
    repr_kept, repr_freed = [], 0
    if args.keep_representative:
        repr_freed, repr_kept = apply_representative(detail)

    print(f"{'目录':<44}{'动作':<22}{'释放':>10}{'留存证据':>10}")
    print('-' * 88)
    for item in detail:
        ev = human(item.get('ev', 0)) if item.get('ev') else '-'
        print(f"{item['dir']:<44}{item['kind']:<22}{human(item['bytes']):>10}{ev:>10}")
    print('-' * 88)
    grand = total['pt'] + total['npz'] + total['junk'] - repr_freed
    print(f"  .pt 删除      {human(total['pt'] - repr_freed):>10}   （{len(PROTECTED_DIRS)} 个保护目录的 .pt "
          f"{human(total['kept_pt'])} 未动）")
    print(f"  .npz 缓存表   {human(total['npz']):>10}")
    print(f"  会话残留目录  {human(total['junk']):>10}")
    if args.keep_representative:
        print(f"  代表性 .pt 保留 {len(repr_kept)} 个（另占 {human(repr_freed)}）：")
        for r in repr_kept:
            print(f"      {r}")
    print(f"  ★ 合计释放    {human(grand):>10}")
    print(f"  ★ 保留证据    {human(total['evidence']):>10}  （全部 .log/.json/.csv/.png，一字不丢）")

    if not args.apply:
        print('\n（dry-run。加 --apply 才真正删除）')
        return 0

    manifest = os.path.join(OUT, 'CLEANUP_MANIFEST.md')
    with open(manifest, 'w', encoding='utf-8') as f:
        f.write('# out/ 清理清单\n\n')
        f.write(f'生成时间：{time.strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        f.write('**只删可重建的派生物**（`.pt` 权重、`.npz` 缓存表），'
                '全部 `.log`/`.json`/`.csv`/`.png` 证据保留。\n\n')
        f.write(f'- 保护目录（.pt 未动）：{", ".join(sorted(PROTECTED_DIRS))}\n')
        f.write(f'- 合计释放：{human(grand)}\n')
        f.write(f'- 保留证据：{human(total["evidence"])}\n\n')
        f.write('| 目录 | 释放 | 留存证据 | 删除的文件 |\n|---|---:|---:|---|\n')
        for item in detail:
            names = ', '.join(rel for rel, _ in item['files']) or '（整目录）'
            if len(names) > 900:
                names = names[:900] + ' …'
            f.write(f"| `{item['dir']}` | {human(item['bytes'])} | "
                    f"{human(item.get('ev', 0)) if item.get('ev') else '-'} | {names} |\n")
    print(f'\n清单已写入 {manifest}')

    removed = 0
    for item in detail:
        if item['kind'] == '会话残留目录（整目录删）':
            shutil.rmtree(os.path.join(OUT, item['dir']), ignore_errors=True)
            removed += 1
        else:
            for rel, _ in item['files']:
                try:
                    os.remove(os.path.join(OUT, rel))
                    removed += 1
                except OSError as e:
                    print(f'  ⚠ 删除失败 {rel}: {e}')
    print(f'✅ 已删除 {removed} 个文件/目录')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
