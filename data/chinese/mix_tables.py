"""生成三张来源占比表（供 DATASET_REPORT.md 使用）：

  A. 修前来源表：audit_prefix.json（双标准切分）+ 修正 c4_zh 的 train 侧 0.5 降采样
  B. 修后来源表：manifest_v2.json 的 source_breakdown
  C. 并排表：旧 val 占比 / 新 val 占比 / 新 train 占比

用法：
    .venv/bin/python data/chinese/mix_tables.py
"""
import json
import os
import random
import sys

DATA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DATA)
import audit_split  # noqa: E402

DIALOGUE = audit_split.PREFIX_DIALOGUE_FILES


def correct_c4(ratio, val_ratio=0.01):
    """精确重算 c4_zh.txt 在 (双标准 val + train 侧 ratio 降采样) 下的 train/val 字符数。"""
    path = os.path.join(DATA, 'c4_zh.txt')
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    blocks = [b.strip() for b in text.split('\n\n') if b.strip()]
    tr, va = audit_split.split_prefix('c4_zh.txt', blocks, val_ratio,
                                      src_ratio=('c4_zh', ratio))
    return sum(len(b) for b in tr), sum(len(b) for b in va)


def md_table(rows, headers):
    out = ['| ' + ' | '.join(headers) + ' |',
           '|' + '|'.join(['---'] + ['---:'] * (len(headers) - 1)) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(str(x) for x in r) + ' |')
    return '\n'.join(out)


def main():
    old = json.load(open(os.path.join(DATA, 'audit_prefix.json')))
    c4_tr, _ = correct_c4(0.5)
    old_rows = []
    for r in old['rows']:
        tc = c4_tr if r['file'] == 'c4_zh.txt' else r['train_chars']
        old_rows.append({'file': r['file'], 'size_mb': r['size_mb'], 'blocks': r['blocks'],
                         'train_chars': tc, 'val_chars': r['val_chars']})
    tot_t = sum(r['train_chars'] for r in old_rows)
    tot_v = sum(r['val_chars'] for r in old_rows)
    for r in old_rows:
        r['train_share'] = r['train_chars'] / tot_t
        r['val_share'] = r['val_chars'] / tot_v

    mani = json.load(open(os.path.join(DATA, 'manifest_v2.json')))
    new_rows = mani['source_breakdown']
    new_tot_t = sum(r['train_chars_raw'] for r in new_rows)
    new_tot_v = sum(r['val_chars_raw'] for r in new_rows)

    print('=' * 100)
    print(f'A. 修前（双标准 val 1%/10% + c4_zh train 侧 ×0.5）：train {tot_t:,} 字符 / val {tot_v:,} 字符')
    print(md_table(
        [[r['file'], r['size_mb'], f"{r['blocks']:,}", f"{r['train_chars']:,}",
          f"{r['train_share']*100:.2f}%", f"{r['val_chars']:,}", f"{r['val_share']*100:.2f}%"]
         for r in sorted(old_rows, key=lambda x: -x['train_chars'])],
        ['来源', '大小MB', 'blocks', 'train字符', 'train占比', 'val字符', 'val占比']))

    print()
    print('=' * 100)
    print(f'B. 修后 v2（统一 val 1%）：train {new_tot_t:,} 字符 / val {new_tot_v:,} 字符')
    print(md_table(
        [[r['file'], f"{r['size_bytes']/1048576:.1f}", f"{r['blocks']:,}",
          f"{r['train_chars_raw']:,}", f"{r['train_share']*100:.2f}%",
          f"{r['val_chars_raw']:,}", f"{r['val_share']*100:.2f}%"]
         for r in sorted(new_rows, key=lambda x: -x['train_chars_raw'])],
        ['来源', '大小MB', 'blocks', 'train字符', 'train占比', 'val字符', 'val占比']))

    print()
    print('=' * 100)
    print('C. 并排：旧 val 占比 / 新 val 占比 / 新 train 占比（按新 train 占比降序）')
    oldv = {r['file']: r['val_share'] for r in old_rows}
    newv = {r['file']: r['val_share'] for r in new_rows}
    newt = {r['file']: r['train_share'] for r in new_rows}
    files = sorted(newt, key=lambda f: -newt[f])
    rows = []
    for f in files:
        rows.append([f, f"{oldv.get(f,0)*100:.2f}%", f"{newv.get(f,0)*100:.2f}%",
                     f"{newt.get(f,0)*100:.2f}%"])
    print(md_table(rows, ['来源', '旧val占比', '新val占比', '新train占比']))
    print()
    print('占比 >15% 的源（按新 train）：')
    for f in files:
        if newt[f] > 0.15:
            print(f"  {f}: train {newt[f]*100:.2f}%  (旧val {oldv.get(f,0)*100:.2f}%, 新val {newv.get(f,0)*100:.2f}%)")


if __name__ == '__main__':
    main()
