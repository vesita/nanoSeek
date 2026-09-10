"""审计语料来源在 train/val 的占比（任务书 #2）。

两种模式：
  --mode=prefix  : 冻结复现"修前"的双标准切分逻辑
                   （DIALOGUE_FILES 10%，其它源 --val-ratio=0.01），
                   用于生成"修前来源占比表"。
  --mode=postfix : 调用 prepare.py 里重构后的 split_one_source()，
                   验证统一比例切分的"修后来源占比表"。
                   （以实际重建产出的 manifest_v2.json 的 source_breakdown 为准，
                     本模式用于交叉核对。）

用法：
    .venv/bin/python data/chinese/audit_split.py --mode prefix \
        --val-ratio 0.01 --out data/chinese/audit_prefix.json
"""
import argparse
import gc
import json
import os
import random

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# 修前逻辑里硬编码的对话文件清单（与历史版本 prepare.py 一致）
PREFIX_DIALOGUE_FILES = {'dailychat_dialogue.txt', 'muice_dialogue.txt',
                         'multi_turn_dialogue.txt', 'agent_dialogue.txt',
                         'zhihu_kol_dialogue.txt'}


def split_prefix(fn, blocks, val_ratio, src_ratio=None):
    """冻结的"修前"切分：对话文件 10% val，其它文件 val_ratio val（只作用于非对话）。

    src_ratio: (name_prefix, ratio) 或 None —— 修前构建实测对 c4_zh 做了 train 侧
    0.5 降采样（见 DATASET_REPORT.md「旧构建参数反推」），只在 train 生效。
    """
    n = len(blocks)
    if fn in PREFIX_DIALOGUE_FILES:
        random.seed(1337)
        random.shuffle(blocks)
        k = int(n * 0.9)
        train_blocks, val_blocks = blocks[:k], blocks[k:]
    else:
        # args.val_all 分支（manifest 记录 val_all=true, val_ratio=0.01）
        random.seed(1337 + sum(ord(c) for c in fn) + 7)
        random.shuffle(blocks)
        n_val = max(1, int(n * val_ratio))
        val_blocks, train_blocks = blocks[:n_val], blocks[n_val:]
    if src_ratio is not None and fn.startswith(src_ratio[0]) and src_ratio[1] < 1.0:
        random.seed(1337 + sum(ord(c) for c in fn) + 1)
        random.shuffle(train_blocks)
        train_blocks = train_blocks[:max(1, int(len(train_blocks) * src_ratio[1]))]
    return train_blocks, val_blocks


def split_postfix(fn, blocks, val_ratio):
    """调用 prepare.py 重构后的函数（统一比例）。"""
    import prepare
    ns = argparse.Namespace(
        val_all=True, val_ratio=val_ratio, source_ratio=[],
        task_ratio=1.0, pretrain=False)
    return prepare.split_one_source(fn, blocks, ns)


def audit(mode, val_ratio, out_path):
    rows = []
    total_train_chars = 0
    total_val_chars = 0
    for fn in sorted(os.listdir(DATA_DIR)):
        if not fn.endswith('.txt'):
            continue
        path = os.path.join(DATA_DIR, fn)
        size_bytes = os.path.getsize(path)
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        blocks = [b.strip() for b in text.split('\n\n') if b.strip()]
        del text
        n_blocks = len(blocks)
        if mode == 'prefix':
            train_blocks, val_blocks = split_prefix(fn, blocks, val_ratio)
        else:
            train_blocks, val_blocks = split_postfix(fn, blocks, val_ratio)
        tc = sum(len(b) for b in train_blocks)
        vc = sum(len(b) for b in val_blocks)
        total_train_chars += tc
        total_val_chars += vc
        rows.append({
            'file': fn, 'size_mb': round(size_bytes / 1048576, 1),
            'blocks': n_blocks, 'train_chars': tc, 'val_chars': vc,
            'train_tokens_est': tc, 'val_tokens_est': vc,
        })
        del blocks, train_blocks, val_blocks
        gc.collect()
    for r in rows:
        r['train_share'] = r['train_chars'] / total_train_chars if total_train_chars else 0
        r['val_share'] = r['val_chars'] / total_val_chars if total_val_chars else 0
    result = {'mode': mode, 'val_ratio': val_ratio,
              'total_train_chars': total_train_chars, 'total_val_chars': total_val_chars,
              'rows': rows}
    if out_path:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    # 控制台 markdown 表
    print(f"mode={mode}  val_ratio={val_ratio}  "
          f"total_train={total_train_chars:,}  total_val={total_val_chars:,}")
    print('| 来源 | 大小MB | blocks | train字符 | train占比 | val字符 | val占比 |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    for r in rows:
        print(f"| {r['file']} | {r['size_mb']} | {r['blocks']} | "
              f"{r['train_chars']:,} | {r['train_share']*100:.2f}% | "
              f"{r['val_chars']:,} | {r['val_share']*100:.2f}% |")
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['prefix', 'postfix'], default='prefix')
    ap.add_argument('--val-ratio', type=float, default=0.01)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    audit(a.mode, a.val_ratio, a.out)
