#!/usr/bin/env python3
"""从魔搭（ModelScope）下载中文对话语料，清洗成统一的对话文本。

输出：data/chinese/<数据集名>_dialogue.txt，每轮对话格式：
    用户：<问题>
    模型：<回答>
    （对话之间空一行）

这些 txt 会被 train_tokenizer.py 收集训 BPE、被 prepare.py 编码成 train.bin。

可靠性设计（吸取之前覆盖/半截文件的教训）：
- 先写 <名>.txt.tmp，全部成功后再原子改名为正式文件——
  中断的下载不会留下半截文件污染 prepare
- 幂等：正式文件存在且够大就跳过；用 --force 强制重下

用法（在项目根目录）：
    uv run python data/chinese/download_dialogue.py                        # 默认下所有数据集
    uv run python data/chinese/download_dialogue.py --datasets zhuangxialie
    uv run python data/chinese/download_dialogue.py --max-samples 50000 --force
"""
import argparse
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
MIN_VALID_SIZE = 1_000_000  # 正式文件小于 1MB 视为无效，重新下载


def extract_shareai(ex):
    """shareAI-Llama3-DPO-zh-en-emoji：question + answer_zh（单轮 QA，中文）。"""
    q = str(ex.get('question', '')).strip()
    a = str(ex.get('answer_zh', '')).strip()
    if not q or not a:
        return None
    return f'用户：{q}\n模型：{a}'


def extract_zhuangxialie(ex):
    """zhuangxialie/Llama3-Chinese-Dataset：conversations = [{from: human/gpt, value}]（多轮，中文）。"""
    conv = ex.get('conversations', [])
    if not conv:
        return None
    turns = []
    for m in conv:
        if not isinstance(m, dict):
            continue
        role = str(m.get('from', ''))
        val = str(m.get('value', '')).strip()
        if not val:
            continue
        if role == 'human':
            turns.append(f'用户：{val}')
        elif role in ('gpt', 'assistant'):
            turns.append(f'模型：{val}')
    return '\n'.join(turns) if turns else None


EXTRACTORS = {
    'shareai': ('shareAI/shareAI-Llama3-DPO-zh-en-emoji', extract_shareai),
    'zhuangxialie': ('zhuangxialie/Llama3-Chinese-Dataset', extract_zhuangxialie),
}


def download_one(dataset_id, extractor, out_path, max_samples, force):
    """流式下载 + 清洗 + 写 txt。先写 tmp 再原子改名。"""
    if not force and os.path.exists(out_path) and os.path.getsize(out_path) >= MIN_VALID_SIZE:
        print(f'跳过（已存在 {os.path.getsize(out_path)/1e6:.1f} MB）：{out_path}')
        return

    from modelscope.msdatasets import MsDataset

    tmp_path = out_path + '.tmp'
    ds = MsDataset.load(dataset_id, split='train', use_streaming=True)
    written = 0
    with open(tmp_path, 'w', encoding='utf-8') as f:
        for i, ex in enumerate(ds):
            if max_samples and i >= max_samples:
                break
            text = extractor(ex)
            if text:
                f.write(text + '\n\n')
                written += 1
            if written % 5000 == 0:
                print(f'  已写 {written} 轮对话...')
    # 全部成功后才原子改名，避免半截文件被 prepare 误读
    os.replace(tmp_path, out_path)
    print(f'完成 ✅ {out_path}  （{written} 轮对话，{os.path.getsize(out_path) / 1e6:.1f} MB）')


def main():
    ap = argparse.ArgumentParser(description='下载魔搭中文对话语料')
    ap.add_argument('--datasets', default='shareai,zhuangxialie',
                    help='逗号分隔的数据集名，可选 shareai / zhuangxialie')
    ap.add_argument('--max-samples', type=int, default=200_000,
                    help='每个数据集最多取多少条样本（默认 20 万）')
    ap.add_argument('--force', action='store_true', help='强制重新下载（覆盖现有文件）')
    args = ap.parse_args()

    for name in [d.strip() for d in args.datasets.split(',')]:
        if name not in EXTRACTORS:
            print(f'未知数据集 {name}，跳过')
            continue
        dataset_id, extractor = EXTRACTORS[name]
        out_path = os.path.join(DATA_DIR, f'{name}_dialogue.txt')
        print(f'\n=== 下载 {dataset_id} ===')
        try:
            download_one(dataset_id, extractor, out_path, args.max_samples, args.force)
        except Exception as e:
            print(f'下载失败：{type(e).__name__}: {e}')


if __name__ == '__main__':
    main()
