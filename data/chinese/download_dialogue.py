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


def extract_muice(ex):
    """Moemuu/Muice-Dataset：{system, conversation=[{human,assistant}]}，中文多轮闲聊。

    多轮对话流是当前数据最缺的（现有全是单轮指令），这是补"对话感"的核心。
    system 是角色设定（如"你是一个名为沐雪的可爱AI女孩子"），训练时丢弃，
    只保留 human/assistant 交替的对话流。
    """
    conv = ex.get('conversation', [])
    if not conv:
        return None
    turns = []
    for m in conv:
        if not isinstance(m, dict):
            continue
        h = str(m.get('human', '')).strip()
        a = str(m.get('assistant', '')).strip()
        if h:
            turns.append(f'用户：{h}')
        if a:
            turns.append(f'模型：{a}')
    return '\n'.join(turns) if turns else None


def extract_dailychat(ex):
    """yyy6778/dailychat：{instruction, input, output}，中文单轮日常对话（口语化）。"""
    q = str(ex.get('instruction', '')).strip()
    inp = str(ex.get('input', '')).strip()
    a = str(ex.get('output', '')).strip()
    full_q = f'{q}\n{inp}'.strip() if inp else q
    if not full_q or not a:
        return None
    return f'用户：{full_q}\n模型：{a}'


def extract_multi_turn(ex):
    """justgo10000/Multi-turn-dialogue：{prompt=[{role,content}...], chosen, rejected}。

    多轮对话流（心理/情感咨询，当前数据最缺的"你来我往"）。
    prompt 是对话历史，但**永远以最后一条用户消息结尾**（DPO 格式），
    chosen/rejected 才是候选回复——所以必须把 chosen 补成结尾的 模型： 回复，
    否则每个样本都以未回答的 用户： 收尾，等于教模型"下一条该输出 用户："。
    """
    prompt = ex.get('prompt') or ex.get('messages')
    if not prompt:
        return None
    turns = []
    for m in prompt:
        if not isinstance(m, dict):
            continue
        role = str(m.get('role', ''))
        content = str(m.get('content', '')).strip()
        if not content:
            continue
        if role == 'user':
            turns.append(f'用户：{content}')
        elif role == 'assistant':
            turns.append(f'模型：{content}')
    chosen = str(ex.get('chosen', '')).strip()
    if chosen:
        turns.append(f'模型：{chosen}')
    return '\n'.join(turns) if turns else None


def extract_zhihu_kol(ex):
    """OmniData/Zhihu-KOL：知乎问答（INSTRUCTION/RESPONSE，单轮闲聊/生活问答）。

    2026-08-19 新增（数据扩充）：真实知乎口语语料，打破话术性格（情感支持类占
    主导的现状）。过滤超长问答（论文级）与 AI 腔开头。
    """
    q = str(ex.get('INSTRUCTION', '')).strip()
    a = str(ex.get('RESPONSE', '')).strip()
    if not q or not a:
        return None
    if len(q) > 300 or len(a) > 600:
        return None
    if a.startswith(('作为', '作为一个', '我是', '首先，')):
        return None
    return f'用户：{q}\n模型：{a}'


EXTRACTORS = {
    'shareai': ('shareAI/shareAI-Llama3-DPO-zh-en-emoji', extract_shareai),
    'zhuangxialie': ('zhuangxialie/Llama3-Chinese-Dataset', extract_zhuangxialie),
    'muice': ('Moemuu/Muice-Dataset', extract_muice),
    'dailychat': ('yyy6778/dailychat', extract_dailychat),
    'multi_turn': ('justgo10000/Multi-turn-dialogue', extract_multi_turn),
    'zhihu_kol': ('OmniData/Zhihu-KOL', extract_zhihu_kol),
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
    bad = 0
    with open(tmp_path, 'w', encoding='utf-8') as f:
        try:
            for i, ex in enumerate(ds):
                if max_samples and i >= max_samples:
                    break
                try:
                    text = extractor(ex)
                except Exception:
                    # 个别坏行（schema 不一致等）跳过，不影响整批下载
                    bad += 1
                    continue
                if text:
                    f.write(text + '\n\n')
                    written += 1
                if written % 5000 == 0:
                    print(f'  已写 {written} 轮对话...')
        except Exception as e:
            # 数据集尾部常混入 schema 不一致的批，流式迭代器会整体抛 CastError。
            # 已写出的内容完整有效，保留前缀提前收尾即可，不算失败。
            print(f'  ⚠ 迭代中断（{type(e).__name__}），保留已写的 {written} 条')
    if bad:
        print(f'  跳过坏行 {bad} 条')
    # 原子改名：即使迭代中断，已写的也是完整样本序列（每条一个 write），可安全使用
    os.replace(tmp_path, out_path)
    print(f'完成 ✅ {out_path}  （{written} 轮对话，{os.path.getsize(out_path) / 1e6:.1f} MB）')


def main():
    ap = argparse.ArgumentParser(description='下载魔搭中文对话语料')
    ap.add_argument('--datasets', default='shareai,zhuangxialie',
                    help='逗号分隔的数据集名，可选 shareai / zhuangxialie / muice / dailychat / multi_turn / zhihu_kol')
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
