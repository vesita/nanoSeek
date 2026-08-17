"""
生成自蒸馏数据（Expert Iteration 轻量版，方向 1）。

动机：交叉熵单步近视目标让模型把"押高频 token"当最优解 → 三连崩坏
（Sinks 训满死循环 / Muon 英文乱码 / 记忆 token 文字拼贴）全是"有稳定统计
但零信息"的捷径。本脚本把"序列质量"这个 CE 看不见的判据转成点对点 CE：
用当前最优不崩坏的 af checkpoint 高温度采样一批对话续写 → 自动过滤 +
人工抽查 → 把"像样的好样本"编码成 data/chinese/distill.bin，供 train.py
以"预算中性混合"方式（p_distill 概率切块）混入训练，让模型学会复现好样本。

流程：
1. 加载 out/chinese-data2-af/best.pt（复用 sample_py.build_model_from_checkpoint）
2. 从对话语料抽真实对话前半段作 prompt（格式 = 训练数据同款"用户：…\n模型：…"）
3. 高温度（T≈1.1）批量采样续写（镜像 Rust sample()：温度→repeat-penalty→top-k→softmax→multinomial）
4. 自动过滤：长度 ≥ 40 token / token 3-gram 重复率 < 0.35 / 中文字符占比 ≥ 0.3
5. 打印过滤后样本（人工抽查是硬门槛——统计过滤器抓不住"语义不连贯"）
6. 通过的样本按对话格式编码成 distill.bin

用法（从项目根目录）：
    uv run python training/self_distill.py --n_prompts 2000
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from inference.scripts.sample_py import build_model_from_checkpoint

# 对话语料（与 prepare.py 的 DIALOGUE_FILES 一致），只从中抽 prompt
DIALOGUE_FILES = ['dailychat_dialogue.txt', 'muice_dialogue.txt', 'multi_turn_dialogue.txt']
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'chinese')
TOKENIZER_PATH = os.path.join(DATA_DIR, 'tokenizer.json')
OUT_BIN = os.path.join(DATA_DIR, 'distill.bin')
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'out', 'chinese-data2-af')

PAD_ID = 0          # BPE 的 id 0 作批量 padding（右对齐，前向时在左边被截掉）
MAX_PROMPT_TOKENS = 96   # prompt 截到 ≤96 token，保证 prompt+续写 ≤ block_size 不截断


def load_prompts(n_prompts, seed):
    """从对话语料抽真实对话前半段作 prompt，每条以 '…模型：' 结尾让模型作答。"""
    rng = random.Random(seed)
    prompts = []
    for fn in DIALOGUE_FILES:
        path = os.path.join(DATA_DIR, fn)
        if not os.path.exists(path):
            continue
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        for block in text.split('\n\n'):
            block = block.strip()
            if not block:
                continue
            lines = block.split('\n')
            # 所有"用户：提问"行的位置；随机选一个非首行的，取它前面的完整轮次 + 该提问
            user_idx = [i for i, l in enumerate(lines) if l.startswith('用户：')]
            candidates = [ui for ui in user_idx if ui > 0]
            if not candidates:
                continue
            ui = rng.choice(candidates)
            prompts.append('\n'.join(lines[:ui + 1]) + '\n模型：')
            if len(prompts) >= n_prompts:
                return prompts
    return prompts


@torch.no_grad()
def generate_batch(model, tok, prompts, max_new_tokens, temperature, top_k,
                   repeat_penalty, batch_size, device):
    """批量生成续写。镜像 Rust sample() 逻辑；批内右对齐 padding 减少污染。
    返回与 prompts 等长的列表 [(prompt, continuation_ids, continuation_text)]。
    """
    # 编码并按长度排序 → 同批长度相近，padding 少
    items = sorted(((p, tok.encode(p).ids) for p in prompts), key=lambda x: len(x[1]))
    # 每个样本截到 MAX_PROMPT_TOKENS（保 prompt+续写 ≤ block_size）
    items = [(p, ids[-MAX_PROMPT_TOKENS:]) for p, ids in items]

    results = [None] * len(items)
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        B = len(batch)
        max_len = max(len(ids) for _, ids in batch)
        cur = torch.full((B, max_len), PAD_ID, dtype=torch.long, device=device)
        seen = []                       # 每样本已见 token（repeat-penalty 用，不含 pad）
        for i, (_, ids) in enumerate(batch):
            cur[i, max_len - len(ids):] = torch.tensor(ids, dtype=torch.long)
            seen.append(ids.copy())
        for _ in range(max_new_tokens):
            logits, _ = model(cur[:, -model.config.block_size:])
            v = logits[:, -1, :] / temperature
            if repeat_penalty > 1.0:
                for i in range(B):
                    for t in set(seen[i]):
                        l = v[i, t]
                        v[i, t] = l / repeat_penalty if l >= 0 else l * repeat_penalty
            if top_k is not None:
                k = min(top_k, v.size(-1))
                topv, _ = torch.topk(v, k, dim=-1)
                v[v < topv[:, [-1]]] = float('-inf')
            probs = F.softmax(v, dim=-1)
            nxt = torch.multinomial(probs, 1)                 # (B, 1)
            for i in range(B):
                seen[i].append(int(nxt[i, 0].item()))
            cur = torch.cat([cur, nxt], dim=1)
        for i, (p, p_ids) in enumerate(batch):
            # seen[i] = 截断后的 prompt ids + 生成 ids，去掉 prompt 即续写
            cont_ids = seen[i][len(p_ids):]
            cont_text = tok.decode(cont_ids)
            results[start + i] = (p, cont_ids, cont_text)
    return results


def filter_sample(cont_ids, cont_text):
    """3 条硬门槛：长度 / token 3-gram 重复率 / 中文字符占比。"""
    n = len(cont_ids)
    if n < 40:
        return False
    # token 3-gram 唯一率：死循环/复读时几乎全重复，唯一率极低
    grams = {tuple(cont_ids[i:i + 3]) for i in range(n - 2)}
    if len(grams) / max(1, n - 2) < 0.35:
        return False
    # 中文字符占比：中文对话模型生成英文/符号堆 = 乱码
    if not cont_text:
        return False
    han = sum(1 for c in cont_text if '一' <= c <= '鿿')
    if han / len(cont_text) < 0.3:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_prompts', type=int, default=2000, help='采样候选条数')
    ap.add_argument('--max-new-tokens', type=int, default=160)
    ap.add_argument('--temperature', type=float, default=1.1)
    ap.add_argument('--top-k', type=int, default=200)
    ap.add_argument('--repeat-penalty', type=float, default=1.2)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--print-n', type=int, default=30, help='打印抽查条数')
    ap.add_argument('--no-write', action='store_true', help='只抽查不写 distill.bin')
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, ckpt = build_model_from_checkpoint(CKPT_DIR)
    model.to(device).eval()
    tok = Tokenizer.from_file(TOKENIZER_PATH)
    print(f'[af checkpoint] {sum(p.numel() for p in model.parameters()):,} 参数 → {device}')

    prompts = load_prompts(a.n_prompts, a.seed)
    print(f'抽到 {len(prompts)} 条 prompt，开始高温度采样（T={a.temperature}）…')
    results = generate_batch(model, tok, prompts, a.max_new_tokens, a.temperature,
                             a.top_k, a.repeat_penalty, a.batch_size, device)

    passed = [(p, c_ids, c_txt) for p, c_ids, c_txt in results
              if filter_sample(c_ids, c_txt)]
    print(f'过滤后保留 {len(passed)}/{len(results)} 条（{len(passed)/max(1,len(results)):.0%}）')

    # 人工抽查（硬门槛）：打印通过样本，剔除明显崩坏
    print(f'\n===== 抽查前 {min(a.print_n, len(passed))} 条通过样本 =====')
    for i, (p, c_ids, c_txt) in enumerate(passed[:a.print_n]):
        first_lines = p.split('\n')[-2:]          # 显示 prompt 尾部（最近的提问）
        print(f'\n--- 样本 {i} | 续写 {len(c_ids)} token ---')
        print('  prompt 尾: ' + ' | '.join(first_lines))
        print('  续写: ' + c_txt.replace('\n', ' / ')[:160])

    if a.no_write:
        print('\n（--no-write，未写 distill.bin）')
        return

    if not passed:
        print('没有样本通过过滤——检查温度/过滤器，或说明 EI 前置条件不成立。')
        return

    # 通过的样本按对话格式拼接（prompt + 续写，样本间空行分隔），编码成 distill.bin
    samples = [p + c_txt for p, _, c_txt in passed]
    text = '\n\n'.join(samples)
    ids = tok.encode(text).ids
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUT_BIN, 'wb') as f:
        np.array(ids, dtype=np.uint16).tofile(f)
    print(f'\n写出 {OUT_BIN}：{len(passed)} 条 / {len(ids):,} token / {os.path.getsize(OUT_BIN):,} B')


if __name__ == '__main__':
    main()
