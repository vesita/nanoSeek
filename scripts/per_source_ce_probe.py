#!/usr/bin/env python
"""逐来源的 held-out CE 探针 —— 回答"基座到底会不会认字"。

为什么不能只看 `results.csv` 的 val：那是**全库混合**值，而混合权重由源的大小决定，
`c4_zh`(19.7%) + `wikipedia_cn`(19.2%) + `deepseek`(27.5%) 三家就占了 val 的 66%。
"识字（通用语言建模）"和"对话模板"是两个能力，混在一个数里读不出来。

口径（**跨 checkpoint 恒定**，不受 `use_loss_masking` 的 step-22000 断裂影响）：
  * 只在**非特殊 token** 上算 CE（屏蔽 id 属于 `<...>` 的槽位），逐 token 加权平均
  * 窗口**块内采样、不跨 block 边界**；所以它评的是源本身的文本，不是拼接缝
  * 三个层次一起报，构成"已知答案"的对照链：
        unigram  —— 只用字符频率（零上下文），从 train bin 统计
        shuffled —— 打乱窗口内字符顺序（保住相邻对 x→y，毁掉长上下文）
        real     —— 原样
    判据：real 必须**明显低于** shuffled 与 unigram，才说明模型在做真正的语言建模。
    若 real ≈ shuffled ≈ unigram ⇒ 模型只是把字符分布记住了，没在"读"。

用法：
  .venv/bin/python scripts/per_source_ce_probe.py \
      --ckpts out/base_v2/ckpt_step_20000.pt out/base_v2/last.pt \
      --windows-per-source 32
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from ckpt_paired_eval import load_model  # noqa: E402
from model import GPT  # noqa: E402

# 分组只影响汇总那一节；逐源明细永远全打印
PROSE = {'c4_zh.txt', 'wikipedia_cn.txt', 'classical_poetry.txt',
         '红楼梦.txt', '三国演义.txt', '水浒传.txt', '西游记.txt'}
KNOW = {'coig_wiki_dialogue.txt', 'zhihu_kol_dialogue.txt'}
CODE = {'code_alpaca_dialogue.txt', 'coig_code_dialogue.txt', 'coig_cqia_dialogue.txt'}
REASON = {'gsm8k_cot_dialogue.txt', 'coig_math_dialogue.txt', 'coig_logic_dialogue.txt'}


def group_of(name):
    if name in PROSE:
        return '散文/语言'
    if name in KNOW:
        return '知识/长文'
    if name in CODE:
        return '代码'
    if name in REASON:
        return '数学/推理'
    return '对话'


def special_mask(tokenizer_path, vocab_size=8192):
    """id -> bool「是不是 <...> 槽位」；用解码名判断，不猜 id 区间。"""
    from tokenizers import Tokenizer
    v = Tokenizer.from_file(tokenizer_path).get_vocab()
    m = np.zeros(vocab_size, dtype=bool)
    for tok, i in v.items():
        if tok.startswith('<') and tok.endswith('>'):
            m[i] = True
    return m


V2_ALIGN_CONTROLS = ('c4_zh.txt', 'classical_poetry.txt', 'deepseek_r1_distill_dialogue.txt')


def align_control_names(spans, order, raw):
    """挑「源对齐对照」用哪几个源，并回传要打印的说明。

    优先用 manifest_v2 的三个对照源（历史存档用了它们，口径可比）；
    换 manifest 时（v3_dlg / v3_know 的源名集不同）退化成**该 manifest 自己的前 3 个源**，
    并回一句说明 —— 免得读者以为对照源还是那三个。
    ★ 没有这条退化路径就会 KeyError 崩在对照上（2026-09-14 实测：在 v3_dlg val 上崩过一次，
      导致"B 段在自己 val 上到底怎样"这个问题整轮没答案）。
    """
    names = [n for n in V2_ALIGN_CONTROLS if n in spans and n in raw]
    if names:
        return names, ''
    fallback = [n for n in order if n in raw][:3]
    return fallback, f"本 manifest 不含 v2 对照源名，改用其前 3 个源：{fallback}"


def source_spans(manifest_path, off):
    """按 manifest 的 source_files 顺序把 val 的 block 区间切成逐源 token 区间。

    正确性对照：`sum(val_blocks) == len(off)-1`（对不上说明 bin 与 manifest 不是同一次构建）。
    """
    m = json.load(open(manifest_path))
    order = [s['file'] for s in m['source_breakdown']]
    blocks = [s['val_blocks'] for s in m['source_breakdown']]
    assert sum(blocks) == len(off) - 1, f"block 数对不上：{sum(blocks)} vs {len(off) - 1}"
    spans, b = {}, 0
    for name, nb in zip(order, blocks):
        spans[name] = (int(off[b]), int(off[b + nb])) if nb > 0 else (0, 0)
        b += nb
    return spans, order


def sample_windows(data, span, n_win, block_size, rng, off_slice):
    """块内随机采样：每个窗口完整落在一个 block 里（不跨拼接缝）。"""
    s0, s1 = span
    if s1 - s0 <= block_size + 1:
        return []
    # off_slice 是本源 block 边界（全局长度的），用来判定窗口不越 block 边界
    outs, tries = [], 0
    lo, hi = 0, len(off_slice) - 2
    while len(outs) < n_win and tries < 40 * n_win:
        tries += 1
        b = int(rng.integers(lo, hi + 1))
        a, e = int(off_slice[b]), int(off_slice[b + 1])
        if e - a <= block_size + 1:
            continue
        st = int(rng.integers(a, e - block_size - 1))
        outs.append(st)
    return outs


def unigram_probs(train_bin, spec, vocab_size, chunk=50_000_000):
    """从 train bin 统计字符频率（屏蔽特殊 token），返回 (V,) float64 概率。"""
    data = np.memmap(train_bin, dtype=np.uint16, mode='r')
    cnt = np.zeros(vocab_size, dtype=np.int64)
    for i in range(0, len(data), chunk):
        c = np.bincount(np.asarray(data[i:i + chunk], dtype=np.int64), minlength=vocab_size)
        cnt += c
    cnt[spec] = 0
    p = cnt.astype(np.float64)
    p += 1.0                      # 拉普拉斯平滑，避免 -log(0)
    return p / p.sum()


@torch.no_grad()
def score(model, starts, data, spec, uni, block_size, batch_size, device, ctx,
          shuffle=False, rng=None):
    """返回 (CE, top1, n_tok)；CE 只统计非特殊 token。"""
    tot, hit, n = 0.0, 0, 0
    uni_tot = 0.0
    # ★ `--no-unigram` 时 uni 是 None；必须真的跳过，否则 `from_numpy(None)` 会崩。
    #   （这个 flag 此前一直是坏的 —— 加"随机初始化对照"时才第一次真正跑到这条路径。）
    uni_t = torch.from_numpy(uni).float().to(device) if uni is not None else None
    # ★ 用 range(0, len, bs) 而不是 range(0, len-bs+1, bs)：后者在 len(starts) < bs 时
    # 直接空转，于是 n=0、CE=nan —— 那是个会静默产出空结果的经典坑（第一次跑就踩了）。
    for b in range(0, len(starts), batch_size):
        chunk = starts[b:b + batch_size]
        x = np.stack([np.asarray(data[s:s + block_size], dtype=np.int64) for s in chunk])
        y = np.stack([np.asarray(data[s + 1:s + 1 + block_size], dtype=np.int64) for s in chunk])
        if shuffle:
            for r in range(x.shape[0]):
                perm = rng.permutation(block_size)
                x[r] = x[r][perm]
                y[r] = y[r][perm]
        xt = torch.from_numpy(x).to(device)
        yt = torch.from_numpy(y).to(device)
        with ctx:
            logits, _ = model(xt, yt)
        V = logits.size(-1)
        ce = F.cross_entropy(logits.view(-1, V), yt.view(-1), reduction='none').view(yt.shape)
        keep = ~torch.from_numpy(spec)[yt.cpu().numpy()].to(device)
        pred = logits.argmax(-1)
        tot += float(ce[keep].sum())
        hit += int((pred[keep] == yt[keep]).sum())
        n += int(keep.sum())
        if uni_t is not None:
            uni_tot += float((-torch.log(uni_t[yt[keep]])).sum())
    if n == 0:
        return float('nan'), float('nan'), float('nan'), 0
    return tot / n, hit / n, (uni_tot / n if uni_t is not None else float('nan')), n


def eval_all_sources(model, args, data, off, spans, order, spec, uni, dev, ctx, seed,
                     dump_path=None):
    """逐来源算 (real, shuffled, unigram) CE。

    抽成函数是为了让**随机初始化对照**复用同一条代码路径 —— 对照必须和实测走同一段代码，
    否则它证明的是另一段代码没坏（`ml-experiment-attribution` §2.2）。

    `dump_path` 非空时，把**实际用到的窗口起点**逐源落盘（JSON）。
    ★ 为什么需要它：要审"这批 val 窗口是不是训练集的近重复"，
      审计脚本必须用**和探针完全同一批窗口**，靠"照抄采样逻辑"很容易差一个 rng 消耗
      （本函数的 rng 还被 shuffled 的 permutation 消耗）。落盘是唯一不会走样的做法。
    """
    rng = np.random.default_rng(seed)
    print(f"{'来源':<36}{'组':<9}{'real CE':>9}{'top1%':>8}{'shuf CE':>9}"
          f"{'unigram':>9}{'real-uni':>10}{'n_tok':>8}")
    rows = {}
    dumped = {}
    for name in order:
        s0, s1 = spans[name]
        if s1 - s0 <= args.block_size + 1:
            continue
        b0 = int(np.searchsorted(off, s0))
        starts = sample_windows(data, (s0, s1), args.windows_per_source,
                                args.block_size, rng, off[b0:])
        if not starts:
            continue
        ce, acc, uce, n = score(model, starts, data, spec, uni, args.block_size,
                                args.batch_size, dev, ctx, shuffle=False)
        if args.no_shuffle:
            sce = float('nan')
        else:
            sce, _, _, _ = score(model, starts, data, spec, uni, args.block_size,
                                 args.batch_size, dev, ctx, shuffle=True, rng=rng)
        # ★ 空结果必须炸出来，不能静默变成 nan（「空测试」的长相）
        if n == 0:
            raise SystemExit(f"错误：{name} 的有效 token 为 0 —— 掩码或采样坏了，"
                             f"不要拿 nan 当结论")
        rows[name] = dict(ce=ce, acc=acc, sce=sce, uce=uce, n=n)
        dumped[name] = [int(s) for s in starts]
        print(f"{name:<36}{group_of(name):<9}{ce:>9.4f}{acc * 100:>8.2f}{sce:>9.4f}"
              f"{uce:>9.4f}{ce - uce:>+10.4f}{n:>8,}")
    if dump_path:
        json.dump(dict(block_size=args.block_size, seed=seed, data=args.data,
                       offsets=args.offsets, manifest=args.manifest,
                       windows_per_source=args.windows_per_source, starts=dumped),
                  open(dump_path, 'w'), ensure_ascii=False, indent=1)
        print(f"已落盘窗口起点 → {dump_path}（{len(dumped)} 源）")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs='+', required=True)
    ap.add_argument('--data', default='data/chinese/val_char_v2.bin')
    ap.add_argument('--offsets', default='data/chinese/val_char_v2.off')
    ap.add_argument('--manifest', default='data/chinese/manifest_v2.json')
    ap.add_argument('--train-bin', default='data/chinese/train_char_v2.bin')
    ap.add_argument('--tokenizer', default='data/chinese/char_tokenizer.json')
    ap.add_argument('--block-size', type=int, default=256)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--windows-per-source', type=int, default=32)
    ap.add_argument('--seed', type=int, default=20260913)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--no-shuffle', action='store_true', help='跳过 shuffled 对照（省时间）')
    ap.add_argument('--no-unigram', action='store_true', help='跳过 unigram 对照')
    ap.add_argument('--control-random', action='store_true',
                    help='★ 已知答案对照：再评一个随机初始化的模型，real 必须约等于 shuffled')
    ap.add_argument('--dump-windows', default=None,
                    help='把**第一个 ckpt** 实际用的窗口起点逐源落盘成 JSON，'
                         '供"val 是否训练集近重复"的审计脚本复用同一批窗口')
    args = ap.parse_args()

    spec = special_mask(args.tokenizer)
    off = np.fromfile(args.offsets, dtype=np.int64)
    spans, order = source_spans(args.manifest, off)
    data = np.memmap(args.data, dtype=np.uint16, mode='r')
    print(f"val={args.data} ({len(data):,} tok)  源={len(order)}  窗口/源={args.windows_per_source}  "
          f"block={args.block_size}  特殊id={int(spec.sum())}")

    # —— 对照 ①：切片是否与源对齐（c4_zh 无回复终止符 → token/chars 应 ≈ 1.0）
    #   ⚠★ 这三个是 **manifest_v2** 的源名。换 manifest 时（如 v3_dlg / v3_know 只有 9 个源、
    #   源名集不同）硬查会 KeyError 直接崩 —— 2026-09-14 实测踩到。
    #   ⇒ 有 v2 对照源就用它（口径可比），没有就退化成"该 manifest 自己的前 3 个源"，
    #     并在输出里**显式注明换过对照源**，免得读者以为还是那三个。
    m = json.load(open(args.manifest))
    raw = {s['file']: s['val_chars_raw'] for s in m['source_breakdown']}
    ctrl_names, ctrl_note = align_control_names(spans, order, raw)
    if ctrl_note:
        print(f"  [对齐对照] {ctrl_note}")
    for name in ctrl_names:
        s0, s1 = spans[name]
        r = (s1 - s0) / max(raw[name], 1)
        print(f"  [对齐对照] {name:<34} token/raw_chars = {r:.4f}")

    need_uni = not args.no_unigram
    uni = unigram_probs(args.train_bin, spec, 8192) if need_uni else None

    dev = torch.device(args.device)
    ctx = (torch.autocast(device_type='cuda', dtype=torch.bfloat16)
           if dev.type == 'cuda' else torch.autocast('cpu', enabled=False))

    all_rows = {}
    cfg_for_control = None
    for ci, path in enumerate(args.ckpts):
        model, ck = load_model(path, dev)
        cfg_for_control = dict(ck['model_args'])
        step = ck.get('iter_num', -1)
        print(f"\n=== step {step}  ({path}) ===")
        all_rows[step] = eval_all_sources(model, args, data, off, spans, order,
                                          spec, uni, dev, ctx, args.seed,
                                          dump_path=(args.dump_windows if ci == 0 else None))
        del model
        torch.cuda.empty_cache()

    # ★ 已知答案对照：**随机初始化**的模型。
    #   它没有学到任何上下文，所以 `real ≈ shuffled ≈ unigram` 必须成立；
    #   若连它也报出 `real << shuffled`，那说明这个指标在量别的东西（token 统计/掩码），
    #   而不是"模型在读上下文"，后面所有结论都作废。
    if args.control_random:
        from model import GPTConfig
        fields = set(GPTConfig.__dataclass_fields__)
        torch.manual_seed(20260913)
        rnd = GPT(GPTConfig(**{k: v for k, v in cfg_for_control.items() if k in fields}))
        rnd.to(dev).eval()
        print("\n=== [对照] 随机初始化权重（应 real ≈ shuffled ≈ unigram）===")
        all_rows['random'] = eval_all_sources(rnd, args, data, off, spans, order,
                                              spec, uni, dev, ctx, args.seed)
        del rnd
        torch.cuda.empty_cache()

    # —— 汇总：按组做 token 加权平均
    print("\n=== 分组汇总（token 加权）===")
    print(f"{'step':>7} " + "".join(f"{g:>13}" for g in ['散文/语言', '知识/长文', '代码', '数学/推理', '对话']))
    for step, rows in all_rows.items():
        line = f"{step:>7} "
        for g in ['散文/语言', '知识/长文', '代码', '数学/推理', '对话']:
            sub = [r for k, r in rows.items() if group_of(k) == g]
            n = sum(r['n'] for r in sub)
            if not sub or n == 0:
                line += f"{'-':>13}"
                continue
            line += f"{sum(r['ce'] * r['n'] for r in sub) / n:>13.4f}"
        print(line)

    print("\n=== 全库 token 加权（= 逐源明细的加权和，可与训练日志交叉核对）===")
    for step, rows in all_rows.items():
        n = sum(r['n'] for r in rows.values())
        ce = sum(r['ce'] * r['n'] for r in rows.values()) / n
        sce = [r['sce'] for r in rows.values() if not np.isnan(r['sce'])]
        uce = sum(r['uce'] * r['n'] for r in rows.values()) / n
        print(f"  step {step:>6}  real {ce:.4f}   shuffled {np.mean(sce):.4f}   unigram {uce:.4f}"
              f"   n_tok {n:,}")


if __name__ == '__main__':
    main()
