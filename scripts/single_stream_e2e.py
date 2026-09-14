#!/usr/bin/env python
"""单流 `<resp>` 格式的**端到端**验收：训练好的权重 → 采样 → 框架读回来。

为什么需要它：`scripts/resp_bin_probe.py` 只验**语料 bin**，`pytest` 只验**纯函数**；
两者都不回答"模型真的能产出这个格式、且框架真的能读回来"。
而这两件事恰恰是本层最容易静默失败的地方 —— 权重在 `<resp>` 之后立刻吐 `<eos>`、
或者输出的根本是 v3_dlg 那套 `A：/B：`，都不会报错，只会"看起来在说话"。

## 三条判据（每条都打印原始证据，不只看摘要）

1. **模型自己收尾**：`P(模型在 max_new_tokens 内吐 <eos>)`。这是 `resp_span` 训练的
   直接目标（`<eos>` 是 loss 区间右端）。★ 只看比例不够，还要看**收尾处**长什么样。
2. **能读回来**：`parse_log(采样文本)` 不抛异常，且
   `parse_log(s.render()).render() == s.render()`（框架自带的不变式）。
3. **轮次交替 + loss 区间对齐**：`history()` 里没有相邻同说话人；
   `loss_token_spans()` 覆盖的区间**只**落在模型轮次上（对方的话不算 loss）。

## ★★ 两个必须写进结论的诚实限制

- **本层模型是背下来的**（86k token / 300 步 ≈ 28 epoch）⇒ 补全得像样**不能**当作
  泛化或人格质量的证据。它证明的是**格式链路**。
- `inference/scripts/sample_py.py` 有**两处会吃掉机制符**（本项目栽过三次的那个坑）：
  ① 命中 `<eos>` 时 `gen = gen[:new_start+i]` **把 `<eos>` 从返回的 ids 里删掉**
  （"EOS 及其后不输出"，见该文件 `generate_ids` 尾部）；
  ② 所有 `tok.decode(...)` 都用默认 `skip_special_tokens=True`。
  ⇒ 采样结果**不能直接**喂 `parse_log`（会是一个没闭合的 `<resp>`）。
  本脚本把那个 `<eos>` **补回去**（只在 `eos_pos >= 0` 时），并用
  `skip_special_tokens=False` 解码 —— 这是**补记**采样器故意删掉的 token，不是伪造。
  该缺口已记入 `TECH_DEBT`，别在这里"顺手"改 `sample_py.py`（会动既有评估的产物）。

用法：
    .venv/bin/python scripts/single_stream_e2e.py --out_dir out/base_v3_persona
    # 可选：--n 6 --max-new-tokens 80 --temperature 0.8 --seed 1337
    #       --out ~/datasets/persona/reports/persona_smoke_e2e.txt

★★ **`--out` 的报告里含模型生成的对话正文** ⇒ 按用户定的边界（人设相关文本只留在
上下文与 `~/datasets/persona/`），**不要把报告写进仓库**：默认只打 stdout，
要落盘请写 `~/datasets/persona/reports/`。2026-09-14 实测时我第一版就写进了
`analysis/`，已挪出仓库。
"""
import argparse
import importlib.util
import io
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from training.dialogue_stream import EOS, TURN_CUE, parse_log  # noqa: E402


def _load_sample_py():
    """按文件路径加载 `inference/scripts/sample_py.py`（它不是包，没有 `__init__.py`）。"""
    path = os.path.join(ROOT, 'inference', 'scripts', 'sample_py.py')
    spec = importlib.util.spec_from_file_location('_ns_sample_py', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # 它有 __main__ guard，导入不会采样
    return mod


def build_prompts(path, n):
    """从语料里取 n 段对话，每段用"对方的话 + `<resp>`"当提示词（= 部署时的输入形状）。"""
    with io.open(path, encoding='utf-8') as f:
        blocks = [b.strip() for b in f.read().split('\n\n') if b.strip()]
    prompts = []
    for b in blocks:
        lines = [ln for ln in b.split('\n') if ln.strip()]
        first = lines[0] if lines else ''
        if not first or first.startswith(TURN_CUE):
            continue
        prompts.append(first + '\n' + TURN_CUE)     # 对方一句 + 请模型接话
        if len(prompts) >= n:
            break
    return prompts


def check_stream(text, encode):
    """返回 (e2e 检查结果 dict, 人类可读的问题列表)。"""
    problems = []
    st = parse_log(text, encode, window=10 ** 9)
    hist = [(sp, s) for sp, s in st.history() if s.strip()]

    # (3a) 相邻同说话人 —— ★ 必须在**渲染行**（= 轮次）层测，不能在句子层测！
    #   `DialogueStream` 内部**按句**存条目，一轮回复本来就是好几条同说话人条目
    #   （`group_turns=True` 时才在渲染时合并成一行）。第一版我拿 `history()` 直接数，
    #   于是每段都报 1~2 处"违规" —— **是我的口径错，不是生成错**（§5.11 同族）。
    #   语料生成器当初查的也是"行级连续同说话人"，这里对齐到同一个口径。
    rendered = [ln for ln in st.render().split('\n') if ln.strip()]
    who = []
    for ln in rendered:
        if ln.startswith(TURN_CUE):
            who.append('self')
        elif '：' in ln:
            who.append(ln.split('：', 1)[0])
        elif who:
            who.append(who[-1])          # 续行接上一条
    same = [(i, who[i]) for i in range(1, len(who)) if who[i] == who[i - 1]]

    # (3b) render 往返（框架自带的等价性不变式）
    r1 = st.render()
    r2 = parse_log(r1, encode, window=10 ** 9).render()
    roundtrip_ok = (r1 == r2)
    if not roundtrip_ok:
        problems.append('render 往返不等价（parse_log(render()) != render()）')

    # (3c) loss 区间：每个区间右端必须紧跟在 `<eos>` 之后；模型轮次必须落在某个区间内
    ids = list(encode(r1))
    eos_ids = list(encode(EOS))
    cue_ids = list(encode(TURN_CUE))
    spans = st.loss_token_spans()
    in_loss = [False] * len(ids)
    for a, b in spans:
        for j in range(a, b):
            in_loss[j] = True
    # 每个 span 的右端（含）必须是 <eos>
    for a, b in spans:
        tail = ids[b - len(eos_ids):b]
        if tail != eos_ids:
            problems.append(f'loss 区间 {(a, b)} 的右端不是 {EOS}（实际 {tail}）')
    # `<resp>` 自身必须永远不在 loss 里
    cue_in_loss = [j for j, t in enumerate(ids)
                   if t in cue_ids and in_loss[j]]
    if cue_in_loss:
        problems.append(f'{len(cue_in_loss)} 个 {TURN_CUE} 落在 loss 区间内（不应发生）')
    n_cov = sum(in_loss)
    return {
        'entries': len(hist),
        'turn_lines': len(who),
        'same_speaker_adjacent': len(same),
        'roundtrip_ok': roundtrip_ok,
        'spans': len(spans),
        'tokens': len(ids),
        'loss_tokens': n_cov,
        'loss_frac': n_cov / max(len(ids), 1),
        'cue_in_loss': len(cue_in_loss),
    }, problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out_dir', default='out/base_v3_persona', help='读它的 best.pt')
    ap.add_argument('--corpus', default=None,
                    help='取提示词的语料（默认 data/chinese/new_sources/persona_identity.txt）')
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--max-new-tokens', type=int, default=80)
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--top-k', type=int, default=200)
    ap.add_argument('--repeat-penalty', type=float, default=1.2)
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', default=None, help='把报告也写到这个文件')
    a = ap.parse_args(argv)

    corpus = a.corpus or os.path.join(ROOT, 'data', 'chinese', 'new_sources',
                                      'persona_identity.txt')
    sp = _load_sample_py()
    torch.manual_seed(a.seed)
    model, ckpt = sp.build_model_from_checkpoint(a.out_dir, a.device)
    tok = sp.load_tokenizer(ckpt)
    eos_id = tok.token_to_id(EOS)
    cue_id = tok.token_to_id(TURN_CUE)
    if eos_id is None or cue_id is None:
        raise SystemExit(f'❌ 词表里找不到 {EOS}/ {TURN_CUE} —— 不是字级词表？')
    encode = lambda s: list(tok.encode(s).ids)      # noqa: E731

    prompts = build_prompts(corpus, a.n)
    lines = [f'模型      : {a.out_dir}（best.pt）',
             f'词表      : {TURN_CUE}={cue_id}  {EOS}={eos_id}',
             f'提示词来源: {os.path.relpath(corpus, ROOT)} 前 {len(prompts)} 段',
             f'采样      : temp={a.temperature} top_k={a.top_k} '
             f'rep={a.repeat_penalty} max_new={a.max_new_tokens} seed={a.seed}',
             '']
    n_eos = 0
    all_problems = []
    agg = []
    for i, p in enumerate(prompts):
        gen, eos_pos = sp.generate_ids(
            model, tok, p, a.max_new_tokens, a.temperature, a.top_k, a.repeat_penalty,
            False, True, False, window=None, no_resume=False)
        # ★ 补回采样器故意删掉的那个 <eos>（见模块 docstring 的说明）
        if eos_pos >= 0:
            n_eos += 1
            gen = list(gen)
            if not gen or gen[-1] != eos_id:
                gen.append(eos_id)
        text = tok.decode(gen, skip_special_tokens=False)
        res, probs = check_stream(text, encode)
        res['eos'] = eos_pos >= 0
        agg.append(res)
        all_problems += probs
        lines.append(f'--- prompt {i} ---')
        lines.append(p)
        lines.append(f'--- 生成 {i}（eos_pos={eos_pos}）---')
        lines.append(text)
        lines.append(f'[自检] 句条目={res["entries"]} 渲染行={res["turn_lines"]} '
                     f'相邻同行={res["same_speaker_adjacent"]} '
                     f'loss 区间={res["spans"]} loss token={res["loss_tokens"]}/{res["tokens"]}'
                     f'（{res["loss_frac"]:.1%}） 往返={res["roundtrip_ok"]} '
                     f'{TURN_CUE} 落在 loss 内={res["cue_in_loss"]}')
        lines.append('')

    lines.insert(4, f'★ 模型自己收尾（吐 {EOS}）: {n_eos}/{len(prompts)}')
    if agg:
        lines.insert(5, f'★ loss 覆盖中位数: '
                        f'{sorted(r["loss_frac"] for r in agg)[len(agg)//2]:.1%}')
    lines.append('=== 汇总 ===')
    lines.append(f'  采样段数                : {len(prompts)}')
    lines.append(f'  模型自己收尾            : {n_eos}/{len(prompts)}')
    lines.append(f'  渲染行级相邻同说话人（应 0）: '
                 f'{sum(r["same_speaker_adjacent"] for r in agg)}')
    lines.append(f'  render 往返全 OK        : {all(r["roundtrip_ok"] for r in agg)}')
    lines.append(f'  {TURN_CUE} 落在 loss 内（应 0）: {sum(r["cue_in_loss"] for r in agg)}')
    lines.append(f'  检查发现的问题          : {len(all_problems)}')
    for pb in all_problems[:8]:
        lines.append(f'    ❌ {pb}')

    report = '\n'.join(lines)
    print(report)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with io.open(a.out, 'w', encoding='utf-8') as f:
            f.write(report + '\n')
        print(f'\n已写入 {a.out}')
    hard = (not all_problems
            and all(r['roundtrip_ok'] for r in agg)
            and all(r['same_speaker_adjacent'] == 0 for r in agg)
            and all(r['cue_in_loss'] == 0 for r in agg))
    print('✅ 三项硬检查通过' if hard else '❌ 有硬检查未通过')
    return 0 if hard else 1


if __name__ == '__main__':
    raise SystemExit(main())
