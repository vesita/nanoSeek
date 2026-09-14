"""按「识字 / 知识 / 对话」三阶段切分语料并各自构建 bin。

为什么需要它：分阶段训练要求每个阶段读**自己那一份**语料，而不是一份混在一起的
900M 语料。项目已有的机制是 `prepare.py --source-dir <目录> --out-prefix <前缀>`
（扫描输入目录、产物仍写 data/chinese/），所以「一个阶段 = 一个目录 + 一个前缀」，
**不需要改训练代码**；训练侧用 `--data-prefix v3_lang` 等切换（`training/schedules.py
::pick_bin_names` 是唯一解析点，train.py 启动时会把真实文件名打出来防配错）。

本脚本只干两件事：
  1. 把（已清洗的 + 新导入的）源文件按阶段映射**软链**到 data/chinese/stages/<stage>/；
     ★ 用符号链接而不是复制：源文件改动会同步反映，且不占额外磁盘。
  2. 逐阶段调用 prepare.py 构建 bin（命令原样打印，便于留痕/复现）。

用法：
    # ★ 原始语料已移出仓库：/home/vesita/datasets/NLP/*.txt（2026-09-13 用户要求）。
    #   清洗前先把「原始语料 + 新导入语料」聚成一个输入目录（软链即可）：
    #     mkdir -p data/chinese/raw_all
    #     ln -sf /home/vesita/datasets/NLP/*.txt data/chinese/raw_all/
    #     ln -sf data/chinese/new_sources/*.txt data/chinese/raw_all/
    #   然后【先脱敏 → 再清洗 → 切阶段】：
    #   ① 脱敏：`new_sources/wildchat_zh.txt` 是**真实用户对话**，含真手机号/邮箱，以及
    #      一个"5 人姓名 + 手机 + 门牌住址"的名单块。实测原文有 **30 个独立 11 位手机号 /
    #      31 个邮箱**；工具默认 dry-run、`--apply` 才写、拒绝原地覆盖；名单块整块删除。
    #      ★ 这一步必须在**上游**做（new_sources/），否则重跑 clean_corpus 会把 PII 灌回 clean_v3。
    #      .venv/bin/python data/chinese/pii_scrub.py \
    #          --in data/chinese/new_sources/wildchat_zh.txt \
    #          --out /tmp/wildchat_zh.clean.txt --apply   # 再把产物替换回原位
    #   ② 清洗：
    .venv/bin/python data/chinese/clean_corpus.py --src data/chinese/raw_all \
        --dst data/chinese/clean_v3 --apply --dedup-blocks \
        --ngram-size 12 --max-ngram-freq 50 --max-ngram-cover 0.5 \
        --filter-nondialogue-rep3 --nondialogue-max-rep3 0.4
    #   ③ 切阶段：
    .venv/bin/python data/chinese/build_stages.py --src data/chinese/clean_v3 \
        --extra data/chinese/new_sources --apply --build
    # ⚠ 上面**故意不带** --max-reply-rep3：它会砍掉 45% 语料，而抽检证明被砍的 CoT 全是合法推理。
    # ⚠ --max-ngram-freq 用 50（不是 20）：20 会让候选爆到 1513 万、越过 --ngram-max-candidates 上限。
    # 两条都是 2026-09-13 实测结论，别照抄任何写着 --max-reply-rep3 0.3 / --max-ngram-freq 20 的旧文档。

构建完 `--build` 会**自动跑终止符位置验收**（`check_terminators`）：块内最后一个 `<eos>/<cont>`
必须落在块尾附近。2026-09-13 实测事故：`v3_know` 的 bin 是在 `annotate_replies` 修好之前构建的，
**32.61%** 的采样块终止符落在回复**中段**（`<eos>` 跟在 `def is_same_tuple(...):` 后面）——
即整份 bin 都在教"回复第一行之后就停"。这个验收就是为了让这类错误**不可能静默通过**。
"""
import argparse
import glob
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, 'data', 'chinese')
PREPARE = os.path.join(DATA, 'prepare.py')

# 阶段 → 源文件名（支持 fnmatch 通配）。划分依据见 PROJECT_STATE §0.5.9。
#   lang  识字：通用书面语（网页/百科/文学）—— 学字、词、句法
#   know  知识：问答 / 指令 / 任务 / 推理链 —— 学事实与推理
#   dlg   对话：有来有回的多轮对话 —— 学轮次结构与对话逻辑
#   persona 人格：**单流 `<resp>` 格式**的人设绑定层（2026-09-14 用户定的新阶段）——
#          语料是 `对象A：…` / `<resp>…<eos>` 逐行交替的**单流**（不是「用户：/模型：」），
#          `annotate_replies` 认不出它，会把每一行**原样保留**（正是我们要的：语料已自带
#          `<resp>`/`<eos>`，不该被二次标注）。⇒ 训练时**必须**配
#          `mask_mode: resp_span`（见 `configs/base_v3_persona.yaml`），否则
#          `build_assistant_mask` 的"整行含终止符"规则会把 `<resp>` 自己也算进 loss。
#  源文件 `persona_identity.txt` 本体**不在仓库里**（放在 ~/datasets/ 下，软链进
#  new_sources/；见 PROJECT_STATE §0.5.14）。文件名刻意保持中性。
STAGES = {
    'v3_lang': [
        'c4_zh.txt', 'wikipedia_cn.txt',
        '西游记.txt', '红楼梦.txt', '三国演义.txt', '水浒传.txt',
        'classical_poetry.txt',
    ],
    'v3_know': [
        'qa_knowledge.txt',                       # 新导入：274k 中文问答
        'deepseek_r1_distill_dialogue.txt', 'qwen3_235b_distill_dialogue.txt',
        'coig_*.txt', 'code_alpaca_dialogue.txt', 'gsm8k_cot_dialogue.txt',
        'zhihu_kol_dialogue.txt', 'muice_dialogue.txt',
        'identity_dialogue.txt',
    ],
    'v3_dlg': [
        'multi_turn_dialogue.txt', 'dailychat_dialogue.txt', 'lccc_dialogue.txt',
        'glm_dialogue.txt', 'kdconv_dialogue.txt',
        'sharegpt_zh_38k.txt',                    # 新导入：38.5k 段真多轮（LCS 承接 +15.1pp）
        'belle_multiturn.txt',                    # 新导入：Belle 0.8M 抽样
        'wildchat_zh.txt',                        # 新导入：WildChat 中文（已过滤 toxic）
        'escov_zh.txt',                           # 新导入：翻译后的多轮对话
    ],
    'v3_persona': [
        'persona_identity.txt',    # 身份事实层（名字/自称/边界，须反复重复才钉得住）
        'persona_monologue.txt',   # 第一人称独白层（给它"有话说"）
        'persona_dialogue.txt',    # 长对话层（**说话方式的主力**）
    ],
}

# 已知但**故意不进任何阶段**的源（写下来是为了让"漏了"和"有意排除"可区分）
EXCLUDED = {'mix_tables.txt', 'persona_contrast_probe.txt'}
# ★ `persona_contrast_probe.txt` 是**评估探针不是训练数据**：同一情境的"希望版/鸡汤版"
#   成对文本。普通 next-token 训练把它放进 loss 只会让模型**同时学会两种答法**
#   （产生不了对比），所以它**故意不进任何阶段**。要真做对比得上 RL/DPO（另一件事）。
#   写进 EXCLUDED 是为了让"漏了"和"有意排除"可区分。


def resolve(src_dirs):
    """{文件名: 绝对路径}；同名文件以先出现的目录为准（--src 优先于 --extra）。"""
    found = {}
    for d in src_dirs:
        if not os.path.isdir(d):
            continue
        for p in sorted(glob.glob(os.path.join(d, '*.txt'))):
            found.setdefault(os.path.basename(p), p)
    return found


def assign(found):
    """返回 (stage→[文件名], 未分配列表, 重叠列表)。

    ★ 2026-09-13：**字面文件名**（不含通配符）匹配不到 → **直接报错退出**，不再只打 ⚠。
    实测事故：`escov_zh.txt` 放进了 `new_sources/`，但重建时漏写 `--extra data/chinese/new_sources`
    ⇒ 脚本打了一行 ⚠ 就继续，**bin 与上一版逐位相同、一个 token 都没变**，而人只看了日志尾巴
    ⇒ "重建成功"其实是"没带上新源"。`STAGES` 是**声明式意图**，字面名不在就等于意图没落实，
    必须大声失败；带通配符的（如 `coig_*.txt`）允许匹配不到，保持 ⚠。
    """
    plan, taken = {}, {}
    for stage, pats in STAGES.items():
        names = []
        for pat in pats:
            hits = [n for n in found if glob.fnmatch.fnmatch(n, pat)]
            if not hits:
                if not any(c in pat for c in '*?['):
                    raise SystemExit(
                        f'❌ 阶段 {stage}: 字面源文件名 {pat!r} 在本批源里不存在。\n'
                        f'   常见原因：该文件不在 --src/--extra 覆盖的目录里 ——\n'
                        f'   若它在 data/chinese/new_sources/，必须显式加 '
                        f'`--extra data/chinese/new_sources`（--extra 默认是空的！）。\n'
                        f'   不修就继续的话，构建会"成功"但 bin 里根本没有这个源。')
                print(f'  ⚠ 阶段 {stage}: 通配模式 {pat!r} 没有匹配到文件（允许，跳过）')
            for n in hits:
                names.append(n)
                taken.setdefault(n, []).append(stage)
        plan[stage] = sorted(set(names))
    unassigned = sorted(n for n in found if n not in taken and n not in EXCLUDED)
    overlap = {n: s for n, s in taken.items() if len(s) > 1}
    return plan, unassigned, overlap


def check_terminators(stage, sample=3000):
    """验收 bin 里 `<eos>/<cont>` 的**位置**：块内最后一个终止符必须落在块尾附近。

    为什么需要它：`annotate_replies` 曾有"终止符插在回复首行之后"的 bug，修好之后
    **旧 bin 不会自己变对** —— 2026-09-13 实测 `v3_know` 的 bin 就是修复前构建的，
    23.6% 的采样块终止符落在回复中段（如 `<eos>` 紧跟 `def is_same_tuple(...):`）。
    文本层有单测钉着（`tests/test_prepare_split.py`），但**产物层此前无人验收**。

    返回 (含终止符的采样块数, rel 中位数, rel<0.5 的占比)；无终止符的纯散文源返回 (0, nan, nan)。
    块之间用 '\\n\\n' 拼接，所以"正确"的 rel 应当是 1 − 3/块长（≈0.97+）。
    """
    import numpy as np
    from tokenizers import Tokenizer

    bin_p = os.path.join(DATA, f'train_char_{stage}.bin')
    off_p = os.path.join(DATA, f'train_char_{stage}.off')
    if not (os.path.exists(bin_p) and os.path.exists(off_p)):
        return None
    v = Tokenizer.from_file(os.path.join(DATA, 'char_tokenizer.json')).get_vocab()
    terms = (v['<eos>'], v['<cont>'])
    off = np.fromfile(off_p, dtype=np.int64)
    d = np.memmap(bin_p, dtype=np.uint16, mode='r')
    n = len(off) - 1
    idx = np.linspace(0, n - 1, min(sample, n)).astype(np.int64)
    rels = []
    for i in idx:
        a, b = int(off[i]), int(off[i + 1])
        w = np.asarray(d[a:b])
        p = np.flatnonzero((w == terms[0]) | (w == terms[1]))
        if len(p):
            rels.append(float(p[-1]) / max(b - a - 1, 1))
    if not rels:
        return 0, float('nan'), float('nan')
    r = np.asarray(rels)
    return len(rels), float(np.median(r)), float((r < 0.5).mean())


def run_terminator_checks(stages, bad_frac_max):
    """对每个阶段跑位置验收；失败就大声退出（不允许静默通过）。"""
    print('\n=== 终止符位置验收（annotate_replies 的位置语义）===')
    bad = []
    for stage in stages:
        r = check_terminators(stage)
        if r is None:
            print(f'  {stage}: 缺 bin/.off，跳过')
            continue
        hit, p50, badf = r
        if hit == 0:
            print(f'  ✅ {stage}: 采样块中无终止符（纯散文源，符合预期）')
            continue
        ok = badf <= bad_frac_max
        print(f'  {"✅" if ok else "❌"} {stage}: 含终止符 {hit} 块  rel_p50={p50:.4f}  '
              f'rel<0.5 占比={badf:.2%}  (上限 {bad_frac_max:.1%})')
        if not ok:
            bad.append(stage)
    if bad:
        raise SystemExit(
            f'❌ 终止符位置验收失败：{bad}\n'
            f'   这些 bin 是在 annotate_replies 修好**之前**构建的 —— 整份都在教"回复首行后就停"。\n'
            f'   重建：.venv/bin/python data/chinese/build_stages.py --apply --build')


def main():
    ap = argparse.ArgumentParser(description='按三阶段切分语料并构建 bin')
    ap.add_argument('--src', default=os.path.join(DATA, 'clean_v3'),
                    help='清洗后的源目录（clean_corpus.py --apply 的产物）')
    ap.add_argument('--extra', action='append', default=[],
                    help='额外源目录（如 data/chinese/new_sources），可重复')
    ap.add_argument('--stages-dir', default=os.path.join(DATA, 'stages'))
    ap.add_argument('--apply', action='store_true', help='真的建软链（默认 dry-run）')
    ap.add_argument('--build', action='store_true', help='真的跑 prepare.py 构建 bin')
    ap.add_argument('--val-ratio', default='0.01')
    ap.add_argument('--seed', default='20260910')
    ap.add_argument('--only', nargs='*', default=None,
                    help='只构建/检查这些阶段（默认全部）。重建单阶段时用，避免白跑另外两个')
    ap.add_argument('--check', action='store_true',
                    help='只跑终止符位置验收（不建软链、不构建）')
    ap.add_argument('--bad-frac-max', type=float, default=0.05,
                    help='终止符位置验收的容忍上限（rel<0.5 的块占比）')
    a = ap.parse_args()

    src_dirs = [a.src] + list(a.extra)
    missing = [d for d in src_dirs if not os.path.isdir(d)]
    if missing:
        raise SystemExit(f'源目录不存在：{missing}\n（先跑 clean_corpus.py --apply）')
    found = resolve(src_dirs)
    print(f'可用源文件 {len(found)} 个（来自 {len(src_dirs)} 个目录）')
    plan, unassigned, overlap = assign(found)

    stages = sorted(a.only) if a.only else sorted(plan)
    unknown = [s for s in stages if s not in plan]
    if unknown:
        raise SystemExit(f'未知阶段 {unknown}；可选：{sorted(plan)}')

    print('\n=== 阶段划分 ===')
    tot = 0
    for stage in sorted(plan):
        chars = sum(os.path.getsize(found[n]) for n in plan[stage])
        tot += chars
        print(f'\n[{stage}] {len(plan[stage])} 个源，{chars/1e6:.1f}MB')
        for n in plan[stage]:
            print(f'    {n:<40} {os.path.getsize(found[n])/1e6:>8.1f}MB')
    print(f'\n三阶段合计 {tot/1e6:.1f}MB（源文件总量 {sum(os.path.getsize(p) for p in found.values())/1e6:.1f}MB）')
    if unassigned:
        print(f'\n⚠ 未被任何阶段覆盖 {len(unassigned)} 个：{unassigned}')
        print('  （若是新加的源，请加进 STAGES；有意排除的写进 EXCLUDED）')
    if overlap:
        print(f'\n⚠ 被多个阶段共用：{overlap}（通常说明划分需要收紧）')

    if a.check:
        run_terminator_checks(stages, a.bad_frac_max)
        if not (a.apply or a.build):
            return

    if not a.apply:
        print('\n（dry-run：未建目录/未构建。加 --apply 建软链，再加 --build 构建 bin）')
        return

    for stage in stages:
        names = plan[stage]
        d = os.path.join(a.stages_dir, stage)
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            if f.endswith('.txt') and f not in names:
                os.remove(os.path.join(d, f))       # 清掉上一轮的残留，避免脏链
        for n in names:
            link = os.path.join(d, n)
            if os.path.islink(link) or os.path.exists(link):
                os.remove(link)
            os.symlink(os.path.abspath(found[n]), link)
        print(f'✅ {stage}: {len(names)} 个软链 → {d}')

    if not a.build:
        print('\n（--apply 完成；未构建 bin。加 --build 才跑 prepare.py）')
        return

    for stage in stages:
        # --emit-offsets：产出 `train_char_<stage>.off` / `val_char_<stage>.off`
        # （块在 bin 里的起始 token 下标），供 train.py 的 `use_doc_packing` 做
        # 块对角注意力掩码。不带它就只能跑全域随机窗口（会跨样本污染）。
        cmd = [sys.executable, PREPARE, '--char-level', '--val-all',
               '--val-ratio', a.val_ratio, '--seed', a.seed,
               '--source-dir', os.path.join(a.stages_dir, stage),
               '--out-prefix', stage, '--emit-offsets']
        print('\n$ ' + ' '.join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            raise SystemExit(f'❌ {stage} 构建失败（退出码 {r.returncode}）')
    run_terminator_checks(stages, a.bad_frac_max)
    print('\n✅ bin 构建完成。训练时用 --data-prefix v3_lang / v3_know / v3_dlg / '
          'v3_persona 切换（★ v3_persona 必须配 mask_mode=resp_span，见 '
          'configs/base_v3_persona.yaml）')


if __name__ == '__main__':
    main()
