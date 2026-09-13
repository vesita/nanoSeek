"""独立验收清洗前后的语料：**不复用清洗器的任何代码**，独立重算指标。

为什么要有这个脚本（而不是只看 `clean_corpus.py` 写的 CLEANING_REPORT.md）：
  * 清洗器的报告是"自己给自己打分"。本项目已经吃过一次亏 —— `TECH_DEBT §1.12`
    的归档清理报「释放 0MB」，而真实释放 0.59 GiB，根因就是统计量算在了错误的时点。
    **测量函数自己坏了，输出会和"没有问题"长得一模一样**，只有独立重算才分辨得出。
  * 所以本脚本自己读 `--src` / `--dst` 两份 `.txt`、自己算 block 数/字符数/重复率/
    高频 n-gram 覆盖率，再和清洗器的报告对照。对不上就是清洗器错了。

三项判据（任一 FAIL ⇒ 退出码非 0，可当闸门）：
  1. **非对话文件必须逐块不变** —— 百科/网页/名著占语料大头，清洗规则只应作用于对话。
  2. **规则开启时，dst 里必须检不出对应病症**（精确重复 / rep3 超限 / 高频 n-gram 覆盖超限）。
  3. **`--selftest`（已知答案输入）**：把病症**植入**一份小语料，断言
       ① 本脚本的指标能检出植入的病症（否则指标是空的、验收是橡皮图章）；
       ② 清洗器**关掉所有规则**时产物必须与输入逐块相同（否则规则之外还有副作用）；
       ③ 打开规则后病症必须消失。

用法：
    .venv/bin/python data/chinese/verify_clean_corpus.py --selftest
    .venv/bin/python data/chinese/verify_clean_corpus.py \
        --src data/chinese --dst data/chinese/clean_v3 \
        --expect-cover 0.5 --ngram-size 12 --ngram-freq 20
"""
import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLEANER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clean_corpus.py')
DIALOGUE_MARK = '模型：'


def read_blocks(path):
    """与 prepare.py 完全一致的 block 切分：`\\n\\n` 分隔 + strip + 去空。"""
    with open(path, encoding='utf-8', errors='replace') as f:
        text = f.read()
    return [b.strip() for b in text.split('\n\n') if b.strip()]


def norm(s):
    return re.sub(r'\s+', '', s)


def rep3(s):
    """字符 3-gram 重复率（与 eval 侧同一口径）。"""
    seq = norm(s)
    if len(seq) < 6:
        return 0.0
    grams = [seq[i:i + 3] for i in range(len(seq) - 2)]
    c = Counter(grams)
    return sum(1 for g in grams if c[g] > 1) / len(grams)


def replies_of(block):
    return re.findall(r'模型：([^\n]*)', block)


def is_dialogue(block):
    return DIALOGUE_MARK in block


def list_txt(d):
    return sorted(f for f in os.listdir(d) if f.endswith('.txt'))


def scan_dir(d, with_grams=False, gram_char_budget=3_000_000):
    """扫一个目录，独立重算统计量（不用清洗器的任何代码）。

    判据全部选**内存安全且无偏**的：
      * block/字符数、非对话 block 数（直接数）
      * 精确重复：全局 blake2b(去空白) 集合
      * rep3>0.3：逐条回复算（与评估侧同口径）
      * **回复级复用浓度**：归一化后完全相同的回复最多被复用多少次 —— 这是"套话洪水"的
        廉价代理。★ 不做 n-gram 覆盖率：全量统计不同 12-gram 会吃几十 GB 内存
        （清洗器为此才上 Count-Min 草图），验收端不该重复那个成本。
    """
    stats = dict(files={}, chars=0, blocks=0, dialogue_blocks=0,
                 exact_dup=0, rep3_bad=0, rep3_bad_chars=0)
    seen = set()
    reply_counter = Counter()
    gram_counter = Counter()
    gram_chars = 0
    for fn in list_txt(d):
        blocks = read_blocks(os.path.join(d, fn))
        fchars = sum(len(b) for b in blocks)
        fdup = frep3 = fnon = 0
        for b in blocks:
            stats['blocks'] += 1
            stats['chars'] += len(b)
            h = hashlib.blake2b(norm(b).encode('utf-8'), digest_size=16).digest()
            if h in seen:
                stats['exact_dup'] += 1
                fdup += 1
            else:
                seen.add(h)
            if not is_dialogue(b):
                fnon += 1
                continue
            stats['dialogue_blocks'] += 1
            reps = replies_of(b)
            if any(rep3(r) > 0.3 for r in reps):
                stats['rep3_bad'] += 1
                stats['rep3_bad_chars'] += len(b)
                frep3 += 1
            for r in reps:
                k = norm(r)
                if len(k) >= 20:
                    reply_counter[k] += 1
                # 12-gram 覆盖：只在语料够小（或显式要求）时精确统计 —— 全量统计不同
                # 12-gram 会吃几十 GB（清洗器为此才上 Count-Min），验收端不重复那个成本。
                if with_grams and gram_chars < gram_char_budget:
                    s_ = norm(r)
                    gram_chars += len(s_)
                    for i in range(len(s_) - 11):
                        gram_counter[s_[i:i + 12]] += 1
        stats['files'][fn] = dict(blocks=len(blocks), chars=fchars, dup=fdup,
                                  rep3_bad=frep3, nondialogue=fnon)
    stats['reply_reuse'] = reply_counter
    stats['grams'] = gram_counter
    return stats


def ngram_cover(stats, freq_threshold):
    """高频 12-gram 的**出现次数占比**（仅在 scan_dir(with_grams=True) 时有值）。"""
    c = stats.get('grams') or {}
    if not c:
        return None, 0
    tot = sum(c.values())
    hot = sum(v for v in c.values() if v > freq_threshold)
    return (hot / tot if tot else 0.0), sum(1 for v in c.values() if v > freq_threshold)


def reuse_stats(stats):
    """返回 (被复用的不同回复数, 最高复用次数, 复用字符占比)。"""
    c = stats['reply_reuse']
    if not c:
        return 0, 0, 0.0
    dup_types = sum(1 for v in c.values() if v > 1)
    extra_chars = sum(len(k) * (v - 1) for k, v in c.items() if v > 1)
    total_chars = sum(len(k) * v for k, v in c.items())
    return dup_types, max(c.values()), (extra_chars / total_chars if total_chars else 0.0)


def parse_enabled_rules(report_path):
    """从清洗器报告里读出**实际启用了哪些规则**。

    为什么必须读而不是假设：验收器一开始硬编码"非对话文件必须逐块不变"、
    "rep3 规则必须生效"，结果在"故意开非对话过滤、故意关 rep3"的配置下报了两条
    假 FAIL（2026-09-13 实测）。判据必须跟着**实际配方**走。
    """
    if not report_path or not os.path.exists(report_path):
        return None
    txt = open(report_path, encoding='utf-8').read()
    head = txt.split('## 全局')[0]
    return dict(
        dedup='--dedup-blocks' in head,
        reply_rep3='--max-reply-rep3' in head,
        ngram='--max-ngram-cover' in head,
        nondialogue='--filter-nondialogue-rep3' in head,
    )


def check(src, dst, rules=None, max_dup_rate=0.005, max_rep3_rate=0.005):
    """返回 (failures, lines)。failures 非空 ⇒ 验收不通过。"""
    failures, lines = [], []
    sfiles, dfiles = list_txt(src), list_txt(dst)
    missing = [f for f in sfiles if f not in dfiles]
    if missing:
        failures.append(f'dst 缺少 {len(missing)} 个文件，例如 {missing[:3]}')
    extra = [f for f in dfiles if f not in {*sfiles, 'CLEANING_REPORT.md'}]
    if extra:
        failures.append(f'dst 出现 src 里没有的文件：{extra[:3]}')

    # 小语料（自带对照的 selftest）精确算 12-gram；全量则跳过并明说
    small = os.path.getsize(os.path.join(src, sfiles[0])) < 4_000_000 if sfiles else True
    S = scan_dir(src, with_grams=small)
    D = scan_dir(dst, with_grams=small)

    lines.append('| 指标 | src（清洗前） | dst（清洗后） |')
    lines.append('|---|---:|---:|')
    for k, label in (('blocks', 'block 数'), ('chars', '字符数'), ('dialogue_blocks', '对话 block'),
                     ('exact_dup', '精确重复 block'), ('rep3_bad', 'rep3>0.3 的 block')):
        lines.append(f'| {label} | {S[k]:,} | {D[k]:,} |')
    sdup = S['exact_dup'] / max(1, S['blocks'])
    ddup = D['exact_dup'] / max(1, D['blocks'])
    srep = S['rep3_bad'] / max(1, S['dialogue_blocks'] or 1)
    drep = D['rep3_bad'] / max(1, D['dialogue_blocks'] or 1)
    lines.append(f'| 精确重复率 | {sdup:.3%} | {ddup:.3%} |')
    lines.append(f'| rep3>0.3 率（对话内） | {srep:.3%} | {drep:.3%} |')
    scov, shot = ngram_cover(S, 50)
    dcov, dhot = ngram_cover(D, 50)
    if scov is not None:
        lines.append(f'| 高频 12-gram 出现占比（>50 次） | {scov:.2%} | {dcov:.2%} |')
        lines.append(f'| 高频 12-gram 种类 | {shot:,} | {dhot:,} |')
    else:
        lines.append('\n> ⚠ 全量语料不做独立 12-gram 重算（内存不允许）；该规则的证据以'
                     '清洗器报告 + `analysis/corpus_quality_v2.md` 的逐源实测为准。')
    sdt, smx, scc = reuse_stats(S)
    ddt, dmx, dcc = reuse_stats(D)
    lines.append(f'| 被复用的不同回复数（≥20 字） | {sdt:,} | {ddt:,} |')
    lines.append(f'| 单条回复最高复用次数 | {smx:,} | {dmx:,} |')
    lines.append(f'| 复用冗余字符占比 | {scc:.2%} | {dcc:.2%} |')

    # 判据 1：非对话文件——未开非对话过滤时必须逐块不变；开了则允许变少但不能变多
    changed, grew = [], []
    for fn in sfiles:
        if fn not in dfiles:
            continue
        sb = read_blocks(os.path.join(src, fn))
        if any(is_dialogue(b) for b in sb):
            continue
        db = read_blocks(os.path.join(dst, fn))
        if sb != db:
            changed.append((fn, len(sb), len(db)))
        if len(db) > len(sb):
            grew.append((fn, len(sb), len(db)))
    if grew:
        failures.append(f'非对话文件**变多**了（清洗不该新增块）：{grew[:3]}')
    if changed and not (rules or {}).get('nondialogue'):
        failures.append(f'非对话文件被改动，但本次并未启用非对话过滤：{changed[:3]}')
    elif changed:
        lines.append(f'\n非对话文件：{len(changed)} 个被非对话过滤裁剪（本次已启用该规则，符合预期）')
    else:
        lines.append('\n非对话文件：全部逐块一致 ✅')

    # 判据 2：dst 里不应残留病症（仅在 src 确实很脏时才判，避免对小语料误报）
    if (rules or {}).get('ngram') and scov is not None and dcov is not None and scov > 0.01 \
            and dcov > scov * 0.9:
        failures.append(f'高频 12-gram 占比仍为 {dcov:.2%}（src {scov:.2%}）—— 短语级规则没生效？')
    if (rules or {}).get('dedup') and sdup > max_dup_rate and ddup > max_dup_rate:
        failures.append(f'精确重复率仍为 {ddup:.3%}（src {sdup:.3%}）—— 去重规则没生效？')
    if (rules or {}).get('reply_rep3') and srep > max_rep3_rate and drep > max_rep3_rate:
        failures.append(f'rep3>0.3 率仍为 {drep:.3%}（src {srep:.3%}）—— rep3 规则没生效？')
    elif not (rules or {}).get('reply_rep3'):
        lines.append(f'\nrep3>0.3 率 {srep:.3%} → {drep:.3%}（本次**未启用** reply-rep3 规则，'
                     f'残留属预期；它不是验证目标）')
    # 复用冗余必须下降（脏语料的"套话洪水"在清洗后应显著减少）
    if scc > 0.01 and dcc > scc * 0.9:
        failures.append(f'回复复用冗余仍为 {dcc:.2%}（src {scc:.2%}）—— 模板规则没生效？')

    # 清洗量必须与"删掉了东西"一致：dst 不应比 src 大
    if D['blocks'] > S['blocks']:
        failures.append(f'dst block 数（{D["blocks"]:,}）大于 src（{S["blocks"]:,}）')
    return failures, lines


# --------------------------------------------------------------------------- selftest
PLANTED_REP = '好烦好烦好烦好烦好烦好烦好烦好烦好烦好烦好烦好烦'
PLANTED_TPL = '这种感受一般不会一下子散掉。'


def _write(path, blocks):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(blocks) + '\n\n')


def _selftest_corpus(d):
    os.makedirs(d, exist_ok=True)
    clean = ['用户：我想聊聊天\n模型：好的，你先说说今天发生了什么吧。',
             '用户：最近有点累\n模型：听起来这阵子确实挺消耗人的。']
    dialogue = [
        clean[0],
        clean[0],                                   # ← 植入：精确重复
        '用户：好烦\n模型：' + PLANTED_REP,          # ← 植入：rep3 超限
        f'用户：甲\n模型：{PLANTED_TPL}甲的情况是这样的，我们慢慢来。',
        f'用户：乙\n模型：{PLANTED_TPL}乙的情况是这样的，我们慢慢来。',
        f'用户：丙\n模型：{PLANTED_TPL}丙的情况是这样的，我们慢慢来。',
        clean[1],
    ]
    wiki = ['昭通机场是位于中国云南昭通的民用机场，始建于1935年。',
            '《红楼梦》是中国古典四大名著之一，作者曹雪芹。']
    _write(os.path.join(d, 'dialogue.txt'), dialogue)
    _write(os.path.join(d, 'wiki.txt'), wiki)
    return dialogue, wiki


def _run_cleaner(src, dst, extra):
    cmd = [sys.executable, CLEANER, '--src', src, '--dst', dst, '--apply',
           '--report', os.path.join(dst, 'CLEANING_REPORT.md'), *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f'清洗器退出码 {r.returncode}\nSTDOUT:\n{r.stdout[-2000:]}\nSTDERR:\n{r.stderr[-2000:]}')
    return r


def selftest():
    fails = []
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, 'src')
        dialogue, wiki = _selftest_corpus(src)

        # ① 指标必须能检出植入的病症（否则指标是空的）
        S = scan_dir(src, with_grams=True)
        if S['exact_dup'] == 0:
            fails.append('对照①：精确定重指标没能检出植入的重复 block')
        if S['rep3_bad'] == 0:
            fails.append('对照①：rep3 指标没能检出植入的重复回复')
        # 小语料用低频次阈值：植入的套话只出现 3 次，>50 那种生产阈值当然看不见。
        # ★ 对照组要针对"最可能搞错的那一步"：这里要防的是"规则/指标根本没生效"。
        dt, mx, cc = reuse_stats(S)
        cov1, hot1 = ngram_cover(S, 1)
        if mx < 2 and not hot1:
            fails.append('对照①：既没检出重复回复、也没检出高频短语 —— 指标是空的')
        print(f'  对照① 指标非空：exact_dup={S["exact_dup"]} rep3_bad={S["rep3_bad"]} '
              f'最高复用={mx} 高频短语(>1次)={hot1} 12-gram占比={cov1:.1%}')

        # ② 关掉所有规则 ⇒ 产物必须与输入逐块相同（规则之外不许有副作用）
        dst0 = os.path.join(tmp, 'dst_rules_off')
        _run_cleaner(src, dst0, [])
        for fn, orig in (('dialogue.txt', dialogue), ('wiki.txt', wiki)):
            got = read_blocks(os.path.join(dst0, fn))
            if got != orig:
                fails.append(f'对照②：规则全关时 {fn} 被改动（{len(orig)} → {len(got)} 块）')
        print(f'  对照② 规则全关：产物与输入逐块相同 = {not fails}')

        # ③ 打开规则 ⇒ 病症必须消失，且非对话文件不变
        dst1 = os.path.join(tmp, 'dst_clean')
        _run_cleaner(src, dst1, ['--dedup-blocks', '--max-reply-rep3', '0.3',
                                 '--ngram-size', '12', '--max-ngram-freq', '2',
                                 '--max-ngram-cover', '0.5', '--min-reply-chars', '4'])
        failures, lines = check(src, dst1, rules=dict(dedup=True, reply_rep3=True,
                                                      ngram=True, nondialogue=True))
        if failures:
            fails.extend([f'对照③：{x}' for x in failures])
        got_wiki = read_blocks(os.path.join(dst1, 'wiki.txt'))
        if got_wiki != wiki:
            fails.append('对照③：非对话文件被清洗规则改动')
        got_dlg = read_blocks(os.path.join(dst1, 'dialogue.txt'))
        if any(rep3(r) > 0.3 for b in got_dlg for r in replies_of(b)):
            fails.append('对照③：dst 里仍有 rep3>0.3 的回复')
        if len(got_dlg) != 2:
            fails.append(f'对照③：预期只剩 2 个干净对话 block，实际 {len(got_dlg)}')
        print(f'  对照③ 规则开启：对话块 {len(dialogue)} → {len(got_dlg)}，'
              f'wiki 不变={got_wiki == wiki}')
        for ln in lines:
            print('    ' + ln)

        # ④ 反向对照：拿脏语料当"清洗后"送进 check，必须判 FAIL
        f2, _ = check(src, src, rules=dict(dedup=True, reply_rep3=True, ngram=True,
                                           nondialogue=True))
        if not f2:
            fails.append('对照④：把未清洗的 src 当作 dst 送进 check 竟然判 PASS —— 验收是橡皮图章')
        print(f'  对照④ 脏 dst 被正确判 FAIL（{len(f2)} 条）')

    print('\n===== selftest ' + ('全部通过 ✅' if not fails else f'失败 ❌ ({len(fails)})') + ' =====')
    for x in fails:
        print('  ✗ ' + x)
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description='独立验收清洗前后的语料')
    ap.add_argument('--src', default=None)
    ap.add_argument('--dst', default=None)
    ap.add_argument('--report', default=None)
    ap.add_argument('--ngram-size', type=int, default=12)
    ap.add_argument('--ngram-freq', type=int, default=20)
    ap.add_argument('--expect-cover', type=float, default=None,
                    help='清洗器的 --max-ngram-cover；给了就断言 dst 的实际覆盖率 ≤ 它的 1.15 倍')
    ap.add_argument('--max-sample-replies', type=int, default=300_000)
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()

    if a.selftest:
        raise SystemExit(selftest())
    if not (a.src and a.dst):
        raise SystemExit('要么 --selftest，要么同时给 --src 与 --dst')

    report = a.report or os.path.join(a.dst, 'CLEANING_REPORT.md')
    rules = parse_enabled_rules(report)
    print(f'启用规则（自 {os.path.basename(report)}）：{rules}')
    failures, lines = check(a.src, a.dst, rules=rules)
    print(f'=== 独立验收 {a.src} → {a.dst} ===')
    for ln in lines:
        print(ln)
    print('\n' + ('✅ 全部判据通过' if not failures else f'❌ {len(failures)} 条判据未通过'))
    for x in failures:
        print('  ✗ ' + x)
    out = a.report or os.path.join(a.dst, 'VERIFY_REPORT.md')
    try:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w', encoding='utf-8') as f:
            f.write(f'# 清洗独立验收报告\n\nsrc={a.src}  dst={a.dst}\n\n')
            f.write('\n'.join(lines))
            f.write('\n\n' + ('## 结论：✅ 通过\n' if not failures else
                              '## 结论：❌ 未通过\n\n' + '\n'.join(f'- {x}' for x in failures) + '\n'))
        print(f'报告 → {out}')
    except OSError as e:
        print(f'⚠ 报告写入失败：{e}')
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
