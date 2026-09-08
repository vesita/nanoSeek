#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nanoSeek 数据诊断（v1）
======================

防「数据占比分布异常导致模型背模板而非真学会语言」的预防性诊断。
本次复盘触发点：100M char-level 预训练在 val 2.19 时，三个不同 prompt 都
17~19 token 就吐 <eos>、输出只是 agent_tools_dialogue.txt 里几条高频模板的复读。
脚本要量化的就是这类隐患。

算法栈（5 类诊断，全部基于纯文本统计，不需要 GPU / 模型）：

1. 体量行数    -- 每文件大小、行数、字符数
2. 长度分布    -- 每条样本的字符长度：均/中/p10/p50/p90/p99/max
                   警惕：超长样本塞爆 block_size；超短样本没梯度信号
3. 文本指纹聚类 -- 按 text fingerprint（去掉所有非空白字符后剩余的「空白模式」
                   加小写 + 去除首尾空白 + 折叠连续空白）做相似度分桶
                   这是业内最便宜的「模板检测」算法，能把「正在调用网络搜索工具检索权威资料...」
                   这类几乎一字不差的固定模板自动聚合
                   警惕：少数指纹占比 ≥5% 几乎一定是模板污染
4. 模板 Top-N   -- 列出 Top-20 高频指纹 + 占比 + 出现次数
5. token 词表覆盖 -- 用项目自带的 char_tokenizer 编码全集合，统计
                   词表里有多少 token 在语料里出现 / 出现 ≥3 次的占比
                   警惕：覆盖率 < 30% 说明数据与词表不匹配；高频 token 集中度
                   高（前 1% token 占全部词频 ≥50%）说明分布偏斜

用法：
    .venv/bin/python training/data_diagnostics.py data/chinese/*.txt
    .venv/bin/python training/data_diagnostics.py --json out/data_diag.json data/chinese/*.txt
    .venv/bin/python training/data_diagnostics.py --top 30 data/chinese/*.txt
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

# ── 可选：项目自带 char tokenizer（覆盖度统计需要）──────────────────────
try:
    from tokenizers import Tokenizer
    _HAS_TOK = True
except ImportError:
    _HAS_TOK = False

# ── 启发式常量 ───────────────────────────────────────────────────────────
_NORM_NUM = re.compile(r"\d+")
_NORM_WS  = re.compile(r"\s+")
_FINGERPRINT_NORM = {
    "，": ",", "。": ".", "：": ":", "；": ";",
    "！": "!", "？": "?", "（": "(", "）": ")",
    "【": "[", "】": "]", "「": '"', "」": '"',
    "、": ",", "—": "-",
}
_TRANSLATE = str.maketrans({k: v for k, v in _FINGERPRINT_NORM.items()})

# 模板污染阈值：单条指纹占比超过此值就报警
TEMPLATE_ALERT_RATIO = 0.02     # 2%
TEMPLATE_ALERT_ABS   = 50       # 且至少出现 50 次（避免小样本误报）

# 词表覆盖度阈值：低于此报警
COVERAGE_ALERT = 0.30           # 30% 的词表 token 至少出现 1 次
HIGHFREQ_CONCENTRATION_ALERT = 0.50   # top1% token 占全部词频 ≥ 50% 视为偏斜


# ── 工具函数 ─────────────────────────────────────────────────────────────
def fingerprint(text: str) -> str:
    """文本指纹：把几乎一字不差的「模板」聚合到同一桶。

    例：
      "正在调用网络搜索工具检索权威资料..."     ⇄
      "正在调用网络搜索工具检索权威资料..."      ← 同一个指纹
    数字会被折叠成 <D>，所以「参数=42」与「参数=43」是同一指纹。
    """
    t = unicodedata.normalize("NFKC", text).translate(_TRANSLATE)
    t = _NORM_NUM.sub("<D>", t)
    t = _NORM_WS.sub(" ", t).strip().lower()
    return t


# ── 单文件诊断 ───────────────────────────────────────────────────────────
def diag_one_file(path: Path) -> dict:
    """对单个文件做 5 类诊断，返回 dict。"""
    text = path.read_text(encoding="utf-8", errors="replace")
    n_chars = len(text)
    # 启发式行分割：agent_tools_dialogue 用「用户：」开头分轮，安全做法是按 \n +
    # 用户/模型 标记切，其它文件大多是 Q/A 或一行一样本也能正确切。
    raw_chunks = re.split(r"\n(?=用户[:：]|User:|Human:|模型[:：]|Model:|A:)", text)
    chunks = [c.strip() for c in raw_chunks if c.strip()]

    # 2) 长度分布
    lens = [len(c) for c in chunks]
    lens_sorted = sorted(lens)

    def pct(p):
        if not lens_sorted:
            return 0
        i = min(len(lens_sorted) - 1, int(len(lens_sorted) * p))
        return lens_sorted[i]

    length_stats = {
        "n_samples": len(chunks),
        "char_total": n_chars,
        "mean": round(sum(lens) / max(len(lens), 1), 1),
        "p10": pct(0.10), "p50": pct(0.50), "p90": pct(0.90), "p99": pct(0.99),
        "max": max(lens) if lens else 0,
    }

    # 3) 指纹聚类
    fp_counter: Counter[str] = Counter()
    fp_examples: dict[str, str] = {}
    for c in chunks:
        fp = fingerprint(c[:200])       # 只取每条前 200 字符算指纹（抗超长偏置）
        fp_counter[fp] += 1
        if fp not in fp_examples:
            fp_examples[fp] = c[:80].replace("\n", "⏎")

    n = max(len(chunks), 1)
    sorted_fp = fp_counter.most_common()

    # 4) 模板 Top-N + 报警
    top_n = []
    template_alerts = []
    for fp, cnt in sorted_fp[:30]:
        ratio = cnt / n
        rec = dict(rank=len(top_n) + 1, count=cnt, ratio=round(ratio, 4),
                   example=fp_examples[fp][:60])
        top_n.append(rec)
        if ratio >= TEMPLATE_ALERT_RATIO and cnt >= TEMPLATE_ALERT_ABS:
            template_alerts.append(rec)

    n_unique_fp = len(fp_counter)
    dedup_ratio = n_unique_fp / n if n else 0

    # 5) 词表覆盖度（若可用）
    coverage = None
    if _HAS_TOK and Path("data/chinese/char_tokenizer.json").exists():
        tok = Tokenizer.from_file("data/chinese/char_tokenizer.json")
        vocab_size = tok.get_vocab_size()
        seen = Counter()
        total_tokens = 0
        for c in chunks[:5000]:           # 上限 5000 条防止太慢
            ids = tok.encode(c[:512]).ids
            seen.update(ids)
            total_tokens += len(ids)
        n_seen = sum(1 for v in seen.values() if v >= 1)
        n_seen3 = sum(1 for v in seen.values() if v >= 3)
        if seen:
            freq_sorted = sorted(seen.values(), reverse=True)
            cutoff = max(1, len(freq_sorted) // 100)
            top1pct_share = sum(freq_sorted[:cutoff]) / max(sum(freq_sorted), 1)
        else:
            top1pct_share = 0.0
        coverage = dict(
            vocab_size=vocab_size,
            seen_at_least_1=n_seen,
            seen_at_least_3=n_seen3,
            coverage_ratio=round(n_seen / vocab_size, 4),
            coverage_at_3_ratio=round(n_seen3 / vocab_size, 4),
            top1pct_concentration=round(top1pct_share, 4),
            sample_tokens=total_tokens,
            alert_coverage=n_seen / vocab_size < COVERAGE_ALERT,
            alert_concentration=top1pct_share >= HIGHFREQ_CONCENTRATION_ALERT,
        )

    return dict(
        path=str(path),
        size_kb=round(path.stat().st_size / 1024, 1),
        n_samples=length_stats["n_samples"],
        length_stats=length_stats,
        n_unique_fingerprints=n_unique_fp,
        dedup_ratio=round(dedup_ratio, 4),
        top_fingerprints=top_n,
        template_alerts=template_alerts,
        coverage=coverage,
    )


# ── 主函数 ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="要诊断的对话 txt 文件（可 glob）")
    ap.add_argument("--top", type=int, default=20, help="每文件 Top-N 模板（默认 20）")
    ap.add_argument("--json", default=None, help="把完整结果写到 JSON 文件")
    args = ap.parse_args()

    all_diag = []
    grand_total_samples = 0
    grand_total_chars = 0
    for f in args.files:
        p = Path(f)
        if not p.exists():
            print(f"⚠️  跳过不存在的文件: {f}", file=sys.stderr)
            continue
        d = diag_one_file(p)
        all_diag.append(d)
        grand_total_samples += d["n_samples"]
        grand_total_chars += d["length_stats"]["char_total"]

    # ── 汇总 ────────────────────────────────────────────────────────────
    print("=" * 78)
    print(f" nanoSeek 数据诊断 · 文件数 {len(all_diag)} · 总样本 {grand_total_samples:,} · "
          f"总字符 {grand_total_chars/1e6:.2f}M")
    print("=" * 78)

    any_alert = False
    for d in all_diag:
        alerts = []
        if d["template_alerts"]:
            alerts.append(f"模板污染: {len(d['template_alerts'])} 条指纹占比 ≥ {TEMPLATE_ALERT_RATIO*100:.0f}%")
        if d["coverage"]:
            if d["coverage"]["alert_coverage"]:
                alerts.append(f"词表覆盖率低: {d['coverage']['coverage_ratio']*100:.1f}% < {COVERAGE_ALERT*100:.0f}%")
            if d["coverage"]["alert_concentration"]:
                alerts.append(f"高频 token 偏斜: top1% 占 {d['coverage']['top1pct_concentration']*100:.0f}% ≥ {HIGHFREQ_CONCENTRATION_ALERT*100:.0f}%")

        head = "🚨" if alerts else "✅"
        if alerts:
            any_alert = True
        print(f"\n{head} {d['path']}  ({d['size_kb']:.0f} KB)")
        ls = d["length_stats"]
        if ls["n_samples"] == 0:
            print("   ⚠️  空文件（0 样本）— 若非占位文件，需排查上游生成是否失败")
            any_alert = True
            continue
        print(f"   样本 {ls['n_samples']:,}  均长 {ls['mean']:.0f}  "
              f"p10/p50/p90/p99/max = {ls['p10']}/{ls['p50']}/{ls['p90']}/{ls['p99']}/{ls['max']}")
        print(f"   唯一指纹 {d['n_unique_fingerprints']:,}  去重率 {d['dedup_ratio']*100:.1f}%")
        if d["coverage"]:
            c = d["coverage"]
            print(f"   词表覆盖 {c['coverage_ratio']*100:.1f}% (≥3次 {c['coverage_at_3_ratio']*100:.1f}%)  "
                  f"top1% 集中度 {c['top1pct_concentration']*100:.0f}%")
        if d["template_alerts"]:
            print(f"   ⚠️  模板污染 Top {min(5, len(d['template_alerts']))}:")
            for a in d["template_alerts"][:5]:
                print(f"      {a['ratio']*100:.2f}% ({a['count']}×)  e.g. {a['example']!r}")
        else:
            print("   模板 Top-3:  ", end="")
            for t in d["top_fingerprints"][:3]:
                print(f"[{t['ratio']*100:.1f}%] ", end="")
            print()
        if alerts:
            print("   ⚠️  报警: " + " | ".join(alerts))

    print("\n" + "=" * 78)
    if any_alert:
        print("🚨 总评: 发现数据分布隐患，建议检查模板占比与词表匹配")
    else:
        print("✅ 总评: 数据分布正常，未发现显著模板污染或偏斜")
    print("=" * 78)

    if args.json:
        Path(args.json).write_text(
            json.dumps(dict(summary=dict(files=len(all_diag),
                                        total_samples=grand_total_samples,
                                        total_chars=grand_total_chars),
                            per_file=all_diag),
                       ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"\nJSON 写入: {args.json}")


if __name__ == "__main__":
    main()
