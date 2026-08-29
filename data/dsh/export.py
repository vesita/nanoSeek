#!/usr/bin/env python3
"""从本地 DSH 会话导出对话语料（训练数据）。

DSH 把每次会话存成 ~/.dsh/sessions/<workspace>/<session-id>/session.jsonl.zstd
（zstd 压缩的 JSONL，记录带 type 字段）。本脚本抽取其中的真实对话：

    user/message       → data.content[].text            （用户完整消息）
    assistant/message  → data.message.content[].text    （助手完整消息，只取 text 块）

过滤规则：
  - 只保留 type=="text" 的内容块：丢弃 reasoning（思维链）/ tool-call / tool/result；
  - 丢弃压缩摘要（compaction）注入的会话历史（内容含 checkpoint 样板）；
  - 单条消息限长（--max-chars 截断）、交换太短或只有一方的丢弃；
  - 全局去重（同一 (用户,助手) 对只留一次）。

输出为数据管线惯用的对话 txt（与 data/chinese/*_dialogue.txt 同格式）：
    用户：<问题>
    模型：<回答>
    （空行分隔）

用法（从项目根目录）：
    uv run python data/dsh/export.py
    # --sessions=~/.dsh/sessions 扫描根；--workspace=nanoSeek 只导出某个工作区
    # --max-chars=4000 单条消息最大字符数；--min-total=40 交换最小总长度
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys

CHECKPOINT_MARKERS = (
    "This is an automatically generated checkpoint",
    "Current runtime context",
    "This snapshot supersedes",
    "<compacted-summary>",
    "compacted-summary",
    "DeepSeek Harness",
    "<system-reminder>",
    "<available_skills>",
    "The following skills are available",
)


def iter_records(zstd_path):
    """逐条解出 JSONL 记录（yield dict）。"""
    raw = subprocess.run(["zstdcat", zstd_path], capture_output=True, check=True).stdout
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def text_of(blocks):
    """从 content 块列表里拼出纯文本（只取 type=='text'）。"""
    parts = []
    if isinstance(blocks, list):
        for b in blocks:
            if isinstance(b, dict) and b.get("type") == "text":
                t = b.get("text") or b.get("content") or ""
                if t:
                    parts.append(t)
    return "\n".join(parts).strip()


def is_boilerplate(text):
    return any(m in text for m in CHECKPOINT_MARKERS)


def export_one(zstd_path, exchanges, max_chars, min_total):
    """把一个会话文件追加成 (user, assistant) 交换列表。"""
    pending = []          # 当前用户消息后累积的助手文本
    cur_user = None
    for r in iter_records(zstd_path):
        t = r.get("type")
        if t == "user/message":
            d = r.get("data", {})
            if d.get("role") != "user":
                continue
            text = text_of(d.get("content"))
            if not text or is_boilerplate(text):
                # 系统注入/压缩摘要：丢弃并重置当前累积（后面用户消息重新起头）
                cur_user = None
                pending = []
                continue
            # 上一个用户消息 + 累积助手文本 → 收成一个交换
            if cur_user is not None:
                _emit(exchanges, cur_user, pending, max_chars, min_total)
            cur_user = text[:max_chars]
            pending = []
        elif t == "assistant/message":
            d = r.get("data", {})
            msg = d.get("message", {})
            if msg.get("role") != "assistant":
                continue
            text = text_of(msg.get("content"))
            if text and cur_user is not None:
                pending.append(text[:max_chars])
    if cur_user is not None:
        _emit(exchanges, cur_user, pending, max_chars, min_total)


def _emit(exchanges, user, pending, max_chars, min_total):
    if not pending:
        return
    asst = "\n".join(pending).strip()
    if len(user) + len(asst) < min_total:
        return
    exchanges.append((user, asst))


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sessions", default=os.path.expanduser("~/.dsh/sessions"))
    ap.add_argument("--workspace", default=None,
                    help="只导出指定工作区目录名（如 nanoSeek）；默认全部")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "dialogue.txt"))
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--min-total", type=int, default=40)
    ap.add_argument("--max-sessions", type=int, default=0, help="限制扫描的会话数（调试用）")
    args = ap.parse_args(argv)

    pattern = os.path.join(args.sessions, "*", "*", "session.jsonl.zstd")
    files = sorted(glob.glob(pattern))
    if args.workspace:
        files = [f for f in files if args.workspace in f]
    if args.max_sessions:
        files = files[:args.max_sessions]
    if not files:
        sys.exit(f"未找到会话文件：{pattern}")

    exchanges = []
    n_user, n_asst, n_bytes = 0, 0, 0
    for i, f in enumerate(files, 1):
        before = len(exchanges)
        export_one(f, exchanges, args.max_chars, args.min_total)
        n_user += sum(1 for _, a in exchanges[before:] if _)
        n_asst += sum(1 for _, a in exchanges[before:] if a)
        n_bytes += os.path.getsize(f)
        print(f"[{i}/{len(files)}] {os.path.relpath(f, args.sessions)}  → +{len(exchanges) - before} 交换")

    # 去重
    seen = set()
    dedup = []
    for u, a in exchanges:
        h = hashlib.md5((u + "\x00" + a).encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            dedup.append((u, a))
    exchanges = dedup

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(f"# dsh 会话导出语料 · {len(exchanges)} 交换 · {n_bytes/1e6:.1f}MB 压缩会话\n")
        for u, a in exchanges:
            fh.write(f"用户：{u}\n模型：{a}\n\n")
    chars = sum(len(u) + len(a) for u, a in exchanges)
    print(f"\n✓ 导出 {len(exchanges)} 个交换（{chars:,} 字符）→ {args.out}")
    print(f"  （用户消息 {n_user} 条 / 助手消息 {n_asst} 条，去重后 {len(exchanges)}）")


if __name__ == "__main__":
    main(sys.argv[1:])
