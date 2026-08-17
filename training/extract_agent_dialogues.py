#!/usr/bin/env python3
"""只读提取 Claude Code / DSH Harness 对话日志 → 清洗 → 中文对话语料。

设计目标（与项目现状对齐）：
- 只读访问 ~/.claude/projects 与 ~/.dsh/sessions，绝不修改源数据。
- 只保留「用户纯文本 → 模型纯文本」的自然语言轮次对，剥掉工具调用、
  代码块、结果块、系统上下文等工程噪音。
- 输出「用户：…\\n模型：…」格式的对话块，可直接作为 prepare.py 的新语料文件
  （data/*/*.txt 已被 .gitignore 排除，不会提交个人日志）。
- 输出统计 JSON，便于按阈值调参（干跑用 --no-write）。

用法（项目根目录）：
    .venv/bin/python training/extract_agent_dialogues.py            # 全量提取+清洗+写文件
    .venv/bin/python training/extract_agent_dialogues.py --no-write # 干跑：只看统计
"""
import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

# ----------------------------------------------------------------------------
# 字段/结构常量（已对实际日志逐条核实，见 dev-notes/25 采样对比之前的探查）
# ----------------------------------------------------------------------------
CLAUDE_USER_CONTENT = "message"          # type=user → message.content（str 或 list）
DSH_USER_MESSAGE = "user/message"        # data.content[].text + data.source.kind == "user"
DSH_SPLICED = "agent/inbox/spliced"      # data.inserted[]，source.kind == "user"
DSH_TEXT_CHUNKS = "text-chunks"          # data.texts 增量流，data.turn / data.index
NON_TEXT_BLOCKS = {"tool_use", "tool_result", "isUesrError", "image", "file"}

STOPWORD_USERS = {"继续", "好", "好的", "可以", "嗯", "对", "行", "ok", "okay",
                  "继续吧", "然后呢", "然后", "是的", "对的", "好呀", "接着说"}

# ----------------------------------------------------------------------------
# 清洗工具
# ----------------------------------------------------------------------------

def strip_noise(text: str) -> str:
    """去掉工程噪音：代码块、行内代码、URL、路径、控制字符，压空白。"""
    t = text
    t = re.sub(r"```.*?```", " ", t, flags=re.S)          # 代码围栏（含内容）
    t = re.sub(r"`[^`\n]{1,80}`", " ", t)                 # 行内代码
    t = re.sub(r"https?://\S+", " ", t)                   # URL
    t = re.sub(r"[Cc]:\\\\[\w.\\]+|[\\/](?:home|Users|root|tmp)[\\/][\w./-]*", " ", t)
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", t)   # 控制字符
    t = re.sub(r"\s+", " ", t).strip()
    return t


def han_ratio(text: str) -> float:
    """中文字符占比（汉字 + 中文标点，分母 = 非空白字符）。"""
    if not text:
        return 0.0
    han = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff" or ch in "，。！？；：、""''（）《》…—")
    denom = sum(1 for ch in text if not ch.isspace())
    return han / denom if denom else 0.0


def char_len(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace())


def is_degenerate(text: str) -> bool:
    """全标点 / 全同一字 / 明显非自然语言。"""
    t = re.sub(r"\s", "", text)
    if not t:
        return True
    if all(ch in "，。！？；：、,.!?;:…—-'\"()[]{}" for ch in t):
        return True
    if len(set(t)) == 1 and len(t) > 2:
        return True
    return False


# ----------------------------------------------------------------------------
# 解析器：Claude Code
# ----------------------------------------------------------------------------

def iter_claude_pairs(root: Path, limit: int):
    """产出 (user_text, assistant_text) 候选对（未过滤）。"""
    for jsonl in sorted(root.rglob("*.jsonl")):
        if limit is not None and limit <= 0:
            return
        last_user = None
        last_user_ok = False
        for line in open(jsonl, encoding="utf-8", errors="replace"):
            if limit is not None and limit <= 0:
                return
            try:
                e = json.loads(line)
            except Exception:
                continue
            t = e.get("type")
            if t == "user":
                msg = e.get(CLAUDE_USER_CONTENT) or {}
                content = msg.get("content") if isinstance(msg, dict) else None
                if isinstance(content, str):
                    last_user, last_user_ok = content, True
                elif isinstance(content, list):
                    blocks = [b for b in content if isinstance(b, dict)]
                    if any(b.get("type") in NON_TEXT_BLOCKS for b in blocks):
                        last_user_ok = False          # 带附件/工具结果，不算干净用户消息
                    else:
                        last_user = " ".join(b.get("text", "") for b in blocks
                                             if b.get("type") == "text").strip()
                        last_user_ok = bool(last_user)
                else:
                    last_user_ok = False
            elif t == "assistant":
                msg = e.get(CLAUDE_USER_CONTENT) or {}
                content = msg.get("content") if isinstance(msg, dict) else None
                if not isinstance(content, list):
                    continue
                blocks = [b for b in content if isinstance(b, dict)]
                if any(b.get("type") in NON_TEXT_BLOCKS or b.get("type") == "thinking" and False
                       for b in blocks):
                    if any(b.get("type") in NON_TEXT_BLOCKS for b in blocks):
                        # 消息里夹着工具调用 → 工作消息，轮次结束不产出
                        last_user_ok = False
                        continue
                texts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
                if not texts:
                    continue
                # 同轮多个纯文本 assistant 消息：保留最后一个（最终答复）
                candidate = " ".join(texts).strip()
                if last_user_ok and candidate:
                    yield last_user, candidate
                    if limit is not None:
                        limit -= 1
                last_user_ok = False


# ----------------------------------------------------------------------------
# 解析器：DSH Harness
# ----------------------------------------------------------------------------

def read_zstd_lines(path: Path):
    """流式读取 zstd JSONL（零额外依赖，复用系统 zstd）。"""
    p = subprocess.Popen(["zstd", "-dc", str(path)], stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True)
    for line in p.stdout:
        yield line
    p.wait()


def iter_dsh_pairs(root: Path, limit: int):
    """按顺序 zip 用户文本与助手文本块（text-chunks 按 (turn,index) 累加）。"""
    users = []
    assts = []          # list[ (turn, text) ]
    block_buf = {}      # (turn, index) -> 累加文本
    block_order = []    # 首次出现的 (turn, index) 顺序
    for zstd in sorted(root.rglob("session.jsonl.zstd")):
        if limit is not None and limit <= 0:
            break
        last_turn = None
        for line in read_zstd_lines(zstd):
            if limit is not None and limit <= 0:
                break
            try:
                e = json.loads(line)
            except Exception:
                continue
            t = e.get("type")
            if t == DSH_USER_MESSAGE:
                d = e.get("data") or {}
                if (d.get("source") or {}).get("kind") == "user":
                    texts = [b.get("text", "") for b in (d.get("content") or [])
                             if b.get("type") == "text"]
                    if texts:
                        users.append(" ".join(texts).strip())
            elif t == DSH_SPLICED:
                for ins in (e.get("data") or {}).get("inserted", []):
                    if (ins.get("source") or {}).get("kind") == "user":
                        texts = [b.get("text", "") for b in (ins.get("content") or [])
                                 if b.get("type") == "text"]
                        if texts:
                            users.append(" ".join(texts).strip())
            elif t == DSH_TEXT_CHUNKS:
                d = e.get("data") or {}
                turn, idx = d.get("turn"), d.get("index")
                frag = "".join(d.get("texts") or [])
                if turn is None or not frag:
                    continue
                key = (turn, idx)
                if key not in block_buf:
                    block_buf[key] = ""
                    block_order.append(key)
                block_buf[key] += frag
                last_turn = turn
            elif t == "turn/end":
                # 一轮结束：把该轮已出现的块按出现顺序拼成一条助手回复
                if last_turn is None:
                    continue
                keys = [k for k in block_order if k[0] == last_turn]
                if keys:
                    text = "".join(block_buf[k] for k in keys).strip()
                    if text:
                        assts.append((last_turn, text))
                block_order = [k for k in block_order if k[0] != last_turn]
                last_turn = None
        # 文件结束：兜底 flush 剩余块（部分会话缺 turn/end 事件）
        if block_order:
            by_turn = {}
            for k in block_order:
                by_turn.setdefault(k[0], []).append(k)
            for turn, keys in by_turn.items():
                text = "".join(block_buf[k] for k in keys).strip()
                if text:
                    assts.append((turn, text))
            block_buf.clear()
            block_order = []
    # 按顺序 zip 用户与助手（两者数量通常一致；取共同前缀）
    n = min(len(users), len(assts))
    for i in range(n):
        yield users[i], assts[i][1]
        if limit is not None:
            limit -= 1


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------

def clean_pairs(pairs, args, stats):
    out = []
    seen = set()
    for user, asst in pairs:
        if not user or not asst:
            continue
        user = strip_noise(user)
        asst = strip_noise(asst)
        stats["candidates"]["after_strip"] += 1
        if len(user) < args.min_user or len(asst) < args.min_assistant:
            stats["drop"]["too_short"] += 1
            continue
        if user in STOPWORD_USERS or (len(user) <= 3 and "继续" in user):
            stats["drop"]["stopword_user"] += 1
            continue
        if char_len(user) > args.max_user or char_len(asst) > args.max_assistant \
                or char_len(user) + char_len(asst) > args.max_combined:
            stats["drop"]["too_long"] += 1
            continue
        ratio = han_ratio(user + " " + asst)
        if ratio < args.han_ratio:
            stats["drop"]["low_han"] += 1
            continue
        if is_degenerate(user) or is_degenerate(asst):
            stats["drop"]["degenerate"] += 1
            continue
        if asst == user or (len(user) >= 6 and user in asst):
            stats["drop"]["echo"] += 1
            continue
        key = (user, asst)
        if key in seen:
            stats["drop"]["dup"] += 1
            continue
        seen.add(key)
        out.append((user, asst))
        stats["kept"] += 1
        if args.limit and len(out) >= args.limit:
            break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--claude-dirs", nargs="*",
                    default=["~/.claude/projects"], help="Claude Code 日志目录（只读）")
    ap.add_argument("--dsh-dirs", nargs="*", default=["~/.dsh/sessions"],
                    help="DSH Harness 会话目录（只读）")
    ap.add_argument("--out", default="data/chinese/agent_dialogue.txt",
                    help="清洗后语料输出路径（data/*/*.txt 已被 gitignore）")
    ap.add_argument("--stats-out", default="out/agent_extract_stats.json")
    ap.add_argument("--no-write", action="store_true", help="干跑：只统计不写文件")
    ap.add_argument("--min-user", type=int, default=3)
    ap.add_argument("--min-assistant", type=int, default=6)
    ap.add_argument("--max-user", type=int, default=200)
    ap.add_argument("--max-assistant", type=int, default=400)
    ap.add_argument("--max-combined", type=int, default=440)
    ap.add_argument("--han-ratio", type=float, default=0.5, help="合并文本中文占比阈值")
    ap.add_argument("--limit", type=int, default=0, help="0=全量")
    ap.add_argument("--preview", type=int, default=8, help="终端预览前 N 条")
    args = ap.parse_args()

    stats = {
        "sources": {"claude": [], "dsh": []},
        "candidates": Counter(),
        "drop": Counter(),
        "kept": 0,
        "raw_pairs": [],
        "pairs": [],   # 只存预览
    }

    for cd in args.claude_dirs:
        root = Path(cd).expanduser()
        if not root.is_dir():
            print(f"⚠ 跳过（不存在）: {root}")
            continue
        jsons = list(root.rglob("*.jsonl"))
        stats["sources"]["claude"] = [str(p) for p in jsons]
        for u, a in iter_claude_pairs(root, None):
            stats["candidates"]["claude_raw"] += 1
            stats["raw_pairs"].append((u, a))
            if args.limit and len(stats["raw_pairs"]) >= args.limit:
                break

    for dd in args.dsh_dirs:
        root = Path(dd).expanduser()
        if not root.is_dir():
            print(f"⚠ 跳过（不存在）: {root}")
            continue
        zstds = list(root.rglob("session.jsonl.zstd"))
        stats["sources"]["dsh"] = [str(p) for p in zstds]
        for u, a in iter_dsh_pairs(root, None):
            stats["candidates"]["dsh_raw"] += 1
            stats["raw_pairs"].append((u, a))
            if args.limit and len(stats["raw_pairs"]) >= args.limit:
                break

    # ---- 统一清洗（一次性过滤 + 全局去重） ----
    pairs = clean_pairs(stats["raw_pairs"], args, stats)
    stats["pairs"] = pairs
    print("=" * 60)
    print("提取统计（只读，源数据未改动）")
    print(f"  Claude Code 文件: {len(stats['sources']['claude'])} · "
          f"DSH 会话: {len(stats['sources']['dsh'])}")
    print(f"  候选轮次: Claude raw {stats['candidates']['claude_raw']} · "
          f"DSH raw {stats['candidates']['dsh_raw']}")
    print(f"  清洗后保留: {len(pairs)} 条轮次对")
    if stats["drop"]:
        print("  丢弃原因:", dict(stats["drop"]))
    total_chars = sum(char_len(u) + char_len(a) for u, a in pairs)
    print(f"  总字符(去空白): {total_chars:,} · 平均每对 {total_chars/max(len(pairs),1):.0f} 字符")
    users = [u for u, _ in pairs]
    print(f"  用户轮次平均 {sum(char_len(u) for u in users)/max(len(users),1):.0f} 字符")

    if not args.no_write and pairs:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write("".join(f"用户：{u}\n模型：{a}\n\n" for u, a in pairs))
        print(f"✅ 已写入 {out}（{len(pairs)} 条，{out.stat().st_size//1024} KB）")
    elif args.no_write:
        print("（干跑模式：未写数据文件）")

    if args.preview and pairs:
        print("\n预览（前 %d 条）：" % args.preview)
        for u, a in pairs[:args.preview]:
            print(f"  用户：{u[:60]}")
            print(f"  模型：{a[:60]}")
            print()

    stats_out = Path(args.stats_out)
    stats_out.parent.mkdir(parents=True, exist_ok=True)
    stats_out.write_text(json.dumps({
        "sources": stats["sources"],
        "candidates": dict(stats["candidates"]),
        "drop": dict(stats["drop"]),
        "kept": len(pairs),
        "preview": [{"user": u[:120], "assistant": a[:120]} for u, a in pairs[:10]],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"统计已写入 {stats_out}")


if __name__ == "__main__":
    main()