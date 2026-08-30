#!/usr/bin/env python3
"""数据分句器（dev-notes/49）：按句边界切分文本，保留标点，超长句二次切。

让训练数据按"句"组织，模型像人一样按完整句子阅读/生成（不硬截断句子）。
- 中文句边界：。！？…（全角）；英文：.!?;（半角句号带数字小数点保护）
- 换行保持为轮次/段落分隔，`用户：/模型：` 行不拆
- 超长句（>max_len 字）再按逗号/分号切，防整段一长句

用法（独立工具）：
    uv run python data/chinese/split_sentences.py --file data/chinese/lccc_dialogue.txt
（输出分句版到同目录 *_split.txt；prepare 集成时用 --split-sentences 开关）
"""
import argparse
import re
import sys
from pathlib import Path

# 句末标点：全角句号/问号/叹号/省略号 + 半角句号/问号/叹号
# （半角 . 要求前后非数字，保护 3.14 / U.S. 这类）
SENT_PAT = re.compile(r"[。！？…?!]|(?<!\d)\.(?!\d)")
SUB_SEP = "，,；;、"


def split_line(line: str, max_len: int = 80) -> list[str]:
    """单行内按句末标点切分（保留标点）；超长句按逗号/分号二次切。

    `用户：`/`模型：` 行若含句末标点也照切（对话内容里的完整句）。
    """
    # 1) 按句末标点切，标点归前一句（split 带捕获组 → 文本,标点,文本,标点...）
    toks = re.split(f"({SENT_PAT.pattern})", line)
    sentences = []
    buf = ""
    for i, t in enumerate(toks):
        if t == "":
            continue
        if i % 2 == 1:  # 标点位 → 拼到前一句
            buf += t
            sentences.append(buf)
            buf = ""
        else:
            buf += t
    if buf.strip():
        sentences.append(buf)
    if not sentences:
        return [line] if line.strip() else []

    # 2) 超长句按逗号/分号二次切
    out = []
    for s in sentences:
        if len(s) <= max_len:
            out.append(s)
            continue
        segs, cur = [], ""
        for ch in s:
            cur += ch
            if ch in SUB_SEP:
                segs.append(cur)
                cur = ""
        if cur:
            segs.append(cur)
        out.extend(segs if segs else [s])
    return out


def split_text(text: str, max_len: int = 80) -> str:
    """整段文本分句：按行保持轮次结构，行内分句后用换行连接。"""
    out_lines = []
    for line in text.split("\n"):
        if not line.strip():
            out_lines.append("")
            continue
        for s in split_line(line, max_len):
            out_lines.append(s)
    return "\n".join(out_lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="语料 txt（空行分隔样本）")
    ap.add_argument("--max-len", type=int, default=80, help="超长句阈值（字）")
    a = ap.parse_args()

    path = Path(a.file)
    text = path.read_text(encoding="utf-8", errors="replace")
    out = split_text(text, a.max_len)
    out_path = path.with_name(path.stem + "_split.txt")
    out_path.write_text(out, encoding="utf-8")
    before = sum(1 for b in text.split("\n\n") if b.strip())
    after = sum(1 for b in out.split("\n\n") if b.strip())
    print(f"{path.name}: {before} 样本 → {after} 样本（分句后）| 输出 {out_path.name}")


if __name__ == "__main__":
    main()
