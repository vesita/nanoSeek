#!/usr/bin/env python3
"""
将外部开源仓库转换为「自然纯文本」语料（不做「用户：/模型：」问答包装）

设计原则（对齐用户"更自然的训练角度"）：
- 诗词 → 就是诗词本身：《词牌名》作者 + 正文
- 代码 → 就是代码本身：docstring 说明 + 实现
- 题解 → 题目描述 + 题解代码（自然阅读流）

这些纯文本与百科、数学等知识数据一起，走 prepare.py --pretrain 模式
（无 loss mask、无对话结构、全 token 语言建模），作为 100M 模型的通识预训练底座。

输出（data/chinese/pretrain/）：
- poetry.txt
- leetcode.txt
- algorithms.txt
"""
import os
import re
import json
import glob
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXT = os.path.join(ROOT, "data", "external")
OUT = os.path.join(ROOT, "data", "chinese", "pretrain")
os.makedirs(OUT, exist_ok=True)


def clean(s):
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", str(s).strip())
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s


def strip_html(s):
    s = re.sub(r"<[^>]+>", "", s)
    s = s.replace("&nbsp;", " ").replace("&lt;", "<").replace("&gt;", ">") \
         .replace("&amp;", "&").replace("&quot;", '"').replace("&#39;", "'")
    return clean(s)


# ---------------------------------------------------------------------------
# 1. 诗词 → 纯文本（《词牌名》作者 + 正文）
# ---------------------------------------------------------------------------
def parse_poetry():
    files = glob.glob(os.path.join(EXT, "chinese-poetry", "**", "*.json"), recursive=True)
    out = os.path.join(OUT, "poetry.txt")
    count = 0
    with open(out, "w", encoding="utf-8") as f:
        for fp in files:
            try:
                with open(fp, encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception:
                continue
            if not isinstance(data, list):
                continue
            for item in data:
                if not isinstance(item, dict):
                    continue
                paragraphs = item.get("paragraphs")
                if not paragraphs:
                    continue
                author = clean(item.get("author", "")) or "佚名"
                rhythmic = clean(item.get("rhythmic", "")) or "诗"
                body = "\n".join(clean(p) for p in paragraphs if clean(p))
                if len(body) < 8:
                    continue
                # 自然格式：标题行 + 正文（无任何问答前缀）
                f.write(f"《{rhythmic}》{author}\n{body}\n\n")
                count += 1
    print(f"✓ 诗词纯文本: {count:,} 篇 -> {out} ({os.path.getsize(out)/1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# 2. LeetCode → 纯文本（题目描述 + 题解代码）
# ---------------------------------------------------------------------------
def parse_leetcode():
    readmes = glob.glob(os.path.join(EXT, "doocs-leetcode", "solution", "**", "README.md"), recursive=True)
    out = os.path.join(OUT, "leetcode.txt")
    count = 0
    with open(out, "w", encoding="utf-8") as f:
        for rp in readmes:
            if rp.endswith("README_EN.md"):
                continue
            try:
                with open(rp, encoding="utf-8") as fh:
                    md = fh.read()
            except Exception:
                continue

            # 题目标题：`# [1. 两数之和](链接)` → 提取「两数之和」
            title_m = re.search(r"^#\s*\[?\d+\.\s*(.+)$", md, re.MULTILINE)
            title = title_m.group(1) if title_m else "算法题"
            title = re.sub(r"\]\(.*?\)", "", title)  # 去掉 ](链接)
            title = re.sub(r"\[", "", title)         # 去掉残留 [
            title = clean(title)

            # 题目描述
            desc = ""
            m = re.search(r"##\s*题目描述\s*(.*?)(?=##\s*(解法|题解|方法)|<!--|$)", md, re.DOTALL)
            if m:
                desc = strip_html(m.group(1))
            if len(desc) < 10:
                desc = title

            # Python 题解代码
            pyfile = os.path.join(os.path.dirname(rp), "Solution.py")
            code = ""
            if os.path.exists(pyfile):
                with open(pyfile, encoding="utf-8") as fh:
                    code = clean(fh.read())

            if len(code) < 5:
                continue
            desc = desc[:800]
            code = code[:1500]
            # 自然格式：题目 + 描述 + 代码（无问答前缀）
            f.write(f"【{title}】\n{desc}\n\n解法（Python）：\n{code}\n\n")
            count += 1
    print(f"✓ LeetCode 纯文本: {count:,} 题 -> {out} ({os.path.getsize(out)/1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# 3. TheAlgorithms/Python → 纯文本（docstring 说明 + 实现）
# ---------------------------------------------------------------------------
def parse_algorithms():
    pys = glob.glob(os.path.join(EXT, "TheAlgorithms-Python", "**", "*.py"), recursive=True)
    out = os.path.join(OUT, "algorithms.txt")
    count = 0
    skip_names = {"__init__.py", "test_", "_test", "setup.py", "sol"}
    with open(out, "w", encoding="utf-8") as f:
        for fp in pys:
            base = os.path.basename(fp)
            if any(s in base for s in skip_names):
                continue
            try:
                with open(fp, encoding="utf-8") as fh:
                    code = fh.read()
            except Exception:
                continue
            code = clean(code)
            if len(code) < 40:
                continue
            # 算法名
            rel = os.path.relpath(fp, os.path.join(EXT, "TheAlgorithms-Python"))
            parts = rel.split(os.sep)
            category = parts[0] if len(parts) > 1 else "算法"
            algo_name = base.replace(".py", "").replace("_", " ").title()
            # 自然格式：分类 + 名称 + 代码
            f.write(f"# {category}：{algo_name}\n{code[:1500]}\n\n")
            count += 1
    print(f"✓ 算法纯文本: {count:,} 个 -> {out} ({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    print("=" * 60)
    print("  外部仓库 → 自然纯文本语料转换")
    print("=" * 60)
    parse_poetry()
    parse_leetcode()
    parse_algorithms()
    print("\n纯文本转换完成！")
