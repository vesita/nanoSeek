#!/usr/bin/env python3
"""
将克隆的外部开源仓库转换为「用户：/模型：」训练语料

输入（data/external/）：
1. chinese-poetry     : 诗词正文（author + rhythmic + paragraphs）
2. doocs/leetcode     : 中文题解（README.md + Solution.py）
3. TheAlgorithms/Python : 算法实现（.py 文件含 docstring）

输出（data/chinese/）：
- poetry_dialogue.txt
- leetcode_dialogue.txt
- algorithms_dialogue.txt

统一规范：空行分隔，「用户：/模型：」格式，适配四区稀疏字级词表。
"""
import os
import re
import json
import glob
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXT = os.path.join(ROOT, "data", "external")
OUT = os.path.join(ROOT, "data", "chinese")


def clean(s):
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", str(s).strip())
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s


def strip_html(s):
    """去除 HTML 标签，保留纯文本。"""
    s = re.sub(r"<[^>]+>", "", s)
    s = s.replace("&nbsp;", " ").replace("&lt;", "<").replace("&gt;", ">") \
         .replace("&amp;", "&").replace("&quot;", '"').replace("&#39;", "'")
    return clean(s)


# ---------------------------------------------------------------------------
# 1. chinese-poetry → 诗词语料
# ---------------------------------------------------------------------------
def parse_poetry():
    files = glob.glob(os.path.join(EXT, "chinese-poetry", "**", "*.json"), recursive=True)
    out = os.path.join(OUT, "poetry_dialogue.txt")
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
                # 包装成问答：请欣赏这首作品
                f.write(f"用户：请欣赏这首《{rhythmic}》，并告诉我它的作者。\n"
                        f"模型：《{rhythmic}》——{author}\n{body}\n\n")
                count += 1
    print(f"✓ 诗词语料: {count:,} 条 -> {out} ({os.path.getsize(out)/1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# 2. doocs/leetcode → 中文题解语料
# ---------------------------------------------------------------------------
def parse_leetcode():
    readmes = glob.glob(os.path.join(EXT, "doocs-leetcode", "solution", "**", "README.md"), recursive=True)
    out = os.path.join(OUT, "leetcode_dialogue.txt")
    count = 0
    for rp in readmes:
        # 跳过英文版
        if rp.endswith("README_EN.md"):
            continue
        try:
            with open(rp, encoding="utf-8") as fh:
                md = fh.read()
        except Exception:
            continue

        # 提取题目标题
        title_m = re.search(r"^#\s*\[?(\d+)\.\s*(.+?)\]?\s*$", md, re.MULTILINE)
        if not title_m:
            title_m = re.search(r"^#\s*(.+)$", md, re.MULTILINE)
        title = clean(title_m.group(2) if title_m and title_m.lastindex and title_m.lastindex >= 2
                      else (title_m.group(1) if title_m else "算法题"))

        # 提取题目描述（## 题目描述 到 ## 解法/题解 之间）
        desc = ""
        m = re.search(r"##\s*题目描述\s*(.*?)(?=##\s*(解法|题解|方法)|<!--|$)", md, re.DOTALL)
        if m:
            desc = strip_html(m.group(1))
        if len(desc) < 10:
            # 退化：用标题作描述
            desc = title

        # 提取 Python 代码（同目录 Solution.py）
        pydir = os.path.dirname(rp)
        pyfile = os.path.join(pydir, "Solution.py")
        code = ""
        if os.path.exists(pyfile):
            with open(pyfile, encoding="utf-8") as fh:
                code = clean(fh.read())

        # 截断过长的描述/代码，控制单条长度
        desc = desc[:800]
        code = code[:1500]

        user = f"请解决这道算法题：{title}\n{desc}"
        model = f"解题思路与代码实现如下：\n\n```python\n{code}\n```"
        if len(code) < 5:
            continue
        with open(out, "a", encoding="utf-8") as f:
            f.write(f"用户：{user}\n模型：{model}\n\n")
        count += 1
    print(f"✓ LeetCode 题解语料: {count:,} 条 -> {out} ({os.path.getsize(out)/1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# 3. TheAlgorithms/Python → 算法实现语料
# ---------------------------------------------------------------------------
def parse_algorithms():
    pys = glob.glob(os.path.join(EXT, "TheAlgorithms-Python", "**", "*.py"), recursive=True)
    out = os.path.join(OUT, "algorithms_dialogue.txt")
    count = 0
    skip_names = {"__init__.py", "test_", "_test", "sol", "setup.py"}
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
            # 提取算法名（目录名 + 文件名）
            rel = os.path.relpath(fp, os.path.join(EXT, "TheAlgorithms-Python"))
            parts = rel.split(os.sep)
            category = parts[0] if len(parts) > 1 else "算法"
            algo_name = base.replace(".py", "").replace("_", " ").title()
            # 提取 docstring 作思路说明
            doc_m = re.search(r'"""(.*?)"""', code, re.DOTALL)
            doc = clean(doc_m.group(1))[:300] if doc_m else f"{category} 类算法实现"

            user = f"请用 Python 实现【{algo_name}】算法（{category}）。"
            model = f"【{algo_name}】实现如下：\n\n```python\n{code[:1200]}\n```\n\n说明：{doc}"
            f.write(f"用户：{user}\n模型：{model}\n\n")
            count += 1
    print(f"✓ 算法实现语料: {count:,} 条 -> {out} ({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    print("=" * 60)
    print("  外部开源仓库 → 训练语料转换")
    print("=" * 60)
    parse_poetry()
    parse_leetcode()
    parse_algorithms()
    print("\n全部转换完成！")
