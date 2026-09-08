# -*- coding: utf-8 -*-
"""清洗用户新下载的中文语料，输出统一 .txt（供 prepare.py --char-level 重新编码）。

输入（HuggingFace Hub 缓存，用户已下载）：
  * 0xDing/wikipedia-cn-20230720-filtered      JSON 数组，字段 completion -> wikipedia_cn.txt
  * allenai/c4 multilingual c4-zh              json.gz 行，字段 text   -> c4_zh.txt

输出在 data/chinese/ 下，格式与现有 *_dialogue.txt 一致：空行分隔的 block。
各 block 内部为正文（多行合并），不做 用户：/模型： 解析（准备脚本按空行切样本即可）。

清洗规则（通用）：
  * NFKC 归一化（全/半角、兼容字符收敛，降低 OOV）
  * 去 HTML 标签 / 维基标记 / 参考文献角标
  * 去除样板（参考文献、外部链接、参见、外部链接模板等标题）
  * 折叠连续空白、去首尾空白
  * 去空块、去 < 20 字符的短块
  * 全局逐字面去重（跨文件游走集合）

c4-zh 额外：
  * 过滤博彩/导航/注册类垃圾（关键词 + 中文占比门槛 + 去站名样板）
  * 降采样到约与维基百科同量级（C4_TARGET_CHARS），避免高噪声淹没语料
"""
import os, re, json, gzip, sys, unicodedata, hashlib

WIKI_JSON = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--0xDing--wikipedia-cn-20230720-filtered/"
    "snapshots/4cef256a3f426ae1d3f6930c8cd59a32d785d99d/wikipedia-cn-20230720-filtered.json")
C4_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--allenai--c4/snapshots/"
    "1588ec454efa1a09f29cd18ddd04fe05fc8653a2/multilingual/")

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
WIKI_OUT = os.path.join(OUT_DIR, "wikipedia_cn.txt")
C4_OUT = os.path.join(OUT_DIR, "c4_zh.txt")
DS_R1_OUT = os.path.join(OUT_DIR, "deepseek_r1_distill_dialogue.txt")
QWEN3_OUT = os.path.join(OUT_DIR, "qwen3_235b_distill_dialogue.txt")

# ---- 魔搭（ModelScope）指令蒸馏数据集 ----
DS_R1_SEARCH = os.path.expanduser(
    "~/.cache/modelscope/hub/datasets/liucong___chinese-deep_seek-r1-distill-data-110k-sft/"
    "default-*/0.0.0/master/*.arrow")
QWEN3_SEARCH = os.path.expanduser(
    "~/.cache/modelscope/hub/datasets/swift___chinese-qwen3-235_b-2507-distill-data-110k-sft/"
    "default-*/0.0.0/master/*.arrow")

MIN_BLOCK = 20
C4_TARGET_CHARS = 191_000_000   # 约与维基百科同量级

# ---- 清洗助手 ----
TAG_RE = re.compile(r"<[^>]+>")
REF_RE = re.compile(r"\[[0-9]+\]")
WIKI_LINK = re.compile(r"\[\[([^\]|]*\|)?([^\]]+)\]\]")
WIKI_TMPL = re.compile(r"\{\{[^{}]*\}\}")
WIKI_MARK = re.compile(r"\'\'\'|\'\'|:{2,}")

# 博彩/导航/垃圾关键词（c4-zh）
SPAM_TOKENS = [
    "电玩城", "博彩", "彩票", "时时彩", "棋牌", "娱乐城", "注册送", "开户送",
    "百家乐", "龙虎", "牛牛", "快三", "六合彩", "赌", "网址导航", "广告",
    "推广", "代理", "加微信", "加qq", "加QQ", "微信号", "返水", "提现",
    "皮肤", "攻略", "排行榜", "最新网址", "手机版", "官网", "首页",
]

# 样板小节标题（维基百科，出现即视为其后是参考/导航区，跳过该块）
BOILERPLATE_HDR = [
    "参考", "参考文献", "外部链接", "参见", "注释", "注记", "来源", "参考资料",
    "脚注", "网址", "扩展阅读", "相关条目",
]

def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = TAG_RE.sub("", s)
    s = REF_RE.sub("", s)
    s = WIKI_LINK.sub(lambda m: m.group(2), s)
    s = WIKI_TMPL.sub("", s)
    s = WIKI_MARK.sub("", s)
    s = re.sub(r"[ \t\u3000]+", " ", s)          # 水平空白折叠
    s = re.sub(r"\n{2,}", "\n", s)                # 折行
    s = re.sub(r"[ \t]+(\n)", r"\1", s)
    return s.strip()

def cn_ratio(s: str) -> float:
    if not s:
        return 0.0
    n = sum(1 for c in s if "\u4e00" <= c <= "\u9fff")
    return n / len(s)

def is_spam(s: str) -> bool:
    low = s.lower()
    head = s[:80].lower()
    if cn_ratio(s) < 0.45:
        return True
    if re.search(r"https?:|www\.", low):
        return True
    for t in SPAM_TOKENS:
        if t in head:
            return True
    return False

def block_ok(s: str) -> bool:
    if s is None:
        return False
    if len(s) < MIN_BLOCK:
        return False
    if cn_ratio(s) < 0.45:
        return False
    for h in BOILERPLATE_HDR:
        if re.match(rf"^{h}\s*[:：]?\s*$", s[:20]):
            return False
    return True


def clean_wikipedia():
    print("=== 清洗维基百科 ===")
    with open(WIKI_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    blocks = []
    for d in data:
        txt = d.get("completion", "")
        c = norm(txt)
        if not block_ok(c):
            continue
        # 单条目内多行合并为一行（更贴近正文序列）
        c = c.replace("\n", "")
        blocks.append(c)
    print(f"  原始条目 {len(data):,} -> 通过 {len(blocks):,}")
    return blocks


def clean_c4():
    print("=== 清洗 c4-zh ===")
    files = sorted(
        f for f in os.listdir(C4_DIR)
        if re.match(r"c4-zh\.tfrecord-\d+-of-01024\.json\.gz$", f))
    print(f"  c4 分片: {len(files)}")
    blocks = []
    for fn in files:
        p = os.path.join(C4_DIR, fn)
        try:
            with gzip.open(p, "rt", encoding="utf-8", errors="replace") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    t = obj.get("text", "")
                    if not t:
                        continue
                    if is_spam(t):
                        continue
                    c = norm(t)
                    if not block_ok(c):
                        continue
                    c = c.replace("\n", "")
                    blocks.append(c)
        except Exception as e:
            print(f"  {fn}: 异常 {e}")
    print(f"  清洗后候选 {len(blocks):,} 块")
    return blocks


def dedupe(blocks):
    seen = set()
    out = []
    for b in blocks:
        k = hashlib.md5(b.encode("utf-8")).hexdigest()
        if k in seen:
            continue
        seen.add(k)
        out.append(b)
    return out


def _conv_clean(s: str) -> str:
    """对话块清洗：NFKC + 去 HTML/引用角标，但保留 <think> 推理链。"""
    s = unicodedata.normalize("NFKC", s)
    s = TAG_RE.sub("", s)            # 去 <ref> 等标签（保留 <think> 需先保护）
    s = REF_RE.sub("", s)
    s = re.sub(r"[ \t\u3000]+", " ", s)
    # 折叠空行：现有语料用「空行」分隔对话块，块内只允许单换行（多行正文）
    s = re.sub(r"\n\s*\n+", "\n", s)
    return s.strip()


def clean_deepseek_r1():
    import pyarrow as pa, glob
    print("=== 清洗 deepseek-r1 蒸馏 ===")
    files = sorted(glob.glob(DS_R1_SEARCH))
    print(f"  arrow 分片: {len(files)}")
    blocks = []
    for f in files:
        r = pa.ipc.open_stream(f)
        for b in r:
            for i in range(b.num_rows):
                inst = str(b.column(0)[i].as_py() or "").strip()
                inp = str(b.column(1)[i].as_py() or "").strip()
                out = str(b.column(2)[i].as_py() or "").strip()
                if not inst or not out:
                    continue
                inst = _conv_clean(inst)
                inp = _conv_clean(inp)
                # 把 <think>...</think> 保护起来再清（TAG_RE 会误删 <think>）
                protected = re.sub(r"<think>|</think>", lambda m: m.group(0).replace("<", "\x01").replace(">", "\x02"), out)
                protected = _conv_clean(protected)
                protected = protected.replace("\x01", "<").replace("\x02", ">")
                # 保留推理链：think 内联到最终回答前（不写 <eos>，prepare.py 统一插入）
                user_txt = inst + (("\n" + inp) if inp else "")
                blocks.append(f"用户：{user_txt}\n模型：{protected}")
    print(f"  -> {len(blocks):,} 条对话框")
    return blocks


def clean_qwen3():
    import pyarrow as pa, glob
    print("=== 清洗 qwen3-235b 蒸馏 ===")
    files = sorted(glob.glob(QWEN3_SEARCH))
    print(f"  arrow 分片: {len(files)}")
    blocks = []
    for f in files:
        r = pa.ipc.open_stream(f)
        for b in r:
            for x in b.column(0):
                msgs = x.as_py() or []
                segs = []
                for m in msgs:
                    role = m.get("role")
                    content = _conv_clean(m.get("content") or "")
                    if not content:
                        continue
                    if role == "user":
                        segs.append(f"用户：{content}")
                    elif role == "assistant":
                        segs.append(f"模型：{content}")
                if not segs or not segs[0].startswith("用户："):
                    continue  # 脏数据：缺首轮 user 消息，丢弃
                blocks.append("\n".join(segs))
    print(f"  -> {len(blocks):,} 条对话框")
    return blocks



def write_blocks(blocks, path, target_chars=None):
    if target_chars is not None:
        # 确定性降采样：按字符顺序游走直到接近目标，同时保分布均匀（隔位抽样后补齐）
        import random
        random.seed(1337)
        order = list(range(len(blocks)))
        random.shuffle(order)
        chosen, acc = [], 0
        for i in order:
            if acc >= target_chars:
                break
            chosen.append(blocks[i])
            acc += len(blocks[i])
        blocks = chosen
        print(f"  降采样到 {len(blocks):,} 块 / ~{acc:,} 字符 (目标 {target_chars:,})")
    else:
        acc = sum(len(b) for b in blocks)
        print(f"  保留 {len(blocks):,} 块 / ~{acc:,} 字符")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(blocks))
        f.write("\n")
    print(f"  已写出 {path} ({os.path.getsize(path)/1e6:.1f} MB)")


def main():
    wiki = clean_wikipedia()
    wiki = dedupe(wiki)
    write_blocks(wiki, WIKI_OUT)

    c4 = clean_c4()
    c4 = dedupe(c4)
    write_blocks(c4, C4_OUT, target_chars=C4_TARGET_CHARS)

    ds_r1 = clean_deepseek_r1()
    ds_r1 = dedupe(ds_r1)
    write_blocks(ds_r1, DS_R1_OUT)

    qwen3 = clean_qwen3()
    qwen3 = dedupe(qwen3)
    write_blocks(qwen3, QWEN3_OUT)

    tot = sum(len(b) for b in wiki) + sum(len(b) for b in c4) \
        + sum(len(b) for b in ds_r1) + sum(len(b) for b in qwen3)
    print(f"\n清洗合计输出 ~{tot:,} 字符")


if __name__ == "__main__":
    main()
