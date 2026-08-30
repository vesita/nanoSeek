#!/usr/bin/env python3
"""下载中文**日常闲聊**对话语料 → lccc_dialogue.txt / kdconv_dialogue.txt。

目标：稀释训练数据里的专业/咨询话术（用户反馈：模型输出偏"心理咨询师套话"）。
两个来源都选口语化、日常生活主题、无认证门槛、可流式下载的：

1) LCCC-base（thu-coai/LCCC，经 silver/lccc 镜像的 jsonl.gz）——主力
   微博/贴吧/豆瓣等平台的**真实用户对话**，最接近"朋友聊天"。
   文件 lccc_base_train.jsonl.gz 约 370MB 压缩 / 680 万条，逐行流式解压，
   读文件前缀即可拿到所需条数（目标 ~15 万条 ≈ 前 2-3%），HTTP 提前断连
   只传了实际读到的字节。
   原始格式：每行一个 JSON 数组 [轮1, 轮2, …]，文本是分词后空格连接
   （'你 去 那儿 竟然 不喊 我'）→ 清洗时去掉所有空白还原。

2) KdConv（thu-coai/KdConv）——补充多轮闲聊
   电影/音乐/旅游三个领域的知识多轮对话，消息形如
   {message, attrs?}：带 attrs（知识标注）的是模型轮，不带的是用户轮。

输出格式与其他语料一致（与 glm_dialogue.txt 兼容）：
    用户：<轮1>
    模型：<轮2>
    （样本间空行）

清洗规则（两个来源统一）：
  - 只保留中文为主的对话（CJK 字符占比 ≥ 0.6，过滤英文/乱码/URL）
  - 每轮 2-200 字，每条对话 2-12 轮
  - 转成 用户/模型 交替（奇偶段），首轮必须为用户；不合法整条丢弃
  - 按完整文本去重（每个来源内部）
  - 输出先写 <名>.txt.tmp 再原子改名（中断不污染 prepare）

幂等：原始字节缓存到临时目录（$TMPDIR/nanoseek_daily_chat_cache/），
重跑（如改 --n-lccc）直接读缓存不再走网络；输出文件已存在且未 --force 则跳过。

用法（项目根目录）：
    uv run python data/chinese/download_daily_chat.py                 # 默认全下
    uv run python data/chinese/download_daily_chat.py --n-lccc 50000
    uv run python data/chinese/download_daily_chat.py --force
"""
import argparse
import gzip
import json
import os
import re
import sys
import tempfile
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent
MIN_VALID_SIZE = 200_000  # 输出小于 200KB 视为无效，重新下载

CACHE_DIR = Path(tempfile.gettempdir()) / "nanoseek_daily_chat_cache"

LCCC_URL = "https://huggingface.co/datasets/silver/lccc/resolve/main/lccc_base_train.jsonl.gz"
KDCONV_BASE = "https://huggingface.co/datasets/thu-coai/KdConv/resolve/main"
KDCONV_FILES = [f"{d}/{s}.json" for d in ("film", "music", "travel") for s in ("train", "dev", "test")]

USER, MODEL = "用户", "模型"


# ---------- 通用清洗 ----------

def _is_cjk(ch: str) -> bool:
    return ("\u4e00" <= ch <= "\u9fff") or ("\u3400" <= ch <= "\u4dbf") or ch in "，。！？、；：""''（）《》…—～·「」"


def chinese_ratio(s: str) -> float:
    """中文占比 = CJK / (CJK + 英文字母)。数字/标点/emoji 不进分母，
    避免"2012年07月14日""票房$975万"这类含数字的闲聊被误杀。"""
    if not s:
        return 0.0
    cjk = sum(1 for ch in s if _is_cjk(ch))
    letters = sum(1 for ch in s if ch.isascii() and ch.isalpha())
    denom = cjk + letters
    return cjk / denom if denom else 0.0


def clean_turn(text: str) -> str:
    """单轮文本：去空白/URL，超长或非中文丢弃。"""
    t = re.sub(r"\s+", "", text)
    t = re.sub(r"https?://\S+", "", t)
    if not (2 <= len(t) <= 200):
        return None
    if chinese_ratio(t) < 0.6:
        return None
    if len(re.findall(r"[<>{}\[\]\\|`]", t)) >= 2:
        return None  # 代码/结构残留（'<div>' 之类）；'#话题#''@某人' 等社交符号不算
    return t


def merge_same_role(roles, texts):
    """连续同角色轮合并成一轮（KdConv 里模型常连续输出多条知识轮）。
    以标点结尾就直接拼接，否则补'，'。"""
    out_r, out_t = [], []
    for r, t in zip(roles, texts):
        if out_r and out_r[-1] == r:
            if out_t[-1] and out_t[-1][-1] in "。？！，；、：…":
                out_t[-1] += t
            else:
                out_t[-1] += "，" + t
        else:
            out_r.append(r)
            out_t.append(t)
    return out_r, out_t


def render_block(turns):
    """['a','b','c'] → '用户：a\n模型：b\n用户：c'（首轮=用户，奇偶交替）。"""
    return "\n".join(f"{USER if i % 2 == 0 else MODEL}：{t}" for i, t in enumerate(turns))


def normalize_dialogue(turns, max_turn=300, max_total=1500):
    """对话级清洗：奇数轮截掉末尾用户轮（与 GLM/multi_turn 脚本同约定：样本不能以
    未回答的用户轮收尾，否则等于教模型"下一条该输出 用户："）；2-12 轮、中文为主、
    非全同轮、长度合理。合法返回最终轮列表，否则返回 None。"""
    if len(turns) % 2 == 1:
        turns = turns[:-1]
    if not (2 <= len(turns) <= 12):
        return None
    if any(len(t) > max_turn for t in turns):
        return None
    joined = "".join(turns)
    if chinese_ratio(joined) < 0.6:
        return None
    if len(joined) > max_total:
        return None
    if len(set(turns)) < 2:  # 全同轮（'哈哈'×3 之类）信息量≈0
        return None
    return turns


# ---------- 来源 1：LCCC-base（jsonl.gz 流式） ----------

def stream_lccc_source(use_cache=True):
    """返回 (fileobj, close_fn)。优先读本地缓存，否则从 HF 流式下载并落缓存。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "lccc_base_train.jsonl.gz"
    if use_cache and cache.exists():
        print(f"  [缓存] 复用 {cache}（{cache.stat().st_size/1e6:.1f} MB）")
        return gzip.GzipFile(fileobj=open(cache, "rb")), lambda: None
    print(f"  [下载] {LCCC_URL}")
    req = urllib.request.Request(LCCC_URL, headers={"User-Agent": "Mozilla/5.0", "Accept-Encoding": "identity"})
    resp = urllib.request.urlopen(req, timeout=60)

    class Tee:
        """边读边把原始字节落盘：只缓存实际消费的前缀（提前断连不会白传）。"""

        def __init__(self, raw, fh):
            self.raw, self.fh = raw, fh

        def read(self, n=-1):
            b = self.raw.read(n)
            if b:
                self.fh.write(b)
            return b

        def readline(self, *a):
            b = self.raw.readline(*a)
            if b:
                self.fh.write(b)
            return b

    fh = open(cache, "wb")
    tee = Tee(resp, fh)

    def close():
        try:
            resp.close()
        finally:
            fh.close()

    return gzip.GzipFile(fileobj=tee), close


def build_lccc(out_path, target, force):
    if not force and out_path.exists() and out_path.stat().st_size >= MIN_VALID_SIZE:
        print(f"跳过（已存在 {out_path.stat().st_size/1e6:.1f} MB）→ {out_path}，--force 重下")
        return 0
    tmp = out_path.with_suffix(".txt.tmp")
    cache = CACHE_DIR / "lccc_base_train.jsonl.gz"
    # 第一次先试缓存（可能只是上次提前截断的前缀）；缓存不够再整段从网络重下
    for attempt in range(2):
        src, close_src = stream_lccc_source(use_cache=(attempt == 0 and cache.exists()))
        seen, written, scanned, bad = set(), 0, 0, 0
        eof = False
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                for line in src:
                    scanned += 1
                    try:
                        raw = json.loads(line)
                    except Exception:
                        bad += 1
                        continue
                    if not isinstance(raw, list):
                        bad += 1
                        continue
                    turns = []
                    for t in raw:
                        c = clean_turn(str(t))
                        if c:
                            turns.append(c)
                    turns = normalize_dialogue(turns)
                    if turns is None:
                        bad += 1
                        continue
                    block = render_block(turns)
                    if block in seen:
                        bad += 1
                        continue
                    seen.add(block)
                    f.write(block + "\n\n")
                    written += 1
                    if written % 20000 == 0:
                        print(f"    LCCC 已写 {written:,} 条（扫 {scanned:,} 行）…")
                    if written >= target:
                        break
                else:
                    eof = True  # 源读完仍未达标
        except EOFError:
            # 缓存是上次提前断连的 gzip 前缀（无 end-of-stream 标记）→ 当作缓存耗尽
            eof = True
        finally:
            close_src()
        if not eof or written >= target:
            break
        print(f"  ⚠ 缓存前缀不足（{written:,} 条 < {target:,}），改为从网络完整下载…")
        cache.unlink(missing_ok=True)
    tmp.rename(out_path)
    print(f"完成 ✅ LCCC {written:,} 条（扫 {scanned:,} 行，过滤 {bad:,}）→ {out_path}（{out_path.stat().st_size/1e6:.1f} MB）")
    return written


# ---------- 来源 2：KdConv（3 领域 × train/dev/test json） ----------

def fetch_cached(url, name):
    """小 json 直接拉进内存并落缓存。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / name
    if cache.exists():
        print(f"  [缓存] {name}")
        return cache.read_bytes()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    cache.write_bytes(data)
    return data


def build_kdconv(out_path, force):
    if not force and out_path.exists() and out_path.stat().st_size >= MIN_VALID_SIZE:
        print(f"跳过（已存在 {out_path.stat().st_size/1e6:.1f} MB）→ {out_path}，--force 重下")
        return 0
    seen, written, bad = set(), 0, 0
    tmp = out_path.with_suffix(".txt.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for rel in KDCONV_FILES:
            name = rel.replace("/", "_")
            data = fetch_cached(f"{KDCONV_BASE}/{rel}", name)
            try:
                entries = json.loads(data)
            except Exception as e:
                print(f"  ⚠ {rel} 解析失败：{e}")
                continue
            for ex in entries:
                msgs = ex.get("messages") if isinstance(ex, dict) else None
                if not msgs:
                    bad += 1
                    continue
                roles, texts = [], []
                for m in msgs:
                    if not isinstance(m, dict) or not m.get("message"):
                        continue
                    c = clean_turn(m["message"])
                    if c is None:
                        texts = None  # 任一轮不合格 → 整条丢弃（保住角色交替完整性）
                        break
                    texts.append(c)
                    roles.append(MODEL if "attrs" in m else USER)
                if not texts:
                    bad += 1
                    continue
                roles, texts = merge_same_role(roles, texts)
                if roles[0] != USER:
                    bad += 1
                    continue
                if roles[-1] != MODEL:
                    texts, roles = texts[:-1], roles[:-1]  # 以用户轮结尾（没答）→ 截掉末轮
                texts = normalize_dialogue(texts)
                if texts is None:
                    bad += 1
                    continue
                block = render_block(texts)
                if block in seen:
                    bad += 1
                    continue
                seen.add(block)
                f.write(block + "\n\n")
                written += 1
    tmp.rename(out_path)
    print(f"完成 ✅ KdConv {written:,} 条（过滤 {bad:,}）→ {out_path}（{out_path.stat().st_size/1e6:.1f} MB）")
    return written


# ---------- 入口 ----------

def main():
    ap = argparse.ArgumentParser(description="下载中文日常闲聊对话语料（LCCC-base / KdConv）")
    ap.add_argument("--datasets", default="lccc,kdconv", help="逗号分隔：lccc / kdconv")
    ap.add_argument("--n-lccc", type=int, default=150_000, help="LCCC 目标条数（默认 15 万）")
    ap.add_argument("--force", action="store_true", help="强制重新下载（覆盖现有文件）")
    a = ap.parse_args()

    wanted = [d.strip() for d in a.datasets.split(",") if d.strip()]
    if "lccc" in wanted:
        print("\n=== LCCC-base（微博/贴吧/豆瓣 真实日常对话）===")
        build_lccc(DATA_DIR / "lccc_dialogue.txt", a.n_lccc, a.force)
    if "kdconv" in wanted:
        print("\n=== KdConv（电影/音乐/旅游 多轮闲聊）===")
        build_kdconv(DATA_DIR / "kdconv_dialogue.txt", a.force)
    print("\n全部完成 ✅（输出位于 data/chinese/，txt 不入库，脚本本身入库）")


if __name__ == "__main__":
    sys.exit(main())
