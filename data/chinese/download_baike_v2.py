# -*- coding: utf-8 -*-
"""通过代理下载并清洗中文维基百科正文 (健壮版)。

修正 download_baike.py 的 JSON 解析：
原版用 buffer.find('{"completion":') 脆弱的文本切割匹配，
但实际数据集是外层 JSON 数组（以 [ 开头，completion 字段带 4 空格缩进），
导致匹配不到任何条目 → 0 有效。这里改用标准 json 全量解析数组遍历。
"""
import os, sys, json, re, requests

OUT = "data/chinese/wikipedia_cn.txt"
TMP = OUT + ".tmp"
URL = "https://hf-mirror.com/datasets/pleisto/wikipedia-cn-20230720-filtered/resolve/main/wikipedia-cn-20230720-filtered.json"
TARGET_CHARS = 30_000_000  # 30M 汉字

# 设置代理（若环境已导出则自动用）
http_proxy = os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY")
if http_proxy:
    proxies = {"http": http_proxy, "https": http_proxy}
else:
    proxies = None
print(f"代理: {http_proxy or '无（直连）'}", flush=True)


def is_valid(text):
    if len(text) < 80:
        return False
    if "消歧义" in text[:30]:
        return False
    cjk = len(re.findall(r'[\u4e00-\u9fff]', text))
    return cjk / max(len(text), 1) >= 0.5


def clean(text):
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


print("开始下载并解析中文维基百科 JSON 数组...", flush=True)
total_chars = 0
total_articles = 0

with requests.get(URL, stream=True, timeout=120, proxies=proxies) as resp:
    resp.raise_for_status()
    buf = b""
    with open(TMP, "w", encoding="utf-8") as out_f:
        # 先完整下载到内存（524MB 可行），再 json.loads
        # 说明：524MB 内存可接受；如需更省可按 json 数组增量解析，但当前规模足够。
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                break
            buf += chunk

print(f"下载完成: {len(buf)/1e6:.1f} MB，开始解析...", flush=True)

try:
    data = json.loads(buf.decode("utf-8", errors="ignore"))
except Exception as e:
    print("JSON 解析失败:", e)
    # 尝试逐行或按 [ 块解析
    print("尝试降级解析（逐条对象）...")
    data = []
    text = buf.decode("utf-8", errors="ignore")
    # 按 {" 分割尝试
    for m in re.finditer(r'\{"completion":\s*"(.*?)",\s*"source"', text, re.DOTALL):
        data.append({"completion": m.group(1)})

if isinstance(data, list):
    print(f"解析到 {len(data):,} 条记录", flush=True)
    with open(TMP, "w", encoding="utf-8") as out_f:
        for item in data:
            if isinstance(item, dict) and "completion" in item:
                text = item["completion"]
                if is_valid(text):
                    out_f.write(clean(text) + "\n\n")
                    total_chars += len(text)
                    total_articles += 1
                    if total_articles % 5000 == 0:
                        print(f"  已处理 {total_articles:,} 条，累计 {total_chars/1e6:.1f}M 字符", flush=True)
                    if total_chars >= TARGET_CHARS:
                        break
else:
    print("数据结构异常:", type(data))

if total_articles > 0:
    os.replace(TMP, OUT)
    print(f"✅ 完成: {OUT} | 有效条目 {total_articles:,} | 文本 {total_chars:,} 字符"
          f" ({os.path.getsize(OUT)/1e6:.1f} MB)", flush=True)
else:
    print("⚠ 仍 0 有效条目，保留 TMP 待排查", flush=True)
