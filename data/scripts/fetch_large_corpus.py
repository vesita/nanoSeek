#!/usr/bin/env python3
"""
nanoSeek 大规模开源语料拉取与清洗主流水线 (ModelScope 通道)

数据源：
1. COIG-CQIA-full.jsonl (149MB, ~46k 条高质量中文指令/问答/百科/逻辑)
2. 数学逻辑题库 (Math23K / GSM8K 已缓存)
3. 代码与算法语料 (CodeAlpaca / 技术问答)

输出规范：
- data/chinese/ 下按领域分文件，空行分隔，「用户：/模型：」格式
- 适配三区字级 WordLevel 分词器
"""

import os
import sys
import json
import unicodedata
import requests
import time

DATA_CHINESE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chinese")
os.makedirs(DATA_CHINESE_DIR, exist_ok=True)

def clean_text(s):
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", str(s).strip())
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s

def download_file(url, dest, timeout=180):
    """流式下载大文件，支持断点续传。"""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  已存在 {dest} ({os.path.getsize(dest)/1e6:.1f} MB)，跳过下载")
        return True
    print(f"  下载中: {url[:80]}...")
    try:
        r = requests.get(url, stream=True, timeout=timeout)
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with open(dest + ".tmp", "wb") as f:
            for chunk in r.iter_content(1024 * 512):
                f.write(chunk)
                done += len(chunk)
        os.rename(dest + ".tmp", dest)
        print(f"  ✓ 下载完成 {dest} ({done/1e6:.1f} MB)")
        return True
    except Exception as e:
        print(f"  ✗ 下载失败: {e}")
        return False

def download_coig_full():
    """下载 COIG-CQIA-full.jsonl 完整数据。"""
    url = "https://modelscope.cn/api/v1/datasets/m-a-p/COIG-CQIA/repo?Source=SDK&Revision=master&FilePath=COIG-CQIA-full.jsonl"
    return download_file(url, os.path.join(DATA_CHINESE_DIR, "_coig_full.jsonl"))

def parse_coig_to_domains():
    """解析 COIG-CQIA-full.jsonl，按 domain/task_type 分流到各领域文件。"""
    src = os.path.join(DATA_CHINESE_DIR, "_coig_full.jsonl")
    if not os.path.exists(src):
        print("COIG-CQIA-full 不存在，跳过解析")
        return
    
    # 领域映射
    domain_files = {
        "wiki": "coig_wiki_dialogue.txt",        # 百科
        "逻辑": "coig_logic_dialogue.txt",       # 逻辑推理
        "数学": "coig_math_dialogue.txt",        # 数学
        "编程": "coig_code_dialogue.txt",        # 代码/技术
        "对话": "coig_chat_dialogue.txt",        # 日常对话
        "other": "coig_other_dialogue.txt",      # 其余
    }
    writers = {}
    counts = {}
    for k, v in domain_files.items():
        writers[k] = open(os.path.join(DATA_CHINESE_DIR, v), "w", encoding="utf-8")
        counts[k] = 0
    
    # 关键字分类
    kw_map = {
        "wiki": ["百科", "知识", "历史", "地理", "科学", "生物", "物理", "化学", "天文", "医学", "法律", "经济"],
        "逻辑": ["逻辑", "推理", "论证", "判断", "悖论", "三段论"],
        "数学": ["数学", "计算", "方程", "几何", "代数", "概率", "统计"],
        "编程": ["编程", "代码", "Python", "算法", "程序", "开发", "软件", "技术", "数据库", "网络"],
    }
    
    total = 0
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            instruction = clean_text(item.get("instruction", ""))
            inp = clean_text(item.get("input", ""))
            out = clean_text(item.get("output", ""))
            user_msg = instruction if not inp else f"{instruction}\n{inp}"
            if len(user_msg) < 4 or len(out) < 8:
                continue
            
            # 分类
            text_all = instruction + " " + out
            domain = item.get("domain", [])
            domain_str = " ".join(domain) if isinstance(domain, list) else str(domain)
            task_type = item.get("task_type", {})
            minor = task_type.get("minor", []) if isinstance(task_type, dict) else []
            minor_str = " ".join(minor) if isinstance(minor, list) else str(minor)
            full_signal = text_all + " " + domain_str + " " + minor_str
            
            category = "other"
            for cat, kws in kw_map.items():
                for kw in kws:
                    if kw.lower() in full_signal.lower():
                        category = cat
                        break
                if category != "other":
                    break
            
            writers[category].write(f"用户：{user_msg}\n模型：{out}\n\n")
            counts[category] += 1
            total += 1
    
    for w in writers.values():
        w.close()
    print(f"  ✓ COIG-CQIA 解析完成，共 {total} 条有效样本")
    for k, v in counts.items():
        fpath = os.path.join(DATA_CHINESE_DIR, domain_files[k])
        print(f"    - {k}: {v} 条 ({os.path.getsize(fpath)/1e6:.1f} MB)")
    
    # 清理原始大文件（节省磁盘）
    os.remove(src)
    print(f"  已删除原始 {src} 释放磁盘")

def download_math23k():
    """下载 Math23K 中文数学应用题。"""
    # 使用公开镜像
    urls = [
        "https://modelscope.cn/api/v1/datasets/AI-ModelScope/Math23K/repo?Source=SDK&Revision=master&FilePath=Math23K.json",
    ]
    dest = os.path.join(DATA_CHINESE_DIR, "_math23k.json")
    for u in urls:
        if download_file(u, dest):
            return True
    return False

if __name__ == "__main__":
    print("=" * 60)
    print("  nanoSeek 大规模开源语料拉取流水线启动")
    print("=" * 60)
    t0 = time.time()
    
    if download_coig_full():
        parse_coig_to_domains()
    
    print(f"\n总耗时 {time.time()-t0:.1f} 秒")
    print("流水线执行完毕！")
