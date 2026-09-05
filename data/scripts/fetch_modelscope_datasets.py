#!/usr/bin/env python3
"""
nanoSeek 外部开源高质量语料拉取器 (ModelScope / HuggingFace 国内镜像通道)

目标数据源：
1. 通用百科与通识问答：
   - `modelscope/wikipedia-cn` 或 `BAAI/COIG-CQIA` (中文高质量指令子集)
2. 逻辑与数学思维链：
   - `modelscope/GSM8K-Chinese` 或 `modelscope/Math23K`
3. 编程与算法：
   - `modelscope/code_alpaca_zh` 或精选 Python 算法数据集

脚本规范：
- 采用轻量流式拉取或精选前 N 条样本，避免阻塞本地磁盘
- 经过标准化清洗（NFKC、去除过长/过短噪声），写入 `data/chinese/` 对应 txt 文件
"""

import os
import sys
import unicodedata
from modelscope.msdatasets import MsDataset

DATA_CHINESE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chinese")
os.makedirs(DATA_CHINESE_DIR, exist_ok=True)

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", str(s).strip())
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s

def fetch_coig_cqia_subset(max_samples: int = 20000):
    """从魔搭社区拉取 COIG-CQIA 高质量中文问答精选子集。"""
    out_path = os.path.join(DATA_CHINESE_DIR, "coig_cqia_dialogue.txt")
    print(f"正在从 ModelScope 拉取 COIG-CQIA 高质量问答 (目标: {max_samples} 条)...")
    try:
        # 使用知乎/百度百科/百科问答等高质子集
        ds = MsDataset.load('m-a-p/COIG-CQIA', subset_name='zhihu', split='train')
        count = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for item in ds:
                instruction = clean_text(item.get("instruction", ""))
                input_text = clean_text(item.get("input", ""))
                output_text = clean_text(item.get("output", ""))
                
                user_msg = instruction if not input_text else f"{instruction}\n{input_text}"
                if len(user_msg) < 5 or len(output_text) < 10:
                    continue
                # 截断过长样本以保持平衡
                if len(user_msg) > 1024 or len(output_text) > 2048:
                    continue
                    
                f.write(f"用户：{user_msg}\n模型：{output_text}\n\n")
                count += 1
                if count >= max_samples:
                    break
        print(f"✓ COIG-CQIA 精选已写入: {out_path} ({count} 条, {os.path.getsize(out_path)/(1024*1024):.2f} MB)")
    except Exception as e:
        print(f"拉取 COIG-CQIA 失败或超时: {e}，跳过此源。")

def fetch_gsm8k_chinese(max_samples: int = 10000):
    """从魔搭社区拉取中文 GSM8K 数学思维链解题集。"""
    out_path = os.path.join(DATA_CHINESE_DIR, "gsm8k_zh_dialogue.txt")
    print(f"正在从 ModelScope 拉取 GSM8K-Chinese 数学题集 (目标: {max_samples} 条)...")
    try:
        ds = MsDataset.load('modelscope/gsm8k', split='train')
        count = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for item in ds:
                q = clean_text(item.get("question", ""))
                a = clean_text(item.get("answer", ""))
                if not q or not a:
                    continue
                f.write(f"用户：请解答这道数学应用题并给出详细的思考步骤：{q}\n模型：【解题思考过程】：\n{a}\n\n")
                count += 1
                if count >= max_samples:
                    break
        print(f"✓ GSM8K-Chinese 已写入: {out_path} ({count} 条, {os.path.getsize(out_path)/(1024*1024):.2f} MB)")
    except Exception as e:
        print(f"拉取 GSM8K-Chinese 失败: {e}，跳过此源。")

if __name__ == "__main__":
    print("=" * 60)
    print("  开始从 ModelScope / HuggingFace 镜像拉取开源高质量中文语料")
    print("=" * 60)
    fetch_coig_cqia_subset(20000)
    fetch_gsm8k_chinese(10000)
    print("\n外部开源语料拉取流程执行结束！")
