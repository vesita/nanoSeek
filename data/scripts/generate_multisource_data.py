#!/usr/bin/env python3
"""
nanoSeek-100M 高质量多源数据集扩充流水线

任务：
1. 扩充四大高质量领域：
   - 领域 A (数学 CoT 思维链): 多位数四则运算、方程应用题、几何计算，带分步推理与最终确认
   - 领域 B (Agent 工具调用): 规范的 Tool API 描述、JSON Schema、<call:tool> 与 <result> 闭环响应
   - 领域 C (中文通识与百科问答): 涵盖天文地理、物理生物、计算机网络、历史哲学的高信息密度问答
   - 领域 D (代码与算法语法): 基础 Python 算法实现、字符串处理、数据结构与详细注释
2. 输出到 `data/chinese/` 对应标准语料文件：
   - `data/chinese/math_cot_dialogue.txt`
   - `data/chinese/agent_tools_dialogue.txt`
   - `data/chinese/baike_qa_dialogue.txt`
   - `data/chinese/code_syntax_dialogue.txt`
3. 遵循 nanoSeek 既有规范：使用空行分隔每条样本，支持「用户：... \n 模型：...」
"""

import os
import sys
import random

# 添加 data/scripts 路径导入生成器
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthetic_generator import MathCoTGenerator, ToolAgentGenerator, GeneralSyntheticGenerator

DATA_CHINESE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chinese")

# 针对代码与算法扩充的轻量模板生成器
CODE_TEMPLATES = [
    ("快速排序算法实现", 
     "def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)",
     "快速排序采用分治法（Divide and Conquer）。平均时间复杂度为 O(n log n)，空间复杂度为 O(log n)。"),
    
    ("二分查找实现",
     "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1",
     "二分查找要求输入数组必须是有序的。时间复杂度为 O(log n)，具有极高的检索效率。"),
     
    ("二叉树前序/中序遍历",
     "class TreeNode:\n    def __init__(self, val=0, left=None, right=None):\n        self.val = val\n        self.left = left\n        self.right = right\n\ndef inorder_traversal(root):\n    res = []\n    def dfs(node):\n        if not node:\n            return\n        dfs(node.left)\n        res.append(node.val)\n        dfs(node.right)\n    dfs(root)\n    return res",
     "中序遍历遵循“左子树 -> 根节点 -> 右子树”的顺序，对于二叉搜索树（BST），中序遍历的结果是有序递增序列。"),
     
    ("LRU 缓存机制 (Least Recently Used)",
     "from collections import OrderedDict\n\nclass LRUCache:\n    def __init__(self, capacity: int):\n        self.cache = OrderedDict()\n        self.capacity = capacity\n\n    def get(self, key: int) -> int:\n        if key not in self.cache:\n            return -1\n        self.cache.move_to_end(key)\n        return self.cache[key]\n\n    def put(self, key: int, value: int) -> None:\n        if key in self.cache:\n            self.cache.move_to_end(key)\n        self.cache[key] = value\n        if len(self.cache) > self.capacity:\n            self.cache.popitem(last=False)",
     "利用双向链表结合哈希表可以在 O(1) 时间复杂度内完成 get 和 put 操作，淘汰最久未被访问的数据项。")
]

def generate_math_dataset(num_samples: int = 25000):
    gen = MathCoTGenerator(seed=1001)
    out_path = os.path.join(DATA_CHINESE_DIR, "math_cot_dialogue.txt")
    print(f"正在生成数学思维链数据 -> {out_path} ({num_samples} 条)...")
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(num_samples):
            item = gen.sample()
            f.write(f"用户：{item['user']}\n模型：{item['model']}\n\n")
    print(f"✓ 数学思维链数据已写入，大小: {os.path.getsize(out_path)/(1024*1024):.2f} MB")

def generate_agent_tools_dataset(num_samples: int = 10000):
    gen = ToolAgentGenerator(seed=2002)
    out_path = os.path.join(DATA_CHINESE_DIR, "agent_tools_dialogue.txt")
    print(f"正在生成工具调用 Agent 数据 -> {out_path} ({num_samples} 条)...")
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(num_samples):
            text = gen.sample()
            f.write(text.strip() + "\n\n")
    print(f"✓ Agent 工具调用数据已写入，大小: {os.path.getsize(out_path)/(1024*1024):.2f} MB")

def generate_baike_dataset(num_samples: int = 15000):
    gen = GeneralSyntheticGenerator(seed=3003)
    out_path = os.path.join(DATA_CHINESE_DIR, "baike_qa_dialogue.txt")
    print(f"正在生成通识百科问答数据 -> {out_path} ({num_samples} 条)...")
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(num_samples):
            item = gen.sample_baike()
            f.write(f"用户：{item['user']}\n模型：{item['model']}\n\n")
    print(f"✓ 百科通识数据已写入，大小: {os.path.getsize(out_path)/(1024*1024):.2f} MB")

def generate_code_dataset(num_samples: int = 8000):
    rng = random.Random(4004)
    out_path = os.path.join(DATA_CHINESE_DIR, "code_syntax_dialogue.txt")
    print(f"正在生成代码语法与算法数据 -> {out_path} ({num_samples} 条)...")
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(num_samples):
            name, code, explain = rng.choice(CODE_TEMPLATES)
            user = f"请使用 Python 实现【{name}】，并给出关键代码与原理解析。"
            model = f"这是关于【{name}】的标准实现方案：\n\n```python\n{code}\n```\n\n【核心逻辑解析】：\n{explain}\n代码规范遵循 PEP 8，且具备良好的边界鲁棒性。"
            f.write(f"用户：{user}\n模型：{model}\n\n")
    print(f"✓ 代码语法数据已写入，大小: {os.path.getsize(out_path)/(1024*1024):.2f} MB")

if __name__ == "__main__":
    print("=" * 60)
    print("  nanoSeek-100M 高质量多源数据集扩充开始")
    print("=" * 60)
    os.makedirs(DATA_CHINESE_DIR, exist_ok=True)
    generate_math_dataset(25000)
    generate_agent_tools_dataset(10000)
    generate_baike_dataset(15000)
    generate_code_dataset(8000)
    print("\n所有增量语料文本已全部生成完成！")
