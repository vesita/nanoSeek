"""
为字符级语言建模准备中文数据集（《西游记》全本，吴承恩/明）。
与 shakespeare_char 同一模式：不做 BPE，直接把汉字映射成整数。
会生成 train.bin、val.bin，以及包含编码器/解码器的 meta.pkl。

用法（从项目根目录）：
    uv run python data/chinese/prepare.py
"""
import os
import pickle
import requests
import numpy as np

# 下载《西游记》全本（来源：github.com/tennessine/corpus，四大名著合集）
# 文件名是中文，URL 里做了百分号编码：西游记.txt
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/tennessine/corpus/master/%E8%A5%BF%E6%B8%B8%E8%AE%B0.txt'
    print(f"正在下载《西游记》……")
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
print(f"数据集字符长度: {len(data):,}")

# 获取这段文本中出现的所有不重复字符
chars = sorted(list(set(data)))
vocab_size = len(chars)
print(f"词汇表大小: {vocab_size:,}")

# 创建从字符到整数的映射
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # 编码器：输入字符串，输出整数列表
def decode(l):
    return ''.join([itos[i] for i in l]) # 解码器：输入整数列表，输出字符串

# 创建训练和验证划分
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# 把两者都编码成整数
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train 有 {len(train_ids):,} 个 token")
print(f"val 有 {len(val_ids):,} 个 token")

# 导出到 bin 文件
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# 同时保存 meta 信息，方便之后编码/解码
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# 期望输出：
# 数据集字符长度: 677,421
# 词汇表大小: 4,507
# train 有 609,679 个 token
# val 有 67,742 个 token
