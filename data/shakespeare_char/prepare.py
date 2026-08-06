"""
为字符级语言建模准备莎士比亚数据集。
所以不是用 GPT-2 BPE token 编码，而是直接把字符映射成整数。
会保存包含 id 的 train.bin、val.bin，以及包含编码器、解码器和
一些其它相关信息的 meta.pkl。
"""
import os
import pickle
import requests
import numpy as np

# 下载 tiny shakespeare 数据集
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"数据集字符长度: {len(data):,}")

# 获取这段文本中出现的所有不重复字符
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("所有不重复字符:", ''.join(chars))
print(f"词汇表大小: {vocab_size:,}")

# 创建从字符到整数的映射
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # 编码器：输入字符串，输出整数列表
def decode(l):
    return ''.join([itos[i] for i in l]) # 解码器：输入整数列表，输出字符串

# 创建训练和测试划分
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

# 数据集字符长度:  1115394
# 所有不重复字符:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# 词汇表大小: 65
# train 有 1003854 个 token
# val 有 111540 个 token
