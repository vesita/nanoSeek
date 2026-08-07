"""
为字符级语言建模准备中文数据集（多部中文经典合集）。
与 shakespeare_char 同一模式：不做 BPE，直接把汉字映射成整数。
会生成 train.bin、val.bin，以及包含编码器/解码器的 meta.pkl。

数据来源：github.com/tennessine/corpus（中文经典合集），共 4 部：
    《西游记》《红楼梦》《三国演义》《水浒传》
每部书名即文件名。prepare.py 会读取 data/chinese/ 下所有 .txt 文件并拼接成一个语料。

用法（从项目根目录）：
    uv run python data/chinese/prepare.py
"""
import os
import pickle
import requests
import numpy as np

# 待补齐的书目：(本地文件名, URL 里的中文书名)
# 文件名是中文，URL 中需做百分号编码
BOOKS = [
    ('西游记.txt',  '西游记'),
    ('红楼梦.txt',  '红楼梦'),
    ('三国演义.txt','三国演义'),
    ('水浒传.txt',  '水浒传'),
]

# 数据源：raw.githubusercontent.com（原站），连不上时用 jsdelivr 镜像兜底
RAW_URL    = 'https://raw.githubusercontent.com/tennessine/corpus/master/{enc}.txt'
MIRROR_URL = 'https://cdn.jsdelivr.net/gh/tennessine/corpus@master/{enc}.txt'

def download_if_missing(local_name, book_name):
    """本地已有就不再下载，避免重复拉取。"""
    path = os.path.join(os.path.dirname(__file__), local_name)
    if os.path.exists(path):
        return
    import urllib.parse
    enc = urllib.parse.quote(book_name)
    for tmpl in (RAW_URL, MIRROR_URL):
        url = tmpl.format(enc=enc)
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(r.text)
                print(f'已下载《{book_name}》 -> {local_name}')
                return
        except Exception:
            continue
    print(f'警告：未能下载《{book_name}》，跳过。')

# 逐个补齐缺失的书目
for local_name, book_name in BOOKS:
    download_if_missing(local_name, book_name)

# 读取 data/chinese/ 下所有 .txt 文件并拼接成一个语料
dir_path = os.path.dirname(__file__)
parts = []
for fn in sorted(os.listdir(dir_path)):
    if fn.endswith('.txt'):
        with open(os.path.join(dir_path, fn), 'r', encoding='utf-8') as f:
            parts.append(f.read())
data = '\n'.join(parts)  # 不同书籍之间用换行分隔
print(f"合并了 {len(parts)} 个文件")
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

# 创建训练和验证划分（90/10）
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
