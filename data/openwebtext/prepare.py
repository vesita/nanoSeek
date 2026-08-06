# 把 openwebtext 数据集保存成用于训练的二进制文件。以下内容很有帮助：
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets

# .map() 调用中的 worker 数量
# 一个合适的数量大约是 CPU 核心数 // 2
num_proc = 8

# load_dataset() 调用中的 worker 数量
# 最佳数量可能与上面的 num_proc 不同，因为它还取决于网络速度。
# 不过通常比 1 好
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    # 在 huggingface .cache 目录占用 54GB，约 800 万个文档（8,013,769）
    dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset)

    # owt 默认只包含 'train' 划分，所以创建一个 test 划分
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') # 把 test 划分重命名为 val

    # 这会得到：
    # >>> split_dataset
    # DatasetDict({
    #     train: Dataset({
    #         features: ['text'],
    #         num_rows: 8009762
    #     })
    #     val: Dataset({
    #         features: ['text'],
    #         num_rows: 4007
    #     })
    # })

    # 现在要对数据集做 tokenize。首先定义编码函数（gpt2 bpe）
    def process(example):
        ids = enc.encode_ordinary(example['text']) # encode_ordinary 会忽略任何特殊 token
        ids.append(enc.eot_token) # 添加文本结束 token，例如 gpt2 bpe 的 50256
        # 注意：我觉得 eot 应该加在开头而不是结尾……嗯。但它叫 "eot"（end of text）……
        out = {'ids': ids, 'len': len(ids)}
        return out

    # 对数据集做 tokenize
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # 把每个数据集中的所有 id 拼接成一个大文件，供训练使用
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16 # （可行，因为 enc.max_token_value == 50256 < 2**16）
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # 把样本批量拼在一起，以便更快写入
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # 写入 mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    # train.bin 约 17GB，val.bin 约 8.5MB
    # train 有约 90 亿个 token（9,035,582,198）
    # val 有约 400 万个 token（4,434,897）

    # 之后读取 bin 文件的方法，例如用 numpy：
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')
