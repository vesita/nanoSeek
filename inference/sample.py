"""
从训练好的模型中采样生成
"""
import os
import sys
import pickle
from contextlib import nullcontext

# 脚本在 inference/ 子目录，把项目根目录插到 sys.path 开头
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import tiktoken
from model import GPTConfig, GPT
from model.config_loader import load_config

# -----------------------------------------------------------------------------
init_from = 'resume' # 从 out_dir 加载 best.pt
out_dir = 'out'
start = "\n" # 或者 "<|endoftext|>" 等。也可以指定一个文件，用法："FILE:prompt.txt"
num_samples = 10 # 要生成的样本数量
max_new_tokens = 500 # 每个样本生成的 token 数量
temperature = 0.8 # 1.0 = 不改变，< 1.0 = 更少随机，> 1.0 = 更多随机，作用于预测
top_k = 200 # 只保留概率最高的 top_k 个 token，其它 token 的概率置为 0
seed = 1337
device = 'cuda' # 示例：'cpu'、'cuda'、'cuda:0'、'cuda:1' 等
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' 或 'bfloat16' 或 'float16'
compile = False # 使用 PyTorch 2.0 编译模型以加速
dump_logits = '' # 非空时把 prompt 最后位置的 logits 落盘（每行一个），用于和 Rust --dump-logits 逐位对拍
load_config(globals()) # 从命令行或配置文件覆盖
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # 在 matmul 上允许 tf32
torch.backends.cudnn.allow_tf32 = True # 在 cudnn 上允许 tf32
device_type = 'cuda' if 'cuda' in device else 'cpu' # 供后面 torch.autocast 使用
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# 模型
assert init_from == 'resume', "本脚本只支持从 out_dir 加载 best.pt"
# 从保存在特定目录中的模型初始化
ckpt_path = os.path.join(out_dir, 'best.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # 需要 PyTorch 2.0（可选）

# 查找数据集文件夹里是否有 BPE 分词器
load_tok = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # 旧版 checkpoint 可能没有这些字段……
    tok_path = os.path.join('data', checkpoint['config']['dataset'], 'tokenizer.json')
    load_tok = os.path.exists(tok_path)
if load_tok:
    print(f"正在从 {tok_path} 加载 BPE 分词器……")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tok_path)
    encode = lambda s: tokenizer.encode(s).ids
    decode = lambda l: tokenizer.decode(l)
else:
    # 好，那就默认使用 gpt-2 编码
    print("未找到 tokenizer.json，假定使用 GPT-2 编码……")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# 对提示（prompt）的开头进行编码
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# 对拍模式：dump 最后位置的 logits（与 Rust --dump-logits 输出格式一致：每行一个）
if dump_logits:
    with torch.no_grad():
        with ctx:
            logits, _ = model(x)  # (1, T, vocab)
    v = logits[0, -1, :].float().cpu().tolist()
    with open(dump_logits, 'w', encoding='utf-8') as f:
        f.write('\n'.join(f'{x}' for x in v) + '\n')
    print(f"已 dump {len(v)} 个 logits → {dump_logits}")
    sys.exit(0)

# 运行生成
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
