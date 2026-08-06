"""
本训练脚本既可以在单 GPU 调试模式下运行，
也可以在更大的分布式数据并行（DDP）训练中使用。

在单 GPU 上运行的示例：
$ python train.py --batch_size=32 --compile=False

在单台机器的 4 张 GPU 上用 DDP 运行的示例：
$ torchrun --standalone --nproc_per_node=4 train.py

在 2 台机器共 8 张 GPU 上用 DDP 运行的示例：
- 在第一个（主）节点上运行，假设 IP 为 123.456.123.456：
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- 在从节点上运行：
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
（如果你的集群没有 Infiniband 互联，请在前面加上 NCCL_IB_DISABLE=1）
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from config_loader import load_config

# -----------------------------------------------------------------------------
# 默认配置，用于在 OpenWebText 上训练一个 gpt2（124M 参数）
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # 如果为 True，脚本在第一次评估后立即退出
always_save_checkpoint = True # 如果为 True，每次评估后总是保存 checkpoint
init_from = 'scratch' # 'scratch' 或 'resume' 或 'gpt2*'
# wandb 日志记录
wandb_log = False # 默认禁用
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# 数据
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # 用于模拟更大的 batch size
batch_size = 12 # 如果 gradient_accumulation_steps > 1，这是微批（micro-batch）大小
block_size = 1024
# 模型
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # 预训练时 0 就很好，微调时可以试试 0.1+
bias = False # 是否在 LayerNorm 和 Linear 层内部使用 bias？
# --- 现代化架构开关（DeepSeek / LLaMA 风格），默认关闭 = 原始 GPT-2 ---
use_rmsnorm = False # 用 RMSNorm 替代 LayerNorm
use_rope = False    # 用 RoPE 替代可学习的位置嵌入
use_swiglu = False  # 用 SwiGLU 替代 GELU MLP
rope_theta = 10000.0 # RoPE 基础频率
# adamw 优化器
learning_rate = 6e-4 # 最大学习率
max_iters = 600000 # 训练总迭代次数
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # 在此值处裁剪梯度，若为 0.0 则禁用
# 学习率衰减设置
decay_lr = True # 是否衰减学习率
warmup_iters = 2000 # 预热多少步
lr_decay_iters = 600000 # 根据 Chinchilla 论文，应约等于 max_iters
min_lr = 6e-5 # 最小学习率，根据 Chinchilla 论文应约等于 learning_rate/10
# DDP 设置
backend = 'nccl' # 'nccl'、'gloo' 等
# 系统
device = 'cuda' # 示例：'cpu'、'cuda'、'cuda:0'、'cuda:1' 等，或在 macbook 上试试 'mps'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32'、'bfloat16' 或 'float16'，后者会自动实现 GradScaler
compile = True # 使用 PyTorch 2.0 编译模型以加速
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
load_config(globals()) # 从 YAML 配置文件或命令行覆盖
config = {k: globals()[k] for k in config_keys} # 对日志记录很有用
# -----------------------------------------------------------------------------

# 各种初始化、派生属性和 I/O 设置
ddp = int(os.environ.get('RANK', -1)) != -1 # 这是 DDP 运行吗？
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # 这个进程将负责日志记录、保存 checkpoint 等
    seed_offset = ddp_rank # 每个进程获得不同的种子
    # world_size 个进程将同时训练，因此我们可以按比例
    # 减少每个进程期望的梯度累积迭代次数
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # 如果不是 DDP，我们在单 GPU 上运行，只有一个进程
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"每次迭代的 token 数将是: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # 在 matmul 上允许 tf32
torch.backends.cudnn.allow_tf32 = True # 在 cudnn 上允许 tf32
device_type = 'cuda' if 'cuda' in device else 'cpu' # 供后面 torch.autocast 使用
# 注意：float16 数据类型会自动使用 GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# 简易数据加载器
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # 我们每个 batch 都重新创建 np.memmap，以避免内存泄漏，参见
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # 固定 x、y 的内存，这样我们可以异步（non_blocking=True）把它们搬到 GPU
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# 在这里初始化，如果 init_from='resume'（即从 checkpoint）可以覆盖
iter_num = 0
best_val_loss = 1e9

# 尝试从数据集推导 vocab_size
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"找到 vocab_size = {meta_vocab_size}（位于 {meta_path}）")

# 模型初始化
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout,
                  use_rmsnorm=use_rmsnorm, use_rope=use_rope, use_swiglu=use_swiglu, rope_theta=rope_theta)
if init_from == 'scratch':
    # 从零初始化一个新模型
    print("正在从零初始化一个新模型")
    # 确定从零训练时使用的 vocab size
    if meta_vocab_size is None:
        print("默认把 GPT-2 的 vocab_size 设为 50304（50257 向上取整以提高效率）")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"正在从 {out_dir} 恢复训练")
    # 从 checkpoint 恢复训练。
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # 强制这些配置属性相等，否则我们根本无法恢复训练
    # 其余属性（如 dropout）可以按命令行里的期望值保持不变
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # modern 开关：老 checkpoint 里可能没这些键，用命令行/默认值兜底
    for k in ['use_rmsnorm', 'use_rope', 'use_swiglu', 'rope_theta']:
        model_args[k] = checkpoint_model_args.get(k, model_args[k])
    # 创建模型
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # 修复 state dictionary 的键 :(
    # 老实说不知道 checkpoint 有时怎么会带上这个前缀，需要再调试一下
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"正在从 OpenAI GPT-2 权重初始化: {init_from}")
    # 从 OpenAI GPT-2 权重初始化
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # 读取创建出的配置参数，以便正确地把它们存进 checkpoint
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size',
              'use_rmsnorm', 'use_rope', 'use_swiglu', 'rope_theta']:
        model_args[k] = getattr(model.config, k)
# 如果需要，用模型“手术”把 block size 裁剪下来
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # 这样 checkpoint 会有正确的值
model.to(device)

# 初始化 GradScaler。如果 enabled=False，scaler 是空操作
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# 优化器
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # 释放内存

# 编译模型
if compile:
    print("正在编译模型……（大约需要一分钟）")
    unoptimized_model = model
    model = torch.compile(model) # 需要 PyTorch 2.0

# 把模型包装进 DDP 容器
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# 通过许多 batch 帮助估算任一划分上任意精度的损失
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# 学习率衰减调度器（带预热的余弦）
def get_lr(it):
    # 1) 线性预热 warmup_iters 步
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) 如果 it > lr_decay_iters，返回最小学习率
    if it > lr_decay_iters:
        return min_lr
    # 3) 中间部分，用余弦衰减下降到最小学习率
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff 取值范围 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# 日志
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# 训练循环
X, Y = get_batch('train') # 获取第一个 batch
t0 = time.time()
local_iter_num = 0 # 本进程生命周期内的迭代次数
raw_model = model.module if ddp else model # 如果需要，解开 DDP 容器
running_mfu = -1.0
while True:

    # 确定并设置本次迭代的学习率
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # 在 train/val 集合上评估损失并保存 checkpoint
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train 损失 {losses['train']:.4f}, val 损失 {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # 换算成百分比
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"正在把 checkpoint 保存到 {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # 前向反向更新，带可选的梯度累积以模拟更大的 batch size
    # 如果数据类型是 float16，则使用 GradScaler
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # 在 DDP 训练中，我们只需要在最后一个微步同步梯度。
            # 官方的做法是用 model.no_sync() 上下文管理器，但
            # 我很不喜欢它让代码膨胀并迫使我们重复代码。
            # 看了那个上下文管理器的源码，它只是切换这个变量。
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # 缩放损失以计入梯度累积
        # 在模型于 GPU 上进行前向传播时，立即异步预取下一个 batch
        X, Y = get_batch('train')
        # 反向传播，如果以 fp16 训练则进行梯度缩放
        scaler.scale(loss).backward()
    # 裁剪梯度
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # 如果以 fp16 训练，则更新优化器和 scaler
    scaler.step(optimizer)
    scaler.update()
    # 尽快清空梯度，不再需要这块内存
    optimizer.zero_grad(set_to_none=True)

    # 计时与日志
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # 把损失转成 float。注意：这是一个 CPU-GPU 同步点
        # 放大以抵消上面的除法，近似真实的总体损失（精确做法应是求和）
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # 让训练循环先稳定一下
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"迭代 {iter_num}: 损失 {lossf:.4f}, 耗时 {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # 终止条件
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
