"""
用于基准测试的 train.py 精简版
"""
import os
from contextlib import nullcontext
import numpy as np
import time
import torch
from model import GPTConfig, GPT
from config_loader import load_config

# -----------------------------------------------------------------------------
batch_size = 12
block_size = 1024
bias = False
real_data = True
seed = 1337
device = 'cuda' # 示例：'cpu'、'cuda'、'cuda:0'、'cuda:1' 等
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' 或 'bfloat16' 或 'float16'
compile = True # 使用 PyTorch 2.0 编译模型以加速
profile = False # 使用 pytorch profiler，还是只做简单基准测试？
load_config(globals()) # 从命令行或配置文件覆盖
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # 在 matmul 上允许 tf32
torch.backends.cudnn.allow_tf32 = True # 在 cudnn 上允许 tf32
device_type = 'cuda' if 'cuda' in device else 'cpu' # 供后面 torch.autocast 使用
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# 数据加载初始化
if real_data:
    dataset = 'openwebtext'
    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    def get_batch(split):
        data = train_data # 注意：基准测试脚本里忽略 split
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x, y
else:
    # 或者，如果想要固定数据而不关心数据加载
    x = torch.randint(50304, (batch_size, block_size), device=device)
    y = torch.randint(50304, (batch_size, block_size), device=device)
    get_batch = lambda split: (x, y)

# 模型初始化
gptconf = GPTConfig(
    block_size = block_size, # 模型往回看多远？即上下文大小
    n_layer = 12, n_head = 12, n_embd = 768, # 模型大小
    dropout = 0, # 为了确定性
    bias = bias,
)
model = GPT(gptconf)
model.to(device)

optimizer = model.configure_optimizers(weight_decay=1e-2, learning_rate=1e-4, betas=(0.9, 0.95), device_type=device_type)

if compile:
    print("正在编译模型……")
    model = torch.compile(model) # pytorch 2.0

if profile:
    # 关于 pytorch profiler 的有用文档：
    # - 教程 https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
    # - API https://pytorch.org/docs/stable/profiler.html#torch.profiler.profile
    wait, warmup, active = 5, 5, 5
    num_steps = wait + warmup + active
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./bench_log'),
        record_shapes=False,
        profile_memory=False,
        with_stack=False, # 会产生额外的开销，不需要时请禁用
        with_flops=True,
        with_modules=False, # 目前只对 torchscript 模型有效
    ) as prof:

        X, Y = get_batch('train')
        for k in range(num_steps):
            with ctx:
                logits, loss = model(X, Y)
            X, Y = get_batch('train')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            lossf = loss.item()
            print(f"{k}/{num_steps} 损失: {lossf:.4f}")

            prof.step() # 在每一步结束时通知 profiler

else:

    # 简单基准测试
    torch.cuda.synchronize()
    for stage, num_steps in enumerate([10, 20]): # 先预热，再基准测试
        t0 = time.time()
        X, Y = get_batch('train')
        for k in range(num_steps):
            with ctx:
                logits, loss = model(X, Y)
            X, Y = get_batch('train')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            lossf = loss.item()
            print(f"{k}/{num_steps} 损失: {lossf:.4f}")
        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1-t0
        mfu = model.estimate_mfu(batch_size * 1 * num_steps, dt)
        if stage == 1:
            print(f"每次迭代耗时: {dt/num_steps*1000:.4f}ms, MFU: {mfu*100:.2f}%")
