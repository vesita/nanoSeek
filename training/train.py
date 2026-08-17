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
import csv
import sys
import time
import math
import pickle
import random
import threading
import hashlib
from contextlib import nullcontext

# 脚本在 training/ 子目录，Python 默认不会把项目根目录加进模块搜索路径。
# 这里把根目录插到 sys.path 开头，才能 `from model import ...`。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from model.config_loader import load_config

# -----------------------------------------------------------------------------
# 默认配置：small 模型在字符级莎士比亚上训练（与 config/train_shakespeare_char.yaml 一致）。
# 推荐通过 config/*.yaml 覆盖运行，这里只是不带配置裸跑时的兜底默认。
# I/O
out_dir = 'out'
eval_interval = 250
log_interval = 10
eval_iters = 200
eval_only = False # 如果为 True，脚本在第一次评估后立即退出
always_save_checkpoint = False # 如果为 True，每次评估后总是保存 checkpoint；否则只在 val 变优时保存
# 早停：val loss 连续 patience 次评估无实质改善就提前终止（不用手动估算步数）
enable_early_stop = True   # 默认开；设为 False 则训满 max_iters
patience = 3               # val 连续 3 次评估不改善就停（激进；保守可调 5-8）
min_val_improve = 0.01     # val 至少下降 0.01 才算"改善"（避免微小抖动干扰）
min_iters = 1000           # 前 1000 步强制不早停（训练早期 loss 波动大，避免误停）
init_from = 'scratch' # 'scratch' 或 'resume' 或 '<路径>.pt'（后训练）
# wandb 日志记录
wandb_log = False # 默认禁用
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt' # 'run' + str(time.time())
# TensorBoard 日志记录（开源、本地，无需账号）。
# 默认关闭：主输出是 YOLO 式 results.csv + loss_curve.png，不需要二进制事件文件。
# 需要多实验曲线叠加对比时再开（事件写到 out/<实验>/tensorboard/ 子目录，不污染主目录）。
tensorboard_log = False
# 训练结束自动生成 results.csv 每评估点一行（step/train/val/lr/mfu/time），
# 纯文本、Excel 可打开、训练中断也能读到已落盘的部分。
# 数据
dataset = 'shakespeare_char'
gradient_accumulation_steps = 1 # 用于模拟更大的 batch size
# 生成自蒸馏（方向 1，Expert Iteration 轻量版）——预算中性混合：
# distill_bin 非空时，train 的每个 batch 以概率 p_distill 从蒸馏数据切块（其余从 train.bin 切）。
# 总 token 量不变，唯一变量 = 训练预算里蒸馏样本的占比。生成脚本见 training/self_distill.py。
distill_bin = ''   # 空 = 关闭；非空 = 蒸馏数据路径（如 data/chinese/distill.bin）
p_distill = 0.0    # 训练预算里蒸馏样本占比（0~1）
batch_size = 64 # 如果 gradient_accumulation_steps > 1，这是微批（micro-batch）大小
block_size = 256
# 模型（small）
n_layer = 6         # 4→6：深度换宽度，加深帮模型学长期依赖（对抗重复坍缩）
n_head = 4
n_embd = 80         # 128→80：与默认 yaml 对齐；深度换宽度，规模持平（head_dim=20）
dropout = 0.2 # 预训练时 0 就很好，微调时可以试试 0.1+
bias = False # 是否在 Linear 层内部使用 bias？
# --- 固定架构：RMSNorm + SwiGLU 硬编码；RoPE 与 wpe 二选一 ---
use_rope = True     # True：RoPE 旋转位置编码；False：可学习位置嵌入 wpe
rope_theta = 1000000.0 # RoPE 基础频率（1e6 表达更远相对距离，DeepSeek 做法）
swiglu_clamp = 0.0  # V4：SwiGLU 门控输出钳制半宽；0 = 关闭
# --- MoE 混合专家（DeepSeek-V3/V4），默认关闭 ---
use_moe = False        # 混合专家：MoE 替换 FFN
n_experts = 8          # 路由专家总数
n_top_k = 2            # 每个 token 激活的专家数
moe_aux_weight = 0.01  # 负载均衡辅助损失权重（Switch 式，use_aux_free_balance=False 时用）
use_shared_expert = False      # V4：始终激活的共享专家
use_aux_free_balance = False   # V4：aux-free 偏置修正替代 Switch aux loss
balance_factor = 0.001         # aux-free 偏置每步更新幅度
use_sqrtsoftplus = False       # V4：路由打分 √softplus 替代 softmax
route_scale = 2.5              # √softplus 打分缩放系数
# --- MLA 多头潜在注意力（DeepSeek-V2），与 CSA 二选一 ---
use_mla = False        # 多头潜在注意力：低秩压缩 KV
kv_lora_rank = 64      # KV 压缩后的潜在维度
qk_rope_head_dim = 16  # 每头参与 RoPE 的维数
# --- MTP 多 token 预测（DeepSeek-V3/V4），默认关闭 ---
use_mtp = True         # 多 token 预测（V3/V4 验证过：训练信号增强，推理零开销）
n_mtp = 1              # 额外预测的 token 数
mtp_weight = 0.3       # MTP 损失权重（DeepSeek-V3 建议 0.3）
# --- V4 优化器：Muon（可选）替代 AdamW ---
use_muon = False       # 矩阵参数用 Muon，embedding/lm_head/norm 用 AdamW
muon_momentum = 0.95   # Muon 动量系数
muon_ns_steps = 10     # Newton-Schulz 迭代次数（默认 8 激进 + 2 经典）
# --- V4 核心：CSA/HCA 压缩稀疏注意力 ---
use_csa = False        # CSA 压缩稀疏注意力（块级 KV 压缩 + top-k 稀疏选择 + 滑窗）
csa_compress = 16      # 块大小：每几个 token 压成一个潜在 KV
csa_topk = 4           # 每个 query 稀疏选几个压缩块
csa_window = 64        # 滑窗：保留最近多少个原始 token
use_hca = False        # HCA 重度压缩全局信号
use_csa_learnable = True   # V4：可学习门控池化替代平均池化
# --- V4 结构设计升级（实验性，默认全关）---
use_attn_sink = True         # Attention Sinks：打破重复坍缩的必要条件（三重 A/B 验证）
use_mhc = False              # mHC 超连接：4 流并行残差
hc_mult = 4                  # mHC 残差流数（V4 原版 = 4）
use_lightning_indexer = False   # 学习型块选择替代 CSA raw top-k
num_hash_layers = 0          # 前 N 层用 hash 路由（0 = 禁用）
block_order = "attn_ffn"     # 计算图重排：块内子层顺序（attn_ffn | ffn_attn）
no_attn_layers = []          # 稀疏注意力布线：跳过注意力的层索引（0-based，空=所有层都有）
n_memory_tokens = 0          # 显式记忆 token：序列前插入 K 个可学习嵌入（0=关闭，实验性）
use_lse_residual = False     # 对数放缩残差：对数域 soft-max 合并替代线性相加（零参数，实验性）
use_lse_gate = False         # 对数放缩门控混合：α·x+(1-α)·LSE(x,F)，α 可学习（每层标量）
# adamw 优化器
learning_rate = 1e-3 # 最大学习率
max_iters = 5000 # 训练总迭代次数
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0 # 在此值处裁剪梯度，若为 0.0 则禁用
# 学习率衰减设置
decay_lr = True # 是否衰减学习率
warmup_iters = 100 # 预热多少步
lr_decay_iters = 5000 # 根据 Chinchilla 论文，应约等于 max_iters
min_lr = 1e-4 # 最小学习率，根据 Chinchilla 论文应约等于 learning_rate/10
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

# 数据溯源：记录 data/<dataset>/manifest.json 的哈希，方便追查“这个模型用的哪版数据”
data_manifest_path = os.path.join(data_dir, 'manifest.json')
if os.path.exists(data_manifest_path):
    try:
        with open(data_manifest_path, 'rb') as _f:
            config['data_manifest_sha256'] = hashlib.sha256(_f.read()).hexdigest()
    except OSError as _e:
        print(f'warning: 读取数据清单 {data_manifest_path} 失败：{_e}')

# 每个 epoch 的步数（YOLO 式进度条显示轮次用）
try:
    _train_tokens = os.path.getsize(os.path.join(data_dir, 'train.bin')) // 2  # uint16
    steps_per_epoch = max(1, _train_tokens // tokens_per_iter)
except OSError:
    steps_per_epoch = None

def get_batch(split):
    # 我们每个 batch 都重新创建 np.memmap，以避免内存泄漏，参见
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        # 预算中性混合：以概率 p_distill 从蒸馏数据切块；distill.bin 太短（< 2*block_size
        # 个字节，即不足一个窗口）时退回 train.bin，避免 randint 越界。
        use_distill = (distill_bin and p_distill > 0 and random.random() < p_distill
                       and os.path.exists(distill_bin)
                       and os.path.getsize(distill_bin) > 2 * block_size)
        path = distill_bin if use_distill else os.path.join(data_dir, 'train.bin')
    else:
        path = os.path.join(data_dir, 'val.bin')
    data = np.memmap(path, dtype=np.uint16, mode='r')
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

# 模型初始化
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout,
                  use_rope=use_rope, rope_theta=rope_theta, swiglu_clamp=swiglu_clamp,
                  use_moe=use_moe, n_experts=n_experts, n_top_k=n_top_k, moe_aux_weight=moe_aux_weight,
                  use_shared_expert=use_shared_expert, use_aux_free_balance=use_aux_free_balance,
                  balance_factor=balance_factor, use_sqrtsoftplus=use_sqrtsoftplus, route_scale=route_scale,
                  use_mla=use_mla, kv_lora_rank=kv_lora_rank, qk_rope_head_dim=qk_rope_head_dim,
                  use_mtp=use_mtp, n_mtp=n_mtp, mtp_weight=mtp_weight,
                  use_muon=use_muon, muon_momentum=muon_momentum, muon_ns_steps=muon_ns_steps,
                  use_csa=use_csa, csa_compress=csa_compress, csa_topk=csa_topk,
                  csa_window=csa_window, use_hca=use_hca, use_csa_learnable=use_csa_learnable,
                  use_attn_sink=use_attn_sink, use_mhc=use_mhc, hc_mult=hc_mult,
                  use_lightning_indexer=use_lightning_indexer, num_hash_layers=num_hash_layers,
                  block_order=block_order, no_attn_layers=no_attn_layers,
                  n_memory_tokens=n_memory_tokens,
                  use_lse_residual=use_lse_residual,
                  use_lse_gate=use_lse_gate)

def _build_model_from_checkpoint(checkpoint):
    """按 checkpoint 里的 model_args 构建模型并加载权重（供 resume / 后训练复用）。"""
    checkpoint_model_args = checkpoint['model_args']
    # 强制这些配置属性等于 checkpoint 里的值（架构必须一致才能加载权重）
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # 架构开关：checkpoint 里没有的键用命令行/默认值兜底
    for k in ['use_rope', 'rope_theta', 'swiglu_clamp',
              'use_moe', 'n_experts', 'n_top_k', 'moe_aux_weight',
              'use_shared_expert', 'use_aux_free_balance', 'balance_factor',
              'use_sqrtsoftplus', 'route_scale',
              'use_mla', 'kv_lora_rank', 'qk_rope_head_dim',
              'use_mtp', 'n_mtp', 'mtp_weight',
              'use_muon', 'muon_momentum', 'muon_ns_steps',
              'use_csa', 'csa_compress', 'csa_topk', 'csa_window',
              'use_hca', 'use_csa_learnable',
              'use_attn_sink', 'use_mhc', 'hc_mult',
              'use_lightning_indexer', 'num_hash_layers', 'block_order', 'no_attn_layers',
              'n_memory_tokens', 'use_lse_residual', 'use_lse_gate']:
        model_args[k] = checkpoint_model_args.get(k, model_args[k])
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # 修复 state dictionary 的键：torch.compile 偶尔会带上 _orig_mod. 前缀
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    return model

if init_from == 'scratch':
    # 从零初始化一个新模型
    # 确定从零训练时使用的 vocab size
    if meta_vocab_size is None:
        print("默认把 GPT-2 的 vocab_size 设为 50304（50257 向上取整以提高效率）")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"正在从 {out_dir} 恢复训练（YOLO 式：自动加载 best.pt）")
    # 从 checkpoint 恢复训练。续训会连同优化器、学习率计划、迭代计数一起恢复。
    ckpt_path = os.path.join(out_dir, 'best.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = _build_model_from_checkpoint(checkpoint)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.endswith('.pt'):
    # 在已有模型上做后训练：加载权重，但从头开始新的优化器/学习率计划
    print(f"正在从 {init_from} 加载已有模型权重（后训练，优化器/学习率重置）")
    checkpoint = torch.load(init_from, map_location=device)
    model = _build_model_from_checkpoint(checkpoint)
    # iter_num / best_val_loss 保持初始值（0 / 1e9），全新训练
else:
    raise ValueError(f"不支持的 init_from：{init_from}（应为 'scratch' / 'resume' / '<路径>.pt'）")
model.to(device)

# 初始化 GradScaler。如果 enabled=False，scaler 是空操作
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

# 优化器
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # 释放内存

# 打印训练启动摘要：把散落的启动日志收敛成一个信息框，再进入 tqdm 进度条
def print_summary():
    attn = 'MLA' if use_mla else f'CSA/HCA (块{csa_compress}·topk{csa_topk}·窗{csa_window})'
    ffn = f'MoE {n_experts}×top{n_top_k}' if use_moe else 'SwiGLU'
    border = "─" * 46
    print()
    print(border)
    print("  训练摘要")
    print(border)
    print(f"  数据集    {dataset} · {model.config.vocab_size} 词表 · 上下文 {block_size}")
    print(f"  模型      {n_layer} 层 · {n_head} 头 · {n_embd} 维 · {attn} · {ffn}")
    opt_name = 'Muon' if use_muon else 'AdamW'
    print(f"  优化器    {opt_name} · lr {learning_rate:g} · wd {weight_decay:g} · betas ({beta1:g}, {beta2:g})")
    epoch_note = f" · ≈{max_iters/steps_per_epoch:.1f} epoch" if steps_per_epoch else ""
    print(f"  训练      {max_iters} 步 · {tokens_per_iter:,} tokens/步{epoch_note}")
    print(f"  设备      {device} · {dtype} · compile {'开' if compile else '关'}")
    print(f"  检查点    best.pt（val 最优）+ last.pt（最新）· 续训自动从 best.pt 恢复")
    print(border)
    print()

if master_process:
    print_summary()

# 编译模型
if compile:
    # 抑制 inductor 在低 SM 数 GPU 上的提示性警告：
    # RTX 5060 只有 30 个 SM（< 68），max_autotune_gemm 用不了，compile 时会反复打
    # "Not enough SMs to use max_autotune_gemm mode"。这是良性提示——只是退回
    # 默认 matmul，不影响正确性。用定向 Filter 只静音这条，不动其他警告。
    import logging
    class _MaxAutotuneGemmFilter(logging.Filter):
        def filter(self, record):
            return "max_autotune_gemm" not in record.getMessage()
    logging.getLogger('torch._inductor').addFilter(_MaxAutotuneGemmFilter())

    print("正在编译模型……这一步比较耗时，请耐心等待")
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

def _backup_old_run(out_dir):
    """重复训练到同一 out_dir 前，把已有旧实验产物归档到 out_dir/old/，仅保留最近一份。

    背景：固定命令格式下，重复跑同一 out_dir 会把上次的 results.csv / ckpt / loss_curve
    整个覆盖掉，想对比/找回旧结果就没了。这里在写任何产物前，先把 out_dir 里已有的
    文件整体挪到 old/ 子目录——想找回时看 old/ 即可。

    "仅保留一个 old"：若 old/ 已存在，先删掉再挪新的（更早的版本丢弃，只留最近一份旧实验）。
    """
    import shutil
    old_dir = os.path.join(out_dir, 'old')
    items = [f for f in os.listdir(out_dir) if f != 'old']
    if not items:
        return  # 全新目录，无需备份
    if os.path.isdir(old_dir):
        shutil.rmtree(old_dir)  # 只保留最近一份 old
    os.makedirs(old_dir, exist_ok=True)
    for f in items:
        shutil.move(os.path.join(out_dir, f), os.path.join(old_dir, f))
    print(f"⚠ 检测到 {out_dir} 已有旧实验产物，已归档到 old/（仅保留最近一份）")


# 覆盖保护：写 results.csv / SummaryWriter 之前执行。
# resume 除外——续训要读回 out_dir/best.pt，不能把老 ckpt 挪走。
if master_process and init_from != 'resume':
    _backup_old_run(out_dir)

writer = None
if tensorboard_log and master_process:
    from torch.utils.tensorboard import SummaryWriter
    # 事件写到 out/<实验>/tensorboard/ 子目录，不污染实验主目录
    #（主目录只留 ckpt.pt / results.csv / loss_curve.png 这些可读文件）
    writer = SummaryWriter(log_dir=os.path.join(out_dir, 'tensorboard'))
    writer.add_text("config", str(config), 0)

# YOLO 式 results.csv：每个评估点一行，纯文本、随时可读、不依赖任何工具
results_csv = None
csv_writer = None
if master_process:
    results_csv = open(os.path.join(out_dir, 'results.csv'), 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(results_csv)
    csv_writer.writerow(['step', 'train/loss', 'val/loss', 'lr', 'mfu', 'time'])

# -----------------------------------------------------------------------------
# 异步 checkpoint 保存
# torch.save 同步写盘会让训练循环卡顿。这里把「序列化 + 磁盘写入」丢给后台线程，
# 主线程只做张量 CPU 快照（~100ms）就立刻返回继续训练。
# 关键安全性：快照是独立张量（to(cpu, copy=True)），后台线程保存期间训练继续
# 修改参数也不会污染它；写临时文件 + 原子改名，保证 ckpt.pt 永远完整。
# -----------------------------------------------------------------------------
_save_threads = []
_save_lock = threading.Lock()

def _checkpoint_to_cpu(ckpt):
    """递归把 checkpoint 里的所有张量拷到 CPU 并脱离计算图，供后台线程安全保存。"""
    out = {}
    for k, v in ckpt.items():
        if isinstance(v, dict):
            out[k] = _checkpoint_to_cpu(v)
        elif isinstance(v, torch.Tensor):
            out[k] = v.detach().to('cpu', copy=True)
        else:
            out[k] = v
    return out

def _save_worker(ckpt, tmp_path, path):
    with _save_lock:  # 同一时刻只写一个文件，避免并发保存互相覆盖
        torch.save(ckpt, tmp_path)
        os.replace(tmp_path, path)  # 原子改名：写一半的文件永远不会被读到

def save_checkpoint_async(ckpt, path):
    """把 checkpoint 丢给后台线程保存，主线程立即返回继续训练。"""
    ckpt_cpu = _checkpoint_to_cpu(ckpt)          # 同步快照（安全），线程只负责写盘
    t = threading.Thread(target=_save_worker, args=(ckpt_cpu, path + '.tmp', path), daemon=True)
    _save_threads.append(t)
    t.start()
    # 清理已完成的线程，防止列表无限累积占内存（eval 多次后列表会很长）
    _save_threads[:] = [x for x in _save_threads if x.is_alive()]

def join_save_threads():
    """等待所有后台保存线程完成（训练结束前调用，确保最后的 checkpoint 落盘）。"""
    for t in _save_threads:
        t.join()

def _plot_loss_curve(loss_history, out_dir, best_val_loss):
    """训练结束后画 train/val loss 曲线到 loss_curve.png（YOLO 式 results.png），
    不用手动开 TensorBoard 也能直接看图。"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # 无显示环境，用非交互后端
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        # 中文字体：图里有中文标签，DejaVu Sans 没有中文字形会打出方块。
        # 按优先级尝试常见 CJK 字体，全找不到就回退默认（图仍能生成，只是中文变方块）。
        for _font in ('Noto Sans CJK SC', 'Source Han Sans CN', 'WenQuanYi Zen Hei',
                      'Microsoft YaHei', 'SimHei'):
            try:
                font_manager.findfont(_font, fallback_to_default=False)
                plt.rcParams['font.family'] = _font
                break
            except Exception:
                continue
        plt.rcParams['axes.unicode_minus'] = False  # 负号用 ASCII 减号，避免显示成方块
        iters = [h[0] for h in loss_history]
        train = [h[1] for h in loss_history]
        val = [h[2] for h in loss_history]
        plt.figure(figsize=(8, 5))
        plt.plot(iters, train, label='train', color='tab:blue')
        plt.plot(iters, val, label='val', color='tab:orange')
        plt.xlabel('迭代步数')
        plt.ylabel('loss')
        plt.title(f'{os.path.basename(out_dir)} · best_val_loss {best_val_loss:.4f}')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        path = os.path.join(out_dir, 'loss_curve.png')
        plt.savefig(path, dpi=120)
        plt.close()
        print(f"已生成 loss 曲线图：{path}")
    except Exception as e:
        print(f"生成 loss 曲线图失败（不影响训练）：{e}")

# 训练循环
X, Y = get_batch('train') # 获取第一个 batch
t0 = time.time()           # t0 在每轮迭代末尾会被重置（用于测单步速度算 MFU）
train_start = time.time()  # 训练总起点，results.csv 里的 time 列用这个（不会随迭代重置）
local_iter_num = 0 # 本进程生命周期内的迭代次数
raw_model = model.module if ddp else model # 如果需要，解开 DDP 容器
running_mfu = -1.0
# tqdm 进度条：DDP 下只有主进程显示
pbar = tqdm(total=max_iters, initial=iter_num, desc="训练中", dynamic_ncols=True) if master_process else None
loss_history = []  # 每个评估点记 (iter, train_loss, val_loss)，训练结束画曲线图用
early_stopped = False  # 早停是否触发（收尾打印用）
no_improve_count = 0   # val 连续无实质改善的评估次数（早停计数）
while True:

    # 确定并设置本次迭代的学习率
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # 在 train/val 集合上评估损失并保存 checkpoint
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        # 用 pbar.write 打印到进度条上方，不打断进度条
        pbar.write(f"step {iter_num}: train 损失 {losses['train']:.4f}, val 损失 {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # 换算成百分比
            })
        if tensorboard_log:
            # 与 wandb 记录同样的指标，写进 TensorBoard
            writer.add_scalar("train/loss", losses['train'], iter_num)
            writer.add_scalar("val/loss", losses['val'], iter_num)
            writer.add_scalar("lr", lr, iter_num)
            writer.add_scalar("mfu", running_mfu*100, iter_num)
        loss_history.append((iter_num, losses['train'], losses['val']))
        if csv_writer is not None:
            csv_writer.writerow([
                iter_num, f"{losses['train']:.4f}", f"{losses['val']:.4f}",
                f"{lr:.6g}", f"{max(running_mfu, 0.0)*100:.2f}", f"{time.time()-train_start:.1f}",
            ])
            results_csv.flush()  # 及时落盘：训练中断也能读到已写出的部分
        if iter_num > 0:
            # YOLO 式：last.pt 每次评估都存（最新状态），best.pt 只在 val 变优时存
            prev_best = best_val_loss                 # 保存本次评估前的 best，早停判断用（修复，见下）
            is_best = losses['val'] < prev_best
            if is_best:
                best_val_loss = losses['val']
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
                'epoch': iter_num / steps_per_epoch if steps_per_epoch else None,
            }
            save_checkpoint_async(checkpoint, os.path.join(out_dir, 'last.pt'))
            if is_best:
                save_checkpoint_async(checkpoint, os.path.join(out_dir, 'best.pt'))
                pbar.write(f"✓ 新最佳 val {best_val_loss:.4f} → best.pt（并已更新 last.pt）")
            # 早停：val 连续 patience 次评估无实质改善 → 提前终止。
            # 改善判定用 min_val_improve 阈值（严格低于才重置计数），避免微小抖动干扰。
            # 注意 is_best 是"比历史 best 低"即算，这里要"比 best 低出 min_val_improve"才算实质改善。
            # 修复（2026-08-12，dev-notes/22）：必须用 prev_best（本次评估前的 best）判断，
            # 用更新后的 best_val_loss 时每次创新低两者相等，永远判"无改善"，patience 次即误停。
            if enable_early_stop and iter_num >= min_iters:
                if losses['val'] < prev_best - min_val_improve:
                    no_improve_count = 0  # 有实质改善，重置计数
                else:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        early_stopped = True
                        pbar.write(f"⏹ 早停：val 连续 {patience} 次评估无实质改善（best {best_val_loss:.4f}），提前终止 @ {iter_num}")
                        break
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

    # 计时与日志：更新 tqdm 进度条
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
        epoch_str = f"{iter_num/steps_per_epoch:.2f}" if steps_per_epoch else "-"
        pbar.set_postfix(轮次=f"{epoch_str}", 损失=f"{lossf:.4f}", MFU=f"{running_mfu*100:.1f}%")
    iter_num += 1
    local_iter_num += 1
    if pbar is not None:
        pbar.update(1)

    # 终止条件
    if iter_num > max_iters:
        break

# 训练结束：等后台保存线程写完，再画 loss 曲线图，收尾 csv
if master_process:
    if early_stopped:
        print(f"训练提前终止：{iter_num} 步（早停，best_val_loss {best_val_loss:.4f}）")
    else:
        print(f"训练完成：{iter_num} 步（达 max_iters {max_iters}）")
join_save_threads()
if results_csv is not None:
    results_csv.close()
if master_process and loss_history:
    _plot_loss_curve(loss_history, out_dir, best_val_loss)
if writer is not None:
    writer.close()
if pbar is not None:
    pbar.close()
if ddp:
    destroy_process_group()
