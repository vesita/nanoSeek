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
import json
import random
import threading
import hashlib
import re
import traceback
from collections import Counter
from contextlib import nullcontext

# --- 基础设施规范（ROCm / AMD GPU 运行规约）---
# 必须在 import torch 之前注入，防止 gfx1032/gfx1030 指令集报错与 SDMA 异步拷贝引发的 PCIe 假死
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("HSA_ENABLE_SDMA", "0")
# PyTorch 2.9+ 用 PYTORCH_ALLOC_CONF（旧 PYTORCH_HIP_ALLOC_CONF 已 deprecated，会刷警告）
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# 脚本在 training/ 子目录，Python 默认不会把项目根目录加进模块搜索路径。
# 这里把根目录插到 sys.path 开头，才能 `from model import ...`。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# --- 在 Inductor / Dynamo 首次初始化之前，提前锁定浮点精度模式 ---
# 必须在任何 .compile() 或首个 matmul 前设置，否则 torch/_inductor 会在编译时刷
# "TensorFloat32 tensor cores ... not enabled" 的 UserWarning（compile_fx.py:312）
torch.backends.cuda.matmul.allow_tf32 = True   # matmul 允许 tf32
torch.backends.cudnn.allow_tf32 = True         # cudnn 允许 tf32
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

from model import GPTConfig, GPT
from model.config_loader import load_config
from training.masking import build_assistant_mask as _build_assistant_mask
from training.masking import build_resp_span_mask as _build_resp_span_mask
# 纯函数抽到 training/schedules.py：本文件是模块级脚本（import 即开训），
# 写在里面的函数无法被 pytest 覆盖。这里只做「读全局 → 转调纯函数」。
from training.schedules import lr_at as _lr_at, pick_bin_names
# 样本打包（document packing）纯函数：.off 边界表读取/校验 + sample_id + 块对齐窗口。
from training.packing import (load_offsets as _load_offsets,
                              offsets_name_for_bin as _offsets_name_for_bin,
                              sample_id_in_window as _sample_id_in_window,
                              mask_cross_sample_labels as _mask_cross_sample_labels,
                              aligned_pack_starts as _aligned_pack_starts)
# 诊断用的纯函数（显存调试行 / 快照触发 / OOM 现场）：抽出来是为了能在 CPU 上
# 单测 —— 诊断代码自己不能变成新的故障源（见 training/diag.py 的说明）。
from training.diag import mem_debug_line, should_dump_snapshot
from training.checkpoints import prune_step_checkpoints_sparse
from training.run_logs import count_csv_rows, open_run_csv

# -----------------------------------------------------------------------------
# 默认配置：small 模型在字符级莎士比亚上训练（与 config/train_shakespeare_char.yaml 一致）。
# 推荐通过 config/*.yaml 覆盖运行，这里只是不带配置裸跑时的兜底默认。
# I/O
out_dir = 'out'
eval_interval = 250
log_interval = 10
eval_iters = 200
# 评估开销优化（2026-08-20 框架提速三件套）：
# eval_train_split=false 时评估点不再重算 train loss（训练 loss 每 log_interval 步已有记录），
# results.csv 的 train/loss 列改用训练侧 EMA 代替——评估开销减半（原 eval 占训练总计算
# ~25-30%，train split 占其中一半）。
eval_train_split = True
eval_only = False # 如果为 True，脚本在第一次评估后立即退出
always_save_checkpoint = False # 如果为 True，每次评估后总是保存 checkpoint；否则只在 val 变优时保存
# 归档检查点的保留策略（2026-09-11 晚：从外部脚本搬进训练内部，见 training/checkpoints.py 的模块文档）
#   ★ 这两个键**必须定义在下面 config_keys 快照之前**，否则 yaml 里设了会被静默忽略
#     （load_config 只覆盖已存在的全局）。旧代码用的是
#     `getattr(config, 'keep_step_ckpts', 5)`，而那个全局从未定义过 ⇒ 该旋钮其实**永远是 5**。
#   策略 = 每 ckpt_sparse_every 步留一个回溯点 + 最新 ckpt_newest_keep 个；best/last 永不删。
#   ≤0 表示关闭对应那条规则；两条都 ≤0 会在 steps_to_keep 里直接抛错（防"删光"）。
ckpt_sparse_every = 5000   # 阶段回溯点间隔（70000 步 ⇒ 约 15 个）
ckpt_newest_keep = 2       # 再额外保留最新的几个，防止"刚写的就被删"
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
moe_hidden_scale = 8 / 3       # MoE 单专家隐层缩放（8/3 粗粒度标准；4/3 细粒度轻量化）
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
muon_ns_steps = 10     # Newton-Schulz 迭代总次数
# 前 muon_ns_aggressive 步用「激进系数」，其余用「经典系数」（混相）。
# 实测（259 个真实动量矩阵、配对检验）：经典10步中位残差 4.53e-05；
# 激进4+经典3=7步 8.79e-06（t=−10.5，显著更好，且 NS 计算省 30%）。0=纯经典=旧行为。
muon_ns_aggressive = 0
muon_lr_scale = 0.2   # Muon 矩阵参数 lr 缩放（DeepSeek/Kimi 惯例：AdamW lr × 0.2）
muon_split = False     # GLM-5 Muon Split：注意力投影按「头」分块做 NS 正交化（修 Muon 短预算收敛差）
# --- V4 核心：CSA/HCA 压缩稀疏注意力 ---
use_csa = False        # CSA 压缩稀疏注意力（块级 KV 压缩 + top-k 稀疏选择 + 滑窗）
csa_compress = 16      # 块大小：每几个 token 压成一个潜在 KV
csa_topk = 4           # 每个 query 稀疏选几个压缩块
csa_window = 64        # 滑窗：保留最近多少个原始 token
use_hca = False        # HCA 重度压缩全局信号
use_csa_learnable = True   # V4：可学习门控池化替代平均池化
use_csa_fused_qkv = True   # CSA 计算优化：Q/K/V 三合一（权重布局变，仅新训练；A/B 2 胜出→默认）
use_csa_bmm = False        # CSA 计算优化：einsum → 显式批量 matmul（逐位等价；A/B 3 更慢→保持关）
use_kv_memory = True  # KV 记忆注意力（P1：GLA 式可学习遗忘/写入状态，替换 HCA 槽位）。
# 记忆 = 项目基线（dev-notes/42 决策）→ 默认开启；无记忆对照实验显式 --use_kv_memory=false
kv_memory_latent = 16      # 记忆 latent 维 l（观测台：K 秩~8/V 秩~5 → 16 够用；1500 步验证 16 > 8）
kv_memory_chunk = 32       # chunk 并行块大小（dev-notes/42-A：C=32 无 checkpoint 实测 1.36 it/s > C=64+checkpoint 1.27，且显存安全）
kv_memory_checkpoint = False   # 梯度检查点：backward 重算块内 D，省内存数学等价（dev-notes/39-#4）
kv_memory_complement_gate = False  # 互补门：写入门 β=1−r（忘记多少写入多少，删独立 mem_write；270 步持平双门未定论）
kv_memory_layers = None            # B组：启用记忆的最后 n 层（None=全部层；300 步 A/B 否定分层）
kv_memory_block = 1                # C组：块级记忆块大小（token 数）。1=逐 token（现行为）；300 步 A/B 否定块级（dev-notes/44）
kv_memory_delta = False               # P2：Delta 擦写律，先擦后写消除键冲突混叠（dev-notes/45；持平但 3× 慢弃用）
kv_memory_output_gate = False         # KV 记忆输出门控 (Output Gate) + 状态 RMSNorm（GLA/RetNet 思想）
sample_boundary_reset = True          # 样本边界重置与因果阻断：遇到 <eos> 时清空记忆黑板并阻断滑窗跨样本注意
# --- V4 结构设计升级（实验性，默认全关）---
use_attn_sink = True         # Attention Sinks：打破重复坍缩的必要条件（三重 A/B 验证）
use_mhc = False              # mHC 超连接：4 流并行残差
hc_mult = 4                  # mHC 残差流数（V4 原版 = 4）
use_cd = False               # 收敛-发散头：主干后权重共享 Block 循环（用户构型 2026-09-19）
cd_iters = 6                 # 收敛循环圈数
use_lightning_indexer = False   # 学习型块选择替代 CSA raw top-k
num_hash_layers = 0          # 前 N 层用 hash 路由（0 = 禁用）
block_order = "attn_ffn"     # 计算图重排：块内子层顺序（attn_ffn | ffn_attn）
no_attn_layers = []          # 稀疏注意力布线：跳过注意力的层索引（0-based，空=所有层都有）
n_memory_tokens = 0          # 显式记忆 token：序列前插入 K 个可学习嵌入（0=关闭，实验性）
# --- NDB（后缀 n-gram 神经数据库，见 model/ngram_ndb.py）共训 ---
# ★ 采用的方案：**读与写都由模型自己的门控决定**，表在训练中在线累积。
#   `ndb_slots = 0` = 完全不启用，训练流程与本文件原行为逐位一致。
#   表是 no_grad 的 token 计数（numpy），**不进 state_dict / 不进 checkpoint**，
#   也**不需要任何离线建库**：每个 run 从空表开始，由本 run 的训练流在线写出来
#   ⇒ 没有库文件要在阶段之间重建，也就没有「基座漂移导致 query/key 空间失配」的问题。
#   梯度路径：读门控从 `read()` 拿（`w_t = sigmoid(write_gate(h))` 是它的输入之一）；
#   `observe()` 自身是 `@torch.no_grad()`，所以「写门控有没有在学」要看**读路径**的梯度。
ndb_slots = 0                # 每级槽位数（0 = 关闭 NDB）；2**26 = 67M
ndb_levels = '8'             # 后缀长度列表，逗号分隔（如 '8' 或 '8,6'）
ndb_top_k = 4                # 每槽保留的续写个数（1 = 只存 top-1，退化为硬检索）
ndb_max_table_gb = 6.0       # 表内存上限，超过直接报错（别跑到一半 OOM）
ndb_lr = 3e-4                # 读/写门控学习率（独立 AdamW，固定，不随基座调度衰减）
ndb_flush_every = 50         # 每多少训练步把热缓冲落进表；每次 NDB 监控前也落一次
ndb_eval_iters = 64          # NDB 监控 eval 的 batch 数
ndb_debug_mem = False        # 诊断：每步打印显存（allocated/peak/reserved/OOM 计数）
mem_snapshot_gb = 0.0        # 诊断：峰值显存超过该值(GB)时 dump 一次显存分配历史快照（0=关）
use_lse_residual = False     # 对数放缩残差：对数域 soft-max 合并替代线性相加（零参数，实验性）
use_lse_gate = False         # 对数放缩门控混合：α·x+(1-α)·LSE(x,F)，α 可学习（每层标量）
use_qk_norm = False          # QK-Norm：q/k L2 归一化 + 每头可学习 scale（近零参数，压重复坍缩）
z_loss_weight = 0.0          # Router Z-Loss 权重；0 = 关闭，建议 1e-4 起步
gradient_checkpointing = False  # 梯度检查点：block 级重算，压降 65%~75% 激活显存（100M 模型 8GB 卡必开）
# --- loss masking（chat 微调惯例：只对 assistant 回复算 loss）---
use_loss_masking = True      # False = 全部 token 参与训练（非对话语料）
# mask 规则（2026-09-14 新增，实现见 training/masking.py）：
#   'eos_line'  = 行内含 <eos>/<cont> ⇒ **整行**算 loss（既有行为，逐位不变）
#   'resp_span' = `<resp>` 之后 → 对应 `<eos>`（含）算 loss（**不含 <resp>**）
#                 —— 「单流 + <resp>」格式（training/dialogue_stream.py）用这个
# ★★ 必须定义在下面 config_keys 快照**之前**（铁律 8），否则 yaml 里设了会被静默改回默认。
mask_mode = 'eos_line'
mask_resp_ids = []
# 去标签 masking（dev-notes/61）：按回复终止符 <eos>/<cont> 定位模型回复行。
# 默认空 = 按模式分支（char/byte）解析；找不到标记会全部 mask → loss NaN（有防护）。
mask_reply_ids = []
mask_sep_ids = []
# 打包非空窗口（2026-09-10）：loss masking 下 ~83% 的 256 窗口整窗全 mask，前向白跑。
# 终止符 <eos>/<cont> 自身必然是有效 token（见 training/masking.py：next_term=自身 < next_nl），
# 所以「窗口内含终止符」⟺「该窗口有 ≥1 个有效 token」。抽样时只抽这种窗口：
# 目标函数逐位不变（有效 token 的集合与权重都不变），只把空窗口的算力还回来。
# 实测：有效 token/步 548 → ~3200（×5.8），eval 有效 token 2.8 万 → 16 万（噪声 ÷2.4）。
pack_nonempty = False
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
# 两阶段训练框架（DeepSeek 路线，2026-08-19）：
#   stage=pretrain：无掩码全 token 语言建模（读 pretrain.bin），学语言+对话结构；
#   stage=sft：对话微调（读 train.bin，build_assistant_mask 只对 assistant 回复算 loss）；
#   stage=full：单阶段对话训练（旧行为，等价 sft）。
# 阶段衔接：pretrain 产物 resume 进 sft（--init_from=out/xxx/best.pt 或续训）。
stage = 'full'                 # pretrain | sft | full
schedule = 'cosine'            # cosine | wsd（WSD=warmup-stable-decay，DeepSeek-V3）
# WSD 参数：stable_frac 之后的 lr_decay_iters 步从 learning_rate 线性/指数衰减到 min_lr
stable_frac = 0.8              # 稳定段占比（前 80% 步保持 learning_rate）
# GLM-5 DSA 配方（2026-02 技术报告）：稀疏注意力适配时先冻结主模型、只训练索引器
# N 步，再放开联合训练——GLM-5 用 20B token（含 1000 步索引器预热）追平 DeepSeek
# 943.7B token 的 DSA 训练效果。只对 lightning indexer 生效，默认关。
indexer_warmup_steps = 0       # >0：前 N 步只训练 idx_q/idx_k（主模型冻结）
# --- 训练中健康体检（你好体检精简版，2026-08-20 框架提速三件套）---
# 背景：dev-notes/14 实证「val 继续降但采样崩」——best.pt 只看 val 会把坍缩模型当冠军。
# health_enabled=true 时每个评估点用固定 prompt 采样体检（EOS 自吐率 / rep3），
# 只有「val 创新低 且 体检合格」才更新 best.pt：体检不合格时不覆盖旧 best（防坍缩保护）。
# 体检口径与 training/health_check_hello.py 一致（temp 0.8 / topk 200 / rep 1.2，
# 原始生成不截断），结果写 out/health.csv。纯框架改动，零模型参数。
health_enabled = False       # True = 计算体检分并门控 best.pt 选点（train_chinese.yaml 开）
health_eval_interval = 0     # 体检步频；0 = 跟随 eval_interval（推荐，避免体检结果过期）
health_prompts = "你好|你是谁|你在哪|今天心情怎么样"  # '|' 分隔多条（去标签，dev-notes/61）
health_seeds = 5             # 每 prompt 采样 seed 数（预算 = prompts×seeds×max_new 次前向）
health_max_new = 120         # 单条生成上限（原始生成，不截断）
health_temp = 0.8            # 采样温度（与你好体检同口径）
health_top_k = 200
health_rep_penalty = 1.2
health_min_eos_rate = 0.6    # EOS 自吐率下限（低于 = 体检不合格）
health_max_rep3 = 0.1        # rep3 上限（>0.1 即坍缩线，dev-notes 口径）
# DDP 设置
backend = 'nccl' # 'nccl'、'gloo' 等
# 系统
device = 'cuda' # 示例：'cpu'、'cuda'、'cuda:0'、'cuda:1' 等，或在 macbook 上试试 'mps'
dtype = ('bfloat16' if torch.cuda.is_bf16_supported() else 'float16') if torch.cuda.is_available() else 'float32' # 'float32'、'bfloat16' 或 'float16'，后者会自动实现 GradScaler；纯 CPU 默认 float32（避免 float16+GradScaler 报错）
compile = True # 默认开（dev-notes/42-A 曾用 3 步短测误判"无收益"改为关；实测 20 步：开 1.29 it/s vs 关 ~1.0 it/s，快 ~25%——1.5 分钟编译开销在长训练摊薄后净赚）
byte_level = False      # 字节直入模式（dev-notes/48，Mamba-Byte 思想）：无 BPE 分词，
                        # 词表 0-255 字节+<eos>=256，读 train_byte.bin/val_byte.bin
char_level = False      # 字级模式（dev-notes/50）：汉字=1 token，读 train_char.bin/val_char.bin
factorized_emb_dim = 0  # 因式分解嵌入维度：>0 启用低秩嵌入（ALBERT 思想），wte 降至 E 维，省参数加深网络
# 数据集文件名后缀：'' = 旧文件（train_char.bin）；'v2' = train_char_v2.bin（prepare.py --out-prefix v2）。
# ★ 必须定义在下面 config_keys 快照**之前**：load_config 只覆盖已存在的全局，
#   定义在它之后会被这里的赋值静默改回默认值（本项目已经栽过一次同类坑）。
data_prefix = ''
# --- document packing（样本打包 + 块对角注意力掩码，analysis/doc_packing.md）---
# 语料是一条扁平 token 流，全域随机窗口会横跨多个样本；<eos> 不足以当边界（非对话 block
# 整块没有 <eos>）。prepare.py --emit-offsets 产出与 bin 对齐的 block 边界表 `.off`，
# 训练时据此算 sample_id 传给模型，注意力只在同一样本内（杜绝跨样本污染）。
# ★ 默认 False = 完全走旧路径（不读 .off、不传 sample_id），逐位向后兼容。
# ★★ 两个键都必须定义在下面 config_keys 快照**之前**（铁律 8），否则 yaml/CLI 设了会被静默改回默认。
use_doc_packing = False   # True：启用样本打包（需同名 .off sidecar，缺失则报错不静默退化）
pack_align = False        # ★★ 2026-09-13 由 True 翻成 False：块对齐模式有**结构性覆盖漏洞** ——
                          # 窗口只有 T 长且必须结束在 block 边界 ⇒ 位置 p 可达 ⟺ p 落在某个边界前
                          # T 个 token 内 ⇒ 比 T 长的块，前 L−T 个 token **永远进不了任何窗口**
                          # （最后一个 block 整体不可达）。实测不可达 train token：
                          # v3_dlg 67.39% / v3_lang 67.05% / v3_know 72.91% / v2 70.67%
                          # （analysis/packing_audit.md；packing.unreachable_token_fraction 可复算）。
                          # False = 全域随机窗口 + **同一张块对角掩码**（覆盖 ≈100%，I1 一样成立；
                          # 代价 analysis/doc_packing.md §3.4：+2.3% vs 块对齐 +5.2%）。
                          # True 仅保留用于复现旧实验 / A-B 对照。
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
load_config(globals()) # 从 YAML 配置文件或命令行覆盖
# ★ 派生开关必须**在这里**算：`load_config` 之前 `ndb_slots` 还是默认值 0，
#   写在键定义处会永远算出 False ⇒ NDB 静默不启用（2026-09-15 冒烟实测踩到）。
use_ndb = int(ndb_slots) > 0     # 启用 NDB 的唯一判据
config = {k: globals()[k] for k in config_keys} # 对日志记录很有用
# 字级模式联动（--char-level=true）：汉字=1 token → block 256（≈BPE 上下文）、
# CSA 默认参数（16 字/块、64 字/窗）、MTP 可开；mask 标记按字级 id。
if char_level:
    block_size = 256
    from tokenizers import Tokenizer as _Tok
    _cv = _Tok.from_file(os.path.join('data', dataset, 'char_tokenizer.json')).get_vocab()
    # 去标签 masking（dev-notes/61）：数据不再用「用户：/模型：」标签，改由每条模型回复
    # 后的终止符 <eos>/<cont> 定位回复行（新三区词表：<eos>=117, <cont>=119）。
    if mask_mode == 'resp_span':
        # resp_span 的真正判定用 <resp>/<eos>；mask_reply_ids 这里只服务"窗口非空"打包判据
        #（窗口内含 <eos> ⟺ 该窗口有有效 token，这条在 resp_span 下依然成立）。
        mask_reply_ids = [_cv['<eos>']]
        mask_resp_ids = [_cv['<resp>']]
    else:
        mask_reply_ids = [_cv['<eos>'], _cv['<cont>']]
    mask_sep_ids = [_cv['\n'], _cv['\n']]
# 字节直入模式联动（--byte-level=true）：中文每字 3 字节 → block 放大保持有效上下文；
# vocab_size 由 meta_byte.pkl（257）提供。CSA 参数作用于**聚合后**的 token（1 聚合=1 字），
# 与 BPE 等价映射：compress 16 字/块、window 64 字/窗。
elif byte_level:
    block_size = 510      # 510 字节 = 170 聚合 token（3 的倍数，≈170 字）
    csa_compress = 16     # 块 16 聚合 token ≈ 16 字（与 BPE 一致）
    csa_window = 64       # 滑窗 64 聚合 token ≈ 64 字（与 BPE 一致）
    use_mtp = False       # MTP 依赖 wte 嵌入，字节模式关
    # loss masking 标记改字节序列（<eos>=256；字级 <cont> 在字节模式暂不启用，只用 eos 定位）
    mask_reply_ids = [0x0100]                       # 256 = <eos>
    mask_sep_ids = [0x0a, 0x0a]                     # \n\n
# mask_mode 合法性：**写错值绝不能静默退回 eos_line** —— 观测到的后果是两回事：
# 'resp_span' 写错成 'resp-spain' 会安静地用整行规则，loss 照样下降（只是多算了 <resp>），
# 而 'eos_line' 写错成 'resp_span' 在非单流语料上会给出**全 False** 的 mask ⇒ loss NaN。
# 断言放在这里（config_keys 之后、消费点之前），配置一加载就炸。
assert mask_mode in ('eos_line', 'resp_span'), (
    f"mask_mode={mask_mode!r} 未知；只认 'eos_line'（行内含 <eos>/<cont> ⇒ 整行算 loss）"
    f" 或 'resp_span'（<resp> 之后 → 对应 <eos> 含，**不含 <resp>**，单流格式用）")
if mask_mode == 'resp_span':
    # 单流格式下 mask_reply_ids 只服务"窗口非空"打包判据；真正判定用 mask_resp_ids。
    assert mask_resp_ids, ("mask_mode='resp_span' 但 mask_resp_ids 为空 —— "
                           "字级模式会在上面按词表填 <resp>；这里为空说明走了 "
                           "BPE/字节分支，而 resp_span 目前只支持字级")
# 打包非空窗口：终止符 id 列表（空 = 不启用，见 get_batch / _sample_nonempty_ix）
_pack_terms = []
if pack_nonempty and use_loss_masking and ('stage' not in globals() or stage != 'pretrain'):
    _pack_terms = [int(t) for t in mask_reply_ids if t is not None]
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

# 抑制 Inductor / Dynamo 重复编译告警，放宽重编译上限
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.recompile_limit = 32

device_type = 'cuda' if 'cuda' in device else 'cpu' # 供后面 torch.autocast 使用
# 注意：float16 数据类型会自动使用 GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# 简易数据加载器
data_dir = os.path.join('data', dataset)

# 数据溯源：记录数据清单的哈希，方便追查“这个模型用的哪版数据”。
# ★ 2026-09-13 修：原来只找 manifest.json，而 v3 的清单叫 manifest_v3_*.json ⇒ **根本没读到**
#   （manifest.json 还是 9 月 8 日 v2 时代的旧文件）。现在按 data_prefix 优先找
#   manifest_<prefix>.json，找不到再回退 manifest.json，并**打印到底读了哪个**（或"没找到"）。
_manifest_candidates = []
if data_prefix:
    _manifest_candidates.append(os.path.join(data_dir, f'manifest_{data_prefix}.json'))
_manifest_candidates.append(os.path.join(data_dir, 'manifest.json'))
data_manifest_path = next((p for p in _manifest_candidates if os.path.exists(p)), None)
if data_manifest_path is not None:
    try:
        with open(data_manifest_path, 'rb') as _f:
            _man_sha = hashlib.sha256(_f.read()).hexdigest()
        config['data_manifest_sha256'] = _man_sha
        config['data_manifest_name'] = os.path.basename(data_manifest_path)
        print(f'数据清单：{data_manifest_path}（sha256 {_man_sha[:16]}…）')
    except OSError as _e:
        print(f'warning: 读取数据清单 {data_manifest_path} 失败：{_e}')
else:
    print(f'warning: 未找到数据清单（试过 {_manifest_candidates}）')

# 数据集文件名统一解析：train / val / meta 三个名字只在这里算一次，
# 下游（get_batch / meta 加载 / summary 打印）一律复用，杜绝"改一处漏一处"。
# data_prefix 已在上面 config_keys 之前定义（见那里的注释）。
_train_bin, _val_bin, _meta_bin = pick_bin_names(
    char_level=char_level, byte_level=byte_level, prefix=data_prefix,
    stage=globals().get('stage'))
try:
    _train_tokens = os.path.getsize(os.path.join(data_dir, _train_bin)) // 2  # uint16
    steps_per_epoch = max(1, _train_tokens // tokens_per_iter)
except OSError:
    steps_per_epoch = None

# --- document packing：加载与 bin 逐 token 对齐的 block 边界表（.off）---
# 路径由 pick_bin_names 的产物名派生（同一解析点，不另写文件名拼接）。
# 缺失 / 与 bin token 数对不上 → **明确报错**（绝不在"打包开着"的假象下静默退回随机窗口）。
_doc_off = {}
if use_doc_packing:
    if distill_bin and p_distill > 0:
        raise SystemExit('错误：use_doc_packing 与 distill_bin/p_distill 混采不兼容'
                         '（蒸馏流没有 .off 边界表，掩码会静默漏掉那条支路）。')
    _off_sha = {}
    for _split, _bn in (('train', _train_bin), ('val', _val_bin)):
        _op = os.path.join(data_dir, _offsets_name_for_bin(_bn))
        if not os.path.exists(_op):
            raise SystemExit(
                f'错误：use_doc_packing=True 但找不到样本边界文件 {_op}。\n'
                f'  .off 是 `prepare.py --emit-offsets` 的产物，与 {_bn} 逐 token 对齐。\n'
                f'  请用同一条 prepare 命令加 --emit-offsets 重建数据；'
                f'不要静默退回随机窗口（那会让人以为打包开着）。')
        _ntok = os.path.getsize(os.path.join(data_dir, _bn)) // 2
        _doc_off[_split] = _load_offsets(_op, _ntok)   # 不合法会大声抛错
        # ★ 2026-09-13：记下 .off 的 sha256 进 config（→ 日志 + checkpoint）。
        # 长度/哨兵校验抓不到"同长度、不同批次"的错配；内容指纹让"这批权重用了哪份边界表"
        # 事后可查、可核对 manifest 里登记的 sha256。
        with open(_op, 'rb') as _f:
            _off_sha[_split] = hashlib.sha256(_f.read()).hexdigest()
    config['doc_off_sha256'] = _off_sha
    print(f"样本打包：train {len(_doc_off['train']) - 1:,} 个 block / "
          f"val {len(_doc_off['val']) - 1:,} 个 block（.off 已校验与 bin 对齐）")
    print(f"  pack_align={bool(pack_align)}"
          f"（True=块对齐，实测有覆盖漏洞；False=全域随机窗口 + 块对角掩码）")
    for _split, _h in _off_sha.items():
        print(f"  .off sha256[{_split}] = {_h}")

def build_assistant_mask(y):
    """(B, T) bool mask 的薄包装：规则与标记 id 都来自配置，实现见 `training/masking.py`。

    - `mask_mode='eos_line'`（默认）：token 所在行内含 `<eos>/<cont>` ⇒ **整行**算 loss；
    - `mask_mode='resp_span'`：`<resp>` 之后 → 对应 `<eos>`（含）算 loss
      （**不含 `<resp>` 本身** —— 它是 harness 喂的，模型不该生成它）。
    """
    if mask_mode == 'resp_span':
        return _build_resp_span_mask(y, mask_resp_ids, mask_reply_ids)
    return _build_assistant_mask(y, mask_reply_ids, mask_sep_ids)


def _sample_nonempty_ix(data):
    """拒绝采样窗口起点，只保留 y 窗口内含回复终止符的（= 该窗口有有效 token）。

    与「均匀抽窗口」在分布上等价：空窗口本来就不贡献任何 loss（训练时整批 continue，
    评估时整批 skip），只是白跑一次前向。只抽非空窗口后，同样算力下拿到 ~5.8× 有效 token。
    用 torch.randint 保持随机流在 torch RNG 里，续训的 RNG 恢复逻辑不用改。
    """
    hi = len(data) - block_size
    out, tries = [], 0
    limit = 200 * batch_size
    while len(out) < batch_size and tries < limit:
        k = max((batch_size - len(out)) * 8, 32)
        for i in torch.randint(hi, (k,)).tolist():
            tries += 1
            w = data[i + 1: i + 1 + block_size]      # 判据看 y（+1 之后）那个窗口
            if any((w == t).any() for t in _pack_terms):
                out.append(i)
                if len(out) == batch_size:
                    break
    if len(out) < batch_size:
        # 兜底：语料里几乎没有终止符时退回均匀采样，绝不死循环
        out = torch.randint(hi, (batch_size,)).tolist()
    return torch.tensor(out, dtype=torch.long)


def get_batch(split):
    # 我们每个 batch 都重新创建 np.memmap，以避免内存泄漏，参见
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        # 预算中性混合：以概率 p_distill 从蒸馏数据切块；distill.bin 太短（< 2*block_size
        # 个字节，即不足一个窗口）时退回主数据，避免 randint 越界。
        # 主数据路径来自 _train_bin（已在上面用 pick_bin_names 统一解析）。
        use_distill = (distill_bin and p_distill > 0 and random.random() < p_distill
                       and os.path.exists(distill_bin)
                       and os.path.getsize(distill_bin) > 2 * block_size)
        path = distill_bin if use_distill else os.path.join(data_dir, _train_bin)
    else:
        path = os.path.join(data_dir, _val_bin)
    data = np.memmap(path, dtype=np.uint16, mode='r')
    sid = None        # 给模型的 (B,T) 样本号（与 x 对齐；未启用打包 = None）
    sid_ext = None    # 给 label 掩码的 (B,T+1) 样本号（多一格，见下面的 -100 注释）
    if use_doc_packing:
        # 样本打包：窗口起点来自 .off 边界（块对齐贪心，仅 pack_align=True）或全域随机；
        # 两种都算块对角 sample_id。用对应 split 的边界表（train/val 各自一份），保证两端口径一致。
        _off = _doc_off['train'] if split == 'train' else _doc_off['val']
        if pack_align:
            ix = torch.from_numpy(_aligned_pack_starts(_off, block_size, batch_size)).long()
        else:
            ix = torch.randint(len(data) - block_size, (batch_size,))
        # ★ 边界：T+1 视图多看一格，所以要求 max(start)+T+1 <= len(data)。
        #   随机路径 start ∈ [0, len-T-1] ⇒ 恰好 <= len；对齐路径
        #   start+T = off[e] <= off[n_blocks-1] < len ⇒ 也成立。这里显式验一次，
        #   免得将来改采样逻辑时静默越界（data[i:i+T] 会切成短片段，stack 才报一句看不懂的错）。
        _max_start = int(ix.max())
        if _max_start + block_size + 1 > len(data):
            raise ValueError(
                f'打包窗口越界：max(start)+T+1 = {_max_start + block_size + 1} > len(data) = {len(data)}'
                f'（split={split}, pack_align={bool(pack_align)}）。'
                f'sample_id 用 T+1 视图需要多看一格。')
        sid_ext = _sample_id_in_window(_off, ix, block_size + 1, device=device)   # (B, T+1)
        sid = sid_ext[:, :block_size]        # ★ 模型侧仍必须是 (B,T)：attention 的 tril 是 T×T
    else:
        ix = _sample_nonempty_ix(data) if _pack_terms else torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if sid_ext is not None:
        # ★ 样本打包下的**边界 label 泄漏**：位置 t 的 label 是 token t+1，若 t 与 t+1
        # 不属于同一样本，这条标签就是在要求"用 A 样本的结尾预测 B 样本的开头" —— 纯噪声
        # （且正是打包要消除的跨样本污染）。掩码只管注意力，管不到 y 的错位，必须显式置 -100。
        # ★★ 2026-09-13：这里传的是 (B, T+1)（比 y 多一格）—— 旧版只算 T 格，导致
        # **每个块对齐窗口的最后一位 label 固定跨样本却漏屏蔽**（实测 100% 命中，
        # 占全部 label 的 0.3906%）。模型侧拿的仍是上面的 (B,T)。
        # 详见 analysis/packing_audit.md 与 packing.py 的 docstring。
        y = _mask_cross_sample_labels(y, sid_ext)
    # loss masking：两阶段框架——pretrain 无掩码全 token 语言建模（DeepSeek 路线）；
    # sft/full 只对 assistant 回复 token 算 loss（chat 微调惯例）。
    # ★ 哨兵值是 **-100**（model/gpt.py 的 F.cross_entropy(ignore_index=-100)）；
    #   旧注释写的 -1 是错的（-1 是合法 token id）。见 analysis/packing_audit.md Q4。
    # use_loss_masking=False 时全部 token 参与训练（非对话语料 / 纯预训练）。
    if use_loss_masking and ('stage' not in globals() or stage != 'pretrain'):
        y[~build_assistant_mask(y)] = -100
    if device_type == 'cuda':
        # 固定 x、y 的内存，这样我们可以异步（non_blocking=True）把它们搬到 GPU
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y, sid

# 在这里初始化，如果 init_from='resume'（即从 checkpoint）可以覆盖
iter_num = 0
best_val_loss = 1e9
raw_best_val = 1e9      # 原始 val 最优（不受体检门控，早停/日志用）
_ndb_resume = None      # NDB 接口参数（若 checkpoint 里存了）
_ndb_opt_resume = None  # NDB 接口优化器状态
_resume_iter = -1       # 续训载入时的 iter_num：跳过该步的评估/存档（否则每次续训都白写 2GB 检查点，且评估会消耗 RNG 打乱数据流）

# 尝试从数据集推导 vocab_size（字节直入模式用 meta_byte.pkl，vocab 257）
meta_path = os.path.join(data_dir, _meta_bin)
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
                  moe_hidden_scale=moe_hidden_scale,
                  use_mla=use_mla, kv_lora_rank=kv_lora_rank, qk_rope_head_dim=qk_rope_head_dim,
                  use_mtp=use_mtp, n_mtp=n_mtp, mtp_weight=mtp_weight,
                  use_muon=use_muon, muon_momentum=muon_momentum, muon_ns_steps=muon_ns_steps,
                  muon_ns_aggressive=muon_ns_aggressive,
                  muon_lr_scale=muon_lr_scale, muon_split=muon_split,
                  use_csa=use_csa, csa_compress=csa_compress, csa_topk=csa_topk,
                  csa_window=csa_window, use_hca=use_hca, use_csa_learnable=use_csa_learnable,
                  use_kv_memory=use_kv_memory, kv_memory_latent=kv_memory_latent,
                  kv_memory_chunk=kv_memory_chunk, kv_memory_checkpoint=kv_memory_checkpoint,
                  kv_memory_complement_gate=kv_memory_complement_gate, kv_memory_layers=kv_memory_layers,
                  kv_memory_block=kv_memory_block, kv_memory_delta=kv_memory_delta,
                  kv_memory_output_gate=kv_memory_output_gate, sample_boundary_reset=sample_boundary_reset,
                  use_csa_fused_qkv=use_csa_fused_qkv, use_csa_bmm=use_csa_bmm,
                  use_attn_sink=use_attn_sink,
                  use_mhc=use_mhc, hc_mult=hc_mult,
                  use_cd=use_cd, cd_iters=cd_iters,
                  use_lightning_indexer=use_lightning_indexer, num_hash_layers=num_hash_layers,
                  block_order=block_order, no_attn_layers=no_attn_layers,
                  n_memory_tokens=n_memory_tokens,
                  use_lse_residual=use_lse_residual,
                  use_lse_gate=use_lse_gate,
                  use_qk_norm=use_qk_norm,
                  z_loss_weight=z_loss_weight,
                  gradient_checkpointing=gradient_checkpointing,
                  byte_level=byte_level, char_level=char_level,
                  factorized_emb_dim=factorized_emb_dim)
# 字级/字节模式下，把 <eos> 的真实 token id 注入 config，供样本边界重置/终止检测使用
# （v3 稀疏词表 <eos>=128；字节模式 <eos>=256；旧字级 <eos>=117/121 等自动对齐）
if char_level:
    model_args['eos_token_id'] = _cv['<eos>']
elif byte_level:
    model_args['eos_token_id'] = 0x0100  # 256 = <eos>
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
              'use_sqrtsoftplus', 'route_scale', 'moe_hidden_scale',
              'use_mla', 'kv_lora_rank', 'qk_rope_head_dim',
              'use_mtp', 'n_mtp', 'mtp_weight',
              'use_muon', 'muon_momentum', 'muon_ns_steps', 'muon_ns_aggressive',
              'muon_lr_scale', 'muon_split',
              'use_csa', 'csa_compress', 'csa_topk', 'csa_window',
              'use_hca', 'use_csa_learnable', 'use_csa_fused_qkv', 'use_csa_bmm',
              'use_kv_memory', 'kv_memory_latent', 'kv_memory_chunk', 'kv_memory_checkpoint',
              'kv_memory_complement_gate', 'kv_memory_layers', 'kv_memory_block', 'kv_memory_delta',
              'kv_memory_output_gate', 'sample_boundary_reset',
              'use_attn_sink', 'use_mhc', 'hc_mult', 'use_cd', 'cd_iters',
              'use_lightning_indexer', 'num_hash_layers', 'block_order', 'no_attn_layers',
              'n_memory_tokens', 'use_lse_residual', 'use_lse_gate',
               'use_qk_norm', 'z_loss_weight', 'factorized_emb_dim']:
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
    # 从 checkpoint 恢复训练：优先 last.pt（最新进度，防止意外中断丢进度），
    # last.pt 缺失时回退 best.pt（兼容旧版只有 best.pt 的情况）。
    last_path = os.path.join(out_dir, 'last.pt')
    best_path = os.path.join(out_dir, 'best.pt')
    ckpt_path = last_path if os.path.exists(last_path) else best_path
    print(f"正在从 {os.path.basename(ckpt_path)} 恢复训练（断点续训：iter/优化器/学习率/RNG 全量恢复）")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = _build_model_from_checkpoint(checkpoint)
    iter_num = checkpoint['iter_num']
    _resume_iter = iter_num  # 该步不重复评估/存档（见评估块的条件）
    best_val_loss = checkpoint['best_val_loss']
    raw_best_val = checkpoint.get('raw_best_val', best_val_loss)  # 早停基准（旧 ckpt 无此字段则用 best_val_loss）
    # 确定性续训：恢复全部随机源状态（缺省字段时静默跳过，兼容旧 checkpoint）。
    # 数据加载是随机采样（无顺序进度指针），靠这四路 RNG 复现采样序列。
    # 注：torch.set_rng_state 严格要求 CPU ByteTensor；torch.load(..., map_location=device) 会将其迁到 GPU，需转回 CPU。
    if checkpoint.get('rng_state') is not None:
        torch.set_rng_state(checkpoint['rng_state'].cpu())
    if checkpoint.get('cuda_rng_state') is not None and torch.cuda.is_available():
        cuda_rng = checkpoint['cuda_rng_state']
        if isinstance(cuda_rng, list):
            cuda_rng = [s.cpu() if hasattr(s, 'cpu') else s for s in cuda_rng]
        torch.cuda.set_rng_state_all(cuda_rng)
    if checkpoint.get('python_rng_state') is not None:
        random.setstate(checkpoint['python_rng_state'])
    if checkpoint.get('numpy_rng_state') is not None:
        np.random.set_state(checkpoint['numpy_rng_state'])
    resume_scaler_state = checkpoint.get('scaler_state')
    _ndb_resume = checkpoint.get('ndb')  # NDB 接口参数（旧 checkpoint 无此字段则为 None）
    _ndb_opt_resume = checkpoint.get('ndb_opt')
elif init_from.endswith('.pt'):
    # 在已有模型上做后训练：加载权重，但从头开始新的优化器/学习率计划
    print(f"正在从 {init_from} 加载已有模型权重（后训练，优化器/学习率重置）")
    checkpoint = torch.load(init_from, map_location=device, weights_only=False)
    model = _build_model_from_checkpoint(checkpoint)
    # iter_num / best_val_loss 保持初始值（0 / 1e9），全新训练
else:
    raise ValueError(f"不支持的 init_from：{init_from}（应为 'scratch' / 'resume' / '<路径>.pt'）")
model.to(device)

# 初始化 GradScaler。如果 enabled=False，scaler 是空操作
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))
# resume 时恢复 GradScaler 的 scale 状态（float16 训练中断恢复不丢失 scale）
if init_from == 'resume' and locals().get('resume_scaler_state') is not None:
    scaler.load_state_dict(resume_scaler_state)

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
    # 把**实际读的文件名**打出来：data_prefix 配错时（例如忘了设就从旧数据开跑 2.7 天）
    # 只看 "数据集 chinese" 是发现不了的 —— 必须是那个真实的文件名。
    _tok_note = f"{_train_tokens/1e6:.1f}M token" if steps_per_epoch else "文件缺失"
    print(f"  数据集    {dataset} · {model.config.vocab_size} 词表 · 上下文 {block_size}")
    print(f"  数据文件  train={_train_bin}（{_tok_note}）  val={_val_bin}")
    print(f"  模型      {n_layer} 层 · {n_head} 头 · {n_embd} 维 · {attn} · {ffn}")
    opt_name = 'Muon' if use_muon else 'AdamW'
    print(f"  优化器    {opt_name} · lr {learning_rate:g} · wd {weight_decay:g} · betas ({beta1:g}, {beta2:g})")
    epoch_note = f" · ≈{max_iters/steps_per_epoch:.1f} epoch" if steps_per_epoch else ""
    print(f"  训练      {max_iters} 步 · {tokens_per_iter:,} tokens/步{epoch_note}")
    print(f"  设备      {device} · {dtype} · compile {'开' if compile else '关'}")
    es_note = (f"开（patience={patience}·min_improve={min_val_improve}·min_iters={min_iters}）"
               if enable_early_stop else "关（训满 max_iters）")
    print(f"  早停      {es_note}")
    eval_note = f"eval_iters {eval_iters} · {'仅 val（train loss 用 EMA）' if ('eval_train_split' in globals() and not eval_train_split) else 'train+val'}"
    if 'health_enabled' in globals() and health_enabled:
        eval_note += " · 体检门控 best.pt（防坍缩）"
    print(f"  评估      {eval_note}")
    print(f"  检查点    best.pt（val 最优" + (" + 体检合格" if ('health_enabled' in globals() and health_enabled) else "") + "）+ last.pt（最新）· 续训自动从 best.pt 恢复")
    print(border)
    print()

if master_process:
    print_summary()

# --- NDB 挂载（ndb_slots=0 时完全跳过；基座与读写门控联合训练）---
ndb = None
ndb_opt = None
_ndb_params = []
_ndb_state = {"on": True, "h": None, "coverage": 0.0}


def _ndb_blend(logits, loss, X, Y):
    """把 loss 里的 CE 项换成「NDB 混合分布」口径，其余附加项（MoE 等）原样保留。

    `p_new = (1-g)·p_model + g·p_ng`（g 与多级混合 α 都由模型门控决定），所以
    `loss = CE(p_model) + 附加项` → `CE(p_new) + 附加项`：先扣掉原 CE 再加新的 CE。
    NDB 未启用、或 `_ndb_state['on']=False`（对照臂）时原样返回。
    """
    if ndb is None or not _ndb_state['on']:
        return loss
    h = _ndb_state.get('h')
    if h is None:      # 还没跑过前向（理论上不会走到）
        return loss
    V = logits.size(-1)
    with torch.no_grad():   # 只用来扣掉原 CE 项，不需要它的梯度
        ce_base = F.cross_entropy(logits.reshape(-1, V), Y.reshape(-1),
                                  ignore_index=-100)
    p_model = F.softmax(logits.float(), dim=-1)
    p_new, st = ndb.read(h, X, p_model, targets=Y)
    # ⚠ 必须 clamp：p_new 是凸组合，理论上 >0，但 fp32 下极小项取 log 会 -inf
    ce_ndb = F.cross_entropy(torch.log(p_new.clamp_min(1e-9)).reshape(-1, V),
                             Y.reshape(-1), ignore_index=-100)
    _ndb_state['coverage'] = float(st.covered)
    return loss - ce_base + ce_ndb


if use_ndb:
    from model.ngram_ndb import NgramNDB
    _ndb_levels = tuple(int(s) for s in str(ndb_levels).strip('()').split(',') if s.strip())
    ndb = NgramNDB(
        n_embd=model.config.n_embd, levels=_ndb_levels, slots=ndb_slots,
        top_k=ndb_top_k, vocab_size=model.config.vocab_size,
        max_table_gb=ndb_max_table_gb).to(device)

    # ★ h = **ln_f 的输入**（走完所有 block、最终 LayerNorm 之前），与探针
    #   `scripts/ndb_online_ab.py:99` 的口径一致；读门控与写门控都吃它。
    #   pre-hook 只做一次张量赋值，用 `_dynamo.disable` 拦住编译图。
    @torch._dynamo.disable
    def _ndb_capture(m, args):
        _ndb_state['h'] = args[0]
        return None

    model.transformer.ln_f.register_forward_pre_hook(_ndb_capture)

    if _ndb_resume is not None:
        # 只恢复门控参数；表**不持久化**（每次 run 从空表在线重建）
        ndb.load_state_dict(_ndb_resume, strict=False)
        if master_process:
            print("  NDB        从 checkpoint 恢复门控参数（表从空开始，在线重建）")
    _ndb_params = [p for p in ndb.parameters() if p.requires_grad]
    # 独立 AdamW：MuonAdamW 不支持 add_param_group；独立优化器还能把门控 lr 固定住
    ndb_opt = torch.optim.AdamW(_ndb_params, lr=ndb_lr, weight_decay=0.0)
    if _ndb_opt_resume is not None:
        try:
            ndb_opt.load_state_dict(_ndb_opt_resume)
        except Exception as _e:  # noqa: BLE001
            if master_process:
                print(f"  NDB        优化器状态载入失败（{_e}）→ 重建（丢弃旧动量）")
            ndb_opt = torch.optim.AdamW(_ndb_params, lr=ndb_lr, weight_decay=0.0)
        else:
            _in_opt = {id(p) for g in ndb_opt.param_groups for p in g['params']}
            _miss = [n for n, p in ndb.named_parameters()
                     if p.requires_grad and id(p) not in _in_opt]
            if _miss:
                if master_process:
                    print(f"  NDB        优化器状态缺 {_miss} → 重建（丢弃旧动量）")
                ndb_opt = torch.optim.AdamW(_ndb_params, lr=ndb_lr, weight_decay=0.0)
    if master_process:
        print(f"  NDB        levels={_ndb_levels} · 每级 {ndb_slots/1e6:.0f}M 槽 · "
              f"top_k {ndb_top_k} · 表 {ndb.table_gb():.2f}GB · "
              f"门控 {sum(p.numel() for p in _ndb_params)} 参数 @ lr {ndb_lr:g} · "
              f"每 {ndb_flush_every} 步落表 · 表不持久化（每个 run 在线重建）")

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
def estimate_loss(splits=('train', 'val')):
    """train/val loss 口径 = **带 NDB**（与部署形态一致）；NDB 的净贡献看 ndb.csv 的配对 Δ。"""
    out = {}
    model.eval()
    for split in splits:
        tot, n = 0.0, 0
        for k in range(eval_iters):
            X, Y, SID = get_batch(split)
            n_i = int((Y != -100).sum().item())
            if n_i == 0:
                continue  # 全 mask 窗口：无有效 token，跳过（否则 loss 为 nan）
            with ctx:
                logits, loss = model(X, Y, sample_id=SID)
            loss = _ndb_blend(logits, loss, X, Y)
            # NaN 防护：train 数据某些窗口无 <eos>/<cont> → mask 全 -100 → loss 为 nan。
            # 跳过这些无效 batch，只对有限 loss 求均值（与训练循环的 step_nan 防护对齐）。
            if torch.isfinite(loss):
                # token 级加权平均：与训练损失同口径（旧版是窗口均值，两者不可比）
                tot += loss.item() * n_i
                n += n_i
        out[split] = torch.tensor(tot / n) if n else float('nan')
    model.train()
    _ndb_state['h'] = None   # 别把验证集前向的图留着（下个训练前向会重新写）
    return out


@torch.no_grad()
def ndb_eval():
    """NDB 监控：mem-on / base-off 两个 val loss（**同一批 val batch 配对**）。

    Δ = mem − base_off     NDB 有没有用（负 = 有用）
    coverage               读的时候有槽命中的位置占比；表冷的时候 Δ 自然 ≈ 0
    """
    if ndb is None:
        return None
    was_training = model.training
    model.eval()

    # 配对评估：两个条件用同一批 val batch。否则各条件抽到不同窗口，
    # 采样噪声（±0.1）会淹没 Δ（±0.03），Δ 变成纯噪声（2026-09-10 实测教训）。
    batches = [get_batch('val') for _ in range(ndb_eval_iters)]

    def _run(on=True):
        _ndb_state['on'] = on
        tot, n = 0.0, 0
        for X, Y, SID in batches:
            n_i = int((Y != -100).sum().item())
            if n_i == 0:
                continue
            with ctx:
                logits, loss = model(X, Y, sample_id=SID)
            loss = _ndb_blend(logits, loss, X, Y)
            if torch.isfinite(loss):
                tot += loss.item() * n_i  # token 级加权，与训练损失同口径
                n += n_i
        return tot / max(n, 1)

    res = {'mem': _run(True), 'off': _run(False)}
    _ndb_state['on'] = True
    _ndb_state['h'] = None      # ★ 绝不能让 val 的隐藏态漏进训练侧写入门控
    if was_training:
        model.train()
    return res


def ngram_rep(s, n):
    """字符级 n-gram 重复率（与 training/health_check_hello.py 同口径）。"""
    seq = re.sub(r"\s+", "", s)
    if len(seq) < 2 * n:
        return 0.0
    g = [seq[i:i + n] for i in range(len(seq) - n + 1)]
    c = Counter(g)
    return sum(1 for x in g if c[x] > 1) / len(g)


def _health_generate(model, tok, prompt, seed):
    """GPU 版原始生成（镜像 inference/scripts/sample_py.generate_ids 的生成语义：
    温度 → 重复惩罚 → top-k → softmax → multinomial；原始生成不截断，EOS 只记录不停止）。

    与 generate_ids 的差异：tensor 显式放模型设备（generate_ids 的
    torch.tensor([context_ids]) 只接受 list，无法传 CUDA 张量）；用独立
    Generator 采样，不污染训练 RNG。EOS 不进入输出，eos_pos = 生成区内位置。
    """
    g = torch.Generator(device=device).manual_seed(seed)
    idx = torch.tensor([tok.encode(prompt).ids], dtype=torch.long, device=device)
    new_start = idx.shape[1]
    seen = idx[0].tolist()
    eos_id = tok.token_to_id("<eos>")
    eos_pos = -1
    for _ in range(health_max_new):
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        logits, _ = model(idx_cond)
        v = logits[0, -1, :].clone() / health_temp
        if health_rep_penalty > 1.0:
            for t in seen:
                l = v[t]
                v[t] = l / health_rep_penalty if l >= 0 else l * health_rep_penalty
        k = min(health_top_k, v.size(-1))
        topv, _ = torch.topk(v, k)
        v[v < topv[-1]] = float('-inf')
        nxt = int(torch.multinomial(torch.softmax(v, dim=-1), 1, generator=g).item())
        if nxt == eos_id:
            eos_pos = len(seen) - new_start
            break
        seen.append(nxt)
        idx = torch.cat((idx, torch.tensor([[nxt]], dtype=torch.long, device=device)), dim=1)
    return seen, eos_pos


def run_health_check(model, tok):
    """固定 prompt 采样体检（你好体检精简版）：EOS 自吐率 / 平均长度 / rep3 / 续轮率。

    口径与 training/health_check_hello.py 一致（temp 0.8 / topk 200 / rep 1.2，
    原始生成不截断）。用未编译模型跑（调用方传 unoptimized_model），避免评估态
    变长序列触发 inductor 逐长度重编译。返回 dict。
    """
    prompts = [p for p in health_prompts.split('|') if p]
    rows = []
    was_training = model.training
    model.eval()
    try:
        for prompt in prompts:
            plen = len(tok.encode(prompt).ids)
            for s in range(health_seeds):
                ids, eos_pos = _health_generate(model, tok, prompt, s)
                text = tok.decode(ids[plen:])
                rows.append(dict(
                    eos=eos_pos >= 0,
                    length=len(ids) - plen,
                    rep3=ngram_rep(text, 3),
                    turns=max(len(re.findall(r'用户[:：]', text)),
                              len(re.findall(r'模型[:：]', text))),
                ))
    finally:
        model.train(was_training)
    n = max(len(rows), 1)
    eos_rate = sum(1 for r in rows if r['eos']) / n
    rep3 = sum(r['rep3'] for r in rows) / n
    return dict(
        eos_rate=eos_rate,
        avg_len=sum(r['length'] for r in rows) / n,
        rep3=rep3,
        turns_rate=sum(1 for r in rows if r['turns'] >= 2) / n,
        health_ok=eos_rate >= health_min_eos_rate and rep3 <= health_max_rep3,
    )

# 学习率衰减调度器（带预热的余弦 / WSD）
# 实现与全部边界条件在 training/schedules.py::lr_at（有单元测试），这里只注入全局。
def get_lr(it):
    return _lr_at(
        it,
        learning_rate=learning_rate, min_lr=min_lr,
        warmup_iters=warmup_iters, lr_decay_iters=lr_decay_iters,
        schedule=schedule, stable_frac=stable_frac,
    )

# 日志
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

def _set_indexer_freeze(model, frozen):
    """GLM-5 索引器预热：frozen=True 时只有 idx_q/idx_k（lightning indexer）可训练。

    其余参数 requires_grad=False → 前向仍计算、反向不再产生梯度，优化器自动跳过
    （grad=None），主模型等效冻结。注意：aux-free 路由偏置在 MoE.forward 里原地
    更新（balance_factor，不经过优化器），预热期仍会微调——幅度 0.001 且自校正，
    可接受（相当于路由偏置顺带预热）。解冻后 requires_grad 恢复 True。
    """
    for n, p in model.named_parameters():
        is_indexer = 'idx_q' in n or 'idx_k' in n
        p.requires_grad = (not frozen) or is_indexer


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
# ★ 续训时**追加**而不是截断（2026-09-11 修）：原来无条件用 'w'，
#   暂停/重启一次就把之前的指标历史清空（实测：step 22000 重启后 967 字节 → 0 字节）。
#   策略实现在 training/run_logs.py，有单测覆盖。
results_csv = None
csv_writer = None
if master_process:
    _rcsv_path = os.path.join(out_dir, 'results.csv')
    _had = count_csv_rows(_rcsv_path) if init_from == 'resume' else 0
    results_csv, csv_writer, _rcsv_appended = open_run_csv(
        _rcsv_path, ['step', 'train/loss', 'val/loss', 'lr', 'mfu', 'time'],
        resuming=(init_from == 'resume'))
    if _rcsv_appended:
        print(f"📈 results.csv 已存在 {_had} 行 → **追加**（不截断历史）")
    elif init_from == 'resume':
        print("📈 results.csv 为空或不存在 → 新建")

# NDB 监控 CSV（启用 NDB 时；每评估点一行）—— 同样的续写策略
ndb_csv = None
ndb_csv_writer = None
if master_process and use_ndb:
    ndb_csv, ndb_csv_writer, _ = open_run_csv(
        os.path.join(out_dir, 'ndb.csv'),
        ['iter', 'base_off', 'mem', 'delta', 'write_gate_bias', 'coverage'],
        resuming=(init_from == 'resume'))

# 健康体检初始化（health_enabled 时加载分词器 + 打开 health.csv；失败则本次跳过体检）
health_tok = None
health_csv = None
health_csv_writer = None
if master_process and health_enabled:
    try:
        from tokenizers import Tokenizer
        # 与模型词表一致：字级/字节直入各有独立 tokenizer（dev-notes/50/61）
        _tk_path = ('byte_tokenizer.json' if byte_level
                    else 'char_tokenizer.json' if char_level
                    else 'tokenizer.json')
        health_tok = Tokenizer.from_file(os.path.join(data_dir, _tk_path))
        health_csv = open(os.path.join(out_dir, 'health.csv'), 'w', newline='', encoding='utf-8')
        health_csv_writer = csv.writer(health_csv)
        health_csv_writer.writerow(['step', 'val/loss', 'eos_rate', 'avg_len', 'rep3', 'turns_rate', 'health_ok'])
    except Exception as e:
        print(f"warning: 健康体检初始化失败，本次训练跳过体检（{e}）")
        health_enabled = False
        health_tok = None
        health_csv = None
        health_csv_writer = None

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
X, Y, SID = get_batch('train') # 获取第一个 batch
t0 = time.time()           # t0 在每轮迭代末尾会被重置（用于测单步速度算 MFU）
train_start = time.time()  # results.csv 里 time 列的零点（不会随迭代重置）
# ⚠️ 注意 `time` 是**本进程内**的累计秒数，**续训会归零**：
#    它只保证「同一次进程生命周期内单调」，不保证跨 resume 单调。
#    实测（2026-09-12）：step 22000 那行是 75120.0，续训后 step 23000 那行变成 3480.0。
#    ⇒ 想算整段训练时长/吞吐，请用 `time` 列的**相邻差值**，不要去和 22000 之前的历史比。
#    （改成跨 resume 累加需要先读上一行的 time 做偏移，属于待办，见 TECH_DEBT P3。）
local_iter_num = 0 # 本进程生命周期内的迭代次数
raw_model = model.module if ddp else model # 如果需要，解开 DDP 容器
running_mfu = -1.0
grad_norm = 0.0     # 首步 OOM/NaN 跳过时日志仍可用
_oom_steps = 0      # 因显存尖峰被跳过的步数（见微步循环里的 OOM 防护）
_oom_dumps = 0      # 已落盘的 OOM 现场次数（限流，避免刷爆磁盘）
_mem_snap_done = False
# 训练侧窗口均值（2026-09-10）：单步 loss 只有 ~550 个有效 token，且其中 ~41% 来自同一个
# 窗口（实测），逐点看就是纯抽样噪声（0.77 / 3.40 / 2.25）。进度条改成「最近 log_interval
# 步的 token 加权均值」才看得见趋势；同时落一份 CSV 便于事后画曲线。
_win_sum, _win_n = 0.0, 0
loss_win_path = os.path.join(out_dir, 'train_loss_window.csv')
if master_process and not os.path.exists(loss_win_path):
    with open(loss_win_path, 'w') as _f:
        _f.write('step,window_mean,window_steps,last_step_loss,grad_norm,lr\n')
if mem_snapshot_gb > 0 and master_process:
    # 显存分配历史：峰值超阈值时 dump，用来定位一次性尖峰的真正来源
    torch.cuda.memory._record_memory_history(max_entries=200000)
    print(f"  [诊断] 显存分配历史已开启，峰值 >{mem_snapshot_gb:g}GB 时 dump 快照")
running_train_loss = None   # 训练侧 loss EMA（eval_train_split=false 时 results.csv 用它）
# tqdm 进度条：DDP 下只有主进程显示
pbar = tqdm(total=max_iters, initial=iter_num, desc="训练中", dynamic_ncols=True) if master_process else None
loss_history = []  # 每个评估点记 (iter, train_loss, val_loss)，训练结束画曲线图用
early_stopped = False  # 早停是否触发（收尾打印用）
no_improve_count = 0   # val 连续无实质改善的评估次数（早停计数）

# --- GLM-5 索引器预热（可选）：冻结主模型，前 N 步只训练 lightning indexer ---
# 配方出处（2026-02 GLM-5 技术报告）：DSA 适配先 1000 步只训索引器、主模型冻结，
# 再 20B token 稀疏适配，追平 DeepSeek 943.7B token 的 DSA 训练效果。
_idx_warmup_active = False
if indexer_warmup_steps > 0:
    assert use_lightning_indexer, \
        "indexer_warmup_steps>0 需要 use_lightning_indexer=True（没有索引器就没东西可预热）"
    idx_names = [n for n, p in raw_model.named_parameters() if 'idx_q' in n or 'idx_k' in n]
    assert idx_names, f"未找到 lightning indexer 参数（idx_q/idx_k），当前模型没有索引器"
    _set_indexer_freeze(raw_model, True)
    _idx_warmup_active = True
    if master_process:
        print(f"GLM-5 索引器预热：前 {indexer_warmup_steps} 步只训练 {len(idx_names)} 个"
              f"索引器参数（主模型冻结），随后解冻联合训练")

while True:

    # 确定并设置本次迭代的学习率
    # 按各参数组各自的 lr_ratio 缩放（Muon=0.2×、AdamW=1.0×），修复此前粗暴覆写导致
    # Muon 实际用 5 倍学习率的 bug（100M 训练 NaN 的根因之一）。
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        if param_group.get('fixed_lr') is not None:  # NDB 接口：固定 lr，不随基座调度衰减
            param_group['lr'] = param_group['fixed_lr']
        else:
            param_group['lr'] = lr * param_group.get('lr_ratio', 1.0)

    # 在 train/val 集合上评估损失并保存 checkpoint
    # 续训载入的那一步跳过：checkpoint 就是这一步存的，重评重存纯浪费（2GB 写盘 +
    # 评估的 val batch 会消耗 CPU RNG → 打乱续训后的数据流，破坏可复现性）。
    if iter_num % eval_interval == 0 and master_process and iter_num != _resume_iter:
        # 评估开销优化：eval_train_split=false 时只评 val，train/loss 用训练侧 EMA 代替
        eval_splits = ('val',) if not eval_train_split else ('train', 'val')
        losses = estimate_loss(eval_splits)
        if 'train' not in losses:
            losses['train'] = running_train_loss if running_train_loss is not None else 0.0
        # 用 pbar.write 打印到进度条上方，不打断进度条
        pbar.write(f"step {iter_num}: train 损失 {losses['train']:.4f}, val 损失 {losses['val']:.4f}")
        # --- NDB 监控：配对 Δ / 门控 / 覆盖率 ---
        if ndb is not None:
            ndb.flush()   # 监控前先落表，保证 Δ 量的是最新表
            _nr = ndb_eval()
            _d = _nr['mem'] - _nr['off']
            _gs = ndb.gate_summary()
            _cov = _ndb_state.get('coverage', 0.0)
            _fill = ndb.n_filled_slots()[0]
            pbar.write(f"[ndb {iter_num}] base_off={_nr['off']:.4f} mem={_nr['mem']:.4f} "
                       f"Δ={_d:+.4f} 槽已填={_fill/1e6:.1f}M 覆盖={_cov*100:.1f}% "
                       f"w_b={_gs['write_gate_bias']:+.3f} r_b={_gs['read_gate_bias']:+.3f} "
                       f"α={[round(a, 2) for a in _gs['level_alpha']]}")
            if ndb_csv_writer is not None:
                ndb_csv_writer.writerow([
                    iter_num, f"{_nr['off']:.4f}", f"{_nr['mem']:.4f}", f"{_d:+.4f}",
                    f"{_gs['write_gate_bias']:+.4f}", f"{_cov*100:.2f}"])
                ndb_csv.flush()
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
        # 健康体检（可选）：固定 prompt 采样，门控 best.pt 选点（防「val 骗低、采样坍缩」）
        # 用未编译模型跑（compile 后 raw_model 是 torch.compile 包装，变长生成会触发
        # inductor 逐长度重编译风暴——见冒烟实测 recompile_limit 告警）。
        health = None
        if health_enabled:
            h_interval = health_eval_interval or eval_interval
            if iter_num % h_interval == 0:
                health_model = unoptimized_model if compile else raw_model
                health = run_health_check(health_model, health_tok)
                pbar.write(
                    f"  体检: EOS 自吐 {health['eos_rate']:.0%} | 平均 len {health['avg_len']:.1f} | "
                    f"rep3 {health['rep3']:.4f} | 续轮 {health['turns_rate']:.0%} | "
                    f"{'✅ 合格' if health['health_ok'] else '⚠ 不合格'}")
                if health_csv_writer is not None:
                    health_csv_writer.writerow([iter_num, f"{losses['val']:.4f}",
                                                f"{health['eos_rate']:.3f}", f"{health['avg_len']:.1f}",
                                                f"{health['rep3']:.4f}", f"{health['turns_rate']:.3f}",
                                                int(health['health_ok'])])
                    health_csv.flush()
        if iter_num > 0:
            # YOLO 式：last.pt 每次评估都存（最新状态），best.pt 只在「val 创新低 且 体检合格」时存。
            # best_val_loss = 门控后的 best（checkpoint 实际选点）；raw_best_val = 原始 val 最优
            # （早停判断用，不受体检门控影响——坍缩模型 val 仍可能降，但 best.pt 不再跟）。
            prev_best = best_val_loss                 # 保存本次评估前的 best（门控后）
            prev_raw = raw_best_val                   # 本次评估前的原始 val 最优
            if losses['val'] < prev_raw:
                raw_best_val = losses['val']
            is_best = losses['val'] < prev_best
            if health is not None:
                is_best = is_best and health['health_ok']
            if is_best:
                best_val_loss = losses['val']
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'raw_best_val': raw_best_val,   # 原始 val 最优（早停判断用，resume 时恢复避免早停计数重置）
                'config': config,
                'epoch': iter_num / steps_per_epoch if steps_per_epoch else None,
                # 确定性续训：保存全部随机源状态，resume 时精确复现数据采样/dropout 顺序
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                'python_rng_state': random.getstate(),
                'numpy_rng_state': np.random.get_state(),
                # float16 训练时 GradScaler 的 scale 状态（bf16 下为 no-op，仍存以保证兼容）
                'scaler_state': scaler.state_dict(),
                'ndb': ndb.state_dict() if ndb is not None else None,  # NDB 接口参数
                'ndb_opt': ndb_opt.state_dict() if ndb is not None else None,
            }
            save_checkpoint_async(checkpoint, os.path.join(out_dir, 'last.pt'))
            # 逢 1000 步归档独立检查点，防止被后续最优覆盖，方便阶段性回溯审查
            if iter_num > 0 and iter_num % 1000 == 0:
                step_ckpt = os.path.join(out_dir, f'ckpt_step_{iter_num}.pt')
                save_checkpoint_async(checkpoint, step_ckpt)
                pbar.write(f"💾 归档检查点 → {step_ckpt}")
                # 保留策略（2026-09-11 晚搬进训练内部）：每 ckpt_sparse_every 步留一个回溯点
                # + 最新 ckpt_newest_keep 个；best.pt / last.pt 是固定文件名，永不删。
                # 起因：每个 ckpt ≈ 0.6GB，70000 步 / 1000 = 70 个 → 42GB。
                # ★ 为什么不再只留「最新 N 个」：那会在 step 25000 之后把
                #   ckpt_step_5000/10000/15000 这些**阶段回溯点**静默删掉。
                # ★ 为什么搬进来：外部 scripts/prune_ckpts.sh 靠巡检看守调用，
                #   而看守会在会话重启时被 SIGKILL（2026-09-11 实测丢过一次训练），
                #   ⇒ 清理不再依赖任何外部进程。策略实现在 training/checkpoints.py（有单测）。
                try:
                    _rm, _kept = prune_step_checkpoints_sparse(
                        out_dir, int(ckpt_sparse_every), int(ckpt_newest_keep))
                    if _rm:
                        pbar.write(f"🧹 清理归档 {len(_rm)} 个（每 {ckpt_sparse_every} 步留一个"
                                   f" + 最新 {ckpt_newest_keep} 个）")
                    pbar.write(f"   现存归档步号 {_kept}")
                except Exception as _e:  # 清理是尽力而为，绝不能有能力搞崩训练
                    pbar.write(f"⚠ 归档清理跳过：{type(_e).__name__}: {_e}")

            if is_best:
                save_checkpoint_async(checkpoint, os.path.join(out_dir, 'best.pt'))
                pbar.write(f"✓ 新最佳 val {best_val_loss:.4f} → best.pt（并已更新 last.pt）")
            elif health is not None and losses['val'] < prev_raw:
                # val 创新低但体检不合格：不覆盖 best.pt（防坍缩保护）
                pbar.write(f"⚠ val 创新低 {losses['val']:.4f} 但体检不合格 → 保持 best.pt（防坍缩保护）")
            # 早停：val 连续 patience 次评估无实质改善 → 提前终止。
            # 改善判定用 min_val_improve 阈值（严格低于才重置计数），避免微小抖动干扰。
            # 用 prev_raw（原始 val 最优）判断，与体检门控解耦。
            # 修复（2026-08-12，dev-notes/22）：必须用本次评估前的值判断，
            # 用更新后的 best/raw 时两者相等，永远判"无改善"，patience 次即误停。
            if enable_early_stop and iter_num >= min_iters:
                if losses['val'] < prev_raw - min_val_improve:
                    no_improve_count = 0  # 有实质改善，重置计数
                else:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        early_stopped = True
                        pbar.write(f"⏹ 早停：val 连续 {patience} 次评估无实质改善（best {best_val_loss:.4f}），提前终止 @ {iter_num}")
                        break
    if iter_num == 0 and eval_only:
        break

    # ★★ 终止判据必须在**优化器步之前**（2026-09-14 修，用户实测发现"多训练一轮"）。
    # 为什么原来会多跑一步：旧代码把 `if iter_num > max_iters: break` 放在**循环末尾**、
    # 且在 `iter_num += 1` **之后**，于是流程是
    #     [eval@k] → [优化器步 k] → k+=1 → (k > max_iters 才 break)
    # ⇒ 跑满 max_iters 之后**还会再走一次前向/反向/optimizer.step()**，那一步：
    #   ① 不产生任何评估、不写任何 ckpt（**白算**，每个 run 固定浪费 1 步）；
    #   ② 但它**改了权重** —— 而 `last.pt` 是在循环顶部 `iter_num == max_iters` 时存的
    #      ⇒ 盘上的权重与内存里的权重差一步（用户看到的"多训练一轮"）。
    #   实测：`--max_iters=3` 打印"训练完成：**4** 步"、tqdm 走到 `4it`（>100%）。
    # 修法：把判据提到评估之后、优化器步之前，判据用 **>=**（不是 >）。
    # 语义（修后）：优化器步**恰好** max_iters 次（iter_num 0..max_iters-1），
    #   最后一步评估/落盘仍发生在 `iter_num == max_iters`（循环顶部），
    #   ⇒ ckpt 编号不变、`results.csv` 不变、`last.pt` 与内存权重**一致**。
    #   结构性断言见 `tests/test_training_loop.py`（AST，不用 import 这个脚本）。
    if iter_num >= max_iters:
        break

    # GLM-5 索引器预热：warmup 结束的当步解冻主模型（只切换一次，避免每步开销）
    if _idx_warmup_active and iter_num >= indexer_warmup_steps:
        _set_indexer_freeze(raw_model, False)
        _idx_warmup_active = False
        if master_process:
            pbar.write(f"✓ 索引器预热结束 @{iter_num}：解冻主模型，联合训练")

    # 前向反向更新，带可选的梯度累积以模拟更大的 batch size
    # 如果数据类型是 float16，则使用 GradScaler
    step_nan = False
    step_loss_val = 0.0
    # 先把整步所有 microbatch 取齐：只有先知道全步的有效 token 总数，才能做正确的
    # token 级归一化。旧写法对每个 microbatch 各取一次 mean 再平均 → 有效 token 少的
    # microbatch 被过度加权；全 mask 的 microbatch 还会返回 NaN 让整步作废。
    # （2026-09-10：本数据集 ~80% 的 256 窗口全 mask，只有 8% 的 token 参与 loss。）
    micro_batches = []
    for _ in range(gradient_accumulation_steps):
        _mx, _my, _msid = get_batch('train')
        micro_batches.append((_mx, _my, _msid))
    micro_counts = [float((Y != -100).sum().item()) for _, Y, _ in micro_batches]
    n_valid_total = sum(micro_counts)
    last_valid_idx = max((i for i, n in enumerate(micro_counts) if n > 0), default=-1)
    if n_valid_total == 0:
        step_nan = True  # 整步全 mask（概率 ~0.1%），无事可做
    for micro_step, ((X, Y, SID), n_i) in enumerate(zip(micro_batches, micro_counts)):
        if n_i == 0:
            continue  # 该 microbatch 无有效 token：跳过（旧版会 NaN 掉整步）
        if ddp:
            # 在 DDP 训练中，我们只需要在最后一个微步同步梯度。
            # 官方的做法是用 model.no_sync() 上下文管理器，但
            # 我很不喜欢它让代码膨胀并迫使我们重复代码。
            # 看了那个上下文管理器的源码，它只是切换这个变量。
            model.require_backward_grad_sync = (micro_step == last_valid_idx)
        try:
            with ctx:
                logits, loss = model(X, Y, sample_id=SID)
                loss = _ndb_blend(logits, loss, X, Y)
                loss = loss * (n_i / n_valid_total)  # token 级加权 → 全步等价于 token 均值
            # ★ 本微步的隐藏态：必须在 backward **之前**取进本地变量。
            #   开了梯度检查点时 backward 会重算前向，pre-hook 会再写一次
            #   `_ndb_state['h']`（值等价但是另一张张量）；我们要的是这一次那一份。
            _h_micro = _ndb_state.get('h') if ndb is not None else None
            # NaN 防护：loss 非有限值（nan/inf）时跳过该微步的反向，
            # 避免 NaN 梯度污染参数（一旦参数变 NaN 就永远救不回来）。
            if not torch.isfinite(loss):
                step_nan = True
                continue
            # 反向传播，如果以 fp16 训练则进行梯度缩放
            scaler.scale(loss).backward()
            step_loss_val += loss.item()
            if ndb is not None and _h_micro is not None:
                # ★ 只在**训练微步**、且**反向之后**写：eval/val 绝不进这个上下文
                #   （`ngram_ndb.py:191` 明确警告：写进验证集 = val 泄漏）。
                with ndb.write_enabled():
                    ndb.observe(_h_micro.detach(), X, Y)
                _ndb_state['h'] = None
        except torch.OutOfMemoryError:
            # 显存尖峰防护（2026-09-10）：少数 batch 会在基座前向里触发一次性 ~3GB
            # 尖峰（NDB 无关，NDB-off 对照同样发生）。把致命崩溃降级成「跳过该步」，
            # 否则一次尖峰就会让数天的训练直接死掉。
            #
            # ★ 2026-09-10 补：旧版这里**把 traceback 丢掉了**，而快照的触发条件是
            #   max_memory_allocated > mem_snapshot_gb，碎片型 OOM（reserved 满、allocated 没到阈值）
            #   两个机制同时沉默 —— 这就是「莫名 OOM 查不出原因」的真正原因。
            #   现在：前 3 次 OOM 落盘完整现场（操作栈 + allocated/reserved/峰值 + 快照）。
            _oom_steps += 1
            if master_process:
                pbar.write(f"⚠ step {iter_num}: 显存不足，跳过该步（累计 {_oom_steps} 次）")
            if master_process and _oom_dumps < 3 and device_type == 'cuda':
                _oom_dumps += 1
                try:
                    _al = torch.cuda.memory_allocated() / 2**30
                    _rs = torch.cuda.memory_reserved() / 2**30
                    _pk = torch.cuda.max_memory_allocated() / 2**30
                    _txt = os.path.join(out_dir, f'oom_dump_{iter_num}.txt')
                    with open(_txt, 'w') as _f:
                        _f.write(f"step={iter_num}  micro_step={micro_step}  iter={iter_num}\n")
                        _f.write(f"allocated={_al:.3f}G  reserved={_rs:.3f}G  "
                                 f"peak_allocated={_pk:.3f}G  mem_snapshot_gb={mem_snapshot_gb:g}\n")
                        _f.write(f"batch_size={batch_size} grad_accum={gradient_accumulation_steps} "
                                 f"block={block_size} tokens/step={tokens_per_iter}\n")
                        _f.write(f"gradient_checkpointing(config)={gradient_checkpointing}  "
                                 f"use_moe={use_moe} use_aux_free_balance={use_aux_free_balance} "
                                 f"use_mhc={use_mhc} hc_mult={hc_mult} use_mtp={use_mtp}\n")
                        _f.write("\n===== 分配现场（谁要的这块内存）=====\n")
                        _f.write(traceback.format_exc())
                    _sp = os.path.join(out_dir, f'oom_snap_{iter_num}.pickle')
                    torch.cuda.memory._dump_snapshot(_sp)
                    pbar.write(f"   ↳ 现场已落盘：{os.path.basename(_txt)} + "
                               f"{os.path.basename(_sp)}（alloc={_al:.2f}G rsv={_rs:.2f}G peak={_pk:.2f}G）")
                except Exception as _e:
                    pbar.write(f"   ↳ OOM 现场落盘失败：{type(_e).__name__}: {_e}")
            optimizer.zero_grad(set_to_none=True)
            if ndb is not None:
                ndb_opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            step_nan = True
            loss = torch.zeros((), device=device)
            break
    if step_nan:
        # 本步含 NaN 微步：丢弃整步梯度（含正常微步累积的部分），跳过优化器更新
        if master_process:
            pbar.write(f"⚠ step {iter_num}: loss 非有限值（nan/inf），已跳过该步优化。"
                       f"检查 use_loss_masking 标记配置 / lr / 新架构数值稳定性")
        optimizer.zero_grad(set_to_none=True)
        if ndb is not None:
            ndb_opt.zero_grad(set_to_none=True)  # NaN 步：接口梯度同样丢弃
        scaler.update()
    else:
        # 裁剪梯度
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
        else:
            grad_norm = 0.0
        # 如果以 fp16 训练，则更新优化器和 scaler
        scaler.step(optimizer)
        scaler.update()
        # 尽快清空梯度，不再需要这块内存
        optimizer.zero_grad(set_to_none=True)
        # NDB 接口：独立优化器（固定 lr），单独裁剪 + 更新
        if ndb is not None:
            if grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(_ndb_params, grad_clip)
            ndb_opt.step()
            ndb_opt.zero_grad(set_to_none=True)
            # 定期把热缓冲落进表：observe 按 (level, slot, token) 一直往缓冲里累加
            if ndb_flush_every > 0 and (iter_num + 1) % ndb_flush_every == 0:
                ndb.flush()

    # 计入窗口统计（NaN/OOM 跳过的步不算，避免把残缺值拉进来）
    if not step_nan:
        _win_sum += step_loss_val
        _win_n += 1

    # 计时与日志：更新 tqdm 进度条
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # 进度条显示「最近 log_interval 步的 token 加权均值」——单步值只有 ~550 个有效
        # token 且高度集中在一两个窗口上，逐点看纯粹是噪声，看不出趋势。
        lossf = _win_sum / _win_n if _win_n else step_loss_val
        # 训练侧 loss EMA（eval_train_split=false 时 results.csv 的 train/loss 用它）
        running_train_loss = lossf if running_train_loss is None \
            else 0.9 * running_train_loss + 0.1 * lossf
        if local_iter_num >= 5: # 让训练循环先稳定一下
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        epoch_str = f"{iter_num/steps_per_epoch:.2f}" if steps_per_epoch else "-"
        # 显存监控与实时吞吐追踪
        mem_gb = torch.cuda.max_memory_allocated() / (1024**3) if device_type == 'cuda' else 0.0
        tps = (tokens_per_iter) / max(dt, 1e-4)
        pbar.set_postfix(轮次=f"{epoch_str}", 损失=f"{lossf:.4f}",
                         梯范=f"{grad_norm:.2f}", 显存=f"{mem_gb:.1f}G", 吞吐=f"{tps:.0f}t/s")
        try:
            with open(loss_win_path, 'a') as _f:
                _f.write(f"{iter_num},{lossf:.6f},{_win_n},{step_loss_val:.6f},"
                         f"{grad_norm:.4f},{get_lr(iter_num):.8g}\n")
        except Exception:
            pass
        _win_sum, _win_n = 0.0, 0
    if master_process and device_type == 'cuda' and not _mem_snap_done \
            and should_dump_snapshot(torch.cuda.max_memory_allocated(),
                                     torch.cuda.memory_reserved(), mem_snapshot_gb):
        # 峰值超阈值：dump 分配历史（含每个 alloc 的调用栈），用于定位一次性尖峰。
        # 触发条件取 max(allocated, reserved)：碎片型尖峰是 reserved 涨上去而
        # allocated 没到阈值，旧实现因此从来不触发（至今 0 个 mem_snap 文件）。
        _snap = os.path.join(out_dir, f'mem_snap_{iter_num}.pickle')
        torch.cuda.memory._dump_snapshot(_snap)
        _mem_snap_done = True
        print(f"  [诊断] 峰值 allocated={torch.cuda.max_memory_allocated()/2**30:.2f}GB "
              f"reserved={torch.cuda.memory_reserved()/2**30:.2f}GB → 显存快照 {_snap}", flush=True)
    # 显存调试行：只在 log_interval 的整数倍打印（原来每步一行，把日志刷成两倍长）；
    # OOM/尖峰有独立的告警与快照，不靠这行发现。
    # ★ 判据用 device_type（实际训练设备），**不能**用 torch.cuda.is_available()：
    #   有显卡的机器上跑 --device=cpu 时后者仍为 True，而 CUDA 分配器没有统计
    #   → memory_stats() 缺键 → KeyError 崩在 step 0（2026-09-10 冒烟实测）。
    if ndb_debug_mem and master_process and iter_num % log_interval == 0:
        _line = mem_debug_line(device_type)
        if _line is not None:
            print(f"[mem {iter_num}] {_line}", flush=True)
    iter_num += 1
    local_iter_num += 1
    if pbar is not None:
        pbar.update(1)

    # ★ 终止判据**不在这里** —— 它已上移到优化器步之前（见那里的长注释）。
    #   留这条注释是为了让"旧写法被删掉了"可被 grep 到，别再把它加回来。

# 训练结束：等后台保存线程写完，再画 loss 曲线图，收尾 csv
if master_process:
    if early_stopped:
        print(f"训练提前终止：{iter_num} 步（早停，best_val_loss {best_val_loss:.4f}）")
    else:
        print(f"训练完成：{iter_num} 步（达 max_iters {max_iters}）")
    print(f"  最终 best_val_loss {best_val_loss:.4f} · 总耗时 {time.time()-train_start:.1f}s")
join_save_threads()
if results_csv is not None:
    results_csv.close()
if health_csv is not None:
    health_csv.close()
if master_process and loss_history:
    _plot_loss_curve(loss_history, out_dir, best_val_loss)
if writer is not None:
    writer.close()
if pbar is not None:
    pbar.close()
if ddp:
    destroy_process_group()
