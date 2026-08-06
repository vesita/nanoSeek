# 现代化架构的 shakespeare 字符级训练配置（实验用）
# 与 config/train_shakespeare_char.py 的超参完全一致，仅把三个 modern 开关打开：
#   use_rmsnorm = True   LayerNorm -> RMSNorm
#   use_rope    = True   wpe(可学习位置编码) -> RoPE(旋转位置编码)
#   use_swiglu  = True   GELU MLP -> SwiGLU
# 这样跑出来的 loss 可以直接和基线（train_shakespeare_char.py）对比，做 A/B 实验。

out_dir = 'out/shakespeare-char-modern'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'shakespeare-char'
wandb_run_name = 'modern-gpt'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# --- 现代化架构开关：全部打开 ---
use_rmsnorm = True
use_rope = True
use_swiglu = True
rope_theta = 10000.0

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially
