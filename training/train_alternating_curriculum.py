"""
training/train_alternating_curriculum.py
=========================================
4.5k 词表下的两段式/交替课程训练：
- 初始冷启动：300 步纯加法算术训练（建立进位逻辑与 Scratchpad 因果回路）
- 交替周期阶段（共 28 轮，累积 2800 步语言 = 1 个自然语言 epoch）：
    每轮推进 100 步自然语言对话，紧随 20 步算术强化回放
- 实时追踪双轨探针：算术 Exact Match (EM) 与自然语言验证集 Loss / 体检
"""

import os
import sys
import time
import math
import pickle
import random
import argparse
from contextlib import nullcontext
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from tokenizers import Tokenizer

# 根目录引用
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import GPT, GPTConfig
from training.masking import build_assistant_mask as _build_assistant_mask
from training.arith_char_dataset import (
    CharArithBatchGenerator,
    compute_addition_scratchpad,
    parse_scratchpad_result,
)


def evaluate_arithmetic(model: GPT, tok: Tokenizer, n_samples: int = 50,
                        max_digits: int = 3, device: str = 'cuda',
                        literal_noise: bool = True) -> Dict[str, float]:
    """评测当前模型在算术加法上的 Exact Match (EM) 准确率。"""
    model.eval()
    correct = 0
    total = 0
    eos_id = tok.token_to_id('<eos>')

    # 固定评测用种子保证各轮指标可比
    rng = random.Random(42)
    ops = ["+", "加", "加上"] if literal_noise else ["+"]
    eqs = ["=", "等于", "得"] if literal_noise else ["="]

    with torch.no_grad():
        for _ in range(n_samples):
            d_a = rng.randint(1, max_digits)
            d_b = rng.randint(1, max_digits)
            a = rng.randint(10 ** (d_a - 1) if d_a > 1 else 0, 10 ** d_a - 1)
            b = rng.randint(10 ** (d_b - 1) if d_b > 1 else 0, 10 ** d_b - 1)
            expected_res = a + b
            expected_sp = compute_addition_scratchpad(a, b)

            op = rng.choice(ops)
            eq = rng.choice(eqs)
            prompt = f"{a}{op}{b}{eq}"
            prompt_ids = tok.encode(prompt).ids
            x = torch.tensor([prompt_ids], dtype=torch.long, device=device)

            # 自回归贪婪采样生成
            max_new = max_digits + 4
            gen_ids = []
            for _step in range(max_new):
                # 裁剪到 block_size 以防超出
                x_cond = x if x.size(1) <= 256 else x[:, -256:]
                logits, _ = model(x_cond)
                next_id = logits[:, -1, :].argmax(dim=-1).item()
                if next_id == eos_id:
                    break
                gen_ids.append(next_id)
                x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)

            # 解析生成的文本
            pred_sp = tok.decode(gen_ids).strip()
            pred_res = parse_scratchpad_result(pred_sp)

            if pred_res == expected_res:
                correct += 1
            total += 1

    model.train()
    em = correct / total if total > 0 else 0.0
    return {"em": em, "correct": correct, "total": total}


def main():
    parser = argparse.ArgumentParser(description="4.5k 词表交替微调流水线")
    parser.add_argument('--out_dir', type=str, default='out/curriculum_arith_dialogue')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--block_size', type=int, default=256)
    parser.add_argument('--init_arith_steps', type=int, default=300, help="冷启动算术步数")
    parser.add_argument('--dialogue_steps_per_cycle', type=int, default=100, help="每轮对话步数")
    parser.add_argument('--arith_steps_per_cycle', type=int, default=20, help="每轮算术步数")
    parser.add_argument('--num_cycles', type=int, default=28, help="交替轮数 (28*100=2800 步约 1 epoch)")
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dtype', type=str, default='bfloat16')
    parser.add_argument('--seed', type=int, default=1337)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # 1. 词表与分词器
    tok_path = 'data/chinese/char_tokenizer.json'
    tok = Tokenizer.from_file(tok_path)
    meta_path = 'data/chinese/meta_char.pkl'
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    vocab_size = meta['vocab_size']
    print(f"载入分词器: {tok_path}, vocab_size = {vocab_size}")

    # 2. 算术生成器
    arith_gen = CharArithBatchGenerator(tokenizer_path=tok_path, max_digits=3, literal_noise_prob=0.5)

    # 3. 自然语言数据集加载
    train_bin = 'data/chinese/train_char.bin'
    val_bin = 'data/chinese/val_char.bin'
    train_data = np.memmap(train_bin, dtype=np.uint16, mode='r')
    val_data = np.memmap(val_bin, dtype=np.uint16, mode='r')
    print(f"载入自然语言: train tokens={len(train_data):,}, val tokens={len(val_data):,}")

    def get_dialogue_batch(split='train'):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - args.block_size, (args.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i + args.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + args.block_size]).astype(np.int64)) for i in ix])
        # Mask assistant
        mask_reply = [tok.token_to_id('<eos>'), tok.token_to_id('<cont>')]
        mask_sep = [tok.token_to_id('\n')]
        mask = _build_assistant_mask(y, mask_reply, mask_sep)
        y[~mask] = -100
        return x.to(args.device), y.to(args.device)

    # 4. 初始化 2.7M 基础模型配置 (scratch 初始化)
    # 对齐项目标准轻量配置
    model_cfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=6,
        n_head=4,
        n_embd=80,
        dropout=0.1,
        bias=False,
        use_rope=True,
        swiglu_clamp=0.5,
        use_moe=True,
        n_experts=8,
        n_top_k=2,
        moe_aux_weight=0.01,
        use_shared_expert=True,
        use_aux_free_balance=True,
        use_sqrtsoftplus=True,
        use_csa=True,
        csa_compress=16,
        csa_topk=4,
        csa_window=64,
        use_hca=True,
        use_kv_memory=True,
        kv_memory_latent=16,
        kv_memory_chunk=32,
        use_attn_sink=True,
        use_mhc=True,
        hc_mult=4,
        use_qk_norm=True,
        char_level=True,
        factorized_emb_dim=24,
    )
    print(f"从 0 初始化 2.7M 基础模型架构...")
    model = GPT(model_cfg)
    model.to(args.device)
    total_params = model.get_num_params()
    print(f"模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")

    # 优化器
    optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate, (0.9, 0.99), device_type)

    total_planned_steps = args.init_arith_steps + args.num_cycles * (args.dialogue_steps_per_cycle + args.arith_steps_per_cycle)
    print(f"计划总步数: {total_planned_steps} 步 (冷启动 {args.init_arith_steps} + {args.num_cycles} 轮 [100 语言 + 20 算术])")

    def get_lr(step):
        # 预热 100 步，之后余弦退火
        warmup_steps = 100
        if step < warmup_steps:
            return args.learning_rate * (step + 1) / warmup_steps
        if step > total_planned_steps:
            return args.min_lr
        decay_ratio = (step - warmup_steps) / (total_planned_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return args.min_lr + coeff * (args.learning_rate - args.min_lr)

    def eval_val_loss(n_eval_iters=20):
        model.eval()
        losses = []
        with torch.no_grad():
            for _ in range(n_eval_iters):
                x_val, y_val = get_dialogue_batch('val')
                with ctx:
                    _, loss = model(x_val, y_val)
                losses.append(loss.item())
        model.train()
        return float(np.mean(losses))

    # 日志记录
    log_csv = os.path.join(args.out_dir, "curriculum_progress.csv")
    with open(log_csv, "w") as f:
        f.write("step,phase,cycle,sub_step,loss,lr,arith_em,val_loss,time_s\n")

    global_step = 0
    start_time = time.time()

    def train_step(data_type: str, phase_name: str, cycle_idx: int, sub_step: int):
        nonlocal global_step
        lr = get_lr(global_step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if data_type == 'arith':
            x, y = arith_gen.get_batch(args.batch_size, args.block_size, device=args.device)
        else:
            x, y = get_dialogue_batch('train')

        with ctx:
            logits, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        global_step += 1
        return loss.item(), lr

    print("\n" + "=" * 60)
    print(">>> 阶段 0: 算术冷启动 (Arithmetic Warmup, 300 步)")
    print("=" * 60)

    for s in range(args.init_arith_steps):
        loss_val, lr_val = train_step('arith', 'warmup', 0, s)
        if (s + 1) % 50 == 0 or (s + 1) == args.init_arith_steps:
            elapsed = time.time() - start_time
            # 抽样评测算术
            arith_eval = evaluate_arithmetic(model, tok, n_samples=40, device=args.device)
            print(f"Step {global_step:4d} | [Warmup Arith {s+1:3d}/{args.init_arith_steps}] | Loss: {loss_val:.4f} | Arith EM: {arith_eval['em']*100:.1f}% | LR: {lr_val:.2e} | Elapsed: {elapsed:.1f}s")
            with open(log_csv, "a") as f:
                f.write(f"{global_step},warmup,0,{s+1},{loss_val:.4f},{lr_val:.6e},{arith_eval['em']:.4f},,-1\n")

    # 保存冷启动模型
    torch.save({
        'model': model.state_dict(),
        'config': model_cfg,
        'step': global_step
    }, os.path.join(args.out_dir, "checkpoint_warmup_300.pt"))
    print(">>> 冷启动完成，已保存 checkpoint_warmup_300.pt")

    print("\n" + "=" * 60)
    print(f">>> 阶段 1~{args.num_cycles}: 周期交替训练 (100 步语言 + 20 步算术)")
    print("=" * 60)

    for cycle in range(1, args.num_cycles + 1):
        c_start = time.time()
        # 1. 语言训练 100 步
        diag_losses = []
        for d_step in range(args.dialogue_steps_per_cycle):
            loss_val, lr_val = train_step('dialogue', 'dialogue', cycle, d_step)
            diag_losses.append(loss_val)
            if (d_step + 1) % 50 == 0:
                print(f"Step {global_step:4d} | [Cycle {cycle:02d}/28 | Dialogue {d_step+1:3d}/100] | Loss: {loss_val:.4f} | LR: {lr_val:.2e}")

        # 语言阶段结束时探测算术遗忘程度
        arith_after_diag = evaluate_arithmetic(model, tok, n_samples=30, device=args.device)
        v_loss = eval_val_loss(n_eval_iters=15)
        print(f"  --> 语言 100 步完成: Val Loss={v_loss:.4f} | 算术 EM 遗忘探测: {arith_after_diag['em']*100:.1f}%")

        # 2. 算术复训 20 步
        arith_losses = []
        for a_step in range(args.arith_steps_per_cycle):
            loss_val, lr_val = train_step('arith', 'arith_replay', cycle, a_step)
            arith_losses.append(loss_val)

        # 算术阶段结束时探测算术唤醒度
        arith_after_replay = evaluate_arithmetic(model, tok, n_samples=40, device=args.device)
        cycle_time = time.time() - c_start
        print(f"  ==> 算术 20 步回放后: 算术 EM 唤醒至: {arith_after_replay['em']*100:.1f}% (耗时 {cycle_time:.1f}s)")

        with open(log_csv, "a") as f:
            f.write(f"{global_step},cycle,{cycle},{args.dialogue_steps_per_cycle + args.arith_steps_per_cycle},{np.mean(diag_losses):.4f},{lr_val:.6e},{arith_after_replay['em']:.4f},{v_loss:.4f},{cycle_time:.1f}\n")

        # 每 4 轮或最终保存一次权重
        if cycle % 4 == 0 or cycle == args.num_cycles:
            ckpt_path = os.path.join(args.out_dir, f"checkpoint_cycle_{cycle:02d}.pt")
            torch.save({
                'model': model.state_dict(),
                'config': model_cfg,
                'step': global_step,
                'cycle': cycle,
                'arith_em': arith_after_replay['em'],
                'val_loss': v_loss,
            }, ckpt_path)
            print(f"  [Checkpoint] 周期 {cycle} 已保存: {ckpt_path}")

    # 最终全面评测
    print("\n" + "=" * 60)
    print(">>> 训练全部完成！正在执行全量终期评测...")
    print("=" * 60)
    final_arith = evaluate_arithmetic(model, tok, n_samples=100, device=args.device)
    final_val_loss = eval_val_loss(n_eval_iters=40)
    print(f"最终结果 (3,660 步交替课程):")
    print(f"  - 最终算术加法 EM (100题抽样): {final_arith['em']*100:.1f}% ({final_arith['correct']}/{final_arith['total']})")
    print(f"  - 最终自然语言 Val Loss: {final_val_loss:.4f}")

    torch.save({
        'model': model.state_dict(),
        'config': model_cfg,
        'step': global_step,
        'final_arith_em': final_arith['em'],
        'final_val_loss': final_val_loss,
    }, os.path.join(args.out_dir, "best.pt"))
    print(f"全套模型权重已保存至: {os.path.join(args.out_dir, 'best.pt')}")


if __name__ == "__main__":
    main()
