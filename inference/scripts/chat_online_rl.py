#!/usr/bin/env python3
"""nanoSeek 运行时在线自进化对话框架 (Online Real-Time RL / Test-Time Adaptation)

核心功能：
1. 实时交互打字机输出（流式自回归，支持字符直入/因式分解嵌入/MoE/KV记忆）
2. 零时延用户反馈回路 (Feedback Loop)：
   - 输入 `++` 或 `+`  ：点赞好评（正向强化梯度，推高该回答模式）
   - 输入 `--` 或 `-`  ：点踩差评（负向抑制梯度，压低套话/生硬表达）
   - 输入 `!corr <回答>`：直接教学/纠错（微单步 SFT 学习正确回答）
   - 输入 `!status`    ：查看本次会话在线更新步数与累计奖励
   - 输入 `!save`      ：将当前在线进化的权重保存为专属 Adapter
   - 输入 `!reset`     ：一键恢复出厂设置（抹平会话改动，基座永久无损）
3. 算法底层：
   - 指数分布奖励塑形 (Exponential Reward Shaping)
   - 终止符 EOS 隐藏层神经元静默正则 (Quiet-State Loss)
   - SFT 基座 KL 散度针 (Anchor KL) 防灾难性遗忘
   - 异步后台微梯度执行线程，前端交互零卡顿

用法：
    .venv/bin/python inference/scripts/chat_online_rl.py --out_dir out/rl_grpo_2epoch
"""
import argparse
import math
import os
import sys
import threading
import time
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn.functional as F
from inference.scripts.sample_py import build_model_from_checkpoint, generate_ids, load_tokenizer


def exponential_reward_shaping(score, tau=1.5):
    """指数分布奖励塑形"""
    sign = 1.0 if score >= 0 else -1.0
    return sign * (math.exp(abs(score) / tau) - 1.0)


class OnlineRLEngine:
    """运行时在线自进化引擎"""

    def __init__(self, model, ref_model, tok, lr=2e-5, beta_kl=0.03, lambda_quiet=0.02, tau=1.5):
        self.model = model
        self.ref_model = ref_model
        self.tok = tok
        self.eos_id = tok.token_to_id("<eos>")
        self.device = next(model.parameters()).device
        
        self.lr = lr
        self.beta_kl = beta_kl
        self.lambda_quiet = lambda_quiet
        self.tau = tau
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=0.01, betas=(0.9, 0.95)
        )
        
        # 统计计数
        self.step_count = 0
        self.pos_count = 0
        self.neg_count = 0
        self.corr_count = 0
        self.lock = threading.Lock()

    def step_rl(self, prompt_ids, reply_ids, raw_score):
        """执行一步异步强化学习策略更新 (Policy Gradient)"""
        with self.lock:
            if not reply_ids:
                return
            
            exp_reward = exponential_reward_shaping(raw_score, tau=self.tau)
            full_ids = prompt_ids + reply_ids
            prompt_len = len(prompt_ids)
            reply_len = len(reply_ids)

            x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=self.device)
            y = torch.tensor([full_ids[1:]], dtype=torch.long, device=self.device)

            self.model.train()
            
            # 捕获隐藏状态以施加 EOS 静默正则
            captured_h = []
            def hook_fn(module, inp, out):
                captured_h.append(out)
            handle = self.model.transformer.ln_f.register_forward_hook(hook_fn)
            logits, _ = self.model(x, targets=y)
            handle.remove()

            log_probs = F.log_softmax(logits, dim=-1)
            target_logprobs = log_probs.gather(2, y.unsqueeze(-1)).squeeze(-1).squeeze(0)
            curr_reply_lp = target_logprobs[prompt_len-1 : prompt_len-1+reply_len]

            with torch.no_grad():
                ref_logits, _ = self.ref_model(x, targets=y)
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                ref_target_lp = ref_log_probs.gather(2, y.unsqueeze(-1)).squeeze(-1).squeeze(0)
                ref_reply_lp = ref_target_lp[prompt_len-1 : prompt_len-1+reply_len]

            # 1. 策略梯度 (以 exp_reward 为杠杆)
            policy_loss = -exp_reward * curr_reply_lp.sum()

            # 2. 基座 KL 散度约束 (防漂移)
            kl_loss = F.kl_div(curr_reply_lp, ref_reply_lp, log_target=True, reduction='sum')

            # 3. 终止符神经元静默损失 (Quiet-State on EOS)
            eos_idx = reply_ids.index(self.eos_id) if self.eos_id in reply_ids else -1
            quiet_loss = torch.tensor(0.0, device=self.device)
            if eos_idx >= 0 and captured_h:
                h_f = captured_h[0]
                eos_pos = prompt_len - 1 + eos_idx
                if eos_pos < h_f.size(1):
                    quiet_loss = torch.mean(torch.abs(h_f[0, eos_pos, :]))

            total_loss = policy_loss + self.beta_kl * kl_loss + self.lambda_quiet * quiet_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            self.step_count += 1
            if raw_score > 0:
                self.pos_count += 1
            else:
                self.neg_count += 1
            self.model.eval()

    def step_correction(self, prompt_ids, correct_text):
        """用户教学纠错：执行一步微 SFT 强化正确答案"""
        with self.lock:
            corr_ids = self.tok.encode(correct_text.strip()).ids
            if not corr_ids:
                return
            if self.eos_id is not None and (not corr_ids or corr_ids[-1] != self.eos_id):
                corr_ids.append(self.eos_id)

            full_ids = prompt_ids + corr_ids
            prompt_len = len(prompt_ids)
            reply_len = len(corr_ids)

            x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=self.device)
            y = torch.tensor([full_ids[1:]], dtype=torch.long, device=self.device)

            self.model.train()
            logits, _ = self.model(x, targets=y)
            log_probs = F.log_softmax(logits, dim=-1)
            target_logprobs = log_probs.gather(2, y.unsqueeze(-1)).squeeze(-1).squeeze(0)
            corr_reply_lp = target_logprobs[prompt_len-1 : prompt_len-1+reply_len]

            # 经典交叉熵正向鼓励
            sft_loss = -corr_reply_lp.sum()

            self.optimizer.zero_grad()
            sft_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            self.step_count += 1
            self.corr_count += 1
            self.model.eval()


def main():
    ap = argparse.ArgumentParser(description="nanoSeek 运行时在线自进化系统")
    ap.add_argument("--out_dir", default="out/rl_grpo_2epoch", help="模型目录")
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.75)
    ap.add_argument("--top_k", type=int, default=200)
    ap.add_argument("--repeat_penalty", type=float, default=1.2)
    ap.add_argument("--lr", type=float, default=3e-5, help="运行时微调学习率")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"▶ 加载模型: {a.out_dir} | 设备: {device}")
    
    model, ckpt = build_model_from_checkpoint(a.out_dir, device=device)
    ref_model, _ = build_model_from_checkpoint(a.out_dir, device=device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
        
    tok = load_tokenizer(ckpt)
    model.eval()

    # 初始化在线强化引擎
    engine = OnlineRLEngine(model, ref_model, tok, lr=a.lr)

    # 备份出厂初始状态 (供 !reset 复原)
    clean_state = deepcopy(model.state_dict())

    print("─" * 60)
    print("🌟 nanoSeek 运行时在线训练框架已就绪！")
    print("【反馈指令】:")
    print("  • 直接输入文本  : 正常多轮对话")
    print("  • ++ 或 +       : 点赞刚才的回答 👍 (触发正向指数强化)")
    print("  • -- 或 -       : 点踩刚才的回答 👎 (触发负向指数打压)")
    print("  • !corr <回答>  : 纠错/教学 (以正确答案微调 1 步)")
    print("  • !status       : 查看会话在线学习统计")
    print("  • !save [路径]  : 保存当前会话学到的权重")
    print("  • !reset        : 恢复出厂设置 (抹平在线修改)")
    print("  • exit / quit   : 退出对话")
    print("─" * 60)

    ctx = ""
    last_prompt_ids = None
    last_reply_ids = None
    last_reply_text = ""

    while True:
        try:
            user_input = input("\n🙂 用户：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见 👋")
            break

        if not user_input:
            continue
        if user_input in ("exit", "quit", "退出"):
            print("再见 👋")
            break

        # ── 1. 指令解析：点赞 👍 ──
        if user_input in ("++", "+", "👍", "赞", "好"):
            if last_prompt_ids is not None and last_reply_ids is not None:
                threading.Thread(
                    target=engine.step_rl, args=(last_prompt_ids, last_reply_ids, +2.5)
                ).start()
                print("⚡ [在线RL] 已记录好评 👍 → 触发正向指数强化 (Advantage > 0)")
            else:
                print("⚠ 还没有可评价的历史回答")
            continue

        # ── 2. 指令解析：点踩 👎 ──
        if user_input in ("--", "-", "👎", "踩", "不好", "差"):
            if last_prompt_ids is not None and last_reply_ids is not None:
                threading.Thread(
                    target=engine.step_rl, args=(last_prompt_ids, last_reply_ids, -2.5)
                ).start()
                print("⚡ [在线RL] 已记录差评 👎 → 触发负向指数惩罚 (Advantage < 0，打压该回路)")
            else:
                print("⚠ 还没有可评价的历史回答")
            continue

        # ── 3. 指令解析：用户教学纠错 !corr ──
        if user_input.startswith("!corr") or user_input.startswith("！corr"):
            corr_text = user_input[5:].strip()
            if not corr_text:
                print("⚠ 请提供正确答案，例如: !corr 天空是蓝色是因为瑞利散射")
                continue
            if last_prompt_ids is not None:
                threading.Thread(
                    target=engine.step_correction, args=(last_prompt_ids, corr_text)
                ).start()
                print(f"🎓 [在线教学] 已接受教学:「{corr_text}」→ 触发微单步 SFT 强化！")
            else:
                print("⚠ 还没有对应的上下文问题")
            continue

        # ── 4. 指令解析：状态查看 !status ──
        if user_input == "!status":
            print(f"📊 [在线统计] 累计微调步数: {engine.step_count} (👍好评: {engine.pos_count} | 👎差评: {engine.neg_count} | 🎓教学: {engine.corr_count})")
            continue

        # ── 5. 指令解析：保存权重 !save ──
        if user_input.startswith("!save"):
            parts = user_input.split(maxsplit=1)
            save_p = parts[1].strip() if len(parts) > 1 else "out/user_adapter.pt"
            os.makedirs(os.path.dirname(save_p) or '.', exist_ok=True)
            save_dict = deepcopy(ckpt)
            save_dict['model'] = model.state_dict()
            torch.save(save_dict, save_p)
            print(f"💾 [在线保存] 已将进化后的权重保存至: {save_p}")
            continue

        # ── 6. 指令解析：出厂重置 !reset ──
        if user_input == "!reset":
            with engine.lock:
                model.load_state_dict(clean_state)
            print("🔄 [出厂重置] 已重置会话权重，恢复出厂默认！")
            continue

        # ── 7. 正常对话生成 ──
        current_turn_prompt = f"用户：{user_input}\n模型："
        ctx += current_turn_prompt
        prompt_ids = tok.encode(ctx).ids

        printed = ""
        gen_tokens = []

        def cb(tid):
            nonlocal printed
            gen_tokens.append(tid)
            text = tok.decode(gen_tokens)
            sys.stdout.write(text[len(printed):])
            sys.stdout.flush()
            printed = text

        print("🤖 模型：", end="", flush=True)
        gen, eos_pos = generate_ids(
            model, tok, ctx, a.max_new_tokens, a.temperature, a.top_k,
            a.repeat_penalty, stop_on_turn=True, stop_on_eos=True, clip_at_sentence=True,
            token_callback=cb
        )
        print()

        # 提取回复内容与 token
        plen = len(tok.encode(ctx).ids)
        reply_ids = gen[plen:]
        reply_text = tok.decode(reply_ids)
        ctx += reply_text + "\n"

        # 记录上一轮状态供点赞/点踩/纠错
        last_prompt_ids = prompt_ids
        last_reply_ids = reply_ids
        last_reply_text = reply_text


if __name__ == "__main__":
    main()
