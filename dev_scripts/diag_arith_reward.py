#!/usr/bin/env python3
"""arith 场景 reward 向量分解诊断探针.

用途: 验证"废话流净分为正、数字作答反而分低/没拉开"这一数学学不会的根因。

对同一道 arith 题, 喂不同作答(答对数字/答错数字/废话流/空/复述), 打印完整
RewardVector 各维度加权分 + 总分 s(塑形前) + exp_R(塑形后), 再模拟 GRPO 组内
归一化, 量化废话流 vs 数字作答的相对差, 判断模型是否有学"憋数字"的梯度。

coherence 对比: 加载基座对各类作答算 mean-logp(语感), 看是否有区分度 →
支撑把 coherence 升级为正向奖励。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from inference.scripts.sample_py import build_model_from_checkpoint, load_tokenizer
from training.rl.reward import RewardEngine, arith_gen
from training.rl.updater import GrpoUpdater
from training.rl.sampler import reward_shaping

EOS_ID = 117
CONT_ID = 119

# 同一道题(40+27=67), 各种作答形态
CASES = {
    "答对数字(等于67)": "等于67",
    "答对裸数字67":      "67",
    "答对中文(六十七)":  "等于六十七",
    "答错数字(等于35)":  "等于35",
    "废话流(中)":        "我觉得有点难但慢慢来总会好起来的吧",
    "废话流(长)":        "有点先让自己起来的那个冒出来，还是今天最明显吗好呀一起去探索吧",
    "复述题目":          "40加上27等于67",
    "空回复":            "",
}


def build_reply_ids(tok, text):
    """编码回复文本并在末尾加 <eos> (与采样后 decode 前的一致形态)。"""
    if not text:
        return [EOS_ID]
    return list(tok.encode(text).ids) + [EOS_ID]


def main():
    base = "out/cont_v1_1epoch"
    prompt_text = "用户：40加上27？\n模型："
    eng = RewardEngine()

    # coherence 探针 (基座)
    print("=== 基座语感 coherence (mean-logp) 区分度 ===")
    model, ckpt = build_model_from_checkpoint(base, device="cuda")
    tok = load_tokenizer(ckpt)
    model.eval()
    pids = tok.encode(prompt_text).ids
    from training.rl.grpo_char import mean_ref_reply_logprob
    coh = {}
    for name, text in CASES.items():
        rids = build_reply_ids(tok, text)
        if not text:
            coh[name] = None
            print(f"  {name:20s}: (空)")
            continue
        mlp = mean_ref_reply_logprob(model, pids, rids, "cuda")
        coh[name] = mlp
        print(f"  {name:20s}: mean-logp={mlp:+.2f}")

    print("\n=== Reward 分解 (同一题 40+27=67) ===")
    kind = "arith"
    kw = arith_gen.gen_prompts(1, 0)[0][1]
    eng0 = RewardEngine()
    dims_order = list(eng0._dims.keys())
    wmap = {
        "fatal_empty": "natural", "coherence": "coherence", "natural": "natural",
        "anti_repeat": "anti_repeat", "anti_robotic": "anti_robotic", "control": "quiet_eos",
        "sentiment": "sentiment", "task": "task", "semantic": "semantic",
        "arith": "arith", "anti_collapse": "anti_collapse",
    }
    print("  维度(权重): " + ", ".join(
        f"{n}={getattr(eng0.weights, 'w_' + wmap[n]):g}" for n in dims_order))

    rows = {}
    for name, text in CASES.items():
        rids = build_reply_ids(tok, text)
        # 真实训练里 grpo_char 传 coherence=None (r_coherence 维度不激活) →
        # 这里分别看 None(实际) 与 传入值 两种情况。
        for mode in ["none", "coh"]:
            cval = coh[name] if mode == "coh" else None
            vec, total_s, _exp = eng.evaluate_reply(
                prompt_text, text, rids, EOS_ID, kind=kind, keywords=kw,
                coherence=cval, cont_id=CONT_ID)
            exp_R = reward_shaping(total_s, shape="exp", tau=1.5)
            rows[f"{name}|{mode}"] = (text, total_s, exp_R, vec, cval)

    names_show = ["答对数字(等于67)", "答对裸数字67", "答对中文(六十七)", "答错数字(等于35)",
                  "废话流(中)", "废话流(长)", "复述题目", "空回复"]
    for mode, tag in [("none", "[实际训练 coherence=None]"), ("coh", "[若加 coherence 原始分]")]:
        print(f"\n  --- {tag} ---")
        for name in names_show:
            key = f"{name}|{mode}"
            if key not in rows:
                continue
            text, total_s, exp_R, vec, c = rows[key]
            bits = []
            seen_w = set()
            for d in dims_order:
                wfield = wmap[d]
                if wfield in seen_w:
                    continue          # fatal_empty/natural 等多函数共写同字段 → 只列一次
                seen_w.add(wfield)
                w = getattr(eng0.weights, 'w_' + wfield)
                val = getattr(vec, 'r_' + wfield) * w
                if abs(val) > 1e-6:
                    bits.append(f"{d}{val:+.1f}")
            print(f"  [{name}]  s={total_s:+6.2f}  exp_R={exp_R:+6.2f}")
            print(f"        {'  '.join(bits)}")
            print(f"        reply={text[:36]!r}")

    # GRPO 组内相对优势 (用实际训练模式 none)
    print("\n=== GRPO 组内相对优势模拟 (coherence=None, 实际训练口径) ===")
    grp = ["答对数字(等于67)", "答对裸数字67", "答错数字(等于35)",
           "废话流(中)", "废话流(长)", "复述题目", "空回复"]
    vals = torch.tensor([rows[f"{g}|none"][2] for g in grp], dtype=torch.float)
    adv = GrpoUpdater.normalize_advantages(vals).tolist()
    for g, a in zip(grp, adv):
        print(f"  {g:16s} exp_R 组内优势 = {a:+.3f}")


if __name__ == "__main__":
    main()
