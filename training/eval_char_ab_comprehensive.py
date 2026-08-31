#!/usr/bin/env python3
"""字符直入 A/B 测试全套多维标准深度评测脚本。

采用项目建立的 4 套评测标准对实验臂进行全方位横向评测：
  1. 训练与收敛标准：Final Train / Best Val / Δ Val
  2. 你好体检标准 (10 Seeds)：EOS 自吐率、平均 EOS 位置、rep2/3/4、d1/d2、ws 骗低率、turns
  3. 多轮压力测试标准 (3 场景 × 4 轮 = 12 对话轮次)：多轮 EOS 率、平均回复长度、多轮 rep3、崩溃次数
  4. 多场景通用开场评估 (6 大意图 Prompt)：Distinct-1/2 多样性、自然 EOS 结束率

用法：
  python training/eval_char_ab_comprehensive.py                # 评测全部臂
  python training/eval_char_ab_comprehensive.py --arms=0,5     # 只评测指定编号的臂
  python training/eval_char_ab_comprehensive.py --json=out/eval.json   # 结果另存 JSON
缺失 best.pt 的臂会警告并跳过，不影响其余臂。
"""
import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from inference.scripts.sample_py import (
    build_model_from_checkpoint,
    generate_ids,
    load_tokenizer,
)

ARMS = [
    ("Arm 0: 基准 (Fact-6L)", "out/char_fact24_300", "2.51M"),
    ("Arm 1: +输出门控", "out/ab_char_gate_300", "2.55M"),
    ("Arm 2: +细粒度MoE (8选2)", "out/ab_char_moe8_300", "2.33M"),
    ("Arm 3: +2-Step MTP", "out/ab_char_mtp2_300", "2.87M"),
    ("Arm 4: 全要素 (7层加深)", "out/ab_char_all_300", "3.02M"),
    ("Arm 5: 全要素组合 6L (1500步)", "out/ab_char_all6_1500", "2.70M"),
]

# 多轮对话 3 大场景
MULTITURN_SCENARIOS = [
    [
        "用户：你好，今天天气怎么样？\n模型：",
        "用户：那适合出门跑步吗？\n模型：",
        "用户：如果下雨该带什么？\n模型：",
        "用户：谢谢你的建议！\n模型：",
    ],
    [
        "用户：我今天心情有点低落。\n模型：",
        "用户：工作上遇到了点挫折。\n模型：",
        "用户：感觉大家都不理解我。\n模型：",
        "用户：听你这么说我好多了。\n模型：",
    ],
    [
        "用户：你能帮我写一首关于秋天的短诗吗？\n模型：",
        "用户：再加点落叶的意象呢？\n模型：",
        "用户：韵脚能不能押得更工整一点？\n模型：",
        "用户：太棒了，给这首诗起个标题吧。\n模型：",
    ],
]

# 通用 6 大意图开场 Prompts
GENERAL_PROMPTS = [
    "用户：你好！\n模型：",
    "用户：请介绍一下你自己。\n模型：",
    "用户：如何学好人工智能？\n模型：",
    "用户：什么是量子计算？\n模型：",
    "用户：推荐两本好看的历史小说。\n模型：",
    "用户：为什么天空是蓝色的？\n模型：",
]


def ngram_rep(s: str, n: int) -> float:
    seq = re.sub(r"\s+", "", s)
    if len(seq) < 2 * n:
        return 0.0
    grams = [seq[i : i + n] for i in range(len(seq) - n + 1)]
    c = Counter(grams)
    return sum(1 for g in grams if c[g] > 1) / max(len(grams), 1)


def distinct_n(s: str, n: int) -> float:
    seq = re.sub(r"\s+", "", s)
    if len(seq) < n:
        return 1.0
    grams = [seq[i : i + n] for i in range(len(seq) - n + 1)]
    return len(set(grams)) / max(len(grams), 1)


def ws_ratio(s: str) -> float:
    return sum(1 for ch in s if ch.isspace()) / max(len(s), 1)


def turns_count(s: str) -> int:
    return max(len(re.findall(r"用户[:：]", s)), len(re.findall(r"模型[:：]", s)))


def eval_hello_10seeds(model, tok):
    prompt = "用户：你好\n模型："
    plen = len(tok.encode(prompt).ids)
    rows = []
    for s in range(10):
        torch.manual_seed(s)
        ids, eos_pos = generate_ids(
            model,
            tok,
            prompt,
            max_new_tokens=60,
            temperature=0.8,
            top_k=100,
            repeat_penalty=1.2,
            stop_on_turn=False,
            stop_on_eos=False,
        )
        text = tok.decode(ids[plen:])
        rows.append(
            {
                "eos": eos_pos,
                "len": len(ids) - plen,
                "rep2": ngram_rep(text, 2),
                "rep3": ngram_rep(text, 3),
                "rep4": ngram_rep(text, 4),
                "d1": distinct_n(text, 1),
                "d2": distinct_n(text, 2),
                "ws": ws_ratio(text),
                "turns": turns_count(text),
            }
        )

    hits = sum(1 for r in rows if r["eos"] >= 0)
    avg_pos = sum(r["eos"] for r in rows if r["eos"] >= 0) / max(hits, 1)
    avg = lambda k: sum(r[k] for r in rows) / len(rows)
    return {
        "eos_rate": f"{hits}/10",
        "avg_eos_pos": avg_pos if hits > 0 else 0.0,
        "avg_len": avg("len"),
        "rep2": avg("rep2"),
        "rep3": avg("rep3"),
        "d1": avg("d1"),
        "d2": avg("d2"),
        "ws": avg("ws"),
        "turns": avg("turns"),
    }


def eval_multiturn(model, tok):
    all_turns = []
    collapse_count = 0
    for scenario in MULTITURN_SCENARIOS:
        context = ""
        mem_state = None
        for u_prompt in scenario:
            context += u_prompt
            plen = len(tok.encode(context).ids)
            torch.manual_seed(1337)
            ids, eos_pos = generate_ids(
                model,
                tok,
                context,
                max_new_tokens=40,
                temperature=0.8,
                top_k=100,
                repeat_penalty=1.2,
                stop_on_turn=True,
                stop_on_eos=True,
                resume_state=mem_state,
            )
            resp_ids = ids[plen:]
            resp_text = tok.decode(resp_ids).strip()
            r3 = ngram_rep(resp_text, 3)
            is_collapse = (r3 > 0.30) or (len(resp_text) <= 1) or (len(resp_ids) >= 39 and eos_pos < 0)
            if is_collapse:
                collapse_count += 1
            all_turns.append(
                {
                    "eos": eos_pos >= 0,
                    "len": len(resp_ids),
                    "rep3": r3,
                    "collapse": is_collapse,
                }
            )
            context += resp_text + "<eos>\n"
            mem_state = model.get_memory_state()

    eos_hits = sum(1 for t in all_turns if t["eos"])
    avg_len = sum(t["len"] for t in all_turns) / len(all_turns)
    avg_rep3 = sum(t["rep3"] for t in all_turns) / len(all_turns)
    return {
        "multi_eos_rate": f"{eos_hits}/{len(all_turns)} ({eos_hits/len(all_turns)*100:.0f}%)",
        "multi_avg_len": avg_len,
        "multi_rep3": avg_rep3,
        "collapse_count": collapse_count,
    }


def eval_general_prompts(model, tok):
    rows = []
    for p in GENERAL_PROMPTS:
        torch.manual_seed(1337)
        ids, eos_pos = generate_ids(
            model,
            tok,
            p,
            max_new_tokens=40,
            temperature=0.7,
            top_k=100,
            repeat_penalty=1.2,
            stop_on_turn=True,
            stop_on_eos=True,
        )
        plen = len(tok.encode(p).ids)
        text = tok.decode(ids[plen:]).strip()
        rows.append(
            {
                "len": len(ids) - plen,
                "d1": distinct_n(text, 1),
                "d2": distinct_n(text, 2),
                "rep3": ngram_rep(text, 3),
                "eos": eos_pos >= 0,
            }
        )
    avg = lambda k: sum(r[k] for r in rows) / len(rows)
    eos_cnt = sum(1 for r in rows if r["eos"])
    return {
        "gen_d1": avg("d1"),
        "gen_d2": avg("d2"),
        "gen_rep3": avg("rep3"),
        "gen_eos": f"{eos_cnt}/{len(rows)}",
    }


def parse_args():
    p = argparse.ArgumentParser(description="字符直入 A/B 全套多维评测")
    p.add_argument(
        "--arms",
        default="",
        help="逗号分隔的臂编号（ARMS 下标），如 0,1,5；默认全部",
    )
    p.add_argument(
        "--json",
        default="",
        help="将评测结果另存为 JSON 文件的路径（如 out/eval_char_ab.json）",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # 臂过滤：--arms=0,5 → [0,5]；未给则全部
    if args.arms.strip():
        try:
            idxs = [int(x.strip()) for x in args.arms.split(",") if x.strip()]
        except ValueError:
            sys.exit("❌ --arms 需为逗号分隔的整数，如 --arms=0,5")
        bad = [i for i in idxs if i < 0 or i >= len(ARMS)]
        if bad:
            sys.exit(f"❌ 臂编号越界: {bad}（有效范围 0~{len(ARMS)-1}）")
    else:
        idxs = list(range(len(ARMS)))

    print("=" * 110)
    print("🔍 字符直入 A/B 架构优化全套多维标准深度评测 (Health Check + Multi-Turn + General Suite)")
    print("=" * 110)

    results = []
    skipped = []
    for i in idxs:
        name, out_dir, params = ARMS[i]
        ckpt_path = os.path.join(ROOT, out_dir, "best.pt")
        if not os.path.exists(ckpt_path):
            print(f"⚠ 跳过 {name} ({out_dir})：找不到 {ckpt_path}")
            skipped.append({"name": name, "out_dir": out_dir, "params": params})
            continue

        print(f"正在评测 {name} ({out_dir})...")
        model, ckpt = build_model_from_checkpoint(out_dir, device="cpu")
        tok = load_tokenizer(ckpt)

        val_loss = float(ckpt.get("best_val_loss", 0.0))
        h_res = eval_hello_10seeds(model, tok)
        m_res = eval_multiturn(model, tok)
        g_res = eval_general_prompts(model, tok)

        results.append(
            {
                "name": name,
                "out_dir": out_dir,
                "params": params,
                "val_loss": val_loss,
                **h_res,
                **m_res,
                **g_res,
            }
        )
        del model, ckpt, tok

    if not results:
        print("\n⚠ 没有可评测的臂（全部缺失 best.pt 或未选中）。")
        return

    # 1. 输出你好体检标准表
    print("\n" + "=" * 110)
    print("【维度 1：你好体检标准 (Health Check Hello - 10 Seeds 鲁棒性)】")
    print("=" * 110)
    print(f"{'实验臂 (Arm)':<26} | {'参数':<5} | {'Val Loss':<8} | {'EOS自吐':<7} | {'均位置':<6} | {'rep2':<6} | {'rep3':<6} | {'d1':<6} | {'d2':<6} | {'ws%':<5} | {'turns':<5}")
    print("-" * 110)
    for r in results:
        print(f"{r['name']:<26} | {r['params']:<5} | {r['val_loss']:<8.4f} | {r['eos_rate']:<7} | {r['avg_eos_pos']:<6.1f} | {r['rep2']:<6.3f} | {r['rep3']:<6.4f} | {r['d1']:<6.3f} | {r['d2']:<6.3f} | {r['ws']:<5.3f} | {r['turns']:<5.1f}")

    # 2. 输出多轮对话压力测试表
    print("\n" + "=" * 110)
    print("【维度 2：多轮压力测试标准 (3 场景 × 4 轮 = 12 对话轮次)】")
    print("=" * 110)
    print(f"{'实验臂 (Arm)':<26} | {'参数':<5} | {'多轮 EOS 自吐率':<18} | {'平均回复长度':<12} | {'多轮 rep3 重复':<14} | {'多轮崩溃次数':<12}")
    print("-" * 110)
    for r in results:
        print(f"{r['name']:<26} | {r['params']:<5} | {r['multi_eos_rate']:<18} | {r['multi_avg_len']:<12.1f} | {r['multi_rep3']:<14.4f} | {r['collapse_count']:<12}")

    # 3. 输出多意图通用评估表
    print("\n" + "=" * 110)
    print("【维度 3：多意图通用开场评测 (6 类真实意图)】")
    print("=" * 110)
    print(f"{'实验臂 (Arm)':<26} | {'参数':<5} | {'Distinct-1 (单字)':<18} | {'Distinct-2 (双字)':<18} | {'通用 rep3':<12} | {'自然 EOS 结束':<12}")
    print("-" * 110)
    for r in results:
        print(f"{r['name']:<26} | {r['params']:<5} | {r['gen_d1']:<18.3f} | {r['gen_d2']:<18.3f} | {r['gen_rep3']:<12.4f} | {r['gen_eos']:<12}")
    print("=" * 110)

    if skipped:
        print(f"\n⚠ 跳过 {len(skipped)} 个臂（无 best.pt）：{', '.join(s['name'] for s in skipped)}")

    if args.json:
        payload = {"results": results, "skipped": skipped}
        jpath = os.path.join(ROOT, args.json) if not os.path.isabs(args.json) else args.json
        os.makedirs(os.path.dirname(jpath), exist_ok=True)
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n📄 结果已保存: {jpath}")


if __name__ == "__main__":
    main()
