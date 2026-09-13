"""
对话智能度评估：用统一的多 prompt + 固定 seed 采样，对多个模型/实验臂做客观对比。

基准采样器复用 sample_py.py（权威 Python 采样器，镜像 Rust）：温度 → 重复惩罚 →
top-k → softmax → 多项式采样。这里在其上批量跑多个 prompt，并计算『对话智能度』
的客观指标，专门针对本项目反复记录的失败模式：

  1. 重复坍缩（dev-notes/14）：输出陷入 n-gram 循环 -> char 级 n-gram 重复率
  2. loss 骗低 / 空格坍缩（dev-notes/14）：押高频低信息 token（全角空格 U+3000、
     标点）抄近道压 loss -> 输出里空白/标点占比
  3. 碎片拼贴（dev-notes/21）：把语料碎片黏在一起、无对话轮次结构 -> 对话轮次
     结构（有无 用户/模型 交替、是否连贯成 `用户：...` 一段）
  4. 多样性 / 长度：distinct-n、平均生成长度

用法（从项目根目录）：
    uv run python inference/scripts/eval_dialogue.py                # 默认跑已有全部 + 新臂
    # 可选 --dirs 显式指定
"""
import argparse
import sys
import re
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from inference.scripts.sample_py import build_model_from_checkpoint, generate_ids, load_tokenizer

# 统一评估用的一批对话 prompt（覆盖寒暄/问询/情绪/建议等不同对话意图，模拟真实对话开场）
#
# ★ 2026-09-11 修（TECH_DEBT P1）：**prompt 的格式必须和模型训练语料一致**，
#   否则等于喂 OOD 输入 —— 表现为 `turns` 假性全 0，而且会在新旧模型之间
#   产生**虚假的主场优势**（谁见过这套标签谁占便宜），足以把结论评反。
#   * v2 语料（dev-notes/61 去标签后）用 `A：` / `B：` —— 已实测确认
#   * v1 及更早的基座用 `用户：` / `模型：`
#   ⇒ 用 `--style` 显式声明，**别混着眼**。
#   ⚠ 真要比较 v1 / v2，**不要**用本脚本的 d1/d2（见 `PROJECT_STATE §0.5.5`），
#     要用**配对 CE**（`scripts/ckpt_paired_eval.py`）—— CE 只吃 token，不受标签格式影响。
PROMPT_STYLES = {
    # v2 主线（当前）
    'ab': [
        "A：你好\nB：",
        "A：最近工作压力好大，怎么办啊？\nB：",
        "A：帮我推荐一本小说吧。\nB：",
        "A：你觉得人生最重要的是什么？\nB：",
        # 真实数据里的对话风格更多是口语短轮次，这里加两个贴近语料的
        "A：吃了没\nB：",
        "A：好想出去玩\nB：",
    ],
    # v1 及更早（带「用户：/模型：」标签，prompt 里不该带引号，语料也没有）
    'user-model': [
        "用户：你好\n模型：",
        "用户：最近工作压力好大，怎么办啊？\n模型：",
        "用户：帮我推荐一本小说吧。\n模型：",
        "用户：你觉得人生最重要的是什么？\n模型：",
        "用户：吃了没\n模型：",
        "用户：好想出去玩\n模型：",
    ],
}
DEFAULT_STYLE = 'ab'
DIALOGUE_PROMPTS = PROMPT_STYLES[DEFAULT_STYLE]

# 各套对话约定下的「轮次标记」。识别时**同时看两套** —— 这样既能兼容新旧模型，
# 也能顺带发现"格式漂移"（用 A：提示、却生成 用户：）。
TURN_MARKERS = {
    'ab': ('A', 'B'),
    'user-model': ('用户', '模型'),
}

MAX_NEW_TOKENS = 200
TEMPERATURE = 0.8
TOP_K = 200
REPEAT_PENALTY = 1.2
SEED = 1337


def unify_ws(s: str) -> str:
    """把全角空格 U+3000 等统一成普通空格，便于统计空白占比。"""
    return s.replace("\u3000", " ")


def ngram_repetition(s: str, n: int) -> float:
    """基于字符 n-gram 的重复率：若 n-gram 全集出现次数 > 1 的比例越高，越『失真』重复。
    返回 (重复 n-gram 的比例)。对中文按字符切分。"""
    seq = re.sub(r"\s+", "", s)
    if len(seq) < 2 * n:
        return 0.0
    grams = [seq[i:i + n] for i in range(len(seq) - n + 1)]
    if not grams:
        return 0.0
    c = Counter(grams)
    repeated = sum(1 for g in grams if c[g] > 1)
    return repeated / len(grams)


def whitespace_ratio(s: str) -> float:
    """输出里空白/全角空格/换行 的占比（loss 骗低信号）。"""
    clean = unify_ws(s)
    if not clean:
        return 0.0
    ws = sum(1 for ch in clean if ch.isspace())
    return ws / len(clean)


def distinct_n(s: str, n: int) -> float:
    """distinct-n：唯一 n-gram / 总 n-gram 占比（多样性）；中文按字符。"""
    seq = re.sub(r"\s+", "", s)
    if len(seq) < n:
        return 0.0
    grams = [seq[i:i + n] for i in range(len(seq) - n + 1)]
    return len(set(grams)) / len(grams)


def dialogue_turn_structure(s: str) -> dict:
    """检查是否形成『交替的对话轮次』结构（碎片拼贴的反面）。

    ★ 2026-09-11：**同时识别两套约定**（`A：/B：` 与 `用户：/模型：`），而不是只看一套。
    理由：
      1. 模型可能"格式漂移"—— 用 `A：` 提示却生成 `用户：`。只看一套会把它误判成
         "完全没有轮次结构"，而真相恰恰相反（结构对，只是标签换了）。
      2. 这样本函数不需要知道调用方用了哪套 prompt，少一个出错面。
    `style_detected` 会告诉我们实际命中哪一套，用于观察漂移。
    """
    clean = unify_ws(s)
    counts = {name: len(re.findall(rf"{u}[:：]", clean))
              for name, (u, m) in TURN_MARKERS.items()}
    turns = {name: len(re.findall(rf"{m}[:：]", clean))
             for name, (u, m) in TURN_MARKERS.items()}
    total = {k: counts[k] + turns[k] for k in counts}
    best = max(total, key=lambda k: total[k]) if total else 'ab'
    # 交替结构：两侧至少各出现一次，才算真的形成了轮次（单侧重复不算）
    has_structure = any(counts[k] >= 1 and turns[k] >= 1 for k in counts)
    return {
        "user_turns": counts[best],
        "model_turns": turns[best],
        "turns": max(counts[best], turns[best]),
        "has_structure": has_structure,
        "style_detected": best if total.get(best, 0) > 0 else "none",
    }


def evaluate_one(model, tok, out_dir: str, prompts: list[str] | None = None) -> dict:
    prompts = DIALOGUE_PROMPTS if prompts is None else prompts
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    all_text = ""
    n_prompts = 0
    tot_tokens = 0
    reps2 = reps3 = reps4 = 0.0
    ws = 0.0
    d1 = d2 = 0.0
    turns_struct = 0
    styles_seen: Counter = Counter()
    all_prompt_samples = []
    for p in prompts:
        n_prompts += 1
        # dev-notes/61：新字级模型支持 <cont> 待续停止，避免吐出 <cont> 后继续自说自话
        gen_ids, _ = generate_ids(model, tok, p, MAX_NEW_TOKENS, TEMPERATURE, TOP_K,
                                  REPEAT_PENALTY, stop_on_eos=True, stop_on_cont=True)
        text = tok.decode(gen_ids[len(tok.encode(p).ids):])
        all_text += text
        tot_tokens += _encode_len(tok, text)
        reps2 += ngram_repetition(text, 2)
        reps3 += ngram_repetition(text, 3)
        reps4 += ngram_repetition(text, 4)
        ws += whitespace_ratio(text)
        d1 += distinct_n(text, 1)
        d2 += distinct_n(text, 2)
        ts = dialogue_turn_structure(text)
        turns_struct += ts["turns"]
        styles_seen[ts["style_detected"]] += 1
        all_prompt_samples.append(text)
    n = n_prompts
    return {
        "out_dir": out_dir,
        "avg_len_tokens": round(tot_tokens / n, 1),
        "rep2": round(reps2 / n, 4),
        "rep3": round(reps3 / n, 4),
        "rep4": round(reps4 / n, 4),
        "ws_ratio": round(ws / n, 4),
        "distinct1": round(d1 / n, 4),
        "distinct2": round(d2 / n, 4),
        "turn_structure": round(turns_struct / n, 1),
        # 生成文本实际命中的对话约定（用于发现"用 A：提示、却生成 用户："的格式漂移）
        "style_detected": styles_seen.most_common(1)[0][0] if styles_seen else "none",
        "samples": all_prompt_samples,
    }


def _encode_len(tok, s: str) -> int:
    return len(tok.encode(s).ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=None,
                    help="要评估的 out/ 子目录；默认 = 新的两臂 + 已有全部")
    ap.add_argument("--style", choices=sorted(PROMPT_STYLES), default=DEFAULT_STYLE,
                    help="prompt 的对话标签格式 **必须和模型训练语料一致**（默认 ab = v2 主线）。"
                         "ab = `A：/B：`（v2 起）；user-model = `用户：/模型：`（v1 及更早）。"
                         "选错 = 喂 OOD 输入，turns 会假性归零（TECH_DEBT「已偿还」那条）")
    a = ap.parse_args()
    prompts = PROMPT_STYLES[a.style]

    default_dirs = [
        # 当前默认模型（v0.2 定版；历史实验在 out/archive/，缺失的自动跳过）
        "out/chinese-reb",
    ]
    historical = []
    dirs = a.dirs if a.dirs else (default_dirs + historical)

    results = []
    for d in dirs:
        if not (Path(d) / "best.pt").exists():
            print(f"⚠ 跳过（无 best.pt）: {d}")
            continue
        try:
            model, ckpt = build_model_from_checkpoint(d)
            # dev-notes/61：按 checkpoint 词表模式选 tokenizer（BPE/字级/字节）
            tok = load_tokenizer(ckpt)
        except Exception as e:
            print(f"⚠ 跳过（checkpoint 加载失败，可能是旧拓扑实验存档不兼容）: {d}")
            print(f"   原因: {type(e).__name__}: {str(e)[:120]}")
            continue
        print(f"\n=== 评估 {d} ===")
        r = evaluate_one(model, tok, d, prompts)
        results.append(r)
        # 打印指标 + 一条代表样本
        print(f"  avg_len={r['avg_len_tokens']} rep2={r['rep2']} rep3={r['rep3']} "
              f"rep4={r['rep4']} ws={r['ws_ratio']} d1={r['distinct1']} d2={r['distinct2']} "
              f"turns={r['turn_structure']} 样式={r['style_detected']}(给了 {a.style})")
        if r['style_detected'] not in (a.style, 'none'):
            print(f"  ⚠ 格式漂移：用 {a.style} 提示、却生成了 {r['style_detected']} 约定")
        print(f"  模型配置: mhc={ckpt['model_args'].get('use_mhc')} "
              f"lse={ckpt['model_args'].get('use_lse_residual')} "
              f"no_attn_layers={ckpt['model_args'].get('no_attn_layers')} "
              f"block_order={ckpt['model_args'].get('block_order')}")
        print(f"  [代表样本] prompt={prompts[0]!r}")
        print(f"    → {r['samples'][0][:150]}")

    # 汇总表
    print("\n\n======== 智能度对比汇总 ========")
    print(f"{'模型':<28} | {'len':>5} | {'rep2':>6} | {'rep3':>6} | {'rep4':>6} | "
          f"{'ws%':>6} | {'d1':>6} | {'d2':>6} | {'turns':>5}")
    print("-" * 100)
    for r in sorted(results, key=lambda x: x["rep3"]):
        print(f"{r['out_dir']:<28} | {r['avg_len_tokens']:>5} | {r['rep2']:>6} | {r['rep3']:>6} | "
              f"{r['rep4']:>6} | {int(r['ws_ratio']*100):>5}% | {r['distinct1']:>6} | "
              f"{r['distinct2']:>6} | {r['turn_structure']:>5}")

    print("\n指标说明：repN=字符 n-gram 重复率(越低越好·重复坍缩) | ws%=空白占比(越低越好·loss骗低) | "
          "d1/d2=多样性(越高越好) | turns=对话轮次结构(越高越好·碎片拼贴反面)")

    # 把完整采样落盘，供人工/后续细看
    import json
    with open("dev-notes/eval_dialogue_samples.json", "w", encoding="utf-8") as f:
        json.dump({r["out_dir"]: {"metrics": {k: v for k, v in r.items() if k != "samples"},
                                   "samples": r["samples"]} for r in results},
                  f, ensure_ascii=False, indent=2)
    print("\n完整样本已写入 dev-notes/eval_dialogue_samples.json")


if __name__ == "__main__":
    main()
