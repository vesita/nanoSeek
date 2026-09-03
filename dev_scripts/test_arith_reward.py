#!/usr/bin/env python3
"""奖励引擎 × 配置驱动数学题 端到端验证:
- 提示词池由 arith_rules.toml 生成 (格式化噪音)
- 答对 (阿拉伯数字) → arith_correct=True, 拿正分
- 答对 (中文数字) → arith_correct=True, 拿正分
- 答错 / 没答 → arith_correct=False, 扣分
- 提示词用的写法, 奖励识别必须认 (闭环)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.rl.reward import RewardEngine
from training.rl.reward import arith_gen

EOS_ID, CONT_ID = 117, 119
eng = RewardEngine()


def evaluate(prompt, reply_text):
    rid = list(reply_text.encode("utf-8")) + [EOS_ID]
    vec, score, exp_r = eng.evaluate_reply(
        prompt, reply_text, rid, EOS_ID, kind="arith",
        keywords=[], cont_id=CONT_ID)
    return vec, score, exp_r


def check(name, prompt, reply, expect_correct):
    vec, score, exp_r = evaluate(prompt, reply)
    arith = arith_gen.extract_arith(prompt)
    ok = vec.r_arith > 0 if expect_correct else vec.r_arith <= 0
    mark = "✓" if ok else "✗"
    print(f"{mark} {name:22} arith={vec.r_arith:+5.2f} total={score:+6.2f} correct={expect_correct} | Q={prompt!r} A={reply!r}")
    if not ok:
        print(f"    !!! 期望 {'正分' if expect_correct else '非正'}, 实际 {vec.r_arith:+.2f} | arith={arith}")
    return ok


fails = 0
# 1. 配置生成池 → 每条提示词都能被奖励识别 (闭环)
prompts = arith_gen.gen_prompts(count=40, seed=7)
for p, kw in prompts:
    ar = arith_gen.extract_arith(p)
    if ar is None:
        print(f"✗ 提示词识别失败: {p!r} kw={kw}")
        fails += 1
    else:
        # 用正确答案作答 → 应判 correct
        ans = str(ar[3])
        if not check(f"gen[{ar[3]}]", p, f"答案是{ans}", True):
            fails += 1

# 2. 中文数字答案
fails += 0 if check("cn-七", "3加4等于几？", "答案是七", True) else 1
fails += 0 if check("cn-二十四", "8乘3等于多少？", "二十四", True) else 1
fails += 0 if check("cn-十", "4加6等于几？", "等于十", True) else 1

# 3. 阿拉伯数字答案 (不同写法)
fails += 0 if check("arab-7", "3加4等于几？", "答案是7", True) else 1
fails += 0 if check("arab-24", "8乘3等于多少？", "24", True) else 1

# 4. 答错 / 没答
fails += 0 if check("wrong-8", "3加4等于几？", "答案是8", False) else 1
fails += 0 if check("empty", "3加4等于几？", "嗯嗯", False) else 1

# 5. 全角/中文写法识别 (格式化噪音闭环)
for s in ["3加上4等于多少？", "15减去7等于？", "12×9等于几？", "7＋8等于多少？", "6乘上7等于几？"]:
    ar = arith_gen.extract_arith(s)
    if ar is None:
        print(f"✗ 变体识别失败: {s!r}")
        fails += 1
    else:
        ok = check(f"variant[{ar[3]}]", s, f"{ar[3]}", True)
        fails += 0 if ok else 1

# 6. 三态判定 (dev-notes/66 拒答惩罚修复): 答对 / 答错(attempted) / 拒答废话(not attempted)
#    拒答惩罚力度应显著重于答错, 且答对 > 答错 > 拒答(废话流).
def r_arith_of(reply):
    v, _, _ = evaluate("3加4等于几？", reply)
    return v.r_arith

fails += 0 if r_arith_of("答案是7") > 0 else 1          # 答对 → 正分
fails += 0 if r_arith_of("答案是8") < r_arith_of("答案是7") else 1  # 答错 < 答对
# 废话流/答非所问 → 拒答重罚, 应显著负于答错(避难所剔除)
fails += 0 if r_arith_of("我觉得慢慢来总会好起来的吧") < r_arith_of("答案是8") else 1
fails += 0 if r_arith_of("有点先让自己起来的那个冒出来") < 0 else 1  # 废话流净负
# has_answer_intent: 废话流汉字"一/一起"不应误判成作答
fails += 0 if arith_gen.has_answer_intent("有点先让自己起来的那个冒出来") is False else 1
fails += 0 if arith_gen.has_answer_intent("答案是35") is True else 1
fails += 0 if arith_gen.has_answer_intent("二十四") is True else 1

print(f"\n=== 结果: {'全部通过' if fails == 0 else f'{fails} 项失败'} ===")
sys.exit(1 if fails else 0)
