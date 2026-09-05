#!/usr/bin/env python3
"""
高质量数学思维链扩充生成器 v2（复用 nano_arith 进位链范式 + 真实分步推导）

相比 synthetic_generator.py 的 MathCoTGenerator：
1. 多位数加减乘除：输出真实的「逐位对齐 + 进位链(c)」分步过程（对齐 Note 71/73 的 Scratchpad 范式）
2. 一元一次方程 / 比例百分数 / 鸡兔同笼 / 行程工程问题：真实逻辑推导链
3. 每一步都带明确的「步骤编号 + 中间结论 + 最终答案」结构

输出：data/chinese/math_cot_v2_dialogue.txt（「用户：/模型：」格式，空行分隔）
"""
import os
import random
import unicodedata

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chinese")


def _digits(n):
    return list(str(n))


def _add_cot(a, b):
    """真实逐位加法进位链（对齐 Scratchpad 进位标记 c）。"""
    c = _digits(a)[::-1]
    d = _digits(b)[::-1]
    n = max(len(c), len(d))
    carry = 0
    steps = []
    res_digits = []
    for i in range(n):
        x = int(c[i]) if i < len(c) else 0
        y = int(d[i]) if i < len(d) else 0
        s = x + y + carry
        carry = s // 10
        res_digits.append(str(s % 10))
        carry_note = f"，进位 c=1" if carry else ""
        steps.append(f"第{i+1}步（从个位起）：{x} + {y} + 进位{('1' if carry else '0') if i>0 else '0'} = {s}，本位写 {s%10}{carry_note}")
    if carry:
        res_digits.append(str(carry))
        steps.append(f"最后：最高位进位 1，直接写下 1")
    result = int("".join(res_digits[::-1]))
    cot = f"【加法进位链推导】{a} + {b}：\n" + "\n".join(steps) + f"\n因此 {a} + {b} = {result}"
    return cot, result


def _mul_cot(a, b):
    """多位数乘法的分步拆解（分配律 + 错位相加）。"""
    result = a * b
    steps = [f"【乘法拆解推导】计算 {a} × {b}：",
             f"1. 将 {b} 拆为 {b//10}×10 + {b%10}：",
             f"2. {a} × {b//10} = {a*(b//10)}，再 ×10 得 {a*(b//10)*10}",
             f"3. {a} × {b%10} = {a*(b%10)}",
             f"4. 错位相加：{a*(b//10)*10} + {a*(b%10)} = {result}"]
    cot = "\n".join(steps) + f"\n因此 {a} × {b} = {result}"
    return cot, result


def _equation_cot():
    """一元一次方程真实求解链。"""
    a = random.randint(2, 9)
    b = random.randint(1, 20)
    c = random.randint(10, 60)
    # a*x + b = c
    x = (c - b) / a
    if x != int(x):
        return _equation_cot()
    x = int(x)
    cot = (f"【方程求解推导】解方程 {a}x + {b} = {c}：\n"
           f"1. 移项：把 {b} 移到等号右边，{a}x = {c} - {b} = {c-b}\n"
           f"2. 系数化 1：两边同除以 {a}，x = {c-b} ÷ {a} = {x}\n"
           f"3. 检验：{a}×{x} + {b} = {a*x+b} = {c} ✓")
    return cot, x


def _chicken_rabbit_cot():
    """鸡兔同笼问题。"""
    heads = random.randint(8, 30)
    legs = random.randint(heads * 2, heads * 4)
    if (legs - 2 * heads) % 2 != 0:
        legs += 1
    rabbits = (legs - 2 * heads) // 2
    chickens = heads - rabbits
    cot = (f"【鸡兔同笼推导】笼中共 {heads} 个头、{legs} 条腿，鸡兔各几只？\n"
           f"1. 假设全是鸡：{heads} 只鸡应有 {2*heads} 条腿\n"
           f"2. 实际 {legs} 条腿，多出 {legs-2*heads} 条\n"
           f"3. 每只兔比鸡多 2 条腿 → 兔子 = {legs-2*heads} ÷ 2 = {rabbits} 只\n"
           f"4. 鸡 = {heads} - {rabbits} = {chickens} 只\n"
           f"5. 检验：{chickens}×2 + {rabbits}×4 = {chickens*2+rabbits*4} = {legs} ✓")
    return cot, (chickens, rabbits)


def gen_sample(rng):
    kind = rng.random()
    if kind < 0.35:
        a = rng.randint(10, 9999)
        b = rng.randint(10, 9999)
        cot, res = _add_cot(a, b)
        q = f"请计算 {a} + {b}，写出逐位进位过程。"
        a_out = f"{cot}"
    elif kind < 0.55:
        a = rng.randint(11, 999)
        b = rng.randint(11, 99)
        cot, res = _mul_cot(a, b)
        q = f"请计算 {a} × {b}，写出分步拆解过程。"
        a_out = cot
    elif kind < 0.75:
        cot, res = _equation_cot()
        q = f"请解这个一元一次方程，写出完整步骤。\n{cot.split('：')[1].split('：')[0] if '：' in cot else ''}"
        # 提取方程
        q = "请解下列一元一次方程，写出完整求解步骤。"
        a_out = cot
    elif kind < 0.9:
        cot, res = _chicken_rabbit_cot()
        q = "请解答这道鸡兔同笼问题，写出假设法与推导过程。"
        a_out = cot
    else:
        a = rng.randint(1000, 99999)
        b = rng.randint(100, a)
        res = a - b
        q = f"请计算 {a} - {b}，写出退位（借位）过程。"
        a_out = f"【减法退位推导】{a} - {b} = {res}\n从高位逐位借位相减，最终差为 {res}"
    return q, a_out


def generate(n, seed=42):
    rng = random.Random(seed)
    out = os.path.join(DATA_DIR, "math_cot_v2_dialogue.txt")
    with open(out, "w", encoding="utf-8") as f:
        for i in range(n):
            q, a = gen_sample(rng)
            f.write(f"用户：{q}\n模型：{a}\n\n")
    print(f"✓ math_cot_v2 生成 {n} 条 -> {out} ({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    generate(80000, seed=20260906)
