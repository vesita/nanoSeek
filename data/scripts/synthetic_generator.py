#!/usr/bin/env python3
"""高质量合成数据集生成器：
1. 数学思维链 (Math CoT):
   - 多位数四则运算 (加、减、乘、除、混合运算)，包含详细竖式/拆解步骤推理
   - 初等代数题 (一元一次方程、二元一次方程组、比例与百分数题、鸡兔同笼、行程问题、工程问题)
   - 平面几何题 (矩形、三角形、圆形面积周长，勾股定理，立体图形体积表面积)
   - 结构化输出：思考过程 (思维链 / 拆解步骤) 与 最终确定答案。
2. 多轮工具调用 Agent 语料 (Tool-Use Agent):
   - 真实场景：天气查询、代码沙盒执行、知识检索(search)、计算器(calculator)、文件读写、数据库查询、定时提醒
   - 格式规范：包含系统引导、用户意图、工具调用触发 (```json\n{"name": ..., "arguments": ...}\n``` 或 <tool_call>)、工具返回结果观察 (Tool Response / Observation) 以及模型最终合成答复。
3. 高质量百科知识与日常对话保底生成器 (用于无外网极端离线环境生成规范样式的冷启动数据)。
"""
from __future__ import annotations

import json
import math
import random
import unicodedata
from typing import Iterator, Dict, Any, List


# ==============================================================================
# 1. 数学思维链 (Math CoT) 生成器
# ==============================================================================

class MathCoTGenerator:
    """生成带 CoT 思考过程的多位数四则运算、代数、几何分步解题题集。"""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def gen_multi_digit_arithmetic(self) -> Dict[str, str]:
        """多位数四则运算，带详细 CoT 竖式或分步拆解。"""
        op = self.rng.choice(['+', '-', '*', '/'])
        if op == '+':
            a = self.rng.randint(100, 99999)
            b = self.rng.randint(100, 99999)
            res = a + b
            prompt = f"请计算 {a} + {b} 的结果，并写出详细的计算过程。"
            cot = (
                f"我们来计算两个多位数的加法：{a} + {b}。\n"
                f"解题思路：\n"
                f"1. 按照数位对齐（个位对齐个位，十位对齐十位等），从最低位（个位）开始逐位相加，逢十进一。\n"
                f"2. 也可以采用按位数拆分法：\n"
            )
            # 提供拆分步骤
            cot += f"   - 原式可以表示为两数各数位权重之和相加。\n"
            cot += f"   - 经逐位相加及处理进位：\n"
            cot += f"   - {a} 加上 {b} 的精确和为 {res}。\n"
            ans = f"最终结果为：{res}。"
            return {"user": prompt, "model": f"{cot}\n所以，{a} + {b} = {res}。\n{ans}"}

        elif op == '-':
            a = self.rng.randint(1000, 99999)
            b = self.rng.randint(100, a)
            res = a - b
            prompt = f"计算 {a} - {b}，请给出清晰的分步推导过程。"
            cot = (
                f"我们需要求解多位数减法算式：{a} - {b}。\n"
                f"思考步骤：\n"
                f"1. 比较两数大小：被减数 {a} 大于减数 {b}，差必然为正数。\n"
                f"2. 从个位依次向高位进行相减，如果某一位被减数不足，则向更高一位借1当10。\n"
                f"3. 验算检查：{res} + {b} = {a}，等式完全成立。\n"
            )
            ans = f"计算得出最终结果为：{res}。"
            return {"user": prompt, "model": f"{cot}\n因此，{a} - {b} = {res}。\n{ans}"}

        elif op == '*':
            a = self.rng.randint(12, 999)
            b = self.rng.randint(11, 99)
            res = a * b
            prompt = f"求算式 {a} × {b} 的乘积，需要展示详细计算步骤。"
            cot = (
                f"计算乘法：{a} × {b}。\n"
                f"分析过程：\n"
                f"1. 将乘数 {b} 拆分为十位和个位：{b} = {b // 10 * 10} + {b % 10}。\n"
                f"2. 分别计算部分积：\n"
                f"   - 第一步：计算 {a} × {b % 10} = {a * (b % 10)}\n"
                f"   - 第二步：计算 {a} × {b // 10 * 10} = {a * (b // 10 * 10)}\n"
                f"3. 将两个部分积累加：{a * (b % 10)} + {a * (b // 10 * 10)} = {res}。\n"
            )
            ans = f"所以，{a} × {b} = {res}。"
            return {"user": prompt, "model": f"{cot}\n{ans}"}

        else: # '/'
            b = self.rng.randint(11, 89)
            res = self.rng.randint(12, 450)
            remainder = self.rng.choice([0, self.rng.randint(1, b - 1)])
            a = b * res + remainder
            prompt = f"请计算除法：{a} ÷ {b}，要求给出商和余数（若有）以及具体推导步骤。"
            cot = (
                f"求解多位数除法：{a} ÷ {b}。\n"
                f"解题步骤（长除法思路）：\n"
                f"1. 确定除数是两位数 {b}，从被减数最高位依次试商。\n"
                f"2. 试商估算：经整除试算，商为 {res}。\n"
                f"3. 计算乘积：{b} × {res} = {b * res}。\n"
                f"4. 求剩余余数：{a} - {b * res} = {remainder}。\n"
            )
            if remainder == 0:
                ans = f"所以，{a} 能被 {b} 整除，商为 {res}，无余数。"
            else:
                ans = f"所以，{a} ÷ {b} 的商为 {res}，余数为 {remainder}（或表示为 {res} 余 {remainder}）。"
            return {"user": prompt, "model": f"{cot}\n{ans}"}

    def gen_algebra_problem(self) -> Dict[str, str]:
        """代数问题：一次方程、方程组、行程/工程应用题。"""
        sub_type = self.rng.choice(['linear_eq', 'chickens_rabbits', 'motion', 'work'])

        if sub_type == 'linear_eq':
            # ax + b = c
            x = self.rng.randint(2, 50)
            a = self.rng.randint(2, 12)
            b = self.rng.randint(5, 100)
            c = a * x + b
            prompt = f"求解一元一次方程：{a}x + {b} = {c}，请写出完整的解题步骤。"
            cot = (
                f"这是一道一元一次方程求解题。\n"
                f"方程：{a}x + {b} = {c}\n"
                f"解题步骤：\n"
                f"1. 移项：将常数项 {b} 移到方程右侧（变号）：\n"
                f"   {a}x = {c} - {b}\n"
                f"   {a}x = {c - b}\n"
                f"2. 系数化为1：方程两边同时除以未知数 x 的系数 {a}：\n"
                f"   x = {c - b} / {a}\n"
                f"   x = {x}\n"
                f"3. 验算：将 x = {x} 代入原方程左边，得 {a} × {x} + {b} = {a * x} + {b} = {c}，等式成立。\n"
            )
            ans = f"因此，方程的解为 x = {x}。"
            return {"user": prompt, "model": f"{cot}\n{ans}"}

        elif sub_type == 'chickens_rabbits':
            # 鸡兔同笼
            chickens = self.rng.randint(5, 50)
            rabbits = self.rng.randint(5, 40)
            heads = chickens + rabbits
            legs = chickens * 2 + rabbits * 4
            prompt = f"经典数学题：现有一个笼子里关着鸡和兔子。从上面数一共有 {heads} 个头，从下面数一共有 {legs} 只脚。问笼中各有鸡和兔子多少只？请给出详细解题过程。"
            cot = (
                f"这是一道经典的“鸡兔同笼”问题。\n"
                f"已知条件：\n"
                f"- 总头数：{heads}（即鸡和兔子的总只数为 {heads}）\n"
                f"- 总脚数：{legs}（每只鸡有 2 只脚，每只兔子有 4 只脚）\n\n"
                f"方法一：假设法（推荐）\n"
                f"1. 假设笼子里全部都是鸡，那么脚的总数应该为：\n"
                f"   {heads} × 2 = {heads * 2} 只脚。\n"
                f"2. 实际脚数比假设多出了：\n"
                f"   {legs} - {heads * 2} = {legs - heads * 2} 只脚。\n"
                f"3. 这是因为每把一只兔子看成鸡，就会少算 4 - 2 = 2 只脚。\n"
                f"4. 因此，兔子的只数为多出的脚数除以每只兔差额：\n"
                f"   兔子 = {legs - heads * 2} ÷ 2 = {rabbits} 只。\n"
                f"5. 鸡的只数为总头数减去兔子的数量：\n"
                f"   鸡 = {heads} - {rabbits} = {chickens} 只。\n\n"
                f"验算：\n"
                f"脚数 = {chickens} × 2 + {rabbits} × 4 = {chickens * 2} + {rabbits * 4} = {legs}，符合题意。"
            )
            ans = f"答：笼中有鸡 {chickens} 只，兔子 {rabbits} 只。"
            return {"user": prompt, "model": f"{cot}\n\n{ans}"}

        elif sub_type == 'motion':
            # 相遇问题
            v1 = self.rng.randint(30, 80)
            v2 = self.rng.randint(35, 90)
            t = self.rng.randint(2, 8)
            s = (v1 + v2) * t
            prompt = f"甲乙两辆汽车同时从相距 {s} 千米的两地相对开出，甲车每小时行驶 {v1} 千米，乙车每小时行驶 {v2} 千米。两车经过几小时能够相遇？请详细写出解题思路与计算。"
            cot = (
                f"这是一道典型的路程相遇问题。\n"
                f"已知数据：\n"
                f"- 两地总距离 S = {s} 千米\n"
                f"- 甲车速度 v1 = {v1} 千米/小时\n"
                f"- 乙车速度 v2 = {v2} 千米/小时\n\n"
                f"解题分析：\n"
                f"1. 速度和：两车相对行驶，每一小时两车共同缩减的距离为两车速度之和。\n"
                f"   速度和 v_sum = v1 + v2 = {v1} + {v2} = {v1 + v2} 千米/小时。\n"
                f"2. 相遇时间公式为：相遇时间 = 总路程 ÷ 速度和。\n"
                f"   t = S ÷ v_sum = {s} ÷ {v1 + v2} = {t} 小时。\n"
            )
            ans = f"答：甲乙两车经过 {t} 小时后相遇。"
            return {"user": prompt, "model": f"{cot}\n{ans}"}

        else: # work
            # 工程问题
            t1 = self.rng.choice([6, 8, 10, 12, 15, 20])
            t2 = self.rng.choice([10, 12, 15, 20, 30])
            lcm = math.lcm(t1, t2)
            eff1 = lcm // t1
            eff2 = lcm // t2
            total_eff = eff1 + eff2
            total_time = round(lcm / total_eff, 2)
            prompt = f"一项工程，甲队单独施工需要 {t1} 天完成，乙队单独施工需要 {t2} 天完成。如果两队合作同时施工，需要多少天才能完成这项工程？请给出思考过程与计算步骤。"
            cot = (
                f"这是一道工程合作问题。\n"
                f"解题思路：\n"
                f"1. 将整项工程的总工作量设为单位“1”。\n"
                f"2. 计算各队的工作效率（即每天完成的工作量）：\n"
                f"   - 甲队每天的工作效率为：1 / {t1}\n"
                f"   - 乙队每天的工作效率为：1 / {t2}\n"
                f"3. 两队合作时，每天的工作效率和为：\n"
                f"   效率和 = 1/{t1} + 1/{t2} = {eff1}/{lcm} + {eff2}/{lcm} = {total_eff}/{lcm}\n"
                f"4. 合作所需时间 = 工作总量 ÷ 效率和：\n"
                f"   时间 = 1 ÷ ({total_eff}/{lcm}) = {lcm} / {total_eff} ≈ {total_time} 天。\n"
            )
            ans = f"答：如果甲乙两队合作，完成这项工程大约需要 {total_time} 天（精确值为 {lcm}/{total_eff} 天）。"
            return {"user": prompt, "model": f"{cot}\n{ans}"}

    def gen_geometry_problem(self) -> Dict[str, str]:
        """几何问题：周长、面积、体积、勾股定理。"""
        g_type = self.rng.choice(['pythagorean', 'circle_area', 'cylinder_volume', 'triangle_area'])

        if g_type == 'pythagorean':
            triples = [(3, 4, 5), (5, 12, 13), (6, 8, 10), (7, 24, 25), (8, 15, 17), (9, 12, 15)]
            base_a, base_b, base_c = self.rng.choice(triples)
            scale = self.rng.randint(1, 4)
            a, b, c = base_a * scale, base_b * scale, base_c * scale
            prompt = f"已知一个直角三角形的两条直角边长分别为 {a} cm 和 {b} cm，请利用勾股定理求出该直角三角形的斜边长度，并给出具体计算步骤。"
            cot = (
                f"这是一道关于直角三角形勾股定理的应用题。\n"
                f"公式回顾：\n"
                f"在直角三角形中，两条直角边 a, b 与斜边 c 满足关系：a² + b² = c²。\n\n"
                f"计算过程：\n"
                f"1. 代入直角边长：a = {a}，b = {b}\n"
                f"2. 分别计算直角边的平方：\n"
                f"   a² = {a}² = {a * a}\n"
                f"   b² = {b}² = {b * b}\n"
                f"3. 计算两平方之和：\n"
                f"   c² = a² + b² = {a * a} + {b * b} = {c * c}\n"
                f"4. 开平方求解斜边 c：\n"
                f"   c = √({c * c}) = {c}\n"
            )
            ans = f"答：该直角三角形的斜边长为 {c} cm。"
            return {"user": prompt, "model": f"{cot}\n{ans}"}

        elif g_type == 'circle_area':
            r = self.rng.randint(2, 20)
            area = round(math.pi * r * r, 2)
            circum = round(2 * math.pi * r, 2)
            prompt = f"一个圆的半径为 {r} 米，请计算该圆的周长和面积（取 π ≈ 3.14），给出分步解题过程。"
            cot = (
                f"圆的几何性质求解：\n"
                f"已知圆的半径 r = {r} 米，圆周率取 π ≈ 3.14。\n\n"
                f"1. 计算圆的周长：\n"
                f"   公式：C = 2πr\n"
                f"   代入数值：C = 2 × 3.14 × {r} = {round(2 * 3.14 * r, 2)} 米。\n\n"
                f"2. 计算圆的面积：\n"
                f"   公式：S = πr²\n"
                f"   代入数值：S = 3.14 × {r}² = 3.14 × {r * r} = {round(3.14 * r * r, 2)} 平方米。\n"
            )
            ans = f"答：该圆的周长约为 {round(2 * 3.14 * r, 2)} 米，面积约为 {round(3.14 * r * r, 2)} 平方米。"
            return {"user": prompt, "model": f"{cot}\n{ans}"}

        elif g_type == 'cylinder_volume':
            r = self.rng.randint(2, 10)
            h = self.rng.randint(3, 15)
            base_area = round(3.14 * r * r, 2)
            volume = round(base_area * h, 2)
            prompt = f"一个圆柱体底面半径为 {r} cm，高为 {h} cm，请计算它的底面积与体积（π取3.14），列出详细步骤。"
            cot = (
                f"圆柱体几何参数计算：\n"
                f"已知底面半径 r = {r} cm，高 h = {h} cm，π ≈ 3.14。\n\n"
                f"步骤 1：计算底面积（圆形）\n"
                f"S_base = π × r² = 3.14 × {r}² = 3.14 × {r * r} = {base_area} cm²。\n\n"
                f"步骤 2：计算体积\n"
                f"圆柱体积公式：V = S_base × h\n"
                f"代入计算：V = {base_area} × {h} = {volume} cm³。\n"
            )
            ans = f"答：该圆柱体的底面积为 {base_area} cm²，体积为 {volume} cm³。"
            return {"user": prompt, "model": f"{cot}\n{ans}"}

        else: # triangle_area
            base = self.rng.randint(4, 30)
            height = self.rng.randint(3, 20)
            area = round(0.5 * base * height, 2)
            prompt = f"已知一个三角形的底边长为 {base} 厘米，高为 {height} 厘米，求该三角形的面积，并写出解题过程。"
            cot = (
                f"三角形面积计算：\n"
                f"公式：面积 S = (底边 × 高) ÷ 2\n"
                f"已知条件：底边 b = {base} cm，高 h = {height} cm\n"
                f"计算：\n"
                f"S = ({base} × {height}) ÷ 2 = {base * height} ÷ 2 = {area} cm²\n"
            )
            ans = f"答：该三角形的面积为 {area} 平方厘米。"
            return {"user": prompt, "model": f"{cot}\n{ans}"}

    def sample(self) -> Dict[str, str]:
        """随机采样一道带 CoT 的数学题。"""
        category = self.rng.choices(['arith', 'algebra', 'geo'], weights=[0.4, 0.4, 0.2])[0]
        if category == 'arith':
            return self.gen_multi_digit_arithmetic()
        elif category == 'algebra':
            return self.gen_algebra_problem()
        else:
            return self.gen_geometry_problem()


# ==============================================================================
# 2. 多轮工具调用 Agent (Tool-Use Agent) 语料生成器
# ==============================================================================

class ToolAgentGenerator:
    """生成带工具定义、工具调用指令、环境反馈观察 (Observation) 的真实 Agent 对话。"""

    TOOLS = [
        {
            "name": "get_weather",
            "description": "查询指定城市的实时天气信息与预报",
            "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "城市名称，如'北京'或'上海'"}}, "required": ["city"]}
        },
        {
            "name": "calculator",
            "description": "高精度数学表达式计算器",
            "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "标准数学运算表达式"}}, "required": ["expression"]}
        },
        {
            "name": "python_interpreter",
            "description": "安全的 Python 代码执行沙盒，用于数据分析、算法计算与绘图",
            "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "待运行的 Python 代码"}}, "required": ["code"]}
        },
        {
            "name": "web_search",
            "description": "通过搜索引擎检索最新的网页资讯、定义与事实",
            "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "搜索关键词"}}, "required": ["query"]}
        }
    ]

    def __init__(self, seed: int = 101):
        self.rng = random.Random(seed)

    def gen_weather_flow(self) -> str:
        cities = ['北京', '杭州', '深圳', '成都', '武汉', '西安', '青岛', '南京', '重庆', '厦门']
        weathers = [
            ('晴朗', '24℃', '北风2级', '湿度45%', '空气质量优'),
            ('多云转小雨', '19℃', '微风', '湿度78%', '空气质量良'),
            ('阴天', '21℃', '东北风3级', '湿度60%', '空气质量良'),
            ('雷阵雨', '26℃', '西南风4级', '湿度85%', '空气质量优')
        ]
        city = self.rng.choice(cities)
        w, temp, wind, hum, aqi = self.rng.choice(weathers)

        tool_call = {
            "name": "get_weather",
            "arguments": {"city": city}
        }
        obs = {
            "city": city,
            "condition": w,
            "temperature": temp,
            "wind": wind,
            "humidity": hum,
            "aqi": aqi
        }

        dialogue = (
            f"用户：你好，请帮我查一下{city}今天的天气情况怎么样，适合洗车或者户外运动吗？\n"
            f"模型：好的，我来为您查询{city}的最新实时天气。\n"
            f"```json\n{json.dumps(tool_call, ensure_ascii=False, indent=2)}\n```\n"
            f"用户：[工具返回结果]\n"
            f"```json\n{json.dumps(obs, ensure_ascii=False, indent=2)}\n```\n"
            f"模型：根据最新的天气实况：\n"
            f"当前{city}天气为{w}，气温 {temp}，{wind}，相对湿度 {hum}，空气质量为{aqi}。\n"
        )
        if '雨' in w:
            dialogue += f"温馨提示：由于今天有降雨可能，路面容易积水湿滑，不建议今天洗车；户外运动建议携带雨具或转为室内锻炼。"
        else:
            dialogue += f"天气条件良好，适宜外出散步或户外锻炼，也非常适合洗车。"
        return dialogue

    def gen_python_calc_flow(self) -> str:
        tasks = [
            ("计算前15个斐波那契数的乘积", "import math\ndef fib(n):\n    a, b = 0, 1\n    res = []\n    for _ in range(n):\n        a, b = b, a + b\n        res.append(a)\n    return res\n\nnums = fib(15)\nprint('nums:', nums)\nprod = math.prod(nums)\nprint('product:', prod)", "nums: [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]\nproduct: 45799987200"),
            ("统计文本中每个词的频率分布", "from collections import Counter\ntext = 'nanoSeek nanoSeek transformer moe csa moe'\ncounts = Counter(text.split())\nprint(dict(counts))", "{'nanoSeek': 2, 'transformer': 1, 'moe': 2, 'csa': 1}"),
            ("计算质数列表及求和", "def get_primes(n):\n    primes = []\n    for x in range(2, n+1):\n        if all(x % p != 0 for p in primes):\n            primes.append(x)\n    return primes\np = get_primes(30)\nprint('primes:', p)\nprint('sum:', sum(p))", "primes: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]\nsum: 129")
        ]
        q, code, out = self.rng.choice(tasks)
        tool_call = {
            "name": "python_interpreter",
            "arguments": {"code": code}
        }
        dialogue = (
            f"用户：请用 Python 帮我实现：{q}。\n"
            f"模型：我将编写并运行一段 Python 脚本来完成该计算任务。\n"
            f"```json\n{json.dumps(tool_call, ensure_ascii=False, indent=2)}\n```\n"
            f"用户：[工具返回结果]\n"
            f"```text\n{out}\n```\n"
            f"模型：代码执行成功！结果分析如下：\n"
            f"{out}\n"
            f"以上计算准确无误，满足您的需求。"
        )
        return dialogue

    def gen_search_flow(self) -> str:
        topics = [
            ("量子纠缠的基本物理定义与实验验证进展", "量子纠缠是量子力学中一种奇特的量子叠加态，两个或多个粒子在相互作用后，各自的状态无法单独描述，必须作为一个整体来描述。近年来潘建伟院士团队通过墨子号卫星实现了千公里级基于纠缠的量子密钥分发与贝尔不等式检验。"),
            ("深度学习中混合专家模型(MoE)的优缺点", "MoE(Mixture of Experts)的核心优势是能够成倍扩展模型参数容量而不显著增加每次推理的计算开销(FLOPs)，通过门控路由网络动态激活部分专家。其挑战主要在于专家负载均衡、训练通信开销以及分布式显存占用。"),
            ("Python中协程 asyncio 的底层事件循环工作原理", "asyncio 基于单线程事件循环(Event Loop)，通过 epoll/kqueue 等多路复用机制监控 I/O 事件。协程通过 yield/await 挂起执行状态，当 I/O 就绪时由事件循环调度唤醒恢复执行，实现高并发。")
        ]
        q, info = self.rng.choice(topics)
        tool_call = {
            "name": "web_search",
            "arguments": {"query": q}
        }
        obs = {"search_results": [{"title": q, "snippet": info, "source": "科技百科"}]}
        dialogue = (
            f"用户：请检索并详细解释：{q}。\n"
            f"模型：正在调用网络搜索工具检索权威资料...\n"
            f"```json\n{json.dumps(tool_call, ensure_ascii=False, indent=2)}\n```\n"
            f"用户：[工具返回结果]\n"
            f"```json\n{json.dumps(obs, ensure_ascii=False, indent=2)}\n```\n"
            f"模型：基于检索到的资料，关于【{q}】的综合解答如下：\n"
            f"{info}\n"
            f"若您对其中某一环节有进一步探究的兴趣，我们可以继续深入讨论。"
        )
        return dialogue

    def sample(self) -> str:
        flow_type = self.rng.choice(['weather', 'python', 'search'])
        if flow_type == 'weather':
            return self.gen_weather_flow()
        elif flow_type == 'python':
            return self.gen_python_calc_flow()
        else:
            return self.gen_search_flow()


# ==============================================================================
# 3. 百科知识问答与日常对话合成器 (离线保底)
# ==============================================================================

class GeneralSyntheticGenerator:
    """生成高质量自然百科、科学文化常识及日常高情商对话。"""

    BAIKE_FACTS = [
        ("太阳系的行星构成", "太阳系是以太阳为中心，以及所有受到太阳引力约束天体的集合体。包括八大行星：水星、金星、地球、火星、木星、土星、天王星和海王星。其中前四颗为类地行星（岩石行星），后四颗为巨行星（木星土星为气态巨行星，天王星海王星为冰巨行星）。"),
        ("光合作用的基本反应过程", "光合作用是绿色植物利用叶绿素等光合色素吸收太阳光能，把二氧化碳和水转化成富能有机物（如葡萄糖），并释放氧气的生物化学过程。它包括光反应（在类囊体薄膜上进行，产生ATP和NADPH，释放O₂）与暗反应/卡尔文循环（在叶绿体基质中进行，固定CO₂合成糖类）。"),
        ("计算机操作系统的进程与线程区别", "进程是操作系统进行资源分配和调度的基本单位，拥有独立的地址空间和内存；而线程是进程内的一个执行分支，是 CPU 调度执行的最小单元。同属一个进程的多个线程共享进程的堆内存、代码段和全局变量，但每个线程拥有独立的调用栈和程序计数器。线程间的上下文切换开销远小于进程间切换。"),
        ("中国二十四节气的由来与划分", "二十四节气是古代中国人通过观察太阳在黄道上的周年运动轨迹而制定的历法指南。将黄道360度划分为24等份，每15度为一个节气，始于立春，终于大寒。它准确反映了季节变化、物候规律与气候特征，深刻指导了中国农耕文明的生产生活。"),
        ("TCP 与 UDP 协议的核心区别", "TCP（传输控制协议）是面向连接、可靠的字节流协议，通过三次握手、四次挥手、确认应答和重传机制保障数据按序、不丢包到达；UDP（用户数据报协议）是无连接的、尽最大努力交付的数据报协议，没有拥塞控制和重传机制，开销小、传输延迟极低，常用于实时音视频流媒体和DNS查询。"),
        ("大语言模型中的 Transformer 架构", "Transformer 是由 Vaswani 等人在 2017 年提出的深度神经网络架构，其核心创新在于完全摒弃了传统的循环神经网络(RNN)和卷积结构，全依靠自注意力机制(Self-Attention)捕捉序列中长距离的 token 依赖关系。多头注意力(Multi-Head Attention)结合残差连接与层归一化，构成了现代大语言模型的基础基石。")
    ]

    DAILY_CHATS = [
        ("今天感觉工作压力好大，有些疲惫，能陪我聊聊吗？", "辛苦了！感到疲倦和压力是很正常的情绪反应，这时候最重要的是允许自己慢下来。不妨先停下手头的事情，深呼吸几次，喝杯温水，或者眺望一下窗外。想聊聊今天遇到什么让你感到棘手或烦心的事情吗？我随时在这里倾听。"),
        ("如何养成坚持早起运动的好习惯？", "培养早起运动习惯的关键在于降低初始阻力，建立正向反馈闭环：第一，循序渐进，先将起床闹钟提前15分钟，而不是一下子激进地提前两小时；第二，前一天晚上备好运动服和水壶，减少早晨决策犹豫；第三，从低强度喜欢的运动开始（如快走、拉伸），让身体逐渐享受运动后内啡肽带来的愉悦感。"),
        ("周末想一个人度过，有什么轻松充实的建议吗？", "一个人过周末是一段非常珍贵的自我对话与充电时光！推荐几个惬意的安排：1. 开启“无目的城市漫步(Citywalk)”，去未曾涉足的街角书店或公园散步；2. 挑选一部高分治愈电影或一本搁置已久的闲书，泡一壶清茶细细品味；3. 亲手为自己做一顿热气腾腾的家常菜。最重要的是按照自己的舒适节奏来。"),
        ("怎样才能提高自己的专注力，减少手机刷短视频的干扰？", "克服碎片化分心的有效方法是物理隔离与时间块化：1. 采用番茄工作法，设定25分钟绝对专注区间，期间将手机调至静音并放到视线之外；2. 给手机娱乐App设置每日使用时间限额；3. 明确当前专注要完成的具体微小目标，目标越具象，大脑越容易进入心流状态。")
    ]

    def __init__(self, seed: int = 202):
        self.rng = random.Random(seed)

    def sample_baike(self) -> Dict[str, str]:
        topic, content = self.rng.choice(self.BAIKE_FACTS)
        user = f"请简要阐述：{topic}"
        return {"user": user, "model": content}

    def sample_chat(self) -> Dict[str, str]:
        q, a = self.rng.choice(self.DAILY_CHATS)
        return {"user": q, "model": a}
