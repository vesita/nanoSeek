"""
arith_char_dataset.py
======================
为 4.5k 字级词表 (char_tokenizer.json) 定制的带进位链加法算术数据流与批次生成器。
支持多种提示词句式（纯算式、问答前缀、用户/模型对话行）与字面噪声扰动。
"""

import random
import torch
import numpy as np
from tokenizers import Tokenizer
from typing import Tuple, List, Optional


def compute_addition_scratchpad(a: int, b: int) -> str:
    """
    计算加法与紧凑进位链表达式（与 nano_arith 对齐）：
    个位 -> 最高位，逢进位用 'c' 记录。
    例如：
      45 + 78 = 123
      5 + 8 = 13 (写 3 进 1, carry=1) -> '3c'
      4 + 7 + 1 = 12 (写 2 进 1, carry=1) -> '2c'
      最高位 1 -> '1'
      返回 scratchpad: '3c2c1' (解析翻转后即为 123)
      
      若无进位：
      12 + 34 = 46 -> '64'
    """
    s_a = str(a)[::-1]
    s_b = str(b)[::-1]
    max_len = max(len(s_a), len(s_b))
    s_a = s_a.ljust(max_len, '0')
    s_b = s_b.ljust(max_len, '0')

    carry = 0
    tokens = []
    for da, db in zip(s_a, s_b):
        sm = int(da) + int(db) + carry
        carry = sm // 10
        rem = sm % 10
        tokens.append(str(rem))
        if carry > 0:
            tokens.append('c')
    if carry > 0:
        tokens.append(str(carry))
    return "".join(tokens)


def parse_scratchpad_result(sp: str) -> Optional[int]:
    """解析 scratchpad 字符串，还原为最终整数计算结果。"""
    digits = []
    i = 0
    while i < len(sp):
        if sp[i].isdigit():
            digits.append(sp[i])
            i += 1
        elif sp[i] == 'c':
            i += 1
        else:
            break
    if not digits:
        return None
    try:
        return int("".join(digits[::-1]))
    except ValueError:
        return None


class CharArithBatchGenerator:
    """
    4.5k 分词器环境下的动态算术批次生成器
    """
    def __init__(self, tokenizer_path: str = 'data/chinese/char_tokenizer.json',
                 max_digits: int = 3, literal_noise_prob: float = 0.6):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_digits = max_digits
        self.literal_noise_prob = literal_noise_prob
        self.eos_id = self.tokenizer.token_to_id('<eos>')
        self.unk_id = self.tokenizer.token_to_id('<unk>')

    def generate_single_sample(self) -> Tuple[str, str]:
        """
        生成一条完整的样本 (prompt, target_completion)
        例如:
          prompt = "45+78="
          target = "3c2c1<eos>"
        """
        d_a = random.randint(1, self.max_digits)
        d_b = random.randint(1, self.max_digits)
        a = random.randint(10 ** (d_a - 1) if d_a > 1 else 0, 10 ** d_a - 1)
        b = random.randint(10 ** (d_b - 1) if d_b > 1 else 0, 10 ** d_b - 1)

        scratchpad = compute_addition_scratchpad(a, b)
        target_str = f"{scratchpad}<eos>"

        if random.random() < self.literal_noise_prob:
            # 引入字面扰动
            op = random.choice(["+", "加", "加上", "加一加"])
            eq = random.choice(["=", "得", "等于", "等于几？", "等于多少？"])
            prefix_style = random.choice(["none", "qa", "dialogue"])
            if prefix_style == "dialogue":
                prompt_str = f"用户：{a}{op}{b}{eq}\n模型："
            elif prefix_style == "qa":
                prompt_str = f"问:{a}{op}{b}{eq}答:"
            else:
                prompt_str = f"{a}{op}{b}{eq}"
        else:
            # 标准精简格式与标准对话格式交替
            if random.random() < 0.5:
                prompt_str = f"{a}+{b}="
            else:
                prompt_str = f"用户：{a}+{b}=\n模型："

        return prompt_str, target_str

    def get_batch(self, batch_size: int = 64, block_size: int = 256, device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        打包成长度为 block_size 的 (X, Y) 批次。
        Prompt 部分 targets 填 -1 (忽略梯度)，只在 Target 区域计算 loss。
        当序列较短时，支持在一个样本中连续拼接多个算术题，以充分利用上下文窗口！
        """
        batch_x = []
        batch_y = []

        for _ in range(batch_size):
            tokens = []
            targets = []
            while len(tokens) < block_size + 1:
                prompt, target = self.generate_single_sample()
                p_ids = self.tokenizer.encode(prompt).ids
                t_ids = self.tokenizer.encode(target).ids

                # 拼接：prompt 对应 target 为 -100；target 自身计算 loss
                for pid in p_ids:
                    tokens.append(pid)
                    targets.append(-100)
                for tid in t_ids:
                    tokens.append(tid)
                    targets.append(tid)

            # 截取恰好 block_size + 1 个 token
            x_ids = tokens[:block_size]
            y_ids = targets[1:block_size + 1]

            batch_x.append(x_ids)
            batch_y.append(y_ids)

        x_tensor = torch.tensor(batch_x, dtype=torch.long, device=device)
        y_tensor = torch.tensor(batch_y, dtype=torch.long, device=device)
        return x_tensor, y_tensor


if __name__ == "__main__":
    gen = CharArithBatchGenerator()
    x, y = gen.get_batch(batch_size=2, block_size=64, device='cpu')
    print("x shape:", x.shape, "y shape:", y.shape)
    print("x[0]:", x[0].tolist())
    print("y[0]:", y[0].tolist())
    print("Masked target count in row 0:", (y[0] == -1).sum().item(), "Active target count:", (y[0] != -1).sum().item())
    print("Prompt/Target sample:")
    for _ in range(3):
        p, t = gen.generate_single_sample()
        print(f"  P: '{p}' | T: '{t}'")
