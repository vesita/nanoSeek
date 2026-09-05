"""
Arithmetic Tokenizer and Dataset Generator for nano_arith.

Key Design Decisions:
1. Compact Vocabulary:
   - Digits: 0-9 (ids 0-9)
   - Operators: +, -, *, = (ids 10-13)
   - Specials: <pad> (14), <eos> (15), <sos> (16)
   Total Vocab Size = 17 tokens.

2. Causal Digit Ordering (Reverse Digits for output):
   - Standard math: 123 + 456 = 579. The most significant digit (5) depends on lower digits!
   - Reverse output: "123+456=975$"
   - Why? In causal autoregressive transformers, to predict the first answer token,
     if it's the least significant digit (ones place 3+6=9), the model only needs to look at the ones place!
     The carry flows forward causally in subsequent tokens (tens, hundreds...).
   - We support both reversed (default) and standard ordering to demonstrate the difference.
"""

import random
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset

PAD_TOKEN = "<pad>"
EOS_TOKEN = "<eos>"
SOS_TOKEN = "<sos>"
CARRY_TOKEN = "c"  # carry indicator token in scratchpad

# 基础纯净词表（向后兼容 Note 70/71/72 历史模型）
VOCAB_BASIC = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "+", "-", "*", "=",
    CARRY_TOKEN,
    PAD_TOKEN, EOS_TOKEN, SOS_TOKEN
]

# 扩展字面噪声词表（支持自然语言算式、全角符号、多样式问句与空格）
# 严格按照 training/rl/reward/arith_rules.toml 设定的表面写法展开
VOCAB = [
    # 1. 阿拉伯数字 0-9
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    # 2. 全角数字 ０-９
    "０", "１", "２", "３", "４", "５", "６", "７", "８", "９",
    # 3. 半角与全角运算符
    "+", "-", "*", "/", "=", "?",
    "＋", "－", "×", "÷", "＝", "？",
    # 4. 中文运算字元（加、加上、加一加、减、减去、乘、除等）
    "加", "上", "一", "减", "去", "掉", "乘", "以", "除",
    # 5. 中文等号/结论/问句字元
    "等", "于", "得", "几", "多", "少", "算", "问", "题", "答", "案", "是",
    # 6. 空格与格式符号
    " ", ":", "：",
    # 7. 进位 Scratchpad 标记
    CARRY_TOKEN,
    # 8. 特殊标记
    PAD_TOKEN, EOS_TOKEN, SOS_TOKEN
]

CHAR_TO_ID = {ch: i for i, ch in enumerate(VOCAB)}
ID_TO_CHAR = {i: ch for i, ch in enumerate(VOCAB)}
PAD_ID = CHAR_TO_ID[PAD_TOKEN]
EOS_ID = CHAR_TO_ID[EOS_TOKEN]
SOS_ID = CHAR_TO_ID[SOS_TOKEN]
VOCAB_SIZE = len(VOCAB)

# 基础词表映射（用于兼容旧权重）
CHAR_TO_ID_BASIC = {ch: i for i, ch in enumerate(VOCAB_BASIC)}
ID_TO_CHAR_BASIC = {i: ch for i, ch in enumerate(VOCAB_BASIC)}
VOCAB_SIZE_BASIC = len(VOCAB_BASIC)

def encode_str(s: str, vocab_type: str = "extended") -> List[int]:
    mapping = CHAR_TO_ID if vocab_type == "extended" else CHAR_TO_ID_BASIC
    res = []
    for c in s:
        if c in mapping:
            res.append(mapping[c])
        else:
            # 宽容处理全角/半角映射
            if c == "+":
                res.append(mapping.get("＋", mapping.get("+", 0)))
            elif c == "=":
                res.append(mapping.get("＝", mapping.get("=", 0)))
            else:
                pass
    return res

def decode_ids(ids: List[int], vocab_type: str = "extended") -> str:
    mapping = ID_TO_CHAR if vocab_type == "extended" else ID_TO_CHAR_BASIC
    pad_val = PAD_ID if vocab_type == "extended" else CHAR_TO_ID_BASIC[PAD_TOKEN]
    eos_val = EOS_ID if vocab_type == "extended" else CHAR_TO_ID_BASIC[EOS_TOKEN]
    sos_val = SOS_ID if vocab_type == "extended" else CHAR_TO_ID_BASIC[SOS_TOKEN]
    res = []
    for i in ids:
        if i == eos_val:
            break
        if i in (pad_val, sos_val):
            continue
        res.append(mapping.get(i, "?"))
    return "".join(res)

def parse_scratchpad_to_number(raw_s: str) -> str:
    """
    Decodes scratchpad format (e.g. '0c4c1' -> reverse of '140' = '140')
    Filter out 'c' tokens, reverse the remaining digits to obtain standard number.
    """
    digits = [ch for ch in raw_s if ch.isdigit()]
    return "".join(digits)[::-1]

def generate_literal_noisy_addition_sample(
    min_digits: int = 1,
    max_digits: int = 4,
    use_scratchpad: bool = False,
    reverse_output: bool = True,
    noise_prob: float = 0.8,
    rng: random.Random = None
) -> Tuple[str, str]:
    """
    生成带有字面/表面噪声（Literal / Formatting Noise）的加法样本。
    支持：
    1. 运算符多样性: '+', '＋', '加', '加上', '加一加'
    2. 等号与问句多样性: '=', '＝', '等于', '得', '等于几?', '等于多少?'
    3. 前缀多样性: '', '算一下', '题目:', '问题:'
    4. 随机空格扰动
    5. 全角数字扰动 (例如 '１２＋３４＝')
    """
    r = rng if rng is not None else random
    len_a = r.randint(min_digits, max_digits)
    len_b = r.randint(min_digits, max_digits)
    
    low_a = 10 ** (len_a - 1) if len_a > 1 else 0
    high_a = (10 ** len_a) - 1
    low_b = 10 ** (len_b - 1) if len_b > 1 else 0
    high_b = (10 ** len_b) - 1
    
    a = r.randint(low_a, high_a)
    b = r.randint(low_b, high_b)
    c = a + b
    
    str_a = str(a)
    str_b = str(b)
    
    # 是否注入字面噪声
    if r.random() < noise_prob:
        # 1. 运算符噪音
        op_candidates = ["+", "＋", "加", "加上", "加一加"]
        op_str = r.choice(op_candidates)
        
        # 2. 等号与提问噪音
        eq_candidates = ["=", "＝", "等于", "得", "等于几？", "等于多少？", "＝几？"]
        eq_str = r.choice(eq_candidates)
        
        # 3. 前缀噪音
        prefix_candidates = ["", "", "", "算:", "题目:", "问:"]
        prefix_str = r.choice(prefix_candidates)
        
        # 4. 全角数字噪音 (10% 概率)
        if r.random() < 0.15:
            full_map = str.maketrans("0123456789", "０１２３４５６７８９")
            str_a = str_a.translate(full_map)
            str_b = str_b.translate(full_map)
            
        # 5. 空格扰动 (20% 概率)
        sp = " " if r.random() < 0.2 else ""
        prompt = f"{prefix_str}{str_a}{sp}{op_str}{sp}{str_b}{sp}{eq_str}"
    else:
        prompt = f"{str_a}+{str_b}="
        
    if use_scratchpad:
        # 按照数字值计算 compact scratchpad
        s_a = str(a)[::-1]
        s_b = str(b)[::-1]
        max_len = max(len(s_a), len(s_b))
        carry = 0
        target_tokens = []
        for i in range(max_len):
            d_a = int(s_a[i]) if i < len(s_a) else 0
            d_b = int(s_b[i]) if i < len(s_b) else 0
            cur_sum = d_a + d_b + carry
            out_digit = cur_sum % 10
            carry = cur_sum // 10
            target_tokens.append(str(out_digit))
            if carry > 0:
                target_tokens.append("c")
        if carry > 0:
            target_tokens.append(str(carry))
        c_str = "".join(target_tokens)
    else:
        c_str = str(c)
        if reverse_output:
            c_str = c_str[::-1]
            
    return prompt, c_str

def generate_addition_sample(min_digits: int = 1, max_digits: int = 4, reverse_output: bool = True, rng: random.Random = None) -> Tuple[str, str]:
    """
    Generate an addition problem: A + B = C
    Allows arbitrary and asymmetric digit lengths (e.g. 1-digit + 4-digit).
    Example (reverse_output=True):
      prompt: "12+4567="
      target: "9754" (4579 reversed)
    """
    r = rng if rng is not None else random
    len_a = r.randint(min_digits, max_digits)
    len_b = r.randint(min_digits, max_digits)
    
    low_a = 10 ** (len_a - 1) if len_a > 1 else 0
    high_a = (10 ** len_a) - 1
    low_b = 10 ** (len_b - 1) if len_b > 1 else 0
    high_b = (10 ** len_b) - 1
    
    a = r.randint(low_a, high_a)
    b = r.randint(low_b, high_b)
    c = a + b
    
    prompt = f"{a}+{b}="
    c_str = str(c)
    if reverse_output:
        c_str = c_str[::-1]
    return prompt, c_str

def generate_scratchpad_addition_sample(min_digits: int = 1, max_digits: int = 5, rng: random.Random = None) -> Tuple[str, str]:
    """
    Generate an addition problem with explicit step-by-step carry scratchpad:
    Example: 87 + 45 =
    Step 0: ones 7+5=12 -> digit 2, carry 1
    Step 1: tens 8+4+c1=13 -> digit 3, carry 1
    Step 2: hundreds 0+0+c1=1 -> digit 1, carry 0
    Format:
      prompt: "87+45="
      target: "2c1 3c1 1c0" or compact "2c3c1" (meaning: output 2 with carry, then 3 with carry, then 1)
    
    Compact Scratchpad Format:
      If carry = 1, append 'c', else direct digit.
      E.g. 7+5 = 12 -> "2c" (digit 2, carry 1)
           3+4 = 7  -> "7"  (digit 7, no carry)
      So: 87+45 -> 7+5=12 ("2c"), 8+4+1=13 ("3c"), 0+0+1=1 ("1")
      target: "2c3c1"
    """
    r = rng if rng is not None else random
    len_a = r.randint(min_digits, max_digits)
    len_b = r.randint(min_digits, max_digits)
    
    low_a = 10 ** (len_a - 1) if len_a > 1 else 0
    high_a = (10 ** len_a) - 1
    low_b = 10 ** (len_b - 1) if len_b > 1 else 0
    high_b = (10 ** len_b) - 1
    
    a = r.randint(low_a, high_a)
    b = r.randint(low_b, high_b)
    
    prompt = f"{a}+{b}="
    
    str_a = str(a)[::-1]
    str_b = str(b)[::-1]
    max_l = max(len(str_a), len(str_b))
    
    target_tokens = []
    carry = 0
    for i in range(max_l):
        d_a = int(str_a[i]) if i < len(str_a) else 0
        d_b = int(str_b[i]) if i < len(str_b) else 0
        s = d_a + d_b + carry
        digit = s % 10
        carry = s // 10
        target_tokens.append(str(digit))
        if carry > 0:
            target_tokens.append("c")
            
    if carry > 0:
        target_tokens.append(str(carry))
        
    target = "".join(target_tokens)
    return prompt, target

def generate_subtraction_sample(min_digits: int = 1, max_digits: int = 4, reverse_output: bool = True, rng: random.Random = None) -> Tuple[str, str]:
    """
    Generate subtraction problem: A - B = C (ensure A >= B >= 0)
    Allows arbitrary and asymmetric digit lengths.
    """
    r = rng if rng is not None else random
    len_a = r.randint(min_digits, max_digits)
    len_b = r.randint(min_digits, max_digits)
    
    low_a = 10 ** (len_a - 1) if len_a > 1 else 0
    high_a = (10 ** len_a) - 1
    low_b = 10 ** (len_b - 1) if len_b > 1 else 0
    high_b = (10 ** len_b) - 1
    
    a = r.randint(low_a, high_a)
    b = r.randint(low_b, high_b)
    if a < b:
        a, b = b, a
    c = a - b
    prompt = f"{a}-{b}="
    c_str = str(c)
    if reverse_output:
        c_str = c_str[::-1]
    return prompt, c_str

class ArithmeticDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 50000,
        min_digits: int = 1,
        max_digits: int = 4,
        ops: Tuple[str, ...] = ("+",),
        reverse_output: bool = True,
        use_scratchpad: bool = False,
        literal_noise: bool = False,
        literal_noise_prob: float = 0.8,
        max_seq_len: int = 48,
        seed: int = 42,
    ):
        super().__init__()
        self.samples = []
        rng = random.Random(seed)
        self.use_scratchpad = use_scratchpad
        
        for _ in range(num_samples):
            op = rng.choice(ops)
            if op == "+":
                if use_scratchpad:
                    if literal_noise:
                        prompt, target = generate_literal_noisy_addition_sample(min_digits, max_digits, use_scratchpad=True, noise_prob=literal_noise_prob, rng=rng)
                    else:
                        prompt, target = generate_scratchpad_addition_sample(min_digits, max_digits, rng=rng)
                else:
                    if literal_noise:
                        prompt, target = generate_literal_noisy_addition_sample(min_digits, max_digits, reverse_output=reverse_output, noise_prob=literal_noise_prob, rng=rng)
                    else:
                        prompt, target = generate_addition_sample(min_digits, max_digits, reverse_output=reverse_output, rng=rng)
            elif op == "-":
                prompt, target = generate_subtraction_sample(min_digits, max_digits, reverse_output=reverse_output, rng=rng)
            else:
                raise ValueError(f"Unsupported op: {op}")
            
            self.samples.append((prompt, target))
            
        self.max_seq_len = max_seq_len
        self.reverse_output = reverse_output

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt, target = self.samples[idx]
        
        # Sequence format: prompt + target + EOS
        prompt_ids = encode_str(prompt)
        target_ids = encode_str(target) + [EOS_ID]
        
        full_ids = prompt_ids + target_ids
        
        # Labels: ignore loss on prompt tokens (-100)
        labels = [-100] * len(prompt_ids) + target_ids
        
        # Truncate if exceeds max_seq_len
        if len(full_ids) > self.max_seq_len:
            full_ids = full_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
            
        # Pad to max_seq_len
        pad_len = self.max_seq_len - len(full_ids)
        input_ids = full_ids + [PAD_ID] * pad_len
        labels = labels + [-100] * pad_len
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "prompt_len": len(prompt_ids),
            "prompt": prompt,
            "target": target,
        }

if __name__ == "__main__":
    ds = ArithmeticDataset(num_samples=5, min_digits=2, max_digits=3, ops=("+",), reverse_output=True)
    print(f"Vocab size: {VOCAB_SIZE}")
    for i in range(5):
        sample = ds[i]
        print(f"Prompt: {sample['prompt']} Target: {sample['target']} Input: {sample['input_ids'][:12].tolist()} Labels: {sample['labels'][:12].tolist()}")
