"""
Python 采样器：为实验性（Rust 未支持）架构提供采样目测。

用途：Rust 运行时只部署官方架构（attn_ffn 全注意力）。实验臂如稀疏布线
（no_attn_layers）、ffn_attn 等无法用 Rust 采样，本脚本直接加载训练 checkpoint
在 Python 里生成，采样逻辑镜像 Rust 的 sample()（model.rs）：温度 → 重复惩罚
（CTRL 做法，对已见 token 施加）→ top-k → softmax → 多项式采样。

用法（从项目根目录）：
    uv run python inference/scripts/sample_py.py \
        --out_dir out/chinese-data2-sparse2 \
        --prompt "用户：最近好累怎么办？\n模型："
    # 可选：--max-new-tokens 300 --temperature 0.8 --top-k 200 --repeat-penalty 1.2 --seed 1337
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import GPTConfig, GPT


def build_model_from_checkpoint(out_dir):
    """复刻 train.py _build_model_from_checkpoint：按 checkpoint 的 model_args 建模型并加载权重。"""
    ckpt = torch.load(Path(out_dir) / "best.pt", map_location="cpu")
    args = dict(ckpt["model_args"])
    conf = GPTConfig(**args)
    model = GPT(conf)
    state = ckpt["model"]
    for k in list(state.keys()):  # 修 torch.compile 的 _orig_mod. 前缀
        if k.startswith("_orig_mod."):
            state[k[len("_orig_mod."):]] = state.pop(k)
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


@torch.no_grad()
def generate(model, tok, prompt, max_new_tokens, temperature, top_k, repeat_penalty):
    idx = tok.encode(prompt).ids
    idx = torch.tensor([idx], dtype=torch.long)
    seen = list(idx[0].tolist())
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        v = logits.squeeze(0).clone()
        if repeat_penalty > 1.0:
            for t in seen:
                l = v[t]
                v[t] = l / repeat_penalty if l >= 0 else l * repeat_penalty
        if top_k is not None:
            k = min(top_k, v.size(-1))
            topv, _ = torch.topk(v, k)
            v[v < topv[-1]] = -float("Inf")
        probs = F.softmax(v, dim=-1)
        nxt = torch.multinomial(probs, 1)
        seen.append(int(nxt.item()))
        idx = torch.cat((idx, nxt.unsqueeze(0)), dim=1)
    return tok.decode(idx[0].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="训练输出目录（读 best.pt）")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--repeat-penalty", type=float, default=1.2)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    model, ckpt = build_model_from_checkpoint(a.out_dir)
    tok = Tokenizer.from_file("data/chinese/tokenizer.json")
    n = sum(p.numel() for p in model.parameters())
    print(f"[{a.out_dir}] {n:,} 参数 | no_attn_layers={ckpt['model_args'].get('no_attn_layers')} | block_order={ckpt['model_args'].get('block_order')}")

    out = generate(model, tok, a.prompt, a.max_new_tokens, a.temperature, a.top_k, a.repeat_penalty)
    # 打印 prompt + 生成全文
    print("--- 生成 ---")
    print(a.prompt + out[len(a.prompt):] if out.startswith(a.prompt) else out)


if __name__ == "__main__":
    main()
