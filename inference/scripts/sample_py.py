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


def _truncate_at_turn(gen_ids, tok):
    """生成内容里出现下一轮标签（\n用户：/\n模型：）→ 截断到标签之前（治喋喋不休）。

    模型学会对话骨架后常自己续写"用户：…"，这是天然轮次边界：话已说"完"才开下一轮。
    在字符层找标签位置，再回退到最近的 token 边界（BPE 标签可能跨 token）。
    """
    text = tok.decode(gen_ids)
    for marker in ("\n用户：", "\n模型：", "\nUser:", "\nModel:"):
        pos = text.find(marker)
        if pos >= 0:
            if pos == 0:
                return [], True
            for k in range(len(gen_ids) + 1):
                if len(tok.decode(gen_ids[:k])) > pos:
                    return gen_ids[:max(k - 1, 0)], True
            return gen_ids, False
    return gen_ids, False


@torch.no_grad()
def generate_ids(model, tok, prompt, max_new_tokens, temperature, top_k, repeat_penalty,
                 stop_on_turn=False, stop_on_eos=False, clip_at_sentence=False):
    """生成并返回 (完整 token 列表, eos_pos)。

    与 generate() 逻辑完全一致，但返回 token 级结果：
    - 完整 token 列表（prompt + 生成，EOS 之前的所有 token）
    - eos_pos：模型自然吐出 <eos> 的生成区位置（-1 = 没吐）
    <eos> 解码为空串，字符串层检测不到，必须在 token 级看。
    """
    idx = tok.encode(prompt).ids
    idx = torch.tensor([idx], dtype=torch.long)
    new_start = idx.shape[1]
    seen = list(idx[0].tolist())
    eos_id = tok.token_to_id("<eos>")
    eos_pos = -1
    for step in range(max_new_tokens):
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
        nxt_id = int(nxt.item())
        if stop_on_eos and nxt_id == eos_id:
            eos_pos = step            # 模型自己说"完了"：记录并停（EOS 不进输出）
            break
        seen.append(nxt_id)
        idx = torch.cat((idx, nxt.unsqueeze(0)), dim=1)
        if stop_on_turn:
            gen_ids = idx[0][new_start:].tolist()
            truncated, hit = _truncate_at_turn(gen_ids, tok)
            if hit:
                keep = torch.tensor([truncated], dtype=torch.long)
                idx = torch.cat([idx[:, :new_start], keep], dim=1)
                if eos_pos == -1:
                    eos_pos = step + 1    # 轮次截断 = 模型自然收尾的中止点
                break
    gen = idx[0].tolist()
    if clip_at_sentence:
        gen_ids = gen[new_start:]
        text = tok.decode(gen_ids).rstrip()
        if text and text[-1] not in "。！？…!?~～":
            pos = max(text.rfind(c) for c in "。！？…!?~～")
            if pos >= 0:
                for k in range(len(gen_ids) + 1):
                    if len(tok.decode(gen_ids[:k])) > pos:
                        gen = gen[:new_start] + gen_ids[:k]
                        break
    # 未开 stop_on_eos 时也找 EOS：模型吐了终止符 → 记录位置并截断
    if eos_pos == -1:
        for i, t in enumerate(gen[new_start:]):
            if t == eos_id:
                eos_pos = i
                gen = gen[:new_start + i]   # EOS 及其后不输出
                break
    return gen, eos_pos


@torch.no_grad()
def generate(model, tok, prompt, max_new_tokens, temperature, top_k, repeat_penalty,
             stop_on_turn=False, stop_on_eos=False, clip_at_sentence=False):
    """生成（返回 prompt + 生成全文，保持旧接口）。轴钮语义见 generate_ids。"""
    ids, _ = generate_ids(model, tok, prompt, max_new_tokens, temperature, top_k,
                          repeat_penalty, stop_on_turn, stop_on_eos, clip_at_sentence)
    return tok.decode(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="训练输出目录（读 best.pt）")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--repeat-penalty", type=float, default=1.2)
    ap.add_argument("--stop-on-turn", action="store_true",
                    help="检测到 \n用户：/\n模型： 标签即截断（轮次边界=结束点，治喋喋不休）")
    ap.add_argument("--stop-on-eos", action="store_true",
                    help="采样到 <eos> 即停止（训练数据里的结束符）")
    ap.add_argument("--clip-sentence", action="store_true",
                    help="预算用尽时回退到最后一个句号/问号/叹号处，不留半句")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    model, ckpt = build_model_from_checkpoint(a.out_dir)
    tok = Tokenizer.from_file("data/chinese/tokenizer.json")
    n = sum(p.numel() for p in model.parameters())
    print(f"[{a.out_dir}] {n:,} 参数 | no_attn_layers={ckpt['model_args'].get('no_attn_layers')} | block_order={ckpt['model_args'].get('block_order')}")

    out = generate(model, tok, a.prompt, a.max_new_tokens, a.temperature, a.top_k,
                   a.repeat_penalty, a.stop_on_turn, a.stop_on_eos, a.clip_sentence)
    # 打印 prompt + 生成全文
    print("--- 生成 ---")
    print(a.prompt + out[len(a.prompt):] if out.startswith(a.prompt) else out)


if __name__ == "__main__":
    main()
