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


def build_model_from_checkpoint(out_dir, device=None, rope_len=None):
    """复刻 train.py _build_model_from_checkpoint：按 checkpoint 的 model_args 建模型并加载权重。

    device=None → 自动选（CUDA 可用则 GPU，否则 CPU）。GPU 推理远快于 CPU（~5-10×），
    负载也远轻于训练（3M 参数单序列前向），默认开 GPU。
    rope_len（dev-notes/46 窗口续传）：扩展 RoPE 表到指定长度（窗口截断推理需要
    绝对位置偏移，表要覆盖整个对话；None = 保持训练长度）。buffer 加载后替换，不影响权重。
    """
    ckpt = torch.load(Path(out_dir) / "best.pt", map_location="cpu")
    args = dict(ckpt["model_args"])
    conf = GPTConfig(**args)
    # 推理无反向传播：梯度检查点纯浪费（profile：每步 414 次 checkpoint 调用）
    if getattr(conf, "kv_memory_checkpoint", False):
        conf.kv_memory_checkpoint = False
    model = GPT(conf)
    state = ckpt["model"]
    for k in list(state.keys()):  # 修 torch.compile 的 _orig_mod. 前缀
        if k.startswith("_orig_mod."):
            state[k[len("_orig_mod."):]] = state.pop(k)
    model.load_state_dict(state)
    if rope_len is not None and conf.use_rope:
        # 窗口续传：RoPE 表扩到 rope_len（绝对位置偏移需要；buffer 非权重，替换无碍）
        from model.attention import CausalSelfAttention
        from model.utils import precompute_rope_freqs
        for m in model.modules():
            if isinstance(m, CausalSelfAttention) and m.use_rope:
                cos, sin = precompute_rope_freqs(m.rope_head_dim, rope_len, conf.rope_theta)
                m.register_buffer("cos", cos)
                m.register_buffer("sin", sin)
    model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        model = model.to(device)
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
                 stop_on_turn=False, stop_on_eos=False, clip_at_sentence=False, window=None,
                 no_resume=False, resume_state=None, token_callback=None):
    """生成并返回 (完整 token 列表, eos_pos)。

    与 generate() 逻辑完全一致，但返回 token 级结果：
    - 完整 token 列表（prompt + 生成，EOS 之前的所有 token）
    - eos_pos：模型自然吐出 <eos> 的生成区位置（-1 = 没吐）
    <eos> 解码为空串，字符串层检测不到，必须在 token 级看。

    window（dev-notes/46 推理状态选择性续传）：非 None 时输入只保留最近 window
    个 token（RoPE 用全局绝对位置偏移），窗口外信息由缓存的记忆状态承接——
    上下文不随对话增长。window=None = 完整上下文（现行为）。
    no_resume：window 模式下禁用状态续传（每步记忆从 0 递推）——对照实验用，
    量化"续传"本身的价值。
    resume_state：跨轮续传的外部记忆状态（None = 本轮从零开始）——多轮评估
    时上一轮结束的状态注入本轮第一步。
    """
    idx = tok.encode(prompt).ids
    device = next(model.parameters()).device        # 与模型同设备（GPU 推理时 idx 也在 GPU）
    idx = torch.tensor([idx], dtype=torch.long, device=device)
    new_start = idx.shape[1]
    seen = list(idx[0].tolist())
    eos_id = tok.token_to_id("<eos>")
    eos_pos = -1
    mem_state = resume_state
    for step in range(max_new_tokens):
        if window is not None:
            win_len = min(idx.size(1), window)
            idx_cond = idx[:, -win_len:]                       # 最近 window 个 token
            rope_offset = idx.size(1) - win_len                # 窗口起点全局绝对位置
        else:
            idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
            rope_offset = 0
        if mem_state is not None and not no_resume:
            model.set_memory_state(mem_state)                  # 续传：记忆从缓存状态继续
        elif no_resume:
            model.set_memory_state(None)                       # 对照：每步从 0 递推
        logits, _ = model(idx_cond, rope_offset=rope_offset)
        if not no_resume:
            mem_state = model.get_memory_state()               # 更新状态缓存（末态）
        logits = logits[:, -1, :] / temperature
        v = logits.squeeze(0).clone()
        if repeat_penalty > 1.0:
            # 向量化（原 O(T²) python 循环：每步遍历全部已见 token）
            seen_t = torch.as_tensor(seen, dtype=torch.long, device=v.device)
            vs = v[seen_t]
            v[seen_t] = torch.where(vs >= 0, vs / repeat_penalty, vs * repeat_penalty)
        if top_k is not None:
            k = min(top_k, v.size(-1))
            topv, _ = torch.topk(v, k)
            v[v < topv[-1]] = -float("Inf")
        probs = F.softmax(v, dim=-1)
        nxt = torch.multinomial(probs, 1)
        nxt_id = int(nxt.item())
        if token_callback is not None:
            token_callback(nxt_id)                 # 流式输出：每步回调（chat.py 打字机）
        if stop_on_eos and nxt_id == eos_id:
            eos_pos = step            # 模型自己说"完了"：记录并停（EOS 不进输出）
            break
        seen.append(nxt_id)
        idx = torch.cat((idx, nxt.unsqueeze(0)), dim=1)
        if stop_on_turn:
            gen_ids = idx[0][new_start:].tolist()
            # 每步全量 decode 是 O(T²)：只在末尾 16-token 窗口里找轮次标记（marker ≤4 字符≈2-4 token），
            # 命中才全量截断。窗口内漏检最多晚几步截断，不影响截断语义。
            if any(m in tok.decode(gen_ids[-16:]) for m in ("\n用户：", "\n模型：", "\nUser:", "\nModel:")):
                truncated, hit = _truncate_at_turn(gen_ids, tok)
                if hit:
                    keep = torch.tensor([truncated], dtype=torch.long, device=idx.device)
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
             stop_on_turn=False, stop_on_eos=False, clip_at_sentence=False, window=None,
             no_resume=False):
    """生成（返回 prompt + 生成全文，保持旧接口）。轴钮语义见 generate_ids。"""
    ids, _ = generate_ids(model, tok, prompt, max_new_tokens, temperature, top_k,
                          repeat_penalty, stop_on_turn, stop_on_eos, clip_at_sentence,
                          window=window, no_resume=no_resume)
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
    ap.add_argument("--device", default=None, help="cuda/cpu；默认自动（有 GPU 用 GPU）")
    ap.add_argument("--no-resume", action="store_true", help="window 模式下禁用状态续传（对照实验）")
    ap.add_argument("--window", type=int, default=None,
                    help="推理状态选择性续传（dev-notes/46）：输入只保留最近 N token，"
                         "窗口外信息由缓存记忆状态承接；None = 完整上下文")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rope_len = 8192 if a.window is not None else None   # 窗口续传需 RoPE 表覆盖绝对位置
    model, ckpt = build_model_from_checkpoint(a.out_dir, a.device, rope_len=rope_len)
    tok = Tokenizer.from_file("data/chinese/tokenizer.json")
    n = sum(p.numel() for p in model.parameters())
    print(f"[{a.out_dir}] {n:,} 参数 | no_attn_layers={ckpt['model_args'].get('no_attn_layers')} | block_order={ckpt['model_args'].get('block_order')}")

    out = generate(model, tok, a.prompt, a.max_new_tokens, a.temperature, a.top_k,
                   a.repeat_penalty, a.stop_on_turn, a.stop_on_eos, a.clip_sentence,
                   window=a.window, no_resume=a.no_resume)
    # 打印 prompt + 生成全文
    print("--- 生成 ---")
    print(a.prompt + out[len(a.prompt):] if out.startswith(a.prompt) else out)


if __name__ == "__main__":
    main()
