"""
Python 采样器：为实验性（Rust 未支持）架构提供采样目测。

用途：Rust 运行时只部署官方架构（attn_ffn 全注意力）。实验臂如稀疏布线
（no_attn_layers）、ffn_attn 等无法用 Rust 采样，本脚本直接加载训练 checkpoint
在 Python 里生成，采样逻辑镜像 Rust 的 sample()（model.rs）：温度 → 重复惩罚
（CTRL 做法，对已见 token 施加）→ top-k → softmax → 多项式采样。

字节直入模型（dev-notes/48，byte_level=True）自动切换 ByteTokenizer，无需手动指定：
词表 257（0-255 字节 + 256=<eos>），输入 = UTF-8 字节流，模型内部 3 字节聚合 1 个
token，每步采样一组 3 字节（max_new_tokens 语义 = 聚合位置数，生成字节数 = 3×step）；
--window 在字节模式下按字节数计（如 192 ≈ 64 字）。

用法（从项目根目录）：
    uv run python inference/scripts/sample_py.py \
        --out_dir out/chinese-data2-sparse2 \
        --prompt "用户：最近好累怎么办？\n模型："
    # 可选：--max-new-tokens 300 --temperature 0.8 --top-k 200 --repeat-penalty 1.2 --seed 1337
    # 字节模型：--out_dir out/byte_daily_300 --prompt $'用户：最近压力好大\n模型：'
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


def load_tokenizer(ckpt):
    """按 checkpoint 的词表模式自动选 tokenizer（推理侧统一入口，chat/eval 复用）。

    char_level=True（dev-notes/50 字级）→ CharTokenizer（汉字=1 token，char_tokenizer.json）；
    byte_level=True（dev-notes/48 字节直入）→ ByteTokenizer（0-255 字节 + 256=<eos>）；
    否则 BPE（data/chinese/tokenizer.json）。
    """
    args = ckpt["model_args"]
    if args.get("char_level"):
        return Tokenizer.from_file("data/chinese/char_tokenizer.json")  # WordLevel 字级
    if args.get("byte_level"):
        from model.byte_tokenizer import ByteTokenizer
        return ByteTokenizer()
    return Tokenizer.from_file("data/chinese/tokenizer.json")


def _truncate_at_turn(gen_ids, tok, byte_mode=False):
    """生成内容里出现下一轮标签（\n用户：/\n模型：）→ 截断到标签之前（治喋喋不休）。

    模型学会对话骨架后常自己续写"用户：…"，这是天然轮次边界：话已说"完"才开下一轮。
    在字符层找标签位置，再回退到最近的 token 边界（BPE 标签可能跨 token）。
    字节模式（byte_mode）：标签按 UTF-8 字节序列在字节流里精确匹配（无跨 token 问题，
    decode 前缀长度匹配退化为字节索引直接截断）。
    """
    if byte_mode:
        for marker in ("\n用户：", "\n模型：", "\nUser:", "\nModel:"):
            mb = list(marker.encode("utf-8"))
            n = len(mb)
            for j in range(len(gen_ids) - n + 1):
                if gen_ids[j:j + n] == mb:
                    return gen_ids[:j], True
        return gen_ids, False
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
                 no_resume=False, resume_state=None, token_callback=None, context_ids=None,
                 stop_on_cont=False, stop_kind_ref=None):
    """生成并返回 (完整 token 列表, eos_pos)。

    stop_kind_ref（dev-notes/61，可选）：传入 list 时把停止原因写进 ref[0]——
    "eos"（说完收尾）/ "cont"（说完递回，期待继续）/ "turn"（轮次截断）/
    "maxlen"（到上限）/ None。默认 None = 不写回，完全向后兼容。

    与 generate() 逻辑完全一致，但返回 token 级结果：
    - 完整 token 列表（prompt + 生成，EOS 之前的所有 token）
    - eos_pos：模型自然吐出 <eos> 的生成区位置（-1 = 没吐）
    <eos> 解码为空串，字符串层检测不到，必须在 token 级看。

    window（dev-notes/46 推理状态选择性续传）：非 None 时输入只保留最近 window
    个 token（RoPE 用全局绝对位置偏移），窗口外信息由缓存的记忆状态承接——
    上下文不随对话增长。window=None = 完整上下文（现行为）。
    字节直入模型（model.config.byte_level，dev-notes/48）自动走字节分支：
    - 输入 = UTF-8 字节流（0-255 + 256=<eos>），长度补足 3 的倍数（头部补 0 对齐聚合组）
    - 每步前向 logits (B, 1, 3, V)，3 个字节位置独立采样 → 一组 3 字节；
      max_new_tokens 语义 = 聚合位置数（生成字节数 = 3×step）
    - token_callback 按聚合组回调（一次 3 字节；EOS 截断组只回调截断前缀）
    - window 参数按字节数计（如 192 ≈ 64 字）；rope_offset = 字节偏移 // 3
    - 返回的 token 列表不含头部对齐补齐字节（= 纯 prompt + 生成）
    no_resume：window 模式下禁用状态续传（每步记忆从 0 递推）——对照实验用，
    量化"续传"本身的价值。
    resume_state：跨轮续传的外部记忆状态（None = 本轮从零开始）——多轮评估
    时上一轮结束的状态注入本轮第一步。
    context_ids：可选。多轮对话时直接传入 token 级上下文。
    """
    byte_mode = bool(getattr(model.config, "byte_level", False))
    if context_ids is not None:
        idx = list(context_ids)
    else:
        idx = tok.encode(prompt).ids
    device = next(model.parameters()).device        # 与模型同设备（GPU 推理时 idx 也在 GPU）
    pad = 0
    if byte_mode:
        # 字节直入要求输入长度是 3 的倍数（3 字节/聚合组，gpt.py 断言）。任意 prompt 的
        # UTF-8 字节数不保证 %3==0 → 头部补 0 字节对齐（训练样本本身任意字节对齐切块，
        # 模型对组内偏移不敏感）；返回前剥掉补齐字节，接口与 BPE 完全一致。
        pad = (-len(idx)) % 3
        idx = [0] * pad + idx
    idx = torch.tensor([idx], dtype=torch.long, device=device)
    new_start = idx.shape[1]
    seen = list(idx[0].tolist())
    eos_id = tok.token_to_id("<eos>")
    cont_id = tok.token_to_id("<cont>") if stop_on_cont else None
    eos_pos = -1
    stop_kind = None           # None / "eos" / "cont"（该轮回复因何终止）
    mem_state = resume_state
    for step in range(max_new_tokens):
        if window is not None:
            win_len = min(idx.size(1), window)
            if byte_mode:
                win_len -= win_len % 3          # 字节窗口对齐到聚合组边界
                win_len = max(win_len, 3)
            idx_cond = idx[:, -win_len:]                       # 最近 window 个 token/字节
            rope_offset = ((idx.size(1) - win_len) // 3 if byte_mode
                           else idx.size(1) - win_len)         # 窗口起点全局绝对位置
        else:
            if byte_mode:
                # block_size 是聚合位置数 → 字节上限 = block_size×3（训练 block 为 3 的倍数）
                limit = model.config.block_size * 3
                idx_cond = idx if idx.size(1) <= limit else idx[:, -limit:]
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
        if byte_mode:
            # 字节直入：logits (B, T, 3, V)，推理只取最后聚合位置 → (B, 3, V)，
            # 3 个字节位置各自 temperature/top_k/repeat_penalty + softmax + 采样。
            v = logits[:, -1, :, :].squeeze(0).clone() / temperature     # (3, V)
        else:
            v = logits[:, -1, :].squeeze(0).clone() / temperature        # (V,)
        if repeat_penalty > 1.0:
            # 向量化（原 O(T²) python 循环：每步遍历全部已见 token/字节）
            seen_t = torch.as_tensor(seen, dtype=torch.long, device=v.device)
            if v.dim() == 1:
                vs = v[seen_t]
                v[seen_t] = torch.where(vs >= 0, vs / repeat_penalty, vs * repeat_penalty)
            else:
                # 字节模式 v 是 (3, V)：沿最后一维 gather（3 个字节位置各自惩罚）
                vs = v[:, seen_t]
                v[:, seen_t] = torch.where(vs >= 0, vs / repeat_penalty, vs * repeat_penalty)
        if top_k is not None:
            k = min(top_k, v.size(-1))
            topv, _ = torch.topk(v, k, dim=-1)
            if v.dim() == 1:
                v[v < topv[-1]] = -float("Inf")
            else:
                v[v < topv[:, -1].unsqueeze(-1)] = -float("Inf")   # 每行各自阈值
        probs = F.softmax(v, dim=-1)
        nxt = torch.multinomial(probs, 1)
        if byte_mode:
            nxt_ids = nxt.squeeze(-1).tolist()     # 一组 3 个字节 id
            cut = nxt_ids.index(eos_id) if (stop_on_eos and eos_id in nxt_ids) else None
            out_ids = nxt_ids if cut is None else nxt_ids[:cut]    # EOS 不进输出
            if token_callback is not None and out_ids:
                token_callback(out_ids)            # 流式输出：按聚合组回调（UTF-8 分片由 decode 增量自愈）
            if cut is not None:
                eos_pos = step * 3 + cut           # EOS 在生成区内的字节位置
                break
            seen.extend(out_ids)
            idx = torch.cat((idx, torch.tensor([out_ids], dtype=torch.long, device=device)), dim=1)
        else:
            nxt_id = int(nxt.item())
            if token_callback is not None:
                token_callback(nxt_id)             # 流式输出：每步回调（chat.py 打字机）
            # 终止符判定：<eos> 或 <cont>(待续) 都标记本轮回复结束点（cont 时 stop_on_cont 开启）
            if (stop_on_eos and nxt_id == eos_id) or (stop_on_cont and cont_id is not None and nxt_id == cont_id):
                eos_pos = step
                stop_kind = "eos" if nxt_id == eos_id else "cont"
                break
            seen.append(nxt_id)
            idx = torch.cat((idx, nxt.unsqueeze(0)), dim=1)
        if stop_on_turn:
            gen_ids = idx[0][new_start:].tolist()
            # 每步全量 decode 是 O(T²)：只在末尾小窗口里找轮次标记（marker ≤4 字符≈2-4 token），
            # 命中才全量截断。窗口内漏检最多晚几步截断，不影响截断语义。
            if byte_mode:
                tail = tok.decode(gen_ids[-48:])   # 48 字节 ≈ 16 字（字节模式放宽窗口）
            else:
                tail = tok.decode(gen_ids[-16:])
            if any(m in tail for m in ("\n用户：", "\n模型：", "\nUser:", "\nModel:")):
                truncated, hit = _truncate_at_turn(gen_ids, tok, byte_mode=byte_mode)
                if hit:
                    keep = torch.tensor([truncated], dtype=torch.long, device=idx.device)
                    idx = torch.cat([idx[:, :new_start], keep], dim=1)
                    if eos_pos == -1:
                        eos_pos = step + 1    # 轮次截断 = 模型自然收尾的中止点
                    if stop_kind is None:
                        stop_kind = "turn"    # 模型自己开下一轮 = 自然收尾（dev-notes/61）
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
                        if byte_mode:
                            # 字节模式：回退到完整字符边界，截断处不留替换符（半截字符）
                            while k > 0 and tok.decode(gen_ids[:k]).endswith("\ufffd"):
                                k -= 1
                        gen = gen[:new_start] + gen_ids[:k]
                        break
    # 未开 stop_on_eos 时也找 EOS：模型吐了终止符 → 记录位置并截断
    if eos_pos == -1:
        for i, t in enumerate(gen[new_start:]):
            if t == eos_id:
                eos_pos = i
                gen = gen[:new_start + i]   # EOS 及其后不输出
                break
    if byte_mode and pad:
        gen = gen[pad:]                     # 剥掉头部对齐补齐字节（返回=纯 prompt+生成）
    if stop_kind_ref is not None:
        # 回写停止原因（dev-notes/61）：循环内已记录的 eos/cont 优先；
        # 其余情况由 eos_pos / 轮次截断 / 步数上限推断。
        if stop_kind is None:
            if eos_pos >= 0:
                stop_kind = "eos"
            elif stop_on_turn and eos_pos >= 0:
                stop_kind = "turn"
            else:
                stop_kind = "maxlen"
        stop_kind_ref[0] = stop_kind
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
    tok = load_tokenizer(ckpt)                          # 字节直入模型自动切 ByteTokenizer
    n = sum(p.numel() for p in model.parameters())
    byte_flag = bool(ckpt["model_args"].get("byte_level"))
    print(f"[{a.out_dir}] {n:,} 参数 | byte_level={byte_flag} | no_attn_layers={ckpt['model_args'].get('no_attn_layers')} | block_order={ckpt['model_args'].get('block_order')}")

    out = generate(model, tok, a.prompt, a.max_new_tokens, a.temperature, a.top_k,
                   a.repeat_penalty, a.stop_on_turn, a.stop_on_eos, a.clip_sentence,
                   window=a.window, no_resume=a.no_resume)
    # 打印 prompt + 生成全文
    print("--- 生成 ---")
    print(a.prompt + out[len(a.prompt):] if out.startswith(a.prompt) else out)


if __name__ == "__main__":
    main()
