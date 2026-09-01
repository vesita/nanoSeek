"""探针: 在给定检查点 + 参数下采样 N 次, 统计非空回复率/短语重复率。
用法: python probe_sampling.py <ckpt_dir> <temp> <rep> [--seed N]
"""
import os, sys, random
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from inference.scripts.sample_py import build_model_from_checkpoint, load_tokenizer
from grpo_char import sample_candidates_batch

def main():
    ckpt_dir = sys.argv[1]
    temp = float(sys.argv[2])
    rep = float(sys.argv[3])
    seed = 42
    if "--seed" in sys.argv:
        seed = int(sys.argv[sys.argv.index("--seed") + 1])
    torch.manual_seed(seed); random.seed(seed)

    model, ckpt = build_model_from_checkpoint(ckpt_dir)
    model.to("cuda"); model.eval()
    tok = load_tokenizer(ckpt)
    eos_id = tok.token_to_id("<eos>")

    prompts = ["你好", "天空为什么是蓝色的？", "帮我起个名字", "今天心情不太好", "我想去旅行"]
    for p in prompts:
        ids = tok.encode(p).ids
        rows = sample_candidates_batch(model, tok, ids, eos_id, group_size=8,
                                       max_new_tokens=50, temperature=temp,
                                       top_k=200, repeat_penalty=rep, device="cuda")
        texts = [tok.decode(r).replace("<eos>", "").strip() for r in rows]
        nonempty = [t for t in texts if t]
        n_empty = len(texts) - len(nonempty)
        top = nonempty[0] if nonempty else "(无)"
        print(f"\nQ: {p}")
        print(f"  非空 {len(nonempty)}/8  空 {n_empty}/8  | 首个非空: {top[:40]!r}")

if __name__ == "__main__":
    main()
