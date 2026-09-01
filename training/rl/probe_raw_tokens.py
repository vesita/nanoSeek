import os, sys, random
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
import torch
sys.path.insert(0, "/home/vesita/coding/my/nanoSeek")
from inference.scripts.sample_py import build_model_from_checkpoint, load_tokenizer
from grpo_char import sample_candidates_batch

model, ckpt = build_model_from_checkpoint("out/curriculum/stage_09_900")
model.to("cuda"); model.eval()
tok = load_tokenizer(ckpt); eos_id = tok.token_to_id("<eos>")

prompts = ["天空为什么是蓝色的？", "你今天心情怎么样呀？", "你好"]
for p in prompts:
    ids = tok.encode(p).ids
    print(f"\n=== {p} (len={len(ids)}) ===")
    for temp in [0.3, 0.6, 1.0]:
        rows = sample_candidates_batch(model, tok, ids, eos_id, group_size=4,
                                       max_new_tokens=55, temperature=temp,
                                       top_k=200, repeat_penalty=1.4, device="cuda")
        ne = sum(1 for r in rows if tok.decode(r).replace("<eos>","").strip())
        print(f"  temp={temp}: 非空 {ne}/4")
        for r in rows[:2]:
            raw = tok.decode(r)
            print(f"      ids={r} raw={raw!r}")
