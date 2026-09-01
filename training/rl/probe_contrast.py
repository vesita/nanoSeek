import os, sys, random
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
import torch
sys.path.insert(0, "/home/vesita/coding/my/nanoSeek")
from inference.scripts.sample_py import build_model_from_checkpoint, load_tokenizer
from grpo_char import sample_candidates_batch

torch.manual_seed(7); random.seed(7)
model, ckpt = build_model_from_checkpoint("out/curriculum/stage_09_900")
model.to("cuda"); model.eval()
tok = load_tokenizer(ckpt); eos_id = tok.token_to_id("<eos>")

prompts = [
    "你好",
    "天空为什么是蓝色的？",
    "今天心情不太好",
    "你叫什么名字呀？给自己起一个喜欢的名字吧！",
    "为什么天空是蓝色的？",
    "如果有机会出去旅游，你最想去哪里玩？",
    "你今天心情怎么样呀？",
    "如果由你来决定自己的名字，你想叫什么？",
]
for p in prompts:
    ids = tok.encode(p).ids
    rows = sample_candidates_batch(model, tok, ids, eos_id, group_size=4,
                                   max_new_tokens=55, temperature=1.0,
                                   top_k=200, repeat_penalty=1.4, device="cuda")
    texts = [tok.decode(r).replace("<eos>","").strip() for r in rows]
    ne = sum(1 for t in texts if t)
    print(f"非空 {ne}/4 | len(prompt_tokens)={len(ids):3d} | {p}")
    for t in texts:
        if t: print(f"      > {t[:40]!r}")
