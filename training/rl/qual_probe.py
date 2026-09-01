import os, sys, random
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
import torch
sys.path.insert(0, "/home/vesita/coding/my/nanoSeek")
from inference.scripts.sample_py import build_model_from_checkpoint, load_tokenizer
from grpo_char import sample_candidates_batch

for ck, label in [("out/curriculum/stage_12_1200", "stage12"), ("out/eos_fix_1epoch", "base2epoch")]:
    model, ckpt = build_model_from_checkpoint(ck)
    model.to("cuda"); model.eval()
    tok = load_tokenizer(ckpt); eos_id = tok.token_to_id("<eos>")
    print(f"\n########## {label} ##########")
    for p in ["为什么天空是蓝色的？", "你今天心情怎么样呀？", "你好", "帮我起个名字"]:
        ids = tok.encode(f"用户：{p}\n模型：").ids
        rows = sample_candidates_batch(model, tok, ids, eos_id, group_size=4,
                                       max_new_tokens=55, temperature=1.0, top_k=200,
                                       repeat_penalty=1.4, device="cuda")
        texts = [tok.decode(r).replace("<eos>","").strip() for r in rows]
        nonempty = [t for t in texts if t]
        print(f"  [{p}] 非空{len(nonempty)}/4")
        for t in nonempty[:2]:
            print(f"      > {t[:44]!r}")
