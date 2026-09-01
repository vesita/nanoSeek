"""验证: 同一高危 prompt, 带"用户：/模型："标签 vs 裸 prompt, 空率差异 (stage_09_900 上)."""
import os, sys, random
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
import torch
sys.path.insert(0, "/home/vesita/coding/my/nanoSeek")
from inference.scripts.sample_py import build_model_from_checkpoint, load_tokenizer
from grpo_char import sample_candidates_batch

torch.manual_seed(3); random.seed(3)
model, ckpt = build_model_from_checkpoint("out/curriculum/stage_09_900")
model.to("cuda"); model.eval()
tok = load_tokenizer(ckpt); eos_id = tok.token_to_id("<eos>")

bare = ["为什么天空是蓝色的？",
        "如果由你来决定自己的名字，你想叫什么？",
        "如果有机会出去旅游，你最想去哪里玩？",
        "早安！今天又是充满无限可能与希望的一天！",
        "什么是量子计算？"]
print("ckpt=stage_09_900  group=4 temp=1.0 rep=1.4 max=55")
for p in bare:
    for label, pp in [("裸", p), ("带标签", f"用户：{p}\n模型：")]:
        ids = tok.encode(pp).ids
        rows = sample_candidates_batch(model, tok, ids, eos_id, group_size=4,
                                       max_new_tokens=55, temperature=1.0,
                                       top_k=200, repeat_penalty=1.4, device="cuda")
        texts = [tok.decode(r).replace("<eos>","").strip() for r in rows]
        ne = sum(1 for t in texts if t)
        sample = next((t for t in texts if t), "")
        print(f"  [{label}] 非空 {ne}/4 | {p[:16]} | 例: {sample[:26]!r}")
