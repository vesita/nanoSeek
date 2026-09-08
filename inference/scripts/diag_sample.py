"""诊断采样：加载 best.pt，计时逐 token 生成，打印 token 级行为与文本。
用途：分辨「推理速度慢」vs「模型陷入模板循环/崩坏」。"""
import sys, time, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from inference.scripts.sample_py import build_model_from_checkpoint, load_tokenizer, generate_ids

OUT = "out/nanoseek_100m"
PROMPT = os.environ.get("PROMPT", "用户：你好\n模型：")

export = lambda: None
device = "cuda"
model, ck = build_model_from_checkpoint(OUT, device)
tok = load_tokenizer(ck)
plen = len(tok.encode(PROMPT).ids)

print(f"设备 {device} · 参数量 {sum(p.numel() for p in model.parameters()):,}")
print(f"prompt: {PROMPT!r}  (plen={plen})")

torch.manual_seed(1337)
t0 = time.time()
ids, eos_pos = generate_ids(model, tok, PROMPT, 200, 0.8, 200, 1.2,
                            stop_on_turn=False, stop_on_eos=False, clip_at_sentence=False)
t1 = time.time()
gen = len(ids) - plen
print(f"\n生成 {gen} token，耗时 {t1-t0:.2f}s，速度 {gen/(t1-t0):.1f} tok/s，EOS@{eos_pos}")
text = tok.decode(ids[plen:])
print("--- 文本 ---")
print(text[:400])
print("--- 末 30 token ---")
print(repr(ids[-30:]))
