"""决定性实验: 用 grpo_char 的【原训练 prompt 池】+【完全相同采样参数】测各 ckpt 非空率。
复现训练采样路径: group_size=4, max_new_tokens=55, temp=1.0, top_k=200, rep=1.4。
用法: python probe_training_pool.py <ckpt_dir> [--temp T] [--rep R] [--rounds N] [--seed S]
"""
import os, sys, random
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from inference.scripts.sample_py import build_model_from_checkpoint, load_tokenizer
from grpo_char import sample_candidates_batch
from grpo_char import IDENTITY_PROMPTS, TRAVEL_DESIRE_PROMPTS, MOOD_DAILY_PROMPTS, \
    HEURISTIC_OPEN_PROMPTS, FACT_PROMPTS

def main():
    ckpt_dir = sys.argv[1]
    temp = 1.0; rep = 1.4; rounds = 5; seed = 42
    if "--temp" in sys.argv: temp = float(sys.argv[sys.argv.index("--temp")+1])
    if "--rep" in sys.argv:  rep  = float(sys.argv[sys.argv.index("--rep")+1])
    if "--rounds" in sys.argv: rounds = int(sys.argv[sys.argv.index("--rounds")+1])
    if "--seed" in sys.argv: seed = int(sys.argv[sys.argv.index("--seed")+1])
    torch.manual_seed(seed); random.seed(seed)

    model, ckpt = build_model_from_checkpoint(ckpt_dir)
    model.to("cuda"); model.eval()
    tok = load_tokenizer(ckpt)
    eos_id = tok.token_to_id("<eos>")

    pool = ([(p, "identity") for p, _ in IDENTITY_PROMPTS] +
            [(p, "travel")   for p, _ in TRAVEL_DESIRE_PROMPTS] +
            [(p, "mood")     for p, _ in MOOD_DAILY_PROMPTS] +
            [(p, "heuristic") for p, _ in HEURISTIC_OPEN_PROMPTS] +
            [(p, "fact")     for p, _ in FACT_PROMPTS])

    print(f"ckpt={ckpt_dir} temp={temp} rep={rep} rounds={rounds} (group=4 max=55 top_k=200)")
    total_empty = 0; total_all = 0
    bad_prompts = []
    for p, kind in pool:
        ids = tok.encode(p).ids
        n_empty = 0; n_all = 0; samples_text = []
        for _ in range(rounds):
            rows = sample_candidates_batch(model, tok, ids, eos_id, group_size=4,
                                           max_new_tokens=55, temperature=temp,
                                           top_k=200, repeat_penalty=rep, device="cuda")
            texts = [tok.decode(r).replace("<eos>", "").strip() for r in rows]
            n_all += len(texts)
            n_empty += sum(1 for t in texts if not t)
            samples_text.extend([t for t in texts if t])
        rate = n_empty / n_all if n_all else 1.0
        total_empty += n_empty; total_all += n_all
        flag = "  <<< 高危" if rate > 0.5 else ""
        print(f"[{kind:9s}] 空 {n_empty:2d}/{n_all:2d} ({rate*100:4.0f}%){flag}  {p[:24]}")
        if rate > 0.5:
            bad_prompts.append((p, rate))
        if samples_text:
            print(f"          样例: {samples_text[0][:36]!r}")
    print(f"\n总计: 空 {total_empty}/{total_all} ({total_empty/total_all*100:.0f}%)")
    print(f"高危 prompt ({len(bad_prompts)}):")
    for p, r in bad_prompts:
        print(f"  {r*100:3.0f}%  {p}")

if __name__ == "__main__":
    main()
