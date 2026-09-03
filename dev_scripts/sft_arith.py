#!/usr/bin/env python3
"""判别实验 v2: 2.7M 模型能否用 SFT 学会"个位数加法"并泛化?

背景: 纯 RL (1000+300步) 数学命中≈0; 基座 argmax 在最简 a+b 上完全不输出数字。
本脚本从基座 warm-start 做监督微调, 只对答案区 "答案是N" 算 loss, 判:
  1. 训练集内学会(记忆)?
  2. 训练集外泛化 (unseen a+b)?
若 SFT 泛化也失败 → 容量天花板, 需改任务或换模型。

样本: "3加4等于几？\n答案是7<eos>", 答案区 = '\n' 之后(含)到 <eos> 前。
"""
import os, sys, random, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from inference.scripts.sample_py import build_model_from_checkpoint, load_tokenizer

EOS = 117


def make_example(tok, a, b, s, nl_id):
    """返回 (x, y, mask): x=ids[:-1], y=ids[1:], mask True=答案区可训。
    文本: "{a}加{b}等于几？\n答案是{s}<eos>", 答案区 = '\n' 之后(含)到 <eos>。"""
    base = tok.encode(f"{a}加{b}等于几？\n答案是{s}").ids
    ids = base + [117]  # <eos>=117 结尾
    # 找 '\n' 位置
    try:
        idx = ids.index(nl_id)
    except ValueError:
        idx = -1
    x = ids[:-1]
    y = ids[1:]
    mask = [False] * len(y)
    for i in range(len(y)):
        if i >= idx and i + 1 < len(ids):
            mask[i] = True   # 预测 ids[i+1], 当其在 '\n' 之后(含 eos 前)
    return x, y, mask
    # 找 '\n' 位置, 答案区 = 其后第一个 token 直到 eos
    try:
        idx = ids.index(nl_id)
    except ValueError:
        idx = -1
    # x 预测 y; x[i] -> y[i] = ids[i+1]
    x = ids[:-1]
    y = ids[1:]
    mask = [False] * len(y)
    for i in range(len(y)):
        # 该位置预测的是 ids[i+1]; 当 i >= idx 时预测的是答案区
        if i >= idx and i + 1 < len(ids):
            mask[i] = True
    return x, y, mask


def build_batch(tok, pairs, dev, nl_id):
    xs, ys, m = [], [], []
    for a, b, s in pairs:
        x, y, mask = make_example(tok, a, b, s, nl_id)
        xs.append(x); ys.append(y); m.append(mask)
    L = max(len(x) for x in xs)
    xt = torch.zeros(len(xs), L, dtype=torch.long, device=dev)
    yt = torch.full((len(xs), L), -100, dtype=torch.long, device=dev)
    for i, (x, y, mask) in enumerate(zip(xs, ys, m)):
        xt[i, :len(x)] = torch.tensor(x, dtype=torch.long)
        for j, ok in enumerate(mask):
            if ok and j < L:
                yt[i, j] = y[j]
    return xt, yt


def greedy_answer(tok, model, a, b, dev, max_n=6):
    text = f"{a}加{b}等于几？\n答案是"
    pids = tok.encode(text).ids
    idx = torch.tensor([pids], dtype=torch.long, device=dev)
    got = ""
    for _ in range(max_n):
        with torch.no_grad():
            lg, _ = model(idx)
        nxt = lg[0, -1, :].argmax().item()
        if nxt == EOS:
            break
        got += tok.decode([nxt])
        idx = torch.cat([idx, torch.tensor([[nxt]], dtype=torch.long, device=dev)], 1)
    return got.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--out", default="out/arith_sft_v2")
    args = ap.parse_args()
    model, ckpt = build_model_from_checkpoint("out/cont_v1_1epoch")
    dev = "cuda"; model.to(dev); model.train()
    tok = load_tokenizer(ckpt)
    nl_id = tok.encode("\n").ids[0]

    # 训练对 (答案用阿拉伯数字; 和<10 便于 +1)
    train_pairs = [(a, b, a+b) for a in range(1, 7) for b in range(1, 7)
                   if a+b < 10 and (a <= 5 and b <= 5)]
    # 泛化对: 从训练集之外 (a 或 b > 5, 但仍和<10)
    test_pairs = [(a, b, a+b) for a in range(1, 9) for b in range(1, 9)
                  if a+b < 10 and (a > 5 or b > 5)]
    print(f"train={len(train_pairs)} test泛化={len(test_pairs)}")
    random.seed(1)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    for step in range(1, args.steps + 1):
        batch = random.sample(train_pairs, args.bs)
        xt, yt = build_batch(tok, batch, dev, nl_id)
        opt.zero_grad()
        lg, loss = model(xt, targets=yt)
        loss.backward(); opt.step()
        if step % 60 == 0 or step == 1:
            model.eval()
            _tr = random.sample(train_pairs, min(15, len(train_pairs)))
            _te = random.sample(test_pairs, min(15, len(test_pairs)))
            tr = sum(1 for a, b, s in _tr if greedy_answer(tok, model, a, b, dev) == str(s))
            te = sum(1 for a, b, s in _te if greedy_answer(tok, model, a, b, dev) == str(s))
            print(f"step {step:4d} loss={loss.item():.4f} | train记忆 {tr}/{len(_tr)} | test泛化 {te}/{len(_te)}", flush=True)
            model.train()

    os.makedirs(args.out, exist_ok=True)
    sd = dict(ckpt); sd['model'] = model.state_dict()
    torch.save(sd, f"{args.out}/best.pt")
    print(f"保存 {args.out}/best.pt")


if __name__ == "__main__":
    main()
