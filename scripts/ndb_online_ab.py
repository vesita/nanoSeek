#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NDB v7 **在线 A/B**：把"离线 Δ=−0.0723 在线能不能兑现"实测出来。

## 为什么必须做这个实验
`scripts/ngram_capacity_probe.py` 量出的 Δ=−0.0723 是**冻结基座上的离线值**。
但 v5/v6 正是死在"基座把记忆吸收掉"（Δ −0.0046 → co-train 后 −0.0002）。
n-gram 路线理论上更安全（写的是**数据统计量**，不经过 `E_p` 过滤），
但如果基座自己学会了这些 n-gram，检索收益同样会被吃掉。
**这个风险只能实测，不能假设。**

## 三臂设计（都从同一个 checkpoint 出发，同预算）

```
off     基座冻结、不挂 NDB                     → 给出基线 CE（应当不变）
frozen  基座冻结、**只训 NDB 门控**             → 回答"模型能不能学会怎么读/写"
joint   基座 + 门控 一起训                      → 回答"会不会被基座吸收"
```

`frozen` 臂是与离线探针最直接可比的一档：探针用的是**手工扫出来的 λ**，
这里用的是**学出来的门控**。如果两者 Δ 接近，说明"模型自己决定"不比人调差。

## 表怎么来（混合方案：底座离线建 + 在线增量）
小预算在线实验里表几乎是空的（300 步 × 1024 token = 30 万次观测 vs 6700 万槽位
→ 覆盖率 0.5%），所以先**离线预热**全量表（与探针同口径：全量语料、L=8、M=268M、K=1），
再在训练中增量写。这正是 v7 最终要用的形态。

## 评测口径（与探针严格一致，可直接比数）
在同一次前向里**同时**算 `CE_off`（纯模型）与 `CE_ndb`（混合后），
报告配对 Δ = CE_ndb − CE_off。固定 256 个 val 窗口（seed 与探针相同）。

用法：
  .venv/bin/python -u scripts/ndb_online_ab.py --steps 300 --prewarm_m 400
"""
import argparse
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("HSA_ENABLE_SDMA", "0")

import numpy as np
import torch
import torch.nn.functional as F

from model.gpt import GPT, GPTConfig
from model.ngram_ndb import NgramNDB
from scripts.ngram_capacity_probe import iter_chunks


def build_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    ma = dict(ck['model_args'])
    ma.pop('eos_token_id', None)
    model = GPT(GPTConfig.from_model_args(ma))
    sd = {(k[10:] if k.startswith('_orig_mod.') else k): v for k, v in ck['model'].items()}
    model.load_state_dict(sd, strict=False)
    return model.to(device).eval(), ma


def val_windows(path, block, n, seed=1234):
    d = np.memmap(path, dtype=np.uint16, mode='r')
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(d) - block - 2, n)
    x = np.stack([np.asarray(d[i:i + block], dtype=np.int64) for i in ix])
    del d
    return torch.from_numpy(x)


def train_batches(path, block, bs, n_steps, seed=7):
    """训练窗口用**独立种子**，且与 val 的 seed 不同（防泄漏）。"""
    d = np.memmap(path, dtype=np.uint16, mode='r')
    rng = np.random.default_rng(seed)
    for _ in range(n_steps):
        ix = rng.integers(0, len(d) - block - 2, bs)
        x = np.stack([np.asarray(d[i:i + block], dtype=np.int64) for i in ix])
        yield torch.from_numpy(x)
    del d


@torch.no_grad()
def evaluate(model, ndb, val_x, bs, device):
    """返回 (CE_off, CE_ndb, mean_gate, stats)。两者在同一次前向里算 → 配对可比。"""
    nll_off = nll_ndb = 0.0
    n_tok = 0
    gate_sum = 0.0
    covered_sum = 0.0
    n_batch = 0
    for i in range(0, val_x.size(0), bs):
        x = val_x[i:i + bs].to(device)
        y = torch.zeros_like(x)
        y[:, :-1] = x[:, 1:]
        y[:, -1] = -100
        cap = {}
        h = model.transformer.ln_f.register_forward_hook(
            lambda m, inp, out: cap.__setitem__('h', out.detach()))
        logits, _ = model(x, y)
        h.remove()
        V = logits.size(-1)
        p_model = F.softmax(logits.float().reshape(-1, V), dim=-1)
        yf = y.reshape(-1)
        valid = yf != -100
        p_new, st = ndb.read(cap['h'], x, p_model.reshape(x.size(0), x.size(1), V),
                             targets=y)
        no = -torch.log(p_model.gather(1, yf.clamp(min=0).unsqueeze(1)).squeeze(1).clamp_min(1e-9))
        nn_ = -torch.log(p_new.reshape(-1, V).gather(1, yf.clamp(min=0).unsqueeze(1)).squeeze(1).clamp_min(1e-9))
        nll_off += float(no[valid].sum()); nll_ndb += float(nn_[valid].sum())
        n_tok += int(valid.sum())
        gate_sum += float(getattr(ndb, '_last_gate', 0.0)); covered_sum += st.covered
        n_batch += 1
        del logits, p_model, p_new
    return (nll_off / max(n_tok, 1), nll_ndb / max(n_tok, 1),
            gate_sum / max(n_batch, 1), covered_sum / max(n_batch, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', default='data/chinese/train_char_v2.bin')
    ap.add_argument('--val', default='data/chinese/val_char_v2.bin')
    ap.add_argument('--ckpt', default='out/ndb_run/last.pt')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--prewarm_m', type=float, default=400.0, help='预热用多少 M token')
    ap.add_argument('--prewarm_chunk_m', type=float, default=20.0)
    ap.add_argument('--L', type=int, default=8)
    ap.add_argument('--slots', type=int, default=268_435_456)
    ap.add_argument('--top_k', type=int, default=1)
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--bs', type=int, default=4)
    ap.add_argument('--block', type=int, default=256)
    ap.add_argument('--val_windows', type=int, default=256)
    ap.add_argument('--gate_lr', type=float, default=1e-3)
    ap.add_argument('--base_lr', type=float, default=1e-4)
    ap.add_argument('--flush_every', type=int, default=50)
    ap.add_argument('--arms', default='off,frozen,joint')
    ap.add_argument('--out_json', default='out/ndb_online/ab.json')
    a = ap.parse_args()

    dev = torch.device(a.device)
    t0 = time.time()

    # ---------- 1) 建表：离线预热（与探针同口径） ----------
    print(f"═══ 1) 离线预热表（{a.prewarm_m:.0f}M token, L={a.L}, M={a.slots/1e6:.0f}M, K={a.top_k}）═══",
          flush=True)
    model, ma = build_model(a.ckpt, dev)
    n_embd = ma['n_embd']
    ndb = NgramNDB(n_embd=n_embd, levels=(a.L,), slots=a.slots, top_k=a.top_k,
                   vocab_size=ma['vocab_size'], max_table_gb=6.0).to(dev)
    print(f"  表内存 {ndb.table_gb():.2f}GB；门控参数 {sum(p.numel() for p in ndb.parameters())} 个", flush=True)
    want = int(a.prewarm_m * 1e6)
    chunk = int(a.prewarm_chunk_m * 1e6)
    seen = 0
    for tok, _off in iter_chunks(a.corpus, chunk, a.L, None, want):
        t = torch.from_numpy(tok).unsqueeze(0)
        ndb.observe_tokens(t, t)
        ndb.flush()
        seen += len(tok)
        if seen % (chunk * 5) < chunk:
            print(f"    预热 {seen/1e6:.0f}M  token  已填槽 "
                  f"{ndb.n_filled_slots()[0]/ndb.slots[0]*100:.1f}%  {time.time()-t0:.0f}s", flush=True)
    print(f"  预热完成：{seen/1e6:.0f}M token，已填槽 "
          f"{ndb.n_filled_slots()[0]/1e6:.1f}M / {ndb.slots[0]/1e6:.0f}M"
          f"（{ndb.n_filled_slots()[0]/ndb.slots[0]*100:.1f}%）", flush=True)

    val_x = val_windows(a.val, a.block, a.val_windows)
    print(f"  val: {tuple(val_x.shape)}（seed 1234，与探针一致）\n", flush=True)

    # ---------- 2) 三臂 ----------
    results = {}
    for arm in a.arms.split(','):
        print(f"═══ 2) 臂 [{arm}] ═══", flush=True)
        # 每臂都从**同一个初始 checkpoint** 重新出发（干净对照）
        model, _ = build_model(a.ckpt, dev)
        for p in model.parameters():
            p.requires_grad = (arm == 'joint')
        if arm == 'joint':
            model.train()
        else:
            model.eval()

        # 门控参数每臂都重置到初始值（否则第二臂继承了第一臂学到的门控）
        torch.manual_seed(0)
        for m in (ndb.write_gate, ndb.read_gate):
            torch.nn.init.zeros_(m.weight)
        torch.nn.init.constant_(ndb.write_gate.bias, 1.0)
        torch.nn.init.constant_(ndb.read_gate.bias, -2.0)
        torch.nn.init.zeros_(ndb.level_weight)
        ndb.zero_grad(set_to_none=True)

        ce_off0, ce_ndb0, g0, cov = evaluate(model, ndb, val_x, 16, dev)
        print(f"  训练前： CE_off={ce_off0:.4f}  CE_ndb={ce_ndb0:.4f}  "
              f"Δ={ce_ndb0-ce_off0:+.4f}  gate={g0:.3f}  覆盖={cov*100:.1f}%", flush=True)

        if arm != 'off':
            params = [p for p in ndb.parameters() if p.requires_grad]
            if arm == 'joint':
                params = list(model.parameters()) + params
            opt = torch.optim.AdamW(params, lr=(a.base_lr if arm == 'joint' else a.gate_lr),
                                    betas=(0.9, 0.95), weight_decay=0.0)
            step_t = time.time()
            for i, x in enumerate(train_batches(a.corpus, a.block, a.bs, a.steps)):
                x = x.to(dev)
                y = torch.zeros_like(x)
                y[:, :-1] = x[:, 1:]
                y[:, -1] = -100
                cap = {}
                hh = model.transformer.ln_f.register_forward_hook(
                    lambda m, inp, out: cap.__setitem__('h', out))
                logits, _ = model(x, y)
                hh.remove()
                V = logits.size(-1)
                p_model = F.softmax(logits.float(), dim=-1)
                p_new, _ = ndb.read(cap['h'], x, p_model, targets=y)
                loss = F.nll_loss(torch.log(p_new.clamp_min(1e-9)).reshape(-1, V),
                                  y.reshape(-1), ignore_index=-100)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                # ★ 写只在训练 micro-batch 里，且用显式上下文（防 val 泄漏）
                with ndb.write_enabled():
                    ndb.observe(cap['h'].detach(), x, y)
                if (i + 1) % a.flush_every == 0:
                    ndb.flush()
                if (i + 1) % 25 == 0 or i == 0:
                    print(f"    step {i+1}/{a.steps}  loss={loss.item():.4f}  "
                          f"gate={ndb._last_gate:.3f}  {time.time()-step_t:.0f}s", flush=True)
            ndb.flush()

        ce_off, ce_ndb, g, cov = evaluate(model, ndb, val_x, 16, dev)
        d_base = ce_off - ce_off0
        d_ndb = ce_ndb - ce_off
        print(f"  训练后： CE_off={ce_off:.4f}（{d_base:+.4f}）  CE_ndb={ce_ndb:.4f}  "
              f"Δ_ndb={d_ndb:+.4f}  gate={g:.3f}", flush=True)
        results[arm] = dict(ce_off_before=ce_off0, ce_ndb_before=ce_ndb0,
                            ce_off=ce_off, ce_ndb=ce_ndb,
                            delta_ndb=d_ndb, delta_base=d_base, gate=g, covered=cov,
                            gate_state=ndb.gate_summary())
        print(flush=True)

    # ---------- 3) 汇总 ----------
    print("═══ 3) 汇总（Δ 是配对差，同一次前向内计算）═══")
    print(f"  {'臂':>8} {'基线CE':>10} {'NDB后CE':>10} {'Δ_ndb':>10} "
          f"{'门控均值':>9} {'覆盖':>7}")
    for arm, r in results.items():
        print(f"  {arm:>8} {r['ce_off']:>10.4f} {r['ce_ndb']:>10.4f} "
              f"{r['delta_ndb']:>+10.4f} {r['gate']:>9.3f} {r['covered']*100:>6.1f}%")
    print(f"\n  离线探针参考（冻结基座 + 手工扫 λ）：Δ=−0.0685 @λ=0.05（M=268M）")
    if 'off' in results and 'frozen' in results:
        gain = results['frozen']['delta_ndb'] - results['off']['delta_ndb']
        print(f"  学出来的门控 vs 纯基线：Δ 改善 {gain:+.4f}")
    if 'joint' in results and 'frozen' in results:
        absorb = results['joint']['delta_ndb'] - results['frozen']['delta_ndb']
        print(f"  联合训练 vs 只训门控：Δ 变化 {absorb:+.4f} "
              f"（正数 = 被基座吸收，这是 v5/v6 的死法）")
    print(f"\n总耗时 {time.time()-t0:.0f}s")

    if a.out_json:
        os.makedirs(os.path.dirname(a.out_json) or '.', exist_ok=True)
        with open(a.out_json, 'w', encoding='utf-8') as f:
            json.dump({'args': vars(a), 'results': results}, f, ensure_ascii=False, indent=2)
        print(f"已写 {a.out_json}")


if __name__ == '__main__':
    main()
