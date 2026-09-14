#!/usr/bin/env python3
"""模型探针：加载 checkpoint，跑前向 + 反向，报告每层激活/梯度统计。

探索新网络结构时的两大杀手：
  1. 数值崩溃 —— 某个模块输出 nan/inf（本工具逐层定位）；
  2. 死层/退化 —— 某层梯度为 0、激活退化成常量（本工具逐个参数检查）。

用法（从项目根目录）：

    uv run python cli.py probe --out_dir=out/test [--device=cuda] [--num_batches=4]
    # 观测台模式（KV/注意力专项统计，输出 stats.json + PNG 图表）：
    uv run python cli.py probe --out_dir=out/obs_kv --stats --num_batches=8 [--stats_out=out/obs_kv/stats]

输出：终端表格（激活统计 / 梯度统计 / 异常汇总），全正常时无异常行；
--stats 时输出 KV 有效秩、范数、距离衰减曲线、头熵、sink、头间相似度。
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn as nn
from model import GPTConfig, GPT
from model.attention import CausalSelfAttention


def parse_args(argv):
    out_dir, device, batch_size, block_size, num_batches = \
        "out/chinese-data2", None, 8, 256, 4
    stats, stats_out = False, None
    passthrough = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--out_dir="):
            out_dir = a.split("=", 1)[1]
        elif a.startswith("--device="):
            device = a.split("=", 1)[1]
        elif a.startswith("--batch_size="):
            batch_size = int(a.split("=", 1)[1])
        elif a.startswith("--block_size="):
            block_size = int(a.split("=", 1)[1])
        elif a.startswith("--num_batches="):
            num_batches = int(a.split("=", 1)[1])
        elif a == "--stats":
            stats = True
        elif a.startswith("--stats_out="):
            stats_out = a.split("=", 1)[1]
        else:
            passthrough.append(a)
        i += 1
    if passthrough:
        print("忽略未知参数: %r" % passthrough)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return out_dir, device, batch_size, block_size, num_batches, stats, stats_out


def load_model(out_dir, device):
    ckpt_path = os.path.join(ROOT, out_dir, "best.pt")
    if not os.path.exists(ckpt_path):
        sys.exit(f"错误：找不到 {ckpt_path}（先用 cli.py train 训练，或用 --out_dir 指定实验目录）")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = GPT(GPTConfig.from_model_args(ckpt["model_args"]))
    state = ckpt["model"]
    unwanted = "_orig_mod."
    for k in list(state):
        if k.startswith(unwanted):
            state[k[len(unwanted):]] = state.pop(k)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, ckpt


def make_batch(ckpt, device, batch_size, block_size):
    """从数据集取一批真实 token；取不到就用随机 id（探针只关心数值统计）。"""
    cfg = ckpt.get("config", {})
    dataset = cfg.get("dataset")
    path = os.path.join(ROOT, "data", dataset, "train.bin") if dataset else None
    if path and os.path.exists(path):
        import numpy as np
        data = np.memmap(path, dtype=np.uint16, mode="r")
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64))
                         for i in ix])
        return x.to(device), y.to(device)
    vocab = ckpt["model_args"].get("vocab_size", 50304)
    return (torch.randint(0, vocab, (batch_size, block_size), device=device),
            torch.randint(0, vocab, (batch_size, block_size), device=device))


def tensor_stats(t):
    t = t.detach().float()
    return dict(mean=t.mean().item(), std=t.std().item(),
                mx=t.max().item(), mn=t.min().item(),
                nan=bool(t.isnan().any()), inf=bool(t.isinf().any()))


def main(argv):
    out_dir, device, batch_size, block_size, num_batches, stats, stats_out = parse_args(argv)
    model, ckpt = load_model(out_dir, device)
    if stats:
        if stats_out is None:
            stats_out = os.path.join(out_dir, "stats")
        else:
            stats_out = os.path.join(ROOT, stats_out)
        run_stats(model, ckpt, device, batch_size, block_size, num_batches, stats_out)
        return
    print(f"探针: {out_dir}/best.pt · {model.get_num_params()/1e6:.2f}M 参数 · "
          f"vocab {ckpt['model_args'].get('vocab_size')} · device {device} · "
          f"batch {batch_size}×{block_size}")

    # ---- 前向 hook：记录每个带参数模块的输出统计 ----
    act = {}
    hooks = []
    for name, m in model.named_modules():
        if name == "" or not any(p.requires_grad for p in m.parameters(recurse=False)):
            continue
        def _hook(mod, inp, out, _name=name):
            t = out[0] if isinstance(out, tuple) else out
            if isinstance(t, torch.Tensor):
                act.setdefault(_name, []).append(tensor_stats(t))
        hooks.append(m.register_forward_hook(_hook))

    x, y = make_batch(ckpt, device, batch_size, block_size)
    with torch.no_grad():
        for _ in range(num_batches):
            model(x, y)  # 前向：填满激活统计
    for h in hooks:
        h.remove()

    # ---- 反向：梯度统计 ----
    grads = {}
    model.zero_grad(set_to_none=True)
    loss = model(x, y)[1]
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is None:
            grads[name] = dict(norm=0.0, nan=False, inf=False, pnorm=p.detach().float().norm().item())
            continue
        g = p.grad.detach().float()
        grads[name] = dict(norm=g.norm().item(), nan=bool(g.isnan().any()),
                           inf=bool(g.isinf().any()), pnorm=p.detach().float().norm().item())

    # ---- 报告 ----
    print("\n========== 激活统计（mean/std/min/max，nan/inf 标记）==========")
    bad_act = 0
    for name, lst in act.items():
        s = lst[-1]
        flag = ""
        if s["nan"] or s["inf"]:
            flag = "  <<< NaN/Inf!"; bad_act += 1
        print(f"  {name:<42} {s['mean']:+.3f} {s['std']:.3f} {s['mn']:+.3f} {s['mx']:+.3f}{flag}")

    print("\n========== 梯度统计（L2 范数，nan/inf/零梯度标记）==========")
    ranked = sorted(grads.items(), key=lambda kv: kv[1]["norm"], reverse=True)
    bad_grad = 0
    for name, s in ranked[:25]:
        flag = ""
        if s["nan"] or s["inf"]:
            flag = "  <<< NaN/Inf!"; bad_grad += 1
        elif s["norm"] == 0.0:
            flag = "  (零梯度)"; bad_grad += 1
        print(f"  {name:<42} {s['norm']:>12.4e}  pnorm={s['pnorm']:.4e}{flag}")
    if len(ranked) > 25:
        print(f"  … 共 {len(ranked)} 个参数张量，仅显示梯度范数最大的 25 个")

    n_nan_act = sum(1 for lst in act.values() if lst[-1]["nan"] or lst[-1]["inf"])
    n_nan_grad = sum(1 for s in grads.values() if s["nan"] or s["inf"])
    n_zero_grad = sum(1 for s in grads.values() if s["norm"] == 0.0)
    print("\n========== 汇总 ==========")
    print(f"  激活 nan/inf 模块数: {n_nan_act}")
    print(f"  梯度 nan/inf 张量数: {n_nan_grad}")
    print(f"  零梯度张量数:       {n_zero_grad}（>0 说明存在死参数；embedding 未用行属正常）")
    if n_nan_act or n_nan_grad:
        print("  ⚠ 存在数值问题：先用 float32 + 小 lr 复现，再看上面定位的模块/参数。")
    else:
        print("  ✓ 数值正常，无 NaN/Inf。")


# =====================================================================
# 观测台模式（probe --stats）：KV/注意力专项统计
# 统计内容：KV 范数分布、KV 有效秩（参与比）、注意力距离衰减曲线、
#           头熵（坍缩检测）、sink 使用、头间相似度、CSA 三路路径贡献。
# =====================================================================

def _eff_rank(t):
    """参与比有效秩：(Σλ)²/Σλ²（λ = 协方差特征值）。t: (B,T,nh,d) → (B,nh)。"""
    B, T, nh, d = t.shape
    tc = t - t.mean(dim=1, keepdim=True)
    cov = torch.einsum('bthd,bthe->bhde', tc, tc) / T          # (B,nh,d,d)
    tr = torch.diagonal(cov, dim1=-2, dim2=-1).sum(-1)          # (B,nh)
    tr2 = (cov * cov).sum(dim=(-1, -2))                         # (B,nh)
    return tr * tr / tr2.clamp_min(1e-12)


def _dist_curve(att):
    """距离衰减：att (B,nh,T,T) → 每个相对距离 δ 上的平均权重 (T,)。"""
    B, nh, T, _ = att.shape
    idx = torch.arange(T, device=att.device)
    curve = torch.zeros(T, device=att.device)
    for delta in range(T):
        w = att[..., idx[delta:], idx[:T - delta]] if delta else att[..., idx, idx]
        curve[delta] = w.mean()
    return curve


def run_stats(model, ckpt, device, batch_size, block_size, num_batches, stats_out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # 图表里的中文需要 CJK 字体，否则标题渲染成方块
    try:
        import matplotlib.font_manager as fm
        avail = {f.name for f in fm.fontManager.ttflist}
        for f in ("Noto Sans CJK SC", "WenQuanYi Zen Hei", "Noto Sans CJK JP",
                  "Noto Sans CJK TC", "Microsoft YaHei"):
            if f in avail:
                matplotlib.rcParams["font.sans-serif"] = [f, "DejaVu Sans"]
                matplotlib.rcParams["axes.unicode_minus"] = False
                break
    except Exception:
        pass

    cfg = model.config
    nh, d = cfg.n_head, cfg.n_embd // cfg.n_head
    m_comp, win = cfg.csa_compress, cfg.csa_window
    use_sink = bool(cfg.use_attn_sink)

    # 只捕获主干注意力层（transformer.h.<i>.attn），不碰 MTP 模块
    attns = []
    for name, m in model.named_modules():
        if isinstance(m, CausalSelfAttention) and name.startswith("transformer.h."):
            m.capture = True
            attns.append((name, m))
    attns.sort(key=lambda kv: int(kv[0].split(".")[2]))
    if not attns:
        print("⚠ 未找到任何主干注意力层，跳过观测台。")
        return
    print(f"\n观测台: {len(attns)} 个注意力层 · CSA={cfg.use_csa} · head={nh} d={d} · "
          f"sink={use_sink} · 批次 {num_batches}×{batch_size}×{block_size}")

    agg = {name: dict(kn=[], vn=[], kr=[], vr=[], sink=[], ent=[], dc=[],
                      wc=[], blk=[], blk_beyond=[], far=[], sim=[], mag=[])
           for name, _ in attns}

    with torch.no_grad():
        for _ in range(num_batches):
            x, _ = make_batch(ckpt, device, batch_size, block_size)
            model(x)
            for name, m in attns:
                a = agg[name]
                k = m._cap_k.float(); v = m._cap_v.float()
                B, T, nh_, dd = k.shape
                a["kn"].append(k.norm(dim=-1))                  # (B,T,nh)
                a["vn"].append(v.norm(dim=-1))
                a["kr"].append(_eff_rank(k))                    # (B,nh)
                a["vr"].append(_eff_rank(v))
                if hasattr(m, "_cap_att"):                      # 标准注意力路径
                    att = m._cap_att.float()                    # (B,nh,T,T')
                    if use_sink:
                        a["sink"].append(att[..., -1].mean(dim=2))       # (B,nh)
                    p = att[..., :T].clamp_min(1e-12)
                    a["ent"].append(-(p * p.log()).sum(-1).mean(dim=2))  # (B,nh)
                    a["dc"].append(_dist_curve(att[..., :T]))           # (T,)
                if hasattr(m, "_cap_win"):                      # CSA 滑窗路径
                    ww = m._cap_win.float().permute(0, 2, 1, 3) # → (B,nh,T,T+1)
                    if use_sink:
                        a["sink"].append(ww[..., -1].mean(dim=2))       # (B,nh)
                    a["wc"].append(_dist_curve(ww[..., :T]))            # (T,)
                if hasattr(m, "_cap_blk"):                      # CSA 块路径
                    blk = m._cap_blk.float(); nb = m._cap_blk_nb
                    real = blk[..., :nb]                        # (B,T,nh,nb)
                    a["blk"].append(real.sum(-1).mean(dim=(1, 2)))      # (B,)
                    if use_sink:
                        a["sink"].append(blk[..., -1].mean(dim=1))      # (B,nh)
                    # 最远选中块 → 距 query 的 token 距离
                    has = real > 0
                    idx = torch.arange(nb, device=real.device).float()
                    far = torch.where(
                        has, idx.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                        torch.full_like(has.float(), float('inf')))
                    a["far"].append(far.min(dim=-1).values)             # (B,T,nh)
                    # 超窗占比：块中点距离 > win 的质量比例
                    t = torch.arange(T, device=real.device).float()
                    blk_dist = t.unsqueeze(-1) - (idx * m_comp + m_comp / 2)  # (T,nb)
                    beyond = (blk_dist > win).unsqueeze(0).unsqueeze(2)
                    frac = (real * beyond.float()).sum(-1) / real.sum(-1).clamp_min(1e-9)
                    a["blk_beyond"].append(frac.mean(dim=(1, 2)))       # (B,)
                if hasattr(m, "_cap_y_heads"):                  # 头间相似度
                    yh = m._cap_y_heads.float().flatten(0, 1)           # (BT,nh,d)
                    yu = yh / yh.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                    a["sim"].append(torch.bmm(yu.transpose(1, 2), yu).mean(dim=0))
                if hasattr(m, "_cap_mag"):                      # CSA 路径贡献
                    a["mag"].append(m._cap_mag)

    # ---- 聚合 + 控制台报告 ----
    print("\n========== KV/注意力统计（跨批次聚合）==========")
    res = {}
    for name, m in attns:
        a = agg[name]
        kn = torch.cat(a["kn"], 0)      # (BN,T,nh)
        vn = torch.cat(a["vn"], 0)
        kr = torch.cat(a["kr"], 0).float()
        vr = torch.cat(a["vr"], 0).float()

        def q(t, p): return torch.quantile(t, p).item()
        L = dict(
            k_norm=dict(mean=kn.mean().item(), std=kn.std().item(),
                        p05=q(kn, .05), p95=q(kn, .95)),
            v_norm=dict(mean=vn.mean().item(), std=vn.std().item(),
                        p05=q(vn, .05), p95=q(vn, .95)),
            k_effrank=dict(mean=kr.mean().item(), std=kr.std().item()),
            v_effrank=dict(mean=vr.mean().item(), std=vr.std().item()),
            k_norm_pos=kn.mean(dim=(0, 2)).tolist(),
            v_norm_pos=vn.mean(dim=(0, 2)).tolist(),
        )
        if a["ent"]:
            L["entropy_per_head"] = torch.cat(a["ent"], 0).float().mean(0).tolist()
        if a["dc"]:
            L["dist_curve"] = torch.stack(a["dc"]).mean(0).tolist()
        if a["wc"]:
            L["win_curve"] = torch.stack(a["wc"]).mean(0).tolist()
        if a["blk"]:
            blk = torch.cat(a["blk"], 0).float()
            beyond = torch.cat(a["blk_beyond"], 0).float()
            far = torch.cat(a["far"], 0).float()                # (BN,T,nh)
            t = torch.arange(far.shape[1], device=far.device).float()
            dist = t.unsqueeze(0).unsqueeze(-1) - (far * m_comp + m_comp / 2)
            valid = torch.isfinite(dist)
            dv = dist[valid]
            L["blk_mass"] = blk.mean().item()
            L["blk_beyond_frac"] = beyond.mean().item()
            L["blk_reach"] = dict(mean=dv.mean().item(),
                                  p50=q(dv, .5), p95=q(dv, .95),
                                  frac_valid=valid.float().mean().item())
        if a["sim"]:
            L["head_sim"] = torch.stack(a["sim"]).mean(0).tolist()
        if a["mag"]:
            keys = set().union(*[d.keys() for d in a["mag"]])
            L["path_mag"] = {kk: sum(d[kk] for d in a["mag"]) / len(a["mag"]) for kk in keys}
        if a["sink"]:
            L["sink_per_head"] = torch.cat(a["sink"], 0).float().mean(0).tolist()
        res[name] = L

        f3 = lambda x: f"{x:.3f}"
        f1 = lambda x: f"{x:.1f}"
        row = (f"L{name.split('.')[2]:>2} |K {f3(L['k_norm']['mean'])}±{f3(L['k_norm']['std'])} "
               f"|V {f3(L['v_norm']['mean'])}±{f3(L['v_norm']['std'])} "
               f"|K秩 {f1(L['k_effrank']['mean'])} |V秩 {f1(L['v_effrank']['mean'])}")
        if "entropy_per_head" in L:
            row += " |熵 " + ",".join(f1(e) for e in L["entropy_per_head"])
        if "sink_per_head" in L:
            row += " |sink " + ",".join(f3(s) for s in L["sink_per_head"])
        if "win_curve" in L:
            wc = L["win_curve"]
            row += " |窗δ1/8/32/63 " + "/".join(f3(wc[i]) for i in (1, 8, 32, 63) if i < len(wc))
        if "blk_reach" in L:
            r = L["blk_reach"]
            row += (f" |块质量 {f3(L['blk_mass'])} 超窗 {f3(L['blk_beyond_frac'])}"
                    f" 最远p50 {f1(r['p50'])} p95 {f1(r['p95'])}")
        if "path_mag" in L:
            pm = L["path_mag"]
            row += f" |路径 c/w/g/m " + "/".join(f3(pm.get(kk, 0.0)) for kk in ("comp", "win", "glob", "mem"))
        print("  " + row)

    # ---- 输出 ----
    os.makedirs(stats_out, exist_ok=True)
    with open(os.path.join(stats_out, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)

    n = len(attns)
    labels = [f"L{i}" for i in range(n)]

    # 1) KV 范数随位置
    fig, axes = plt.subplots((n + 2) // 3, 3, figsize=(12, 2.6 * ((n + 2) // 3)))
    axes = np.array(axes).flatten()
    for i, (name, _) in enumerate(attns):
        L = res[name]
        axes[i].plot(L["k_norm_pos"], color="#d62728", label="K")
        axes[i].plot(L["v_norm_pos"], color="#1f77b4", label="V")
        axes[i].set_title(f"{labels[i]} ||k||/||v|| vs 位置"); axes[i].legend(fontsize=7)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(stats_out, "kv_norm_pos.png"), dpi=110); plt.close(fig)

    # 2) KV 有效秩
    fig, ax = plt.subplots(figsize=(9, 3.2))
    xs = np.arange(n); wd = 0.35
    ax.bar(xs - wd/2, [res[nm]["k_effrank"]["mean"] for nm, _ in attns], wd,
           yerr=[res[nm]["k_effrank"]["std"] for nm, _ in attns], label="K", color="#d62728", capsize=2)
    ax.bar(xs + wd/2, [res[nm]["v_effrank"]["mean"] for nm, _ in attns], wd,
           yerr=[res[nm]["v_effrank"]["std"] for nm, _ in attns], label="V", color="#1f77b4", capsize=2)
    ax.axhline(d, color="gray", ls="--", lw=0.8)
    ax.set_xticks(list(xs)); ax.set_xticklabels(labels)
    ax.set_ylim(0, d * 1.15); ax.set_ylabel("有效秩"); ax.legend()
    ax.set_title("KV 有效秩（参与比；虚线=满秩 d）")
    fig.tight_layout(); fig.savefig(os.path.join(stats_out, "kv_effrank.png"), dpi=110); plt.close(fig)

    # 3) 距离衰减（对数 y）
    fig, ax = plt.subplots(figsize=(8, 3.8))
    cmap = plt.get_cmap("viridis")
    for i, (name, _) in enumerate(attns):
        L = res[name]; c = cmap(i / max(1, n - 1))
        if "win_curve" in L:
            ax.plot(range(len(L["win_curve"])), L["win_curve"], label=f"{labels[i]} 滑窗", color=c)
        if "dist_curve" in L:
            ax.plot(range(len(L["dist_curve"])), L["dist_curve"], label=f"{labels[i]} 全距", color=c, ls="--")
    ax.set_xlabel("相对距离 δ"); ax.set_ylabel("平均注意力权重"); ax.set_yscale("log")
    ax.legend(fontsize=7); ax.grid(alpha=.3); ax.set_title("注意力距离衰减曲线（对数 y）")
    fig.tight_layout(); fig.savefig(os.path.join(stats_out, "dist_decay.png"), dpi=110); plt.close(fig)

    # 4) 头熵（仅标准路径有）
    if any("entropy_per_head" in res[nm] for nm, _ in attns):
        ent = np.array([res[nm]["entropy_per_head"] for nm, _ in attns])
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        im = ax.imshow(ent, cmap="viridis", aspect="auto")
        ax.set_xticks(range(nh)); ax.set_xticklabels([f"H{h}" for h in range(nh)])
        ax.set_yticks(range(n)); ax.set_yticklabels(labels)
        ax.set_title("注意力熵（低=头坍缩）"); fig.colorbar(im)
        fig.tight_layout(); fig.savefig(os.path.join(stats_out, "attn_entropy.png"), dpi=110); plt.close(fig)

    # 5) 头间相似度
    cols = max(1, (n + 1) // 2)
    fig, axes = plt.subplots(2, cols, figsize=(3.4 * cols, 6.6))
    axes = np.array(axes).flatten()
    for i, (name, _) in enumerate(attns):
        sim = np.array(res[name]["head_sim"])
        im = axes[i].imshow(sim, cmap="coolwarm", vmin=-1, vmax=1)
        axes[i].set_title(f"{labels[i]} 头间余弦相似度")
        axes[i].set_xticks(range(nh)); axes[i].set_yticks(range(nh))
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(stats_out, "head_sim.png"), dpi=110); plt.close(fig)

    # 6) CSA 三路路径贡献
    if any("path_mag" in res[nm] for nm, _ in attns):
        pm = [res[nm]["path_mag"] for nm, _ in attns]
        keys = sorted(set().union(*[p.keys() for p in pm]))
        colors = {"comp": "#d62728", "win": "#1f77b4", "glob": "#2ca02c", "mem": "#ff7f0e"}
        labels_p = {"comp": "块压缩路径", "win": "滑窗路径", "glob": "HCA 全局", "mem": "KV记忆"}
        fig, ax = plt.subplots(figsize=(8, 3.2))
        bottom = [0.0] * n
        for kk in ("comp", "win", "glob", "mem"):
            if kk not in keys:
                continue
            vals = [p.get(kk, 0.0) for p in pm]
            ax.bar(xs, vals, bottom=bottom, label=labels_p[kk], color=colors[kk])
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_xticks(list(xs)); ax.set_xticklabels(labels)
        ax.set_ylabel("平均 token 输出范数"); ax.legend(fontsize=8)
        ax.set_title("CSA 路径贡献（comp 块 / win 滑窗 / glob HCA / mem 记忆）")
        fig.tight_layout(); fig.savefig(os.path.join(stats_out, "csa_paths.png"), dpi=110); plt.close(fig)

    print(f"\n✓ 统计与图表已写入 {stats_out}/ （stats.json + PNG）")


if __name__ == "__main__":
    main(sys.argv[1:])
