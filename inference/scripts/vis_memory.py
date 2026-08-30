"""记忆网络可视化：真实加载 checkpoint，把"黑板"（状态矩阵 S）画出来。

用法：
    uv run python inference/scripts/vis_memory.py --out_dir out/obs_zh_mem1500 --save_dir out/vis_memory

产出三张图（out/vis_memory/）：
  s_final.png    联想状态 A_final + 持久记忆 P 的热力图（"黑板最后长什么样"）
  gates.png      保留率 r / 写入门 β 沿 token 位置的曲线（"什么时候忘、什么时候写"）
  evolution.png  一个头的状态矩阵快照演化（"黑板被逐渐写满"）
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tokenizers import Tokenizer

from inference.scripts.sample_py import build_model_from_checkpoint

# CJK 字体回退（probe.py 同款）
avail = {f.name for f in fm.fontManager.ttflist}
for f in ("Noto Sans CJK SC", "WenQuanYi Zen Hei", "Noto Sans CJK JP", "Noto Sans CJK TC", "Microsoft YaHei"):
    if f in avail:
        matplotlib.rcParams["font.sans-serif"] = [f, "DejaVu Sans"]
        break
matplotlib.rcParams["axes.unicode_minus"] = False

PROMPT = ("用户：最近工作压力好大，晚上总是睡不着，怎么办？\n"
          "模型：听起来你很累。这件事是最近才冒出来的，还是已经拖了很久了？\n"
          "用户：主要是项目 deadline 压着，心里一直吊着放不下。\n"
          "模型：你先把最想讲的那一块说出来，不用一次讲完。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="out/obs_zh_mem1500")
    ap.add_argument("--save_dir", default="out/vis_memory")
    ap.add_argument("--layer", type=int, default=2, help="画 S 热力图和演化图用的层")
    ap.add_argument("--head", type=int, default=2, help="演化图用的头")
    a = ap.parse_args()

    model, ckpt = build_model_from_checkpoint(a.out_dir)
    tok = Tokenizer.from_file("data/chinese/tokenizer.json")
    ids = torch.tensor([tok.encode(PROMPT).ids])
    texts = [tok.decode([i]) for i in ids[0].tolist()]

    # 打开所有 attention 的观测台
    attns = []
    for block in model.transformer.h:
        if hasattr(block, "attn") and hasattr(block.attn, "mem_qkv"):
            attns.append(block.attn)
            block.attn.capture = True
    with torch.no_grad():
        model(ids)
    T = ids.shape[1]

    save = Path(a.save_dir); save.mkdir(parents=True, exist_ok=True)
    print(f"prompt {T} token → 每层 {len(attns)} 个注意力（含记忆路径）\n")

    # ---- 表 1：每层每头的门控/状态统计 ----
    print(f"{'层':<4}{'头':<4}{'保留率r̄':<9}{'写入门β̄':<9}{'|A|':<9}{'|P|':<9}{'读写比|o|/|x|'}")
    for li, at in enumerate(attns):
        r = at._cap_mem_r[0]        # (T,nh,l)
        w = at._cap_mem_w[0]        # (T,nh,1)
        A = at._cap_mem_A[-1][0]    # 最后快照 (nh,l,l)
        P = at.mem_persist.detach().cpu()
        for h in range(r.shape[1]):
            print(f"{li:<4}{h:<4}{r[:,h].mean():<9.3f}{w[:,h].mean():<9.3f}"
                  f"{A[h].norm():<9.3f}{P[h].norm():<9.3f}{r[:,h].mean()*w[:,h].mean():<9.4f}")

    # ---- 图 1：一层各头的 S = A_final + P 热力图 ----
    at = attns[a.layer]
    A = at._cap_mem_A[-1][0]        # (nh,l,l)
    P = at.mem_persist.detach().cpu()
    S = A + P
    nh = S.shape[0]; l = S.shape[1]
    vmax = S.abs().max().item() + 1e-9
    fig, axes = plt.subplots(2, nh, figsize=(3.2 * nh, 6.4))
    for h in range(nh):
        axes[0, h].imshow(S[h].numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[0, h].set_title(f"头{h}  S = 联想+持久", fontsize=11)
        axes[1, h].imshow(P[h].numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[1, h].set_title(f"头{h}  仅持久记忆 P", fontsize=11)
        for ax in (axes[0, h], axes[1, h]):
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"第{a.layer}层记忆黑板  S∈R^{{{l}×{l}}}（latent 维，深色=大值）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save / "s_final.png", dpi=130)
    plt.close(fig)

    # ---- 图 2：门控沿 token 位置 ----
    r = at._cap_mem_r[0]   # (T,nh,l)
    w = at._cap_mem_w[0]   # (T,nh,1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    step = max(1, T // 18)
    for h in range(nh):
        ax1.plot(r[:, h].mean(-1).numpy(), lw=0.8, alpha=0.6, label=f"头{h}")
        ax2.plot(w[:, h, 0].numpy(), lw=0.8, alpha=0.6, label=f"头{h}")
    ax1.plot(r.mean((1, 2)).numpy(), "k-", lw=2, label="平均")
    ax2.plot(w.mean((1, 2)).numpy(), "k-", lw=2, label="平均")
    ax1.axhline(0.5, ls="--", c="gray", lw=0.8); ax2.axhline(0.5, ls="--", c="gray", lw=0.8)
    ax1.set_ylabel("保留率 r（1−遗忘门）"); ax1.legend(fontsize=7, ncol=5)
    ax2.set_ylabel("写入门 β"); ax2.legend(fontsize=7, ncol=5)
    ax2.set_xticks(range(0, T, step))
    ax2.set_xticklabels(["".join(t for t in texts[i:i+1]) if texts[i] else "·" for i in range(0, T, step)],
                        rotation=90, fontsize=8)
    ax1.grid(alpha=.3); ax2.grid(alpha=.3)
    fig.suptitle(f"第{a.layer}层记忆门控：r 高=记得久，β 高=此刻写入多", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save / "gates.png", dpi=130)
    plt.close(fig)

    # ---- 图 3：一个头的状态演化（每个 chunk 后的快照）----
    hist = at._cap_mem_A                       # list of (B,nh,l,l)
    n = len(hist)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.6))
    vmax = max(hh[0, a.head].abs().max().item() for hh in hist) + 1e-9
    for k, hh in enumerate(hist):
        axes[k].imshow(hh[0, a.head].numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[k].set_title(f"chunk{k+1} 后（t≈{min((k+1)*64, T)}）", fontsize=10)
        axes[k].set_xticks([]); axes[k].set_yticks([])
    fig.suptitle(f"第{a.layer}层·头{a.head}：联想黑板被逐渐写满（16×16）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save / "evolution.png", dpi=130)
    plt.close(fig)

    print(f"\n图已保存到 {save}/")


if __name__ == "__main__":
    main()
