#!/usr/bin/env python3
"""
模型归档/索引工具：把 out/ 下散落的实验目录变成可查询的模型档案。

作用：
  1. 扫描 out/ 下所有含 best.pt / results.csv 的实验目录；
  2. 读取 results.csv 提取训练曲线信息（step / best val / 最终 val）；
  3. 读取 checkpoint 的 model_args / config 提取架构与训练配置；
  4. 给每个实验生成小体积 manifest.json（避免以后每次扫描都加载 38MB checkpoint）；
  5. 汇总生成 out/index.json，并按 val loss 排序，方便横向对比实验臂。

用法（从项目根目录）：
    uv run python inference/scripts/archive.py                 # 查看现有索引/扫描摘要
    uv run python inference/scripts/archive.py --write         # 生成/更新 manifest.json + out/index.json
    uv run python inference/scripts/archive.py --dir out/chinese-data2-gate
    uv run python inference/scripts/archive.py --json          # 输出 JSON
"""
import argparse
import csv
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_ROOT = ROOT / "out"
INDEX_PATH = OUT_ROOT / "index.json"

# 核心架构开关，展示时优先展示这些
ARCH_KEYS = [
    "n_layer", "n_head", "n_embd", "block_size", "vocab_size",
    "use_rope", "use_moe", "n_experts", "n_top_k", "use_shared_expert",
    "use_aux_free_balance", "use_sqrtsoftplus",
    "use_mla", "use_csa", "use_hca", "csa_compress", "csa_topk", "csa_window",
    "use_mtp", "use_muon", "use_attn_sink", "use_mhc", "hc_mult",
    "use_lightning_indexer", "num_hash_layers", "block_order",
    "no_attn_layers", "n_memory_tokens",
    "use_lse_residual", "use_lse_gate", "swiglu_clamp", "rope_theta",
]

# 训练/实验信息，展示时优先从 config 取
META_KEYS = [
    "dataset", "out_dir", "max_iters", "eval_interval", "eval_iters",
    "batch_size", "block_size", "learning_rate", "min_lr", "warmup_iters",
    "gradient_accumulation_steps", "dropout", "wandb_run_name",
    "enable_early_stop", "patience", "min_val_improve",
]


def read_results_csv(path: Path) -> list[dict]:
    """读 results.csv，返回数值化的行列表；文件不存在返回 []。"""
    if not path.exists():
        return []
    rows = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(row for row in f if row.strip())
            for row in reader:
                rec = {}
                for k, v in row.items():
                    if k is None:
                        continue
                    try:
                        rec[k] = float(v)
                    except (TypeError, ValueError):
                        rec[k] = v
                rows.append(rec)
    except Exception as e:
        print(f"⚠ 解析 {path} 失败：{e}")
    return rows


def summarize_results(rows: list[dict]) -> dict:
    if not rows:
        return {"has_results": False}
    steps = [r.get("step") for r in rows if r.get("step") is not None]
    val_losses = [r.get("val/loss") for r in rows if r.get("val/loss") is not None]
    best_idx = min(range(len(val_losses)), key=val_losses.__getitem__) if val_losses else None
    return {
        "has_results": True,
        "rows": len(rows),
        "first_step": int(steps[0]) if steps else None,
        "last_step": int(steps[-1]) if steps else None,
        "best_val_loss": float(val_losses[best_idx]) if best_idx is not None else None,
        "best_val_step": int(steps[best_idx]) if best_idx is not None and steps else None,
        "last_val_loss": float(val_losses[-1]) if val_losses else None,
    }


def _plain(obj):
    """把 checkpoint 里常见的 tensor/numpy 转成 JSON 友好标量。"""
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def load_checkpoint_meta(ckpt_path: Path) -> dict:
    """只提取 checkpoint 里的轻量元数据（config/model_args），不保留大张量。"""
    import torch
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_args = _plain(ck.get("model_args", {}))
    config = _plain(ck.get("config", {}))
    meta = {
        "model_args": model_args,
        "config": config,
        "iter_num": _plain(ck.get("iter_num")),
        "best_val_loss_ckpt": _plain(ck.get("best_val_loss")),
        "epoch": _plain(ck.get("epoch")),
    }
    del ck
    return meta


def _git_commit() -> str | None:
    """返回当前仓库短 commit；不在 git 仓库里时返回 None。"""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip() or None
    except Exception:
        return None


def experiment_manifest(exp_dir: Path, force: bool = False) -> dict:
    manifest_path = exp_dir / "manifest.json"
    if manifest_path.exists() and not force:
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"⚠ 读取 {manifest_path} 失败，重新生成：{e}")

    name = exp_dir.name
    rows = read_results_csv(exp_dir / "results.csv")
    summary = summarize_results(rows)

    manifest = {
        "name": name,
        "path": str(exp_dir.relative_to(ROOT)),
        "updated_at": None,
        "git_commit": _git_commit(),
        "has_best": (exp_dir / "best.pt").exists(),
        "best_pt_size_mb": round((exp_dir / "best.pt").stat().st_size / 1024 / 1024, 1)
                           if (exp_dir / "best.pt").exists() else None,
        "results": summary,
        "model_args": None,
        "config": None,
    }

    ckpt_path = exp_dir / "best.pt"
    if ckpt_path.exists():
        try:
            manifest["checkpoint_mtime"] = datetime.datetime.fromtimestamp(
                ckpt_path.stat().st_mtime
            ).astimezone().isoformat(timespec="seconds")
            meta = load_checkpoint_meta(ckpt_path)
            manifest["model_args"] = meta["model_args"]
            manifest["config"] = meta["config"]
            manifest["iter_num"] = meta["iter_num"]
            manifest["best_val_loss_ckpt"] = meta["best_val_loss_ckpt"]
            manifest["epoch"] = meta["epoch"]
        except Exception as e:
            manifest["load_error"] = str(e)
            print(f"⚠ 加载 {ckpt_path} 失败：{e}")

    manifest["updated_at"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def iter_experiments(out_root: Path):
    if not out_root.exists():
        return
    for child in sorted(out_root.iterdir()):
        if not child.is_dir():
            continue
        # 只归档“看起来是实验目录”的路径：有 best.pt 或 results.csv
        if (child / "best.pt").exists() or (child / "results.csv").exists():
            yield child


def build_archive(out_root: Path, write: bool, only_dir: str | None, force: bool) -> list[dict]:
    records = []
    for exp_dir in iter_experiments(out_root):
        if only_dir and str(exp_dir.relative_to(ROOT)) not in (only_dir, only_dir.strip("./")):
            continue
        if write:
            rec = experiment_manifest(exp_dir, force=force)
        else:
            rec = experiment_manifest(exp_dir, force=False)  # 有 manifest 就复用
        # 从 results 补一个总览用的 best_val
        rec["_best_val"] = rec.get("results", {}).get("best_val_loss") or rec.get("best_val_loss_ckpt")
        records.append(rec)

    # 方便人工查看：按 val loss 排序，None 放最后
    records.sort(key=lambda r: (r.get("_best_val") is None, r.get("_best_val") or float("inf")))
    return records


def display_summary(records: list[dict]) -> None:
    print(f"共 {len(records)} 个实验目录：\n")
    header = f"{'实验':<28}{'best_val':>10}{'steps':>8}{'规模':>14}{'架构亮点':<40}"
    print(header)
    print("-" * len(header))
    for r in records:
        ma = r.get("model_args") or {}
        size = f"{ma.get('n_layer')}×{ma.get('n_embd')}@{ma.get('vocab_size')}" if ma else "-"
        tags = []
        if ma.get("use_moe"): tags.append("MoE")
        if ma.get("use_shared_expert"): tags.append("Shared")
        if ma.get("use_csa"): tags.append("CSA")
        if ma.get("use_hca"): tags.append("HCA")
        if ma.get("use_mla"): tags.append("MLA")
        if ma.get("use_mtp"): tags.append("MTP")
        if ma.get("use_mhc"): tags.append("mHC")
        if ma.get("use_attn_sink"): tags.append("Sinks")
        if ma.get("use_lse_gate"): tags.append("LSE-gate")
        if ma.get("use_lse_residual"): tags.append("LSE")
        if ma.get("n_memory_tokens"): tags.append(f"Mem{ma['n_memory_tokens']}")
        if ma.get("no_attn_layers"): tags.append(f"noAttn{ma['no_attn_layers']}")
        if ma.get("block_order") == "ffn_attn": tags.append("ffn_attn")
        best = r.get("_best_val")
        best_s = f"{best:.4f}" if best is not None else "-"
        steps = r.get("results", {}).get("last_step") if r.get("results") else None
        steps_s = f"{steps}" if steps is not None else "-"
        print(f"{r['name']:<28}{best_s:>10}{steps_s:>8}{size:>14}{' '.join(tags):<40}")
    if records:
        print("\n提示：用 --dir out/<实验名> 看单个实验详情；--json 输出机器可读 JSON。")
    else:
        print("（没有发现实验目录）")


def display_detail(record: dict) -> None:
    print(json.dumps(record, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser(description="模型归档/索引工具")
    ap.add_argument("--dir", default=None, help="只看某个实验目录，如 out/chinese-data2-gate")
    ap.add_argument("--write", action="store_true", help="生成/更新每个实验的 manifest.json 和 out/index.json")
    ap.add_argument("--force", action="store_true", help="--write 时即使已有 manifest 也重新从 checkpoint 提取")
    ap.add_argument("--json", action="store_true", help="输出 JSON 而不是表格")
    ap.add_argument("--out-root", default=str(OUT_ROOT), help="实验根目录（默认 out/）")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    if not out_root.exists():
        sys.exit(f"错误：实验目录不存在 {out_root}")

    records = build_archive(out_root, write=args.write, only_dir=args.dir, force=args.force)

    if args.write:
        INDEX_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已更新索引：{INDEX_PATH}")

    if args.dir:
        if not records:
            sys.exit(f"没有找到实验目录：{args.dir}")
        display_detail(records[0])
    elif args.json:
        print(json.dumps(records, ensure_ascii=False, indent=2))
    else:
        display_summary(records)


if __name__ == "__main__":
    main()
