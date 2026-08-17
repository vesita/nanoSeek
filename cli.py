#!/usr/bin/env python3
"""
nanoSeek 统一命令行入口。

把散落在 model/ training/ inference/ data/ 下的多个脚本收敛成一条命令：

    uv run python cli.py <子命令> [参数]

每条子命令都内置「开箱即用的默认预设」——不用记长路径、不用背参数，
直接跑就能得到项目推荐行为，再按需覆盖。

子命令：
    data      准备数据（下载对话语料 → 训分词器 → 编码）
    train     训练模型。默认预设 = 正式中文对话配置 train_chinese.yaml
              （--preset smoke 可快速冒烟；--preset test 同 test.yaml）
    sample    Python 采样（默认 out/chinese-data2）
    eval      对话智能度评估
    smoke     冒烟测试（架构前向/反向 + 参数量）
    bench     性能基准
    convert   把 checkpoint 转换成 Rust 推理权重
    package   打包成独立部署目录
    distill   生成自蒸馏数据（Expert Iteration）
    compare   Rust/Python 逐位对拍
    archive   模型归档/索引（扫描 out/，生成 manifest + index.json）
    selftest  快速自检（命令/预设/配置继承/归档）
    presets   列出可用训练预设

示例：
    uv run python cli.py train --preset smoke          # 几秒冒烟训练
    uv run python cli.py train                          # 正式训练（默认预设）
    uv run python cli.py train --max-iters 1000        # 覆盖任意配置项
    uv run python cli.py sample --prompt "悟空"
    uv run python cli.py eval
    uv run python cli.py convert
    uv run python cli.py package
    uv run python cli.py data
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def _run(*args: str, cwd: str = ROOT) -> int:
    """在当前项目根目录跑一个子进程，把输出透传给终端。

    命令回显打到 stderr，避免干扰 --json 等机器可读 stdout。
    """
    print(f"\n> python {' '.join(args)}", file=sys.stderr)
    return subprocess.call([sys.executable, *args], cwd=cwd)


# ---------------------------------------------------------------------------
# train / sample / bench 用的是「poor man's config」风格：
#   第一个不带 '=' 的参数 = 配置文件（可选）
#   其余 --key=value 覆盖（下划线命名）
# 这里统一把用户给的 `--key value` 转成 `--key=value`，并把连字符转下划线。
# ---------------------------------------------------------------------------
def _kv_args(argv: list[str]) -> list[str]:
    """把 argparse 风格的 --key value / --key=value 规整成训练脚本的 --key=value。"""
    out: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--") and "=" not in a:
            key = a[2:].replace("-", "_")
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                out.append(f"--{key}={argv[i + 1]}")
                i += 2
                continue
            # 裸开关：布尔 true
            out.append(f"--{key}=True")
            i += 1
            continue
        out.append(a)
        i += 1
    return out


# 各数据准备脚本
DATA_STEPS = [
    ("download", os.path.join("data", "chinese", "download_dialogue.py")),
    ("tokenizer", os.path.join("data", "chinese", "train_tokenizer.py")),
    ("prepare", os.path.join("data", "chinese", "prepare.py")),
]


def cmd_data(argv: list[str]) -> int:
    """准备中文对话数据：下载语料 → 训 BPE 分词器 → 编码成 train/val.bin。"""
    step = argv[0] if argv and argv[0] in ("download", "tokenizer", "prepare", "all") else "all"
    if step not in ("download", "tokenizer", "prepare", "all"):
        print("用法: cli.py data [download|tokenizer|prepare|all] [额外参数]")
        return 2
    targets = {"download": [0], "tokenizer": [1], "prepare": [2], "all": [0, 1, 2]}[step]
    if step == "all" and len(argv) > 1:
        print("用法: cli.py data all 不接受额外参数；需要传参请分别运行：")
        print("  uv run python cli.py data download [参数]")
        print("  uv run python cli.py data tokenizer [参数]")
        print("  uv run python cli.py data prepare [参数]")
        return 2
    args = argv[1:]  # 去掉子命令本身，剩下的原样透传给每个脚本
    for idx in targets:
        path = DATA_STEPS[idx][1]
        code = _run(path, *args)
        if code != 0:
            return code
    return 0


_TRAIN_PRESETS = {
    # 预设名 -> (yaml 配置, 说明)
    "default": ("training/config/train_chinese.yaml", "正式中文对话模型（CSA/HCA + MoE + mHC + Sinks + MTP）"),
    "smoke":   ("training/config/test.yaml",            "秒级冒烟训练（几十步，验证代码不崩）"),
    "test":    ("training/config/test.yaml",            "同 smoke"),
}


def cmd_presets(argv: list[str]) -> int:
    """列出当前可用的训练预设。"""
    print("可用训练预设：")
    for name, (cfg, desc) in _TRAIN_PRESETS.items():
        print(f"  {name:<8} {cfg}  — {desc}")
    return 0


def cmd_train(argv: list[str]) -> int:
    """训练模型。默认预设 = 正式中文对话配置；--preset smoke/test 可快速冒烟。"""
    # 解析 --preset
    preset = "default"
    rest: list[str] = []
    it = iter(argv)
    for a in it:
        if a == "--preset":
            preset = next(it, "default")
        elif a.startswith("--preset="):
            preset = a.split("=", 1)[1]
        else:
            rest.append(a)

    if preset not in _TRAIN_PRESETS:
        print(f"未知预设: {preset}，可选: {', '.join(_TRAIN_PRESETS)}")
        return 2

    cfg, desc = _TRAIN_PRESETS[preset]
    print(f"预设 [{preset}]: {desc}")
    # 先把 --key value 规整成 train.py 的 --key=value，再判断有没有显式配置文件
    norm = _kv_args(rest)
    has_config = any(not a.startswith("--") and "=" not in a for a in norm)
    if not has_config:
        # 用户没给位置参数配置文件 → 用预设配置 + 覆盖参数
        cfg_path = os.path.join(ROOT, cfg)
        if not os.path.exists(cfg_path):
            print(f"错误：预设配置不存在 {cfg_path}")
            return 1
        return _run(os.path.join("training", "train.py"), cfg, *norm)
    # 用户显式给了配置文件（第一个非 -- 参数），尊重它
    return _run(os.path.join("training", "train.py"), *norm)


def cmd_sample(argv: list[str]) -> int:
    """Python 采样。默认加载当前最佳模型 out/chinese-data2（--out_dir 覆盖）。

    cli 层把更友好的 --prompt 转发成底层 sample.py 的 --start。
    """
    # 先归一化，这样 --out-dir out/x 也能被识别，避免重复注入默认值
    norm = _kv_args(argv)
    # 统一 --prompt/--start：采样脚本内部变量叫 start，对外统一叫 prompt 更直观
    norm = [("--start=" + a.split("=", 1)[1] if a.startswith("--prompt=") else a) for a in norm]
    if not any(a.startswith("--out_dir=") for a in norm):
        norm = ["--out_dir=out/chinese-data2", *norm]
    # 友好预检：采样前确认 out_dir/best.pt 存在，避免直接吐 torch.load 堆栈
    out_dir = next((a.split("=", 1)[1] for a in norm if a.startswith("--out_dir=")), None)
    if out_dir:
        ckpt = os.path.join(ROOT, out_dir, "best.pt")
        if not os.path.exists(ckpt):
            print(f"错误：找不到 {ckpt}\n"
                  f"请先训练，或用 --out_dir 指定已有实验目录（如 out/xxx）。")
            return 1
    return _run(os.path.join("inference", "sample.py"), *norm)


def cmd_smoke(argv: list[str]) -> int:
    return _run(os.path.join("inference", "scripts", "smoke_test.py"), *argv)


def cmd_bench(argv: list[str]) -> int:
    return _run(os.path.join("inference", "bench.py"), *_kv_args(argv))


def cmd_eval(argv: list[str]) -> int:
    """对话智能度评估（对多个实验臂做客观对比）。"""
    return _run(os.path.join("inference", "scripts", "eval_dialogue.py"), *argv)


def cmd_convert(argv: list[str]) -> int:
    """转换 checkpoint → Rust 推理权重。默认转换 out/chinese-data2/best.pt。"""
    args = list(argv)
    if not any(a in ("-h", "--help") for a in args):
        has_ckpt = any(a.startswith("--ckpt") or a.startswith("--ckpt=") for a in args)
        if not has_ckpt:
            args = ["--ckpt", "out/chinese-data2/best.pt", *args]
        # 友好预检：默认/指定 checkpoint 不存在时直接提示，而不是等 convert.py 报错
        ckpt_val = None
        for i, a in enumerate(args):
            if a == "--ckpt" and i + 1 < len(args):
                ckpt_val = args[i + 1]
            elif a.startswith("--ckpt="):
                ckpt_val = a.split("=", 1)[1]
        if ckpt_val and not os.path.exists(os.path.join(ROOT, ckpt_val)):
            print(f"错误：找不到 checkpoint {ckpt_val}\n"
                  f"请先训练，或用 --ckpt 指定已有 .pt 文件。")
            return 1
    return _run(os.path.join("inference", "runtime", "scripts", "convert.py"), *args)


def cmd_package(argv: list[str]) -> int:
    """一键打包成独立部署目录（Linux 默认；--target windows 交叉编译）。"""
    return _run(os.path.join("inference", "scripts", "package.py"), *argv)


def cmd_compare(argv: list[str]) -> int:
    """Rust/Python 逐位对拍（convert → build → 双端 dump → diff）。"""
    return _run(os.path.join("inference", "scripts", "compare_logits.py"), *argv)


def cmd_archive(argv: list[str]) -> int:
    """模型归档/索引：扫描 out/ 生成 manifest 与 index.json。"""
    return _run(os.path.join("inference", "scripts", "archive.py"), *argv)


def cmd_selftest(argv: list[str]) -> int:
    """快速自检：命令注册、预设存在、配置继承、归档脚本可用。"""
    errors: list[str] = []

    # 1) 所有子命令都有可调用实现
    for name, (fn, _desc) in COMMANDS.items():
        if not callable(fn):
            errors.append(f"子命令 {name} 没有可调用实现")

    # 2) 训练预设对应的配置文件都存在
    for preset, (cfg, _desc) in _TRAIN_PRESETS.items():
        if not os.path.exists(os.path.join(ROOT, cfg)):
            errors.append(f"预设 {preset} 的配置不存在: {cfg}")

    # 3) 配置继承能正确合并（以 train_chinese_fa 为例）
    try:
        from model.config_loader import _load_yaml_with_inheritance
        merged = _load_yaml_with_inheritance(
            os.path.join("training", "config", "train_chinese_fa.yaml")
        )
        if merged.get("block_order") != "ffn_attn":
            errors.append("配置继承结果不正确：train_chinese_fa 的 block_order 应为 ffn_attn")
        if "extends" in merged or "base" in merged:
            errors.append("配置继承结果不应包含 extends/base 键")
    except Exception as e:
        errors.append(f"配置继承自检失败: {e}")

    # 4) 归档脚本存在
    archive_py = os.path.join("inference", "scripts", "archive.py")
    if not os.path.exists(os.path.join(ROOT, archive_py)):
        errors.append(f"归档脚本不存在: {archive_py}")

    if errors:
        print("自检失败：")
        for e in errors:
            print(f"  ✗ {e}")
        return 1
    print("自检通过 ✅ 命令/预设/配置继承/归档脚本均正常")
    return 0


def cmd_distill(argv: list[str]) -> int:
    """生成自蒸馏数据（Expert Iteration）。"""
    return _run(os.path.join("training", "self_distill.py"), *argv)


COMMANDS = {
    "data":     (cmd_data,     "准备数据（下载/分词/编码）"),
    "train":    (cmd_train,    "训练（默认 train_chinese 预设；--preset smoke 冒烟）"),
    "presets":  (cmd_presets,  "列出可用训练预设"),
    "sample":   (cmd_sample,   "Python 采样（默认 out/chinese-data2）"),
    "eval":     (cmd_eval,     "对话智能度评估"),
    "smoke":    (cmd_smoke,    "冒烟测试"),
    "bench":    (cmd_bench,    "性能基准"),
    "convert":  (cmd_convert,  "转换 → Rust 推理权重"),
    "package":  (cmd_package,  "打包独立部署目录"),
    "distill":  (cmd_distill,  "生成自蒸馏数据"),
    "compare":  (cmd_compare,  "Rust/Python 对拍"),
    "archive":  (cmd_archive,  "模型归档/索引（out/ → manifest + index.json）"),
    "selftest": (cmd_selftest, "快速自检（命令/预设/继承/归档）"),
}

# 这些子命令的底层脚本本身就是 argparse，直接把 --help 透传过去看详细选项；
# 其余脚本（train/sample/bench/smoke/data）由统一帮助兜底。
ARGPARSE_COMMANDS = {"convert", "package", "eval", "distill", "compare", "archive"}


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        print("子命令：")
        for name, (_, desc) in COMMANDS.items():
            print(f"  {name:<10} {desc}")
        print("\n每条命令可用 --help 查看具体参数和用法。")
        return 0

    cmd, *rest = argv
    if cmd not in COMMANDS:
        print(f"未知子命令: {cmd}\n可用: {', '.join(COMMANDS)}")
        return 2

    runner, desc = COMMANDS[cmd]
    if any(a in ("-h", "--help") for a in rest):
        if cmd in ARGPARSE_COMMANDS:
            # argparse 子命令：直接看底层脚本的详细参数
            return runner(["--help"])
        # 其余脚本由本入口统一打印用法，避免 --help 被当成配置文件或直接跑掉
        doc = (runner.__doc__ or "").strip()
        print(f"用法: cli.py {cmd} [参数]")
        print(f"\n{desc}")
        if doc:
            print(f"\n{doc}")
        print("\n底层脚本的参数可以直接透传；train/sample/bench 支持 --preset / --key=value。")
        return 0
    return runner(rest)


if __name__ == "__main__":
    sys.exit(main())
