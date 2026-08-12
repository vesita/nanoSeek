#!/usr/bin/env python3
"""一键打包：把训练好的 checkpoint 变成一个可独立运行的 release 目录。

流程：convert.py（转权重）→ cargo build --release（编二进制）→ 组装 release/<实验名>/
release 目录包含：推理程序 + 模型权重 + 架构配置 + 词表 + 运行脚本，
拷走就能跑，不依赖 Python / GPU / 源码。

支持 Windows 交叉编译：Linux 上用 zig + cargo-zigbuild 直接编出 Windows .exe
（--target windows），生成的 release 目录里有 运行.bat。

用法（在项目根目录）：
    uv run python inference/scripts/package.py                            # 默认打当前最佳模型（Linux）
    uv run python inference/scripts/package.py --target windows           # 打 Windows 版
    uv run python inference/scripts/package.py --ckpt out/chinese-data2/best.pt --name csa-v1
"""
import argparse
import os
import shutil
import subprocess
import sys

# 项目根目录 = inference/scripts/ 的上三级（脚本从任何位置运行都能定位）
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNTIME_DIR = os.path.join(ROOT, 'inference', 'runtime')
CONVERT_PY = os.path.join(RUNTIME_DIR, 'scripts', 'convert.py')
RELEASE_ROOT = os.path.join(ROOT, 'release')


def run(cmd, cwd=None, env=None):
    """运行子进程，把输出透传给终端；失败直接抛错终止。"""
    print(f'\n> {" ".join(cmd)}')
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def dir_size(path):
    """递归统计目录总字节数。"""
    total = 0
    for dirpath, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total


def fmt(nbytes):
    for unit in ('B', 'KB', 'MB', 'GB'):
        if nbytes < 1024 or unit == 'GB':
            return f'{nbytes:.1f} {unit}'
        nbytes /= 1024


def main():
    ap = argparse.ArgumentParser(description='打包成独立部署目录')
    ap.add_argument('--ckpt', default='out/chinese-data2/best.pt',
                    help='训练产物路径（默认当前最佳模型）')
    ap.add_argument('--dataset', default='chinese', help='数据集名（找 data/<名>/meta.pkl 取词表）')
    ap.add_argument('--name', default=None, help='release 目录名（默认从 ckpt 路径自动提取）')
    ap.add_argument('--target', default='linux', choices=['linux', 'windows'],
                    help='打包目标平台：linux（默认，本机二进制）/ windows（zig 交叉编译 .exe）')
    ap.add_argument('--no-build', action='store_true',
                    help='跳过 cargo build（二进制已是最新时用）')
    args = ap.parse_args()

    is_windows = args.target == 'windows'

    ckpt = os.path.join(ROOT, args.ckpt)
    if not os.path.exists(ckpt):
        sys.exit(f'错误：checkpoint 不存在 {ckpt}')

    name = args.name or os.path.basename(os.path.dirname(ckpt))
    # Windows 版和 Linux 版放不同目录，避免互相覆盖
    if is_windows:
        name = f'{name}-windows'
    out_dir = os.path.join(RELEASE_ROOT, name)

    # 1) 转换权重到 release 目录（convert.py 内部会 mkdir）
    print(f'\n=== 1/3 转换权重 → {out_dir} ===')
    os.makedirs(out_dir, exist_ok=True)
    run([sys.executable, CONVERT_PY, '--ckpt', ckpt, '--dataset', args.dataset, '--out', out_dir])

    # 2) 编译 Rust 二进制
    print(f'\n=== 2/3 编译 Rust 运行时（target={args.target}）===')
    if is_windows:
        # Windows 交叉编译：需要 zig + cargo-zigbuild（详见下方依赖检查）。
        # 目标 x86_64-pc-windows-gnu：zig 完整内置 mingw-w64 的 CRT 头（含 new.h），
        # 不需要 Visual Studio。用 -msvc 目标会缺 MSVC 专属头（zig 不内置，要装 VS）。
        # tokenizers 的 C/C++ 依赖（onig_sys / esaxx-rs）由 zig 的 cc 接管。
        TARGET = 'x86_64-pc-windows-gnu'
        if not args.no_build:
            # 需要 zig 可执行文件路径：cargo-zigbuild 认 CARGO_ZIGBUILD_ZIG_PATH 环境变量
            # （uv pip install ziglang 后，二进制在 venv 的 site-packages/ziglang/zig）
            zig = os.environ.get('CARGO_ZIGBUILD_ZIG_PATH') or shutil.which('zig')
            if not zig:
                sys.exit('错误：找不到 zig。先装：uv pip install ziglang，并把\n'
                         '  CARGO_ZIGBUILD_ZIG_PATH 指向 venv/lib/*/site-packages/ziglang/zig')
            run(['cargo', 'zigbuild', '--release', '--target', TARGET], cwd=RUNTIME_DIR,
                env={**os.environ, 'CARGO_ZIGBUILD_ZIG_PATH': zig})
        bin_path = os.path.join(RUNTIME_DIR, 'target', TARGET, 'release', 'nanoseek-runtime.exe')
        bin_name = 'nanoseek-runtime.exe'
        run_name = '运行.bat'
    else:
        if not args.no_build:
            run(['cargo', 'build', '--release'], cwd=RUNTIME_DIR)
        bin_path = os.path.join(RUNTIME_DIR, 'target', 'release', 'nanoseek-runtime')
        bin_name = 'nanoseek-runtime'
        run_name = '运行.sh'
    if not os.path.exists(bin_path):
        sys.exit(f'错误：找不到编译产物 {bin_path}\n'
                 f'  --target windows 需要先安装：cargo install cargo-zigbuild &&\n'
                 f'  uv pip install ziglang（zig 二进制在 venv 的 site-packages/ziglang/zig，\n'
                 f'  用 CARGO_ZIGBUILD_ZIG_PATH 环境变量指向它）\n'
                 f'  或先跑一次 cargo build')

    # 3) 组装 release 目录
    print(f'\n=== 3/3 组装 release 目录 ===')
    shutil.copy2(bin_path, os.path.join(out_dir, bin_name))
    if is_windows:
        # Windows 批处理：用 %~dp0 定位脚本所在目录（相当于 Linux 的 dirname $0）。
        # 字符串里已带 \r\n，newline='' 禁止 Python 再翻译（否则 \r\n 变 \r\r\n 双重回车）
        run_script = os.path.join(out_dir, run_name)
        with open(run_script, 'w', encoding='utf-8', newline='') as f:
            f.write('@echo off\r\n'
                    'rem 自动定位到本脚本所在目录，再运行推理程序\r\n'
                    'cd /d "%~dp0"\r\n'
                    'nanoseek-runtime.exe %*\r\n'
                    'rem 用法：\r\n'
                    'rem   "运行.bat" --prompt "悟空"         # 一次性生成\r\n'
                    'rem   "运行.bat" --repeat-penalty 1.2    # 抑制复述（1.0 关闭）\r\n'
                    'rem   "运行.bat"                         # 进入对话 REPL\r\n'
                    'pause\r\n')
    else:
        run_script = os.path.join(out_dir, run_name)
        with open(run_script, 'w', encoding='utf-8') as f:
            f.write('#!/bin/sh\n'
                    '# 自动定位到本脚本所在目录，再运行推理程序（这样相对路径的模型文件才找得到）\n'
                    'cd "$(dirname "$0")"\n'
                    'exec ./nanoseek-runtime "$@"\n'
                    '# 用法：\n'
                    '#   sh 运行.sh --prompt "悟空"          # 一次性生成\n'
                    '#   sh 运行.sh --repeat-penalty 1.2     # 抑制复述（1.0 关闭）\n'
                    '#   sh 运行.sh                          # 进入对话 REPL\n')
        os.chmod(run_script, 0o755)

    # 汇总
    files = [
        (bin_name, '推理程序（CPU，无需 Python/GPU）'),
        ('model.safetensors', '模型权重'),
        ('model_config.json', '架构配置'),
        ('tokenizer.json', 'BPE 分词器'),
        (run_name, '启动脚本'),
    ]
    print(f'\n打包完成 ✅  {out_dir}  （共 {fmt(dir_size(out_dir))}）')
    for fn, desc in files:
        p = os.path.join(out_dir, fn)
        if os.path.exists(p):
            print(f'  {fn:<22} {fmt(os.path.getsize(p)):>8}  {desc}')
    if is_windows:
        print('\n运行方式（Windows）：')
        print(f'  把 {os.path.relpath(out_dir, ROOT)} 整个目录拷到 Windows 电脑')
        print('  双击 运行.bat 进入对话 REPL，或命令行：')
        print('    nanoseek-runtime.exe --prompt "悟空"')
    else:
        print('\n运行方式：')
        print(f'  cd {os.path.relpath(out_dir, ROOT)}')
        print('  ./nanoseek-runtime --prompt "悟空"      # 一次性生成')
        print('  ./nanoseek-runtime                      # 对话 REPL')
        print('  sh 运行.sh --prompt "悟空"             # 或通过启动脚本')


if __name__ == '__main__':
    main()
