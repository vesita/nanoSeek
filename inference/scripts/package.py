#!/usr/bin/env python3
"""一键打包：把训练好的 checkpoint 变成一个可独立运行的 release 目录。

流程：convert.py（转权重）→ cargo build --release（编二进制）→ 组装 release/<实验名>/
release 目录包含：推理程序 + 模型权重 + 架构配置 + 词表 + 运行.sh，
拷走就能跑，不依赖 Python / GPU / 源码。

用法（在项目根目录）：
    uv run python inference/scripts/package.py                        # 默认打全特性模型
    uv run python inference/scripts/package.py --ckpt out/chinese-all/best.pt
    uv run python inference/scripts/package.py --ckpt out/chinese/best.pt --name csa-v1
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


def run(cmd, cwd=None):
    """运行子进程，把输出透传给终端；失败直接抛错终止。"""
    print(f'\n> {" ".join(cmd)}')
    subprocess.run(cmd, cwd=cwd, check=True)


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
    ap.add_argument('--ckpt', default='out/chinese-all/best.pt',
                    help='训练产物路径（默认全特性模型）')
    ap.add_argument('--dataset', default='chinese', help='数据集名（找 data/<名>/meta.pkl 取词表）')
    ap.add_argument('--name', default=None, help='release 目录名（默认从 ckpt 路径自动提取）')
    ap.add_argument('--no-build', action='store_true',
                    help='跳过 cargo build（二进制已是最新时用）')
    args = ap.parse_args()

    ckpt = os.path.join(ROOT, args.ckpt)
    if not os.path.exists(ckpt):
        sys.exit(f'错误：checkpoint 不存在 {ckpt}')

    name = args.name or os.path.basename(os.path.dirname(ckpt))
    out_dir = os.path.join(RELEASE_ROOT, name)

    # 1) 转换权重到 release 目录（convert.py 内部会 mkdir）
    print(f'\n=== 1/3 转换权重 → {out_dir} ===')
    os.makedirs(out_dir, exist_ok=True)
    run([sys.executable, CONVERT_PY, '--ckpt', ckpt, '--dataset', args.dataset, '--out', out_dir])

    # 2) 编译 Rust 二进制
    print(f'\n=== 2/3 编译 Rust 运行时 ===')
    if not args.no_build:
        run(['cargo', 'build', '--release'], cwd=RUNTIME_DIR)
    bin_path = os.path.join(RUNTIME_DIR, 'target', 'release', 'nanoseek-runtime')
    if not os.path.exists(bin_path):
        sys.exit(f'错误：找不到编译产物 {bin_path}（先跑一次 cargo build）')

    # 3) 组装 release 目录
    print(f'\n=== 3/3 组装 release 目录 ===')
    shutil.copy2(bin_path, os.path.join(out_dir, 'nanoseek-runtime'))
    run_sh = os.path.join(out_dir, '运行.sh')
    with open(run_sh, 'w', encoding='utf-8') as f:
        f.write('#!/bin/sh\n'
                '# 自动定位到本脚本所在目录，再运行推理程序（这样相对路径的模型文件才找得到）\n'
                'cd "$(dirname "$0")"\n'
                'exec ./nanoseek-runtime "$@"\n'
                '# 用法：\n'
                '#   sh 运行.sh --prompt "悟空"     # 一次性生成\n'
                '#   sh 运行.sh                     # 进入对话 REPL\n')
    os.chmod(run_sh, 0o755)

    # 汇总
    files = [
        ('nanoseek-runtime', '推理程序（CPU，无需 Python/GPU）'),
        ('model.safetensors', '模型权重'),
        ('model_config.json', '架构配置'),
        ('tokenizer.json', 'BPE 分词器'),
        ('运行.sh', '启动脚本'),
    ]
    print(f'\n打包完成 ✅  {out_dir}  （共 {fmt(dir_size(out_dir))}）')
    for fn, desc in files:
        p = os.path.join(out_dir, fn)
        if os.path.exists(p):
            print(f'  {fn:<22} {fmt(os.path.getsize(p)):>8}  {desc}')
    print('\n运行方式：')
    print(f'  cd {os.path.relpath(out_dir, ROOT)}')
    print('  ./nanoseek-runtime --prompt "悟空"      # 一次性生成')
    print('  ./nanoseek-runtime                      # 对话 REPL')
    print('  sh 运行.sh --prompt "悟空"             # 或通过启动脚本')


if __name__ == '__main__':
    main()
