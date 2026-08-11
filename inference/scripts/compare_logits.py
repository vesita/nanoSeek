"""
Rust/Python 逐位对拍工具：同步部署端的"守门员"。

一条命令完成：convert → cargo build → Python dump logits → Rust dump logits → diff。
任何一次 Python 侧改架构后，跑它就能立刻知道 Rust 端有没有同步上
（最大误差应 < 0.01；top-5 排序应完全一致）。

用法（从项目根目录）：
    uv run python inference/scripts/compare_logits.py                          # 默认 out/chinese-all/best.pt
    uv run python inference/scripts/compare_logits.py --ckpt out/chinese/best.pt
    uv run python inference/scripts/compare_logits.py --prompt "用户：你好\n模型："
"""
import argparse
import subprocess
import tempfile
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNTIME = os.path.join(ROOT, 'inference', 'runtime')
BIN = os.path.join(RUNTIME, 'target', 'release', 'nanoseek-runtime')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='out/chinese-all/best.pt', help='训练好的 checkpoint 路径')
    ap.add_argument('--dataset', default='chinese', help='数据集名（找 tokenizer.json）')
    ap.add_argument('--prompt', default='悟空', help='对拍用的 prompt（默认短 prompt，快）')
    ap.add_argument('--max-err', type=float, default=0.01,
                    help='允许的最大绝对误差（默认 0.01；mHC 有 f32 舍入累积，逐层 1e-6、最终 ~1e-2）')
    args = ap.parse_args()

    # 1) convert：checkpoint → safetensors + config + tokenizer
    print(f'[1/5] 转换 checkpoint {args.ckpt} ...')
    r = subprocess.run(
        ['uv', 'run', 'python', os.path.join(RUNTIME, 'scripts', 'convert.py'),
         '--ckpt', args.ckpt, '--dataset', args.dataset],
        cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print('convert.py 失败：\n', r.stdout, r.stderr)
        sys.exit(1)

    # 2) cargo build --release（增量编译，秒级）
    print('[2/5] 编译 Rust release ...')
    r = subprocess.run(['cargo', 'build', '--release'], cwd=RUNTIME,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print('cargo build 失败：\n', r.stdout, r.stderr)
        sys.exit(1)

    # 3) Python dump logits（fp32，和 Rust 的 F32 对齐；自动加 --dtype=float32）
    print('[3/5] Python dump logits ...')
    py_out = os.path.join(tempfile.mkdtemp(), 'py_logits.txt')
    r = subprocess.run(
        ['uv', 'run', 'python', os.path.join(ROOT, 'inference', 'sample.py'),
         f'--out_dir={os.path.dirname(args.ckpt)}', f'--start={args.prompt}',
         f'--dump_logits={py_out}', '--dtype=float32'],
        cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print('sample.py 失败：\n', r.stdout, r.stderr)
        sys.exit(1)

    # 4) Rust dump logits
    print('[4/5] Rust dump logits ...')
    rust_out = os.path.join(os.path.dirname(py_out), 'rust_logits.txt')
    r = subprocess.run([BIN, '--dump-logits', rust_out, '--prompt', args.prompt],
                       cwd=RUNTIME, capture_output=True, text=True)
    if r.returncode != 0:
        print('Rust 端失败：\n', r.stdout, r.stderr)
        sys.exit(1)

    # 5) diff
    print('[5/5] 对比 logits ...')
    import numpy as np  # noqa: E402 （脚本要求 uv 环境，numpy 有）
    a = np.loadtxt(py_out)
    b = np.loadtxt(rust_out)
    max_err = float(np.abs(a - b).max())
    mean_err = float(np.abs(a - b).mean())
    top_p = np.argsort(a)[-5:][::-1]
    top_r = np.argsort(b)[-5:][::-1]
    top_same = np.array_equal(top_p, top_r)

    print(f'  最大绝对误差 = {max_err:.5f}（阈值 {args.max_err}）')
    print(f'  平均绝对误差 = {mean_err:.5f}')
    print(f'  top-5 排序一致 = {top_same}')
    print(f'  Python top5: {top_p.tolist()}')
    print(f'  Rust   top5: {top_r.tolist()}')

    ok = max_err < args.max_err and top_same
    if ok:
        print('\n对拍通过 ✅ Rust 与 Python 一致')
    else:
        print('\n对拍失败 ❌ Rust 端没同步上！检查：')
        print('  1. Rust 是否实现了该 checkpoint 的架构开关（model.rs/attention.rs）')
        print('  2. 用 --prompt 换更长 prompt 复测（覆盖 CSA 块路径）')
        sys.exit(1)


if __name__ == '__main__':
    main()
