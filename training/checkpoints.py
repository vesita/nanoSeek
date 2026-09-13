"""归档检查点的保留策略（纯函数，可在 CPU 上毫秒级单测）。

**为什么需要它**：`train.py` 逢 1000 步写一个 `ckpt_step_<N>.pt`，每个约 0.6GB。
70000 步 ⇒ 70 个归档 ⇒ **约 42GB**。2026-09-11 巡检时发现 `out/` 已占 74GB、
分区只剩 87GB，而这个归档逻辑**没有任何清理**，长期训练会把盘写满。

`best.pt` / `last.pt` 是固定文件名，**不受本策略影响**；续训只依赖 `last.pt`，
历史归档仅用于阶段回溯（例如「用第 30k 步的基座测一次 NDB 的 Δ」）。

抽成独立模块的理由与 `training/diag.py` 一致：**会删文件的代码必须被测试覆盖**，
逻辑写错就是数据丢失，不能只靠起一次真训练来验证。

## 策略演进（2026-09-11 晚）

第一版是「只保留最新 N 个」（`prune_step_checkpoints(keep=N)`）。它有个**静默的副作用**：
训练超过 `(N+1)×1000` 步后，早期归档会被当作"旧"删掉 —— 于是
`ckpt_step_5000/10000/15000` 这些**阶段回溯点**在 step 25000 左右全部消失。
真正想要的策略是 `scripts/prune_ckpts.sh` 里的那条：**每 `sparse_every` 步留一个 + 最新几个**。

那条策略原先只活在 shell 脚本里、由外部巡检调用，而巡检本身依赖看守进程活着
（2026-09-11 会话重启把看守和训练一起 SIGKILL 掉了）。
⇒ 现在**统一到本模块**，`train.py` 每次存档后自己调用；shell 脚本退化为薄包装。

**唯一决策函数是 `steps_to_keep`**，其余都是它的薄包装 ——
避免"Python 一套 + shell 一套"两个实现慢慢分叉。
"""

from __future__ import annotations

import os
import re

_STEP_CKPT_RE = re.compile(r'ckpt_step_(\d+)\.pt')

DEFAULT_KEEP_STEP_CKPTS = 5
# 稀疏保留：每 5000 步留一个（阶段回溯点）+ 最新 2 个（防"刚写的就被删"）。
# 与 scripts/prune_ckpts.sh 的历史默认值保持一致，避免行为静默变化。
DEFAULT_SPARSE_EVERY = 5000
DEFAULT_NEWEST_KEEP = 2


def step_checkpoint_steps(out_dir: str) -> list[int]:
    """按步号升序列出 ``out_dir`` 里的归档检查点步号（忽略 best/last 等其它文件）。"""
    steps: list[int] = []
    for name in os.listdir(out_dir):
        m = _STEP_CKPT_RE.fullmatch(name)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def steps_to_keep(
    steps: list[int],
    sparse_every: int = DEFAULT_SPARSE_EVERY,
    newest_keep: int = DEFAULT_NEWEST_KEEP,
) -> set[int]:
    """★ 唯一决策函数：给定已有归档步号，返回**应保留**的步号集合。

    两条规则（并集）：
      1. ``step % sparse_every == 0`` —— 稀疏回溯点（``sparse_every <= 0`` 时关闭）
      2. 步号最大的 ``newest_keep`` 个 —— 防"刚写的就被删"（``newest_keep <= 0`` 时关闭）

    两条规则**都关闭**时抛 ``ValueError``：那样的策略含义是"全删"，
    而"全删"绝不该是任何人配置出来的意图 —— 宁可崩，也不要静默删掉全部历史。

    ``newest_keep`` 大于现有数量时返回全部，不报错（训练早期属正常）。
    """
    if sparse_every <= 0 and newest_keep <= 0:
        raise ValueError(
            'sparse_every 与 newest_keep 不能同时 <= 0：那等于"删光所有归档"'
        )
    ordered = sorted(steps)
    keep: set[int] = set()
    if sparse_every > 0:
        keep |= {s for s in ordered if s % sparse_every == 0}
    if newest_keep > 0:
        keep |= set(ordered[-newest_keep:])
    return keep


def _remove(out_dir: str, steps: list[int]) -> list[str]:
    """尽力而为地删除给定步号的归档，返回**实际删除**的文件名列表。

    任何单个文件删除失败（权限、被占用）都跳过，**绝不抛异常**：
    清理是尽力而为，不能有能力搞崩训练。
    """
    removed: list[str] = []
    for step in steps:
        name = f'ckpt_step_{step}.pt'
        try:
            os.remove(os.path.join(out_dir, name))
        except OSError:
            continue
        removed.append(name)
    return removed


def prune_step_checkpoints(out_dir: str, keep: int = DEFAULT_KEEP_STEP_CKPTS) -> list[str]:
    """只保留最近 ``keep`` 个归档检查点，返回**实际删除**的文件名列表。

    这是 ``steps_to_keep(sparse_every=0, newest_keep=keep)`` 的薄包装，保留给
    「不需要稀疏回溯点、只要最新几个」的场景。

    - ``keep <= 0`` 视作「永久保留」，不做任何事（**不会**走到 steps_to_keep 的报错分支）。
    - 步号比较用**整数**而非文件名字典序（否则 ``ckpt_step_9000`` 会被误判为新于
      ``ckpt_step_10000``，删错文件）。
    """
    if keep <= 0:
        return []
    steps = step_checkpoint_steps(out_dir)
    keep_set = steps_to_keep(steps, sparse_every=0, newest_keep=keep)
    return _remove(out_dir, [s for s in steps if s not in keep_set])


def prune_step_checkpoints_sparse(
    out_dir: str,
    sparse_every: int = DEFAULT_SPARSE_EVERY,
    newest_keep: int = DEFAULT_NEWEST_KEEP,
) -> tuple[list[str], list[int]]:
    """生产用的稀疏保留策略：每 ``sparse_every`` 步留一个 + 最新 ``newest_keep`` 个。

    返回 ``(实际删除的文件名, 保留的步号)`` —— 把"保留了什么"也交回给调用方，
    好让训练日志能印出来（**看不见的清理策略等于没有策略**）。
    """
    steps = step_checkpoint_steps(out_dir)
    keep_set = steps_to_keep(steps, sparse_every, newest_keep)
    removed = _remove(out_dir, [s for s in steps if s not in keep_set])
    return removed, sorted(keep_set)


def main(argv: list[str] | None = None) -> int:
    """命令行入口：``python -m training.checkpoints <out_dir> [sparse_every] [newest_keep]``。

    存在的意义：让 `scripts/prune_ckpts.sh` 退化成薄包装，
    这样**策略只有一份实现**，shell 与 Python 不会慢慢分叉。
    加 ``--dry-run`` 只报告不删除。
    """
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    dry = '--dry-run' in args
    args = [a for a in args if a != '--dry-run']
    if not args:
        print('用法: python -m training.checkpoints <out_dir> [sparse_every] [newest_keep] [--dry-run]')
        return 2
    out_dir = args[0]
    sparse_every = int(args[1]) if len(args) > 1 else DEFAULT_SPARSE_EVERY
    newest_keep = int(args[2]) if len(args) > 2 else DEFAULT_NEWEST_KEEP

    if not os.path.isdir(out_dir):
        print(f'✗ 目录不存在：{out_dir}')
        return 1

    steps = step_checkpoint_steps(out_dir)
    if not steps:
        print('（无归档检查点）')
        return 0

    keep_set = steps_to_keep(steps, sparse_every, newest_keep)
    doomed = [s for s in steps if s not in keep_set]

    # ★ 大小必须在**删除之前**取。旧代码是删完再 getsize，于是每次都 OSError →
    #   freed 恒为 0（2026-09-12 实测：删掉一个 0.59GB 的 ckpt 却报「释放 0MB」）。
    #   这是典型的「测量函数自己坏了却看不出」——只有对照之下才暴露。
    sizes: dict[int, int] = {}
    for step in doomed:
        try:
            sizes[step] = os.path.getsize(
                os.path.join(out_dir, f'ckpt_step_{step}.pt'))
        except OSError:
            pass  # 已经被并发的另一路清理删掉了

    removed_names = ([f'ckpt_step_{s}.pt' for s in doomed] if dry
                     else _remove(out_dir, doomed))
    removed_set = set(removed_names)
    freed = sum(size for step, size in sizes.items()
                if f'ckpt_step_{step}.pt' in removed_set)

    verb = '将删除' if dry else '删除'
    print(f'🧹 归档稀疏化（每 {sparse_every} 步留一个 + 最新 {newest_keep} 个）：'
          f'{verb} {len(removed_names)} 个，释放 {freed / 1048576:.0f}MB')
    print(f'   保留 {len(keep_set)} 个：{sorted(keep_set)}')
    print(f'   现存归档：{len(steps) - len(removed_names)} 个')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
