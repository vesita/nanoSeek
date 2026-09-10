"""归档检查点的保留策略（纯函数，可在 CPU 上毫秒级单测）。

**为什么需要它**：`train.py` 逢 1000 步写一个 `ckpt_step_<N>.pt`，每个约 0.6GB。
70000 步 ⇒ 70 个归档 ⇒ **约 42GB**。2026-09-11 巡检时发现 `out/` 已占 74GB、
分区只剩 87GB，而这个归档逻辑**没有任何清理**，长期训练会把盘写满。

`best.pt` / `last.pt` 是固定文件名，**不受本策略影响**；续训只依赖 `last.pt`，
历史归档仅用于阶段回溯（例如「用第 30k 步的基座测一次 NDB 的 Δ」）。

抽成独立模块的理由与 `training/diag.py` 一致：**会删文件的代码必须被测试覆盖**，
逻辑写错就是数据丢失，不能只靠起一次真训练来验证。
"""

from __future__ import annotations

import os
import re

_STEP_CKPT_RE = re.compile(r'ckpt_step_(\d+)\.pt')

DEFAULT_KEEP_STEP_CKPTS = 5


def step_checkpoint_steps(out_dir: str) -> list[int]:
    """按步号升序列出 ``out_dir`` 里的归档检查点步号（忽略 best/last 等其它文件）。"""
    steps: list[int] = []
    for name in os.listdir(out_dir):
        m = _STEP_CKPT_RE.fullmatch(name)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def prune_step_checkpoints(out_dir: str, keep: int = DEFAULT_KEEP_STEP_CKPTS) -> list[str]:
    """只保留最近 ``keep`` 个归档检查点，返回**实际删除**的文件名列表。

    - ``keep <= 0`` 视作「永久保留」，不做任何事。
    - 步号比较用**整数**而非文件名字典序（否则 ``ckpt_step_9000`` 会被误判为新于
      ``ckpt_step_10000``，删错文件）。
    - 任何单个文件删除失败（权限、被占用）都跳过，**绝不抛异常**：
      清理是尽力而为，不能有能力搞崩训练。
    """
    if keep <= 0:
        return []
    steps = step_checkpoint_steps(out_dir)
    removed: list[str] = []
    for step in steps[:-keep]:
        name = f'ckpt_step_{step}.pt'
        try:
            os.remove(os.path.join(out_dir, name))
        except OSError:
            continue
        removed.append(name)
    return removed
