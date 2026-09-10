"""归档检查点保留策略的单测。

对应 `training/checkpoints.py`。这块逻辑**会删文件**，写错就是数据丢失，
所以必须有离线覆盖，不能只靠"起一次真训练看看"。
"""

from __future__ import annotations

import os

import pytest

from training.checkpoints import (
    DEFAULT_KEEP_STEP_CKPTS,
    prune_step_checkpoints,
    step_checkpoint_steps,
)


def _mk(out_dir, name, content=b'x'):
    p = os.path.join(out_dir, name)
    with open(p, 'wb') as f:
        f.write(content)
    return p


def test_lists_only_step_checkpoints(tmp_path):
    d = str(tmp_path)
    _mk(d, 'ckpt_step_1000.pt')
    _mk(d, 'ckpt_step_3000.pt')
    _mk(d, 'best.pt')           # 固定文件名，不该被当成归档
    _mk(d, 'last.pt')
    _mk(d, 'ckpt_step_1000.pt.tmp')   # 异步保存的中间态
    _mk(d, 'results.csv')
    assert step_checkpoint_steps(d) == [1000, 3000]


def test_prune_keeps_newest_by_numeric_order(tmp_path):
    """★ 核心回归：必须按整数步号比较，不能按文件名字典序。

    字典序下 'ckpt_step_9000.pt' > 'ckpt_step_10000.pt'（'9' > '1'），
    于是「保留最近的」会删掉真正最新的 10000，留下 9000 —— 且不报错。
    """
    d = str(tmp_path)
    for s in (9000, 10000, 20000):
        _mk(d, f'ckpt_step_{s}.pt')

    removed = prune_step_checkpoints(d, keep=2)

    assert removed == ['ckpt_step_9000.pt']
    assert step_checkpoint_steps(d) == [10000, 20000]


def test_prune_never_touches_best_or_last(tmp_path):
    d = str(tmp_path)
    for s in (1000, 2000, 3000, 4000):
        _mk(d, f'ckpt_step_{s}.pt')
    _mk(d, 'best.pt')
    _mk(d, 'last.pt')

    prune_step_checkpoints(d, keep=1)

    assert sorted(os.listdir(d)) == [
        'best.pt', 'ckpt_step_4000.pt', 'last.pt',
    ]


def test_prune_zero_or_negative_keeps_everything(tmp_path):
    """keep<=0 是「永久保留」，不是「删光」。"""
    d = str(tmp_path)
    for s in (1000, 2000, 3000):
        _mk(d, f'ckpt_step_{s}.pt')

    assert prune_step_checkpoints(d, keep=0) == []
    assert prune_step_checkpoints(d, keep=-1) == []
    assert step_checkpoint_steps(d) == [1000, 2000, 3000]


def test_prune_is_idempotent_and_noop_when_under_limit(tmp_path):
    d = str(tmp_path)
    for s in (1000, 2000):
        _mk(d, f'ckpt_step_{s}.pt')

    assert prune_step_checkpoints(d, keep=DEFAULT_KEEP_STEP_CKPTS) == []
    assert prune_step_checkpoints(d, keep=DEFAULT_KEEP_STEP_CKPTS) == []
    assert step_checkpoint_steps(d) == [1000, 2000]


def test_prune_survives_unremovable_file(tmp_path, monkeypatch):
    """删除失败必须被吞掉 —— 清理是尽力而为，不能有能力搞崩训练。"""
    d = str(tmp_path)
    for s in (1000, 2000, 3000):
        _mk(d, f'ckpt_step_{s}.pt')

    real_remove = os.remove

    def fake_remove(path):
        if path.endswith('ckpt_step_1000.pt'):
            raise PermissionError('模拟：文件被占用')
        return real_remove(path)

    monkeypatch.setattr(os, 'remove', fake_remove)

    removed = prune_step_checkpoints(d, keep=1)

    # 1000 删失败、2000 删成功；3000 保留
    assert removed == ['ckpt_step_2000.pt']
    assert sorted(os.listdir(d)) == ['ckpt_step_1000.pt', 'ckpt_step_3000.pt']


def test_prune_empty_dir(tmp_path):
    assert prune_step_checkpoints(str(tmp_path), keep=2) == []


def test_prune_nonincreasing_keep_sequence_converges(tmp_path):
    """连续归档 + 每步清理，长期应稳定在 keep 个（模拟真实训练循环）。"""
    d = str(tmp_path)
    for step in range(1000, 11000, 1000):
        _mk(d, f'ckpt_step_{step}.pt')
        prune_step_checkpoints(d, keep=3)

    assert step_checkpoint_steps(d) == [8000, 9000, 10000]


@pytest.mark.parametrize('keep', [1, 2, 3])
def test_prune_keeps_exactly_n(tmp_path, keep):
    d = str(tmp_path)
    for s in range(1000, 8000, 1000):
        _mk(d, f'ckpt_step_{s}.pt')

    prune_step_checkpoints(d, keep=keep)

    assert len(step_checkpoint_steps(d)) == keep
