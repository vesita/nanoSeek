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
    prune_step_checkpoints_sparse,
    step_checkpoint_steps,
    steps_to_keep,
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


# ==========================================================================
# 稀疏保留策略（2026-09-11 晚新增：从外部 shell 脚本搬进训练内部）
#
# 为什么要有这一组：旧的「只留最新 N 个」策略有一个**静默的破坏性副作用** ——
# 训练超过 (N+1)×1000 步后，早期归档会被当成"旧"删掉，于是
# ckpt_step_5000/10000/15000 这些**阶段回溯点**在 step 25000 左右全部消失。
# 这类 bug 不报错、不告警，只是让"想回到第 10k 步的基座测一次"变成不可能。
# ==========================================================================

def test_steps_to_keep_sparse_rule_only():
    """只开稀疏规则时，只留 step % sparse_every == 0 的。"""
    steps = [1000, 5000, 7000, 10000, 12000]
    assert steps_to_keep(steps, sparse_every=5000, newest_keep=0) == {5000, 10000}


def test_steps_to_keep_newest_rule_only():
    """关上稀疏规则（0）后，退化成旧的「只留最新 N 个」语义。"""
    steps = [1000, 5000, 7000, 10000, 12000]
    assert steps_to_keep(steps, sparse_every=0, newest_keep=2) == {10000, 12000}


def test_steps_to_keep_both_rules_is_union():
    steps = [1000, 5000, 7000, 10000, 12000]
    assert steps_to_keep(steps, sparse_every=5000, newest_keep=2) == {5000, 10000, 12000}


def test_steps_to_keep_newest_larger_than_list_keeps_all():
    """训练早期归档还不够多，newest_keep 超出数量时不该报错，应返回全部。"""
    assert steps_to_keep([1000, 2000], sparse_every=5000, newest_keep=2) == {1000, 2000}


def test_steps_to_keep_refuses_policy_that_deletes_everything():
    """★ 两条规则都关 = "删光全部归档"。宁可崩，也不要静默把历史删干净。"""
    with pytest.raises(ValueError):
        steps_to_keep([1000, 2000], sparse_every=0, newest_keep=0)
    with pytest.raises(ValueError):
        steps_to_keep([1000, 2000], sparse_every=-1, newest_keep=-1)


def test_sparse_prune_never_removes_early_sparse_points(tmp_path):
    """★ 核心回归：走到 step 25000 时，5000/10000/15000 必须还在。

    旧策略（keep=5）在这里会把它们全删掉 —— 这是本次改动的直接原因。
    """
    d = str(tmp_path)
    for s in range(1000, 26000, 1000):
        _mk(d, f'ckpt_step_{s}.pt')

    removed, kept = prune_step_checkpoints_sparse(d, sparse_every=5000, newest_keep=2)

    # 5000/10000/15000/20000 是稀疏点，24000/25000 是最新两个
    assert set(kept) == {5000, 10000, 15000, 20000, 24000, 25000}
    assert 'ckpt_step_5000.pt' not in removed
    assert 'ckpt_step_15000.pt' not in removed
    assert step_checkpoint_steps(d) == [5000, 10000, 15000, 20000, 24000, 25000]


def test_sparse_prune_never_touches_best_or_last(tmp_path):
    d = str(tmp_path)
    for s in range(1000, 8000, 1000):
        _mk(d, f'ckpt_step_{s}.pt')
    _mk(d, 'best.pt')
    _mk(d, 'last.pt')

    prune_step_checkpoints_sparse(d, sparse_every=5000, newest_keep=1)

    assert 'best.pt' in os.listdir(d)
    assert 'last.pt' in os.listdir(d)


def test_sparse_prune_converges_over_a_long_run(tmp_path):
    """模拟真实训练：每 1000 步归档并就地清理，最终稳定在「每 5000 一个 + 最新 2 个」。"""
    d = str(tmp_path)
    for step in range(1000, 31000, 1000):
        _mk(d, f'ckpt_step_{step}.pt')
        prune_step_checkpoints_sparse(d, sparse_every=5000, newest_keep=2)

    # 30000 既是稀疏点又是最新点；29000 是最新两个里的另一个
    assert step_checkpoint_steps(d) == [5000, 10000, 15000, 20000, 25000, 29000, 30000]


def test_sparse_prune_is_idempotent(tmp_path):
    d = str(tmp_path)
    for s in range(1000, 12000, 1000):
        _mk(d, f'ckpt_step_{s}.pt')

    first, _ = prune_step_checkpoints_sparse(d, 5000, 2)
    second, _ = prune_step_checkpoints_sparse(d, 5000, 2)

    assert first != []
    assert second == []          # 第二次无事可做


# ---------------------------------------------------------------------------
# CLI 报告的数字必须是真的
# ---------------------------------------------------------------------------

def _mk_mib(out_dir, name, mib=1):
    return _mk(out_dir, name, b'z' * (mib * 1048576))


def test_main_reports_bytes_actually_freed(tmp_path, capsys):
    """★ 回归：`freed` 必须在删除**之前**取大小。

    旧实现是「先 `_remove` 再 `os.path.getsize`」，于是每个文件都抛 OSError、
    `freed` 恒为 0 —— 删掉一个 0.59GB 的 ckpt 也报「释放 0MB」。
    这类"测量函数自己坏了却看不出"的 bug，只有拿**已知答案的输入**当对照才暴露：
    这里写 1MiB 的文件，删 2 个就必须正好报 2MB。
    """
    from training.checkpoints import main

    d = str(tmp_path)
    for step in (5000, 6000, 7000):
        _mk_mib(d, f'ckpt_step_{step}.pt')

    before = 3 * 1048576
    assert main([d, '5000', '1']) == 0
    out = capsys.readouterr().out

    # 保留集 = 稀疏点 {5000} ∪ 最新 1 个 {7000}；6000 既非 5000 倍数也非最新 → 只删它
    assert '删除 1 个' in out, out
    assert '释放 1MB' in out, out
    assert sorted(os.listdir(d)) == ['ckpt_step_5000.pt', 'ckpt_step_7000.pt']

    after = sum(
        os.path.getsize(os.path.join(d, f))
        for f in os.listdir(d)
    )
    assert before - after == 1 * 1048576, '实际释放字节数应与报告一致'


def test_main_dry_run_reports_would_be_freed_and_deletes_nothing(tmp_path, capsys):
    """dry-run 也要报出**会**释放多少（旧实现这里同样恒报 0MB）。"""
    from training.checkpoints import main

    d = str(tmp_path)
    for step in (5000, 6000, 7000):
        _mk_mib(d, f'ckpt_step_{step}.pt')

    assert main([d, '5000', '1', '--dry-run']) == 0
    out = capsys.readouterr().out

    assert '将删除 1 个' in out, out
    assert '释放 1MB' in out, out
    assert sorted(os.listdir(d)) == [
        'ckpt_step_5000.pt', 'ckpt_step_6000.pt', 'ckpt_step_7000.pt']


def test_main_freed_is_zero_when_nothing_to_delete(tmp_path, capsys):
    """对照：没有可删对象时必须报 0，别把"算不出"包装成"释放了很多"。"""
    from training.checkpoints import main

    d = str(tmp_path)
    _mk_mib(d, 'ckpt_step_5000.pt')

    assert main([d, '5000', '2']) == 0
    out = capsys.readouterr().out
    assert '删除 0 个' in out, out
    assert '释放 0MB' in out, out
