"""`training/diag.py` 的测试：诊断代码自己不能成为故障源。

## 背景（真实事故）
2026-09-10 的 `train.py` 冒烟测试实测抓到：

    configs/base_v2.yaml（`ndb_debug_mem: true`） +  `--device=cpu`
    → KeyError: 'allocated_bytes.all.current'  → 训练在 step 0 直接崩

根因：守卫写的是 `torch.cuda.is_available()`，而**在有显卡的机器上它永远是 True**，
即使实际训练设备是 CPU。于是走进了 CUDA 分配器统计，而无 CUDA 上下文时
`torch.cuda.memory_stats()` 返回的 dict 里没有那些键。

这类 bug 起一次真训练要几十秒才能发现；抽成纯函数后，在 CPU 上毫秒级就能测。
"""
import torch

from training.diag import mem_debug_line, should_dump_snapshot


# ==========================================================================
# 1) mem_debug_line：非 CUDA 必须直接返回 None
# ==========================================================================
def test_mem_debug_line_returns_none_off_cuda():
    """★ 回归测试：`device_type != 'cuda'` 时**必须**返回 None。

    这条就是那个 KeyError 的直接防线。本机 `torch.cuda.is_available()` 为 True，
    所以"有卡但用 CPU 训练"是**真实可达**的组合，不是理论情况。
    """
    assert mem_debug_line('cpu') is None
    assert mem_debug_line('mps') is None
    assert mem_debug_line('') is None


def test_mem_debug_line_is_a_string_on_cuda():
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("本机没有 CUDA/ROCm")
    line = mem_debug_line('cuda')
    assert isinstance(line, str) and line
    # 所有字段都要在，且用的是 .get 兜底（不会 KeyError）
    for field in ('alloc=', 'peak=', 'rsv=', 'oom=', 'retry=', 'dynamo='):
        assert field in line, f"缺少字段 {field}：{line}"


def test_mem_debug_line_does_not_raise_on_this_machine():
    """无论本机是什么设备，两个分支都不能抛异常。"""
    for dt in ('cpu', 'cuda'):
        mem_debug_line(dt)          # 不崩即通过


# ==========================================================================
# 2) should_dump_snapshot：取 max(allocated, reserved)
# ==========================================================================
def test_snapshot_disabled_when_threshold_zero():
    """`mem_snapshot_gb = 0` = 关闭（配置默认如此）。"""
    assert not should_dump_snapshot(10 * 2**30, 10 * 2**30, 0)
    assert not should_dump_snapshot(10 * 2**30, 10 * 2**30, -1)


def test_snapshot_triggers_on_allocated_above_threshold():
    assert should_dump_snapshot(7 * 2**30, 0, 6.0)


def test_snapshot_triggers_on_reserved_only():
    """★ 旧实现只看 allocated，碎片型尖峰（reserved 涨、allocated 没到）永不触发。

    实测：项目里至今 0 个 `mem_snap_*.pickle`，根因就是这个。
    """
    assert should_dump_snapshot(1 * 2**30, 7 * 2**30, 6.0), \
        "只看 allocated 会让碎片型尖峰漏掉"


def test_snapshot_not_triggered_below_threshold():
    assert not should_dump_snapshot(5 * 2**30, 5 * 2**30, 6.0)


def test_snapshot_boundary_is_strictly_greater():
    """恰好等于阈值不触发（用 `>` 而不是 `>=`）—— 与旧实现一致，避免语义漂移。"""
    assert not should_dump_snapshot(6 * 2**30, 0, 6.0)
    assert should_dump_snapshot(6 * 2**30 + 1, 0, 6.0)


def test_snapshot_only_dumps_once():
    """`already_done=True` 后不再触发（避免刷爆磁盘）。"""
    assert not should_dump_snapshot(100 * 2**30, 100 * 2**30, 6.0, already_done=True)


def test_snapshot_threshold_is_in_gib_not_gb():
    """阈值单位是 GiB（2**30）—— 与 train.py 里的换算保持一致。"""
    just_below = int(6.0 * 2**30) - 1
    assert not should_dump_snapshot(just_below, 0, 6.0)
    assert should_dump_snapshot(just_below + 2, 0, 6.0)
