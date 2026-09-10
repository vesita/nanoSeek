"""训练诊断的**纯函数**：显存调试行、快照触发判定、OOM 现场快照。

## 为什么单独成模块
这些是"出问题时才用得上"的诊断代码，但它们**自己不能成为新的故障源**。
2026-09-10 的冒烟测试实测抓到一例：

    ndb_debug_mem: true  +  --device=cpu  →  KeyError: 'allocated_bytes.all.current'
    → 训练在 step 0 直接崩

根因是守卫写成了 `torch.cuda.is_available()`。**在有显卡的机器上它永远是 True**，
即使实际训练设备是 CPU —— 于是走进了 CUDA 分配器统计，而没有 CUDA 上下文时
`memory_stats()` 返回的 dict 里根本没有那些键。

正确判据是「**实际训练设备**」，不是「机器上有没有卡」。抽成纯函数后，
这类错误可以用 CPU 上的小张量在毫秒级测出来，不需要真起一次训练。
"""
import torch

__all__ = ['mem_debug_line', 'should_dump_snapshot']

_GIB = 2 ** 30


def mem_debug_line(device_type):
    """返回一行显存调试字符串；**不在 CUDA 上训练时返回 None**。

    `device_type` 必须是实际训练设备（`'cuda'` / `'cpu'`），不是
    `torch.cuda.is_available()`。所有取值都用 `.get(..., 0)` 兜底：
    诊断字符串少几个数字无所谓，崩掉训练不行。
    """
    if device_type != 'cuda':
        return None
    st = torch.cuda.memory_stats()
    try:
        import torch._dynamo.utils as _du
        dc = dict(_du.counters.get('stats', {}))
    except Exception:
        dc = {}
    return (
        f"alloc={st.get('allocated_bytes.all.current', 0) / _GIB:.2f} "
        f"peak={st.get('allocated_bytes.all.peak', 0) / _GIB:.2f} "
        f"rsv={st.get('reserved_bytes.all.current', 0) / _GIB:.2f} "
        f"oom={st.get('num_ooms', 0)} retry={st.get('num_alloc_retries', 0)} "
        f"dynamo={dc}"
    )


def should_dump_snapshot(allocated_bytes, reserved_bytes, threshold_gb,
                         already_done=False):
    """峰值是否越过快照阈值（以及是否已经 dump 过）。

    取 `max(allocated, reserved)` 而不是只看 allocated：碎片型尖峰是
    **reserved 涨上去而 allocated 没到阈值**，旧实现因此从来不触发
    （项目里至今 0 个 `mem_snap_*.pickle`）。
    """
    if already_done or threshold_gb <= 0:
        return False
    return max(allocated_bytes, reserved_bytes) / _GIB > threshold_gb
