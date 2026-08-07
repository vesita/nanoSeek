# 10 · torch.compile 的 inductor 良性警告（Not enough SMs）

## 现象
训练开始时 `torch.compile` 打印：
```
W0807 ... torch/_inductor/utils.py:1953] [3/0] Not enough SMs to use max_autotune_gemm mode
```

## 根因
inducer 决定是否启用 GEMM 自动调优（`max_autotune_gemm`）时，要求 GPU 至少 **68 个 SM**（RTX 3080 级别）。
RTX 5060 只有 **30 个 SM**，达不到阈值，于是打印提示并**退回默认 matmul 实现**。

判断 GPU SM 数：
```python
torch.cuda.get_device_properties(0).multi_processor_count
```

## 修复（定向压制，不动其他警告）
注意：这是 **logging** 不是 warnings，所以 `warnings.filterwarnings` 没用。
在 compile 前给 `torch._inductor` logger 加一个定向 Filter：

```python
import logging
class _MaxAutotuneGemmFilter(logging.Filter):
    def filter(self, record):
        return "max_autotune_gemm" not in record.getMessage()
logging.getLogger('torch._inductor').addFilter(_MaxAutotuneGemmFilter())
```

Filter 会被 fork 继承，compile 的 worker 进程也会带上，一次全覆盖。

## 怎么避免
- 先判断警告是"影响性能"还是"影响正确性"。这条只让 matmul 少用自动调优，
  **完全良性**，压掉即可。
- 压制用 logging Filter **定向匹配消息**，别把整个 logger 调到 ERROR（会误伤其他有用警告）。
- 想知道为什么某条日志压不掉：先查它走的是 logging 还是 warnings，两条通道处理方式不同。
