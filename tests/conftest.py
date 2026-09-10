"""pytest 公共配置：路径注入、设备探测、共享夹具。

## 三条硬约定（新人改测试前先读）
1. **所有测试必须能在纯 CPU 上跑通**。需要显卡的用例加 `@pytest.mark.gpu`，
   无卡机器由 `pytest_collection_modifyitems` 自动 skip —— 保证"没卡就红一片"
   不会发生，否则大家会开始无视测试结果。
2. **不要 import `training/train.py`**。它是模块级脚本，import 就会直接开训
   （48 秒起，还会占显存）。要测的纯函数在 `training/schedules.py`。
3. **不要写依赖真实语料/checkpoint 的测试**。需要真实产物的用例要 `skipif`
   文件不存在，且标 `slow`，避免换机器就挂。
"""
import os
import pathlib
import sys

import pytest

# 让 tests/ 下的文件可以直接 `from model import ...`（pytest 的 pythonpath 也做了，
# 这里再保一道，方便单独 `python tests/xxx.py` 调试）。
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 与 training/train.py 一致的 ROCm 环境，必须在 import torch 之前设好。
# gfx1030 是 RDNA2，ROCm 6.4 已不在官方支持列表，靠这两个变量硬撑。
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("HSA_ENABLE_SDMA", "0")

import torch  # noqa: E402  (必须在环境变量之后)

HAS_GPU = torch.cuda.is_available()


def pytest_collection_modifyitems(config, items):
    """没有 GPU 时，把 @pytest.mark.gpu 的用例标成 skip 而不是让它失败。"""
    if HAS_GPU:
        return
    skip = pytest.mark.skip(reason="需要 GPU（本机 torch.cuda.is_available() 为 False）")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def device():
    """可用设备字符串。"""
    return "cuda" if HAS_GPU else "cpu"


@pytest.fixture
def tiny_config():
    """最小但**全分支打开**的 GPTConfig。

    刻意把 MLA / MoE / 共享专家 / aux-free / MTP / mHC / qk_norm / attn_sink
    全打开：这些是本项目真正的架构复杂度所在，只测默认配置等于没测。
    n_embd=64 / n_head=4 → head_dim=16，mla 的 qk_rope_head_dim=8 ≤ 16 才成立。
    """
    from model.config import GPTConfig
    return GPTConfig(
        block_size=64, vocab_size=256, n_layer=2, n_head=4, n_embd=64,
        dropout=0.0, bias=False, use_rope=True, rope_theta=10000.0,
        swiglu_clamp=10.0,
        use_moe=True, n_experts=4, n_top_k=2, use_shared_expert=True,
        use_aux_free_balance=True, balance_factor=0.001,
        use_sqrtsoftplus=True, route_scale=2.5, moe_hidden_scale=4 / 3,
        use_mla=True, kv_lora_rank=16, qk_rope_head_dim=8,
        use_mtp=True, n_mtp=1, mtp_weight=0.3,
        use_muon=True, muon_split=True, muon_ns_steps=5, muon_ns_aggressive=2,
        use_mhc=True, hc_mult=2,
        use_attn_sink=True, use_qk_norm=True,
        char_level=True, eos_token_id=128,
    )


@pytest.fixture
def tiny_model(tiny_config):
    """`tiny_config` 实例化的模型（fp32，CPU，eval 之前的干净状态）。"""
    from model.gpt import GPT
    torch.manual_seed(1234)
    return GPT(tiny_config)


@pytest.fixture(scope="session")
def load_module_from_path():
    """按文件路径加载一个脚本模块。

    为什么需要：`local/` 和 `data/chinese/` 下的脚本不在任何包里，而且模块顶层
    有副作用（`sys.path.insert`、设 ROCm 环境变量）。用 importlib 显式按路径加载，
    再把模块注册进 `sys.modules`（`spec.loader.exec_module` 里的相对/自引用要靠它）。
    """
    import importlib.util

    def _load(path, name):
        spec = importlib.util.spec_from_file_location(name, str(path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    return _load
