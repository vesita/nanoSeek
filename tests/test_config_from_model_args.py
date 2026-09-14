# -*- coding: utf-8 -*-
"""`GPTConfig.from_model_args` —— 让"删配置字段"不再炸历史 checkpoint。

## 为什么需要这个文件
2026-09-15 删掉旧 PK-NDB 的 6 个 `GPTConfig` 字段后，`local/build_chunk_store.py`
当场崩在 `GPTConfig(**args)`：

    TypeError: GPTConfig.__init__() got an unexpected keyword argument 'use_neural_db'

`checkpoint['model_args']` 存的是**当时那套**字段，删字段就会让**所有历史 checkpoint
都加载不了**。修法是给 `GPTConfig` 一个过滤掉未知键的构造入口，并把所有
"从 checkpoint 拿 model_args"的地方都改成走它。

★ 注意 `training/train.py` 本来就不受影响 —— 它是 `model_args[k] = ckpt.get(k, ...)`
**逐个已知键**取的。出事的是那些直接 `**` 展开的地方（共 12 处，已统一）。
⇒ 这也是为什么"改完 train.py 跑了冒烟"**不足以**证明没破坏加载路径：
冒烟走的是 train.py 那条已经过滤过的路。
"""
import dataclasses

import pytest

from model.config import GPTConfig

# 这些键曾经是 GPTConfig 的字段，2026-09-15 随旧 PK-NDB 一起删除。
REMOVED_KEYS = {
    "use_neural_db": False,
    "neural_db_layer": 6,
    "neural_db_sub_keys": 512,
    "neural_db_top_k": 32,
    "neural_db_gc_interval": 200,
    "neural_db_usage_aux_scale": 0.1,
}


def test_unknown_keys_are_dropped():
    """老 checkpoint 的 model_args 里带着已删除的键 ⇒ 必须能安静地建出配置。"""
    args = {"n_layer": 4, "n_embd": 64, **REMOVED_KEYS}

    cfg = GPTConfig.from_model_args(args)

    assert cfg.n_layer == 4 and cfg.n_embd == 64, "认识的键必须照常生效"
    for k in REMOVED_KEYS:
        assert not hasattr(cfg, k), f"{k} 不该出现在配置上"
    # 仍与 dataclass 的字段集合一致（没有多余属性）
    assert {f.name for f in dataclasses.fields(cfg)} == {
        f.name for f in dataclasses.fields(GPTConfig)}


def test_known_keys_are_not_silently_altered():
    """已知键必须**原样**传进去 —— 别把"过滤"做成"顺手改值"。"""
    cfg = GPTConfig.from_model_args({"n_layer": 7, "dropout": 0.3, "use_moe": True})
    assert (cfg.n_layer, cfg.dropout, cfg.use_moe) == (7, 0.3, True)


def test_plain_construction_still_rejects_unknown_keys():
    """★ 负向对照：证明上面那条测试**不是恒真的**。

    `from_model_args` 必须真的在做过滤，而不是因为 `GPTConfig` 恰好能吞掉未知键。
    如果哪天有人给 GPTConfig 加了 `**kwargs`，这条会失败 —— 那是好事，
    说明那时应当重新审视"过滤"这件事还需不需要。
    """
    with pytest.raises(TypeError):
        GPTConfig(**{"n_layer": 4, **REMOVED_KEYS})


def test_non_dict_input_raises():
    """传错类型要当场报错，别静默建出一个全默认配置。"""
    with pytest.raises(TypeError):
        GPTConfig.from_model_args([("n_layer", 4)])


def test_empty_args_gives_all_defaults():
    """空 dict 是合法输入（等价于默认配置）。"""
    assert GPTConfig.from_model_args({}) == GPTConfig()
