"""训练调度与路径拼接的**纯函数**（从 train.py 抽出，可单元测试）。

## 为什么要单独一个模块
`training/train.py` 是「模块级脚本」——1482 行全部在模块顶层执行，**import 它
就等于直接开始训练**。所以写在里面的任何函数（`get_lr`、`_ds` …）都无法被
pytest 覆盖：测试只能靠起子进程跑完整训练，不可能。

这里把两类**无副作用、只依赖入参**的逻辑抽出来：
- `lr_at`             学习率调度（warmup + cosine / WSD）
- `with_data_prefix`  数据集文件名加前缀（'' / 'v2' / …）

`train.py` 里的 `get_lr(it)` / `_ds(base)` 退化成「读全局 → 转调这里」的薄包装，
**行为逐位不变**（tests/test_schedules.py 有与手算值的回归对照）。
"""
import math

__all__ = ['lr_at', 'with_data_prefix', 'pick_bin_names', 'SCHEDULES']

#: 支持的调度名。写错要**报错**而不是静默退回 cosine —— 静默退回会让一次
#: 2.7 天的训练白跑（本项目已有过多起"配置写错但没人报错"的事故）。
SCHEDULES = ('cosine', 'wsd')


def lr_at(it, *, learning_rate, min_lr, warmup_iters, lr_decay_iters,
          schedule='cosine', stable_frac=0.8):
    """返回第 `it` 步的学习率。

    参数
    ----
    it              当前步（0-based）
    learning_rate   峰值学习率
    min_lr          最低学习率
    warmup_iters    线性预热步数
    lr_decay_iters  衰减总步数
    schedule        'cosine'（余弦退火）| 'wsd'（warmup-stable-decay）
    stable_frac     WSD 稳定段占比：前 `stable_frac·lr_decay_iters` 步保持峰值

    三段结构（两种调度共有前两段）：
      1. `it < warmup_iters`：线性预热，从 `learning_rate/(warmup_iters+1)` 起
      2. WSD：稳定段恒为 `learning_rate`，之后线性降到 `min_lr`
         cosine：余弦从 `learning_rate` 降到 `min_lr`
      3. 超出 `lr_decay_iters`：夹在 `min_lr`（cosine 分支显式夹；WSD 靠
         `min(decay_ratio, 1.0)` 夹）

    返回值恒在 `[min_lr, learning_rate]` 内（预热段可能更低，属预期）。
    """
    if schedule not in SCHEDULES:
        raise ValueError(f"未知 schedule={schedule!r}，可选 {SCHEDULES}")

    # 1) 线性预热
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)

    # 2) WSD：稳定段 + 末段线性衰减
    if schedule == 'wsd':
        decay_start = int(lr_decay_iters * stable_frac)
        if it <= decay_start:
            return learning_rate
        decay_ratio = (it - decay_start) / max(lr_decay_iters - decay_start, 1)
        decay_ratio = min(decay_ratio, 1.0)
        return learning_rate + (min_lr - learning_rate) * decay_ratio

    # 3) 余弦退火
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1, f"cosine decay_ratio 越界：{decay_ratio}"
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def with_data_prefix(base, prefix):
    """把 `'train_char.bin'` + prefix `'v2'` → `'train_char_v2.bin'`。

    只处理 `.bin` / `.pkl` 两种扩展名（本项目数据产物的全部形态）；
    `prefix` 为空时原样返回，其他扩展名也原样返回。
    """
    if not prefix:
        return base
    for ext in ('.bin', '.pkl'):
        if base.endswith(ext):
            return base[:-len(ext)] + f'_{prefix}' + ext
    return base


def pick_bin_names(char_level=False, byte_level=False, prefix='', stage=None):
    """返回 `(train_bin, val_bin, meta_bin)` 三个**文件名**（不含 data_dir）。

    这是「用哪份数据」的**唯一**解析点。此前那个
    `'train_char.bin' if char_level else 'train_byte.bin' if byte_level else 'train.bin'`
    三元表达式在 `train.py` 里重复了 5 遍（train / val / meta / summary / 一行死代码），
    加一个新模式要改 5 个地方 —— 漏改任何一处都会让"换数据集"**静默失效**
    （训练照跑，只是还在读旧文件）。现在收敛成一个可单测的函数。

    `stage='pretrain'` 时 train 固定为 `'pretrain.bin'`（不分级、不加前缀），
    与旧行为逐字一致。
    """
    if char_level:
        train, val, meta = 'train_char.bin', 'val_char.bin', 'meta_char.pkl'
    elif byte_level:
        train, val, meta = 'train_byte.bin', 'val_byte.bin', 'meta_byte.pkl'
    else:
        train, val, meta = 'train.bin', 'val.bin', 'meta.pkl'
    if stage == 'pretrain':
        return 'pretrain.bin', with_data_prefix(val, prefix), with_data_prefix(meta, prefix)
    return (with_data_prefix(train, prefix),
            with_data_prefix(val, prefix),
            with_data_prefix(meta, prefix))
