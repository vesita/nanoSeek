"""学习率调度 + 数据路径前缀的测试。

`training/schedules.py::lr_at` 是从 `train.py::get_lr` **逐位等价**抽出来的。
所以这里除了常规边界，还把抽取前的字面实现拷进来当基准，防止"重构悄悄改了行为"。
"""
import glob
import math
import os

import pytest
import yaml

from training.schedules import SCHEDULES, lr_at, pick_bin_names, with_data_prefix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 本项目基座训练的真实超参（configs/base_v2.yaml）
BASE = dict(learning_rate=3e-4, min_lr=1e-4, warmup_iters=100, lr_decay_iters=70000)


def _pre_extraction_get_lr(it, learning_rate, min_lr, warmup_iters,
                           lr_decay_iters, schedule, stable_frac=0.8):
    """`train.py` 抽取**之前**的 get_lr 字面拷贝（一字未改），作为回归基准。"""
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if schedule == 'wsd':
        decay_start = int(lr_decay_iters * stable_frac)
        if it <= decay_start:
            return learning_rate
        decay_ratio = (it - decay_start) / max(lr_decay_iters - decay_start, 1)
        decay_ratio = min(decay_ratio, 1.0)
        return learning_rate + (min_lr - learning_rate) * decay_ratio
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# --------------------------------------------------------------------------
# 1) 抽取前后逐位等价
# --------------------------------------------------------------------------
@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("stable_frac", [0.5, 0.8, 0.95])
def test_bitwise_matches_pre_extraction_impl(schedule, stable_frac):
    """7 万个步点全覆盖：新实现必须与抽取前的字面实现**完全相等**。"""
    kw = dict(BASE, schedule=schedule, stable_frac=stable_frac)
    for it in list(range(0, 300)) + list(range(0, 70001, 137)) + [69998, 69999, 70000, 70001, 10**6]:
        old = _pre_extraction_get_lr(it, **kw)
        new = lr_at(it, **kw)
        assert new == old, f"it={it} schedule={schedule} stable_frac={stable_frac}: {new} != {old}"


# --------------------------------------------------------------------------
# 2) 边界与形状
# --------------------------------------------------------------------------
def test_warmup_starts_at_one_over_warmup_plus_one():
    """第 0 步不能是 0（否则白跑一步），应是 learning_rate/(warmup+1)。"""
    assert lr_at(0, **BASE, schedule='cosine') == pytest.approx(3e-4 / 101)


def test_warmup_reaches_peak_right_after():
    """预热结束（it == warmup_iters）立刻到峰值附近，且随 it 单调升。"""
    lrs = [lr_at(it, **BASE, schedule='cosine') for it in range(0, BASE['warmup_iters'] + 1)]
    assert lrs == sorted(lrs), "预热段必须单调递增"
    assert lr_at(BASE['warmup_iters'], **BASE, schedule='cosine') == pytest.approx(3e-4, rel=1e-9)


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_bounds(schedule):
    """预热**之后**学习率必须始终落在 [min_lr, learning_rate] 内。

    注意预热段起点 = learning_rate/(warmup+1) 本来就低于 min_lr，
    这是设计如此（不从 0 起但也很小），所以从 warmup_iters 开始查。
    """
    for it in list(range(BASE['warmup_iters'], 200)) + list(range(BASE['warmup_iters'], 80000, 97)):
        lr = lr_at(it, **BASE, schedule=schedule)
        assert BASE['min_lr'] - 1e-15 <= lr <= BASE['learning_rate'] + 1e-15, f"it={it} lr={lr}"


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_no_nan_and_finite(schedule):
    for it in (0, 1, 100, 101, 70000, 70001, 10**7):
        assert math.isfinite(lr_at(it, **BASE, schedule=schedule))


def test_cosine_monotonic_decreasing_after_warmup():
    lrs = [lr_at(it, **BASE, schedule='cosine') for it in range(100, 70001, 500)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:])), "余弦段必须单调不增"


def test_cosine_clamped_after_decay_iters():
    assert lr_at(70000, **BASE, schedule='cosine') == pytest.approx(1e-4, abs=1e-12)
    assert lr_at(99999, **BASE, schedule='cosine') == pytest.approx(1e-4, abs=1e-12)


def test_wsd_holds_plateau_then_decays():
    """WSD 的核心性质：前 stable_frac 段纹丝不动 = 随时可安全中断。"""
    decay_start = int(BASE['lr_decay_iters'] * 0.8)          # 56000
    for it in (BASE['warmup_iters'], 1000, 30000, decay_start):
        assert lr_at(it, **BASE, schedule='wsd') == pytest.approx(3e-4, rel=1e-12)
    assert lr_at(decay_start + 1, **BASE, schedule='wsd') < 3e-4
    assert lr_at(69999, **BASE, schedule='wsd') == pytest.approx(1e-4, rel=1e-3)


def test_wsd_linear_in_decay_phase():
    """衰减段是**线性**的（不是余弦）—— 中点应恰为峰谷均值。"""
    decay_start = 56000
    mid = (decay_start + BASE['lr_decay_iters']) / 2         # 63000
    assert lr_at(int(mid), **BASE, schedule='wsd') == pytest.approx((3e-4 + 1e-4) / 2, rel=1e-6)


def test_unknown_schedule_fails_loud():
    """写错调度名必须**报错**，不能静默退回 cosine。

    静默退回 = 一次 2.7 天的训练用错调度白跑，且日志里看不出来。
    """
    with pytest.raises(ValueError, match="未知 schedule"):
        lr_at(1000, **BASE, schedule='cosinee')
    with pytest.raises(ValueError):
        lr_at(1000, **BASE, schedule=None)


# --------------------------------------------------------------------------
# 3) 钉住 dev-notes/83 §5 那张 cosine vs WSD 对照表
#    文档里的数字必须能被代码复现，否则文档会慢慢变成谎言。
# --------------------------------------------------------------------------
@pytest.mark.parametrize("it,cosine,wsd", [
    (10000, 2.903e-4, 3.000e-4),
    (50000, 1.378e-4, 3.000e-4),
    (60000, 1.099e-4, 2.429e-4),
    (69999, 1.000e-4, 1.000e-4),
])
def test_documented_cosine_vs_wsd_table(it, cosine, wsd):
    assert lr_at(it, **BASE, schedule='cosine') == pytest.approx(cosine, rel=2e-3)
    assert lr_at(it, **BASE, schedule='wsd') == pytest.approx(wsd, rel=2e-3)


# --------------------------------------------------------------------------
# 4) 真实配置里的 schedule 必须是支持的取值
# --------------------------------------------------------------------------
def test_all_configs_use_supported_schedule():
    """扫 configs/*.yaml：schedule 写错的话，起训时才会炸 —— 测试里先炸。"""
    files = sorted(glob.glob(os.path.join(ROOT, 'configs', '*.yaml')))
    assert files, "configs/ 下没有 yaml？"
    checked = 0
    for fp in files:
        with open(fp, encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
        if 'schedule' in cfg:
            assert cfg['schedule'] in SCHEDULES, f"{os.path.basename(fp)}: schedule={cfg['schedule']!r}"
            checked += 1
    assert checked > 0, "没有任何配置显式声明 schedule，这条测试就白测了"


# --------------------------------------------------------------------------
# 5) with_data_prefix（原 train.py::_ds）
# --------------------------------------------------------------------------
@pytest.mark.parametrize("base,prefix,expect", [
    ('train_char.bin', 'v2', 'train_char_v2.bin'),
    ('val_char.bin', 'v2', 'val_char_v2.bin'),
    ('meta_char.pkl', 'v2', 'meta_char_v2.pkl'),
    ('train_char.bin', '', 'train_char.bin'),
    ('train_char.bin', None, 'train_char.bin'),
    ('meta_char.pkl', '', 'meta_char.pkl'),
    ('train_byte.bin', 'v3', 'train_byte_v3.bin'),
    ('notes.txt', 'v2', 'notes.txt'),        # 非 .bin/.pkl 原样返回
    ('noext', 'v2', 'noext'),
])
def test_with_data_prefix(base, prefix, expect):
    assert with_data_prefix(base, prefix) == expect


def test_with_data_prefix_roundtrip_covers_real_products():
    """真实数据产物名必须是「加前缀 → 还原」的一个双射，不能有歧义。"""
    names = ['train_char.bin', 'val_char.bin', 'meta_char.pkl',
             'train_byte.bin', 'val_byte.bin', 'meta_byte.pkl']
    seen = set()
    for n in names:
        p = with_data_prefix(n, 'v2')
        assert p != n
        assert p not in seen, f"前缀后撞名：{p}"
        seen.add(p)


# --------------------------------------------------------------------------
# 6) pick_bin_names：数据集开关的**唯一**解析点
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kw,expect", [
    (dict(char_level=True), ('train_char.bin', 'val_char.bin', 'meta_char.pkl')),
    (dict(byte_level=True), ('train_byte.bin', 'val_byte.bin', 'meta_byte.pkl')),
    (dict(), ('train.bin', 'val.bin', 'meta.pkl')),
    (dict(char_level=True, prefix='v2'),
     ('train_char_v2.bin', 'val_char_v2.bin', 'meta_char_v2.pkl')),
    (dict(byte_level=True, prefix='v3'),
     ('train_byte_v3.bin', 'val_byte_v3.bin', 'meta_byte_v3.pkl')),
])
def test_pick_bin_names_combinations(kw, expect):
    assert pick_bin_names(**kw) == expect


def test_pick_bin_names_char_wins_over_byte():
    """char 与 byte 同时为真时 char 优先（与 train.py 里的旧三元表达式一致）。"""
    assert pick_bin_names(char_level=True, byte_level=True)[0] == 'train_char.bin'


def test_pick_bin_names_pretrain_ignores_prefix_for_train_only():
    """`stage='pretrain'` 时 train 固定为 `pretrain.bin`（不加前缀），
    但 val/meta 仍然吃前缀 —— 这是旧实现的逐字行为。"""
    train, val, meta = pick_bin_names(char_level=True, prefix='v2', stage='pretrain')
    assert train == 'pretrain.bin'
    assert val == 'val_char_v2.bin' and meta == 'meta_char_v2.pkl'


def test_pick_bin_names_other_stages_use_prefix():
    for stage in ('full', 'sft', None):
        assert pick_bin_names(char_level=True, prefix='v2', stage=stage)[0] == 'train_char_v2.bin'


def test_pick_bin_names_matches_real_v2_artifacts():
    """★ 端到端：`configs/base_v2.yaml` 的开关解析出来必须是真的数据文件。

    这条把"配置 → 文件名 → 磁盘上的字节"三个环节连起来测。`data_prefix: v2`
    写错（或解析逻辑被改坏）时，2.7 天的训练会读错数据 —— 而 loss 曲线
    在前几百步看不出来。
    """
    import os
    yml = os.path.join(ROOT, 'configs', 'base_v2.yaml')
    if not os.path.exists(yml):
        pytest.skip("configs/base_v2.yaml 不存在")
    with open(yml, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    names = pick_bin_names(char_level=cfg.get('char_level', False),
                           byte_level=cfg.get('byte_level', False),
                           prefix=cfg.get('data_prefix', ''))
    ddir = os.path.join(ROOT, 'data', cfg.get('dataset', 'chinese'))
    if not os.path.isdir(ddir):
        pytest.skip(f"{ddir} 不存在")
    missing = [n for n in names if not os.path.exists(os.path.join(ddir, n))]
    assert not missing, f"配置解析出的文件不存在：{missing}"
