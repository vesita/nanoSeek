"""项目布局与配置的**不变量**测试：把运维铁律从"文档里的教训"变成"跑不过就红"。

## 为什么需要这一类测试
本项目最贵的错误不是算法错，而是**配置/布局错**：
- 配置放在 `out_dir` 里 → 被 `_backup_old_run` 当作旧实验产物挪进 `old/`
- 全局变量定义在 `config_keys` 快照**之后** → YAML 里的值被静默改回默认
- `data_prefix` 没生效 → 换了数据集但训练照跑，白烧两天 GPU

这三条都已经在文档里写过，但文档拦不住手滑。下面的测试用 AST 静态分析
`train.py`，让这些错误在 `pytest` 阶段就暴露，而不是在 65 小时后。
"""
import ast
import glob
import os
import pathlib

import pytest

from training.schedules import pick_bin_names

ROOT = pathlib.Path(__file__).resolve().parent.parent
TRAIN_PY = ROOT / 'training' / 'train.py'
CONFIGS = sorted(glob.glob(str(ROOT / 'configs' / '*.yaml')))


def load_yaml(path):
    from model.config_loader import _load_yaml_with_inheritance
    return _load_yaml_with_inheritance(path)


def train_py_names_before(marker_name):
    """返回 `train.py` 里在 `marker_name` 这行**之前**被赋值的顶层名字集合。

    只看 `tree.body`（模块顶层语句），因为 `config_keys`/`load_config` 也只看得见
    模块级全局。`marker_name` 既可以是赋值目标（`config_keys = [...]`），
    也可以是函数调用（`load_config(globals())`）。
    """
    tree = ast.parse(TRAIN_PY.read_text(encoding='utf-8'))
    marker = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, 'id', None) == marker_name for t in node.targets):
            marker = node.lineno
            break
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and getattr(node.value.func, 'id', None) == marker_name):
            marker = node.lineno
            break
    assert marker is not None, f"train.py 里找不到 {marker_name}（赋值或调用）"
    names = set()
    for node in tree.body:
        if node.lineno >= marker:
            continue
        if isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names, marker


def test_configs_exist():
    assert CONFIGS, "configs/*.yaml 一个都没有？"


# ==========================================================================
# 1) ★ 配置不能放在自己的 out_dir 里
# ==========================================================================
@pytest.mark.parametrize("cfg_path", CONFIGS, ids=os.path.basename)
def test_config_file_lives_outside_its_own_out_dir(cfg_path):
    """★ 运维铁律 3：`_backup_old_run` 会把 out_dir 里**所有**文件挪进 `old/`。

    `init_from != 'resume'` 时（`scratch` 就是）train.py 会无条件归档 out_dir。
    如果配置文件本身放在 out_dir 里，第一次启动它就被挪走 —— 之后想复现
    那次训练，配置已经不在原处了。本项目真的发生过一次。
    """
    cfg = load_yaml(cfg_path)
    out_dir = cfg.get('out_dir')
    if not out_dir:
        pytest.skip("该配置没写 out_dir")
    cfg_abs = pathlib.Path(cfg_path).resolve()
    out_abs = (ROOT / out_dir).resolve() if not os.path.isabs(out_dir) else pathlib.Path(out_dir)
    assert not cfg_abs.is_relative_to(out_abs), (
        f"配置文件 {cfg_abs} 位于它自己的 out_dir {out_abs} 内 —— "
        f"`_backup_old_run` 会把它挪进 old/")


# ==========================================================================
# 2) ★ 每个配置键都必须是「可覆盖的全局」
# ==========================================================================
@pytest.mark.parametrize("cfg_path", CONFIGS, ids=os.path.basename)
def test_every_config_key_is_overridable(cfg_path):
    """★ 运维铁律 4：配置键必须在 `load_config(globals())` **之前**定义。

    `config_loader` 的逻辑是 `if k in g: 覆盖 else: 注入`。所以一个定义在
    `load_config` **之后**的名字，会先被"注入"、随后又被那句 `name = 默认值`
    静默覆盖 —— 配置看起来生效了（加载日志会打印它），实际训练用的是默认值。

    `data_prefix` 和 `gradient_checkpointing` 都栽在这上面。
    """
    overridable, marker = train_py_names_before('load_config')
    cfg = load_yaml(cfg_path)
    missing = sorted(set(cfg) - overridable)
    assert not missing, (
        f"{os.path.basename(cfg_path)} 的这些键在 train.py 第 {marker} 行"
        f"（load_config）之后才定义，会被静默改回默认值：{missing}")


@pytest.mark.parametrize("cfg_path", CONFIGS, ids=os.path.basename)
def test_every_config_key_is_in_the_logged_snapshot(cfg_path):
    """配置键还必须在 `config_keys` 快照之前定义，否则不会被写进日志/checkpoint。

    `config_keys` 只收 `int/float/bool/str` 类型的全局，所以 list/None 类型的键
    （如 `no_attn_layers`、`kv_memory_layers`）本来就不在里面 —— 这类键
    **跳过**，只查标量键。
    """
    snapshot, marker = train_py_names_before('config_keys')
    cfg = load_yaml(cfg_path)
    # 只查 config_keys 会收的类型（与 train.py 里的 isinstance 过滤保持一致）
    scalar = {k: v for k, v in cfg.items() if isinstance(v, (int, float, bool, str))}
    missing = sorted(set(scalar) - snapshot)
    assert not missing, (
        f"{os.path.basename(cfg_path)} 的这些标量键在 train.py 第 {marker} 行"
        f"（config_keys 快照）之后：{missing}")


# ==========================================================================
# 3) ★ 新基座配置必须与 PROJECT_STATE §5 记录的一致
# ==========================================================================
def test_base_v2_matches_documented_decisions():
    """把 PROJECT_STATE §5 那张表变成断言 —— 文档与配置不允许悄悄分叉。

    这张表是「基座配方」的唯一依据（每一项都有实测支撑，见 §2.3）。
    配置被改动而文档没改（或反过来）都会让下一个人按错误的前提做决定。
    """
    path = ROOT / 'configs' / 'base_v2.yaml'
    if not path.exists():
        pytest.skip("configs/base_v2.yaml 不存在")
    cfg = load_yaml(path)
    expected = {
        'data_prefix': 'v2',
        'out_dir': 'out/base_v2',
        # 2026-09-11 暂停在 step 22000 后由 'scratch' 改成 'resume'，这是**有意**的：
        # 'resume' 是更安全的重启默认 —— 若写 'scratch'，误用 §0.4 那条（不带
        # --init_from 的）重启命令会触发 `_backup_old_run` 把整个 run 静默移进 old/。
        # 本断言的作用正是逼着"文档和配置一起改"，所以以后要改这个值，
        # **必须同时改 PROJECT_STATE §5 那张表**，不要只改这里。
        'init_from': 'resume',
        'batch_size': 4,
        'gradient_accumulation_steps': 8,
        'use_mhc': False,
        'use_mtp': False,
        'swiglu_clamp': 10.0,
        'muon_ns_steps': 7,
        'muon_ns_aggressive': 4,
        'lr_decay_iters': 70000,
        'ndb_store': '',
        'ndb_heldout': '',
    }
    wrong = {k: (cfg.get(k, '<缺失>'), v) for k, v in expected.items() if cfg.get(k, '<缺失>') != v}
    assert not wrong, f"配置与 PROJECT_STATE §5 不符（键: (实际, 期望)）：{wrong}"


def test_base_v2_decay_covers_full_run():
    """退火必须覆盖整个训练：`lr_decay_iters` == `max_iters`。

    旧基线是 30000/70000 —— 退火在 30000 结束、之后 40000 步平在 min_lr，
    PROJECT_STATE §5 明确列为要修的项。
    """
    path = ROOT / 'configs' / 'base_v2.yaml'
    if not path.exists():
        pytest.skip("configs/base_v2.yaml 不存在")
    cfg = load_yaml(path)
    assert cfg['lr_decay_iters'] == cfg['max_iters'], \
        f"退火 {cfg['lr_decay_iters']} != 总步数 {cfg['max_iters']}"


# ==========================================================================
# 4) ★ 配置指向的数据文件必须真的存在
# ==========================================================================
@pytest.mark.parametrize("cfg_path", CONFIGS, ids=os.path.basename)
def test_config_data_files_resolve(cfg_path):
    """`data_prefix` + 级别开关解析出来的三个文件名必须存在。

    这条直接保护"白烧两天 GPU"的场景：`data_prefix: v2` 写错成 `v3` 时，
    训练会在**启动时**就报 `FileNotFoundError`（train.py 有 exists 检查），
    但 `meta` 文件缺失只是让 vocab 回退 —— 这条测试把它也拦住。
    """
    cfg = load_yaml(cfg_path)
    dataset = cfg.get('dataset', 'chinese')
    prefix = cfg.get('data_prefix', '')
    train, val, meta = pick_bin_names(
        char_level=cfg.get('char_level', False),
        byte_level=cfg.get('byte_level', False),
        prefix=prefix, stage=cfg.get('stage'))
    ddir = ROOT / 'data' / dataset
    missing = [n for n in (train, val, meta) if not (ddir / n).exists()]
    if missing and not ddir.exists():
        pytest.skip(f"data/{dataset} 不存在（换机器/未准备数据）")
    assert not missing, (
        f"{os.path.basename(cfg_path)} 解析出的文件在 data/{dataset}/ 下不存在：{missing}"
        f"（现有 v2 文件：{sorted(p.name for p in ddir.glob('*_v2.*'))}）")


def test_pick_bin_names_drives_get_batch(tmp_path):
    """`get_batch` 的数据路径必须来自 `pick_bin_names`，不能有第二处硬编码。

    这条用 AST 查 `train.py` 里是否残留 `_ds(`/硬编码的三元表达式 ——
    之前那个 `'train_char.bin' if char_level else ...` 重复了 5 遍，
    正是"改一处漏一处"的温床。
    """
    src = TRAIN_PY.read_text(encoding='utf-8')
    tree = ast.parse(src)
    # 收集所有字符串常量里出现的裸数据文件名
    data_names = {'train.bin', 'val.bin', 'meta.pkl', 'train_char.bin', 'val_char.bin',
                  'meta_char.pkl', 'train_byte.bin', 'val_byte.bin', 'meta_byte.pkl'}
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value in data_names:
            hits.append((node.lineno, node.value))
    assert not hits, (
        f"train.py 里仍有硬编码的数据文件名（应当只由 training/schedules.py::"
        f"pick_bin_names 解析）：{sorted(hits)}")


# ==========================================================================
# 5) gradient_checkpointing 是空开关（已知、已文档化）
# ==========================================================================
def test_gradient_checkpointing_is_a_noop_under_aux_free_moe():
    """★ 记录一个**空开关**：`use_moe + use_aux_free_balance` 时梯度检查点被强制关闭。

    见 `model/gpt.py:167`：aux-free 的 `router_bias` 在 forward 里就地更新
    （no_grad 副作用），与 checkpoint 的 recompute 不兼容（重算时 bias 已变 →
    路由结果不一致 → CheckpointError），所以自动降级。

    后果：配置里写 `gradient_checkpointing: true` **不会**省显存，
    而本项目曾用两个 A/B 臂比较"开/关检查点" —— 那是**空测试**
    （两臂其实完全一样），结论已撤回。

    这条测试从源码里核实短路条件仍然存在。如果哪天短路被去掉，
    这个断言会红，提醒你"空测试"这个坑又要回来了。
    """
    src = (ROOT / 'model' / 'gpt.py').read_text(encoding='utf-8')
    assert 'use_ckpt = False' in src, "gpt.py 里找不到梯度检查点的短路逻辑"
    # 短路条件必须仍然包含 use_aux_free_balance
    idx = src.index('use_ckpt = False')
    window = src[max(0, idx - 400):idx]
    assert 'use_aux_free_balance' in window, "短路条件变了：不再依赖 aux-free"
    assert 'use_moe' in window


# ==========================================================================
# 6) 测试卫生：不允许 import train.py
# ==========================================================================
def test_no_test_imports_train_py():
    """`training/train.py` 是模块级脚本，**import 它就会直接开训**。

    所以测试里绝不能 import 它。要测的纯函数在 `training/schedules.py`
    （以及 `training/masking.py`）。这条防止后来者无意中把测试变成"启动训练"。
    """
    offenders = []
    for tp in (ROOT / 'tests').glob('test_*.py'):
        tree = ast.parse(tp.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name in ('training.train', 'train'):
                        offenders.append((tp.name, node.lineno, a.name))
            elif isinstance(node, ast.ImportFrom):
                if node.module in ('training.train', 'train'):
                    offenders.append((tp.name, node.lineno, f"from {node.module}"))
    assert not offenders, f"测试里 import 了 train.py（会直接开训）：{offenders}"
