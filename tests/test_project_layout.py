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
        # 2026-09-11 由 true 改成 false（全量语料预训练）。这是**有意**的路线切换，
        # 不是笔误：masking=true 时只有 6.14% 的语料能产生梯度（c4_zh 一个终止符都没有）。
        # 详见 PROJECT_STATE §0.5 与 configs/base_v2.yaml 里那段注释。
        'use_loss_masking': False,
        # ★ NDB 是训练的默认组件（2026-09-15 用户定）。采用的方案是 `model/ngram_ndb.py`：
        # 读与写都由模型门控、表在训练中**在线**累积 ⇒ 没有 `ndb_store` 这类离线库文件。
        # 超参取自离线 A/B 探针验过的那套（`scripts/ndb_online_ab.py` 的默认值）。
        'ndb_slots': 268435456,
        'ndb_levels': '8',
        'ndb_top_k': 1,
        'ndb_lr': 0.001,
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


# ==========================================================================
# 7) ★ v3 三阶段配方的不变量（2026-09-13 新增）
# ==========================================================================
V3_STAGE_CONFIGS = sorted(glob.glob(str(ROOT / 'configs' / 'base_v3_*.yaml')))

# 每段的步数在这里再写一遍，逼着"改配置就必须同时改这里"（和 §5 配方表同一个套路）。
# 依据 PROJECT_STATE §0.5.10：
#   - A 段 `base_v3_know.yaml`（3k，v3_know 知识/CoT）**已被用户 2026-09-13 拍板跳过**
#     —— 它唯一能实测的理由"冲刷 <eos> 先验"被 `scripts/eos_prior_probe.py` 推翻。
#     ★ **配置保留**（配方不删，将来补知识段仍用它），所以步数继续登记在这里；
#       但它不再是 B 段的前置条件。
#   - B 段 `base_v3_dlg.yaml`（14k = 1 epoch 对话专修）**直接接 `out/base_v2/last.pt`**。
#   - 人格层 `base_v3_persona.yaml`（**单流 `<resp>` 格式**，接 `out/base_v3_dlg/last.pt`）
#     2026-09-14 新增。★ 它有三处**有意的例外**，都写在下面对应断言旁边：
#     ① 必须开 `use_loss_masking`（+ `mask_mode: resp_span`）；
#     ② `data_prefix == 'v3_persona'`；
#     ③ 步数在 2026-09-15 由**冒烟 300 步**改成正式配方 **900 步**（语料已补到 809,598
#        token ⇒ 98.8 步/epoch，900 步 = 9.1 epoch）。
#   - ★ 2026-09-15 新增两段，计划是 `persona(1) → 通识 → persona(2)`：
#     `base_v3_know2.yaml`（18,000 步 = v3_know 的 **0.302 epoch**，接 persona1）
#     与 `base_v3_persona2.yaml`（900 步，接 know2，**最后一段 ⇒ 最终人设由它决定**）。
#     ⚠ `base_v3_know.yaml`（A 段，3k，接 base_v2）**仍然保留**、仍然登记——
#       它是"补知识段"的备用配方，与新加的 `base_v3_know2.yaml`（接 persona1）不是一回事。
V3_STAGE_STEPS = {
    'base_v3_know.yaml': 3000,
    'base_v3_dlg.yaml': 14000,
    'base_v3_persona.yaml': 900,
    'base_v3_know2.yaml': 18000,
    'base_v3_persona2.yaml': 900,
}


def test_v3_stage_configs_exist():
    """三阶段配方文件必须存在（否则下面那些参数化测试会静默变成 0 项、恒真通过）。"""
    assert V3_STAGE_CONFIGS, "找不到 configs/base_v3_*.yaml —— 三阶段配方没落地"


def test_v3_stage_configs_enable_ndb():
    """★ **NDB 是训练的默认组件**（2026-09-15 用户定）⇒ 每个 v3 阶段解析后都必须启用它。

    这条把"要记得启用 NDB"变成断言。动机是本项目反复踩的同一类坑：
    一个**看起来配好了、实际没生效**的开关（`keep_step_ckpts`、`gradient_checkpointing`、
    `use_neural_db` 都是），跑完几十小时才发现 NDB 根本没开。

    判据用**解析 `extends` 之后**的 `ndb_slots` 非 0 —— 它是 `train.py` 里唯一的启用判据
    （`use_ndb = ndb_slots > 0`，见 train.py 顶部那族 `ndb_*` 键）。
    阶段配置通常**不写**这些键、靠 `base_v2.yaml` 继承，所以必须查解析后的值：
    查原文件会把"正确继承"误判成"没启用"。
    """
    missing = []
    for cfg_path in V3_STAGE_CONFIGS:
        if int(load_yaml(cfg_path).get('ndb_slots', 0)) <= 0:
            missing.append(os.path.basename(cfg_path))
    assert not missing, (
        f"这些 v3 阶段配置解析后 NDB 是关闭的（`ndb_slots <= 0`）：{missing}\n"
        f"★ NDB 现在是训练的默认组件：继承 `configs/base_v2.yaml` 即可；\n"
        f"  只有确实要**关掉**它的实验才显式写 `ndb_slots: 0` 并写明理由。")


def test_all_config_out_dirs_are_pairwise_distinct():
    """所有配置（含 base_v2）的 out_dir 必须互不相同。

    两个 run 共用一个 out_dir ⇒ 后启动的会触发 `_backup_old_run` 把前一个的产物
    整体挪进 `old/`。
    """
    seen = {}
    for cfg_path in CONFIGS:
        cfg = load_yaml(cfg_path)
        out_dir = os.path.abspath(cfg.get('out_dir', ''))
        name = os.path.basename(cfg_path)
        assert out_dir not in seen, (
            f"{name} 与 {seen[out_dir]} 共用 out_dir={cfg.get('out_dir')!r} —— "
            f"后启动的那个会触发 _backup_old_run 把前一个挪进 old/")
        seen[out_dir] = name


@pytest.mark.parametrize("cfg_path", V3_STAGE_CONFIGS, ids=os.path.basename)
def test_v3_stage_config_safety(cfg_path):
    """★ v3 分阶段配方的五条硬不变量。

    这条测试的存在理由是一次**真实险情**：方案文档（PROJECT_STATE §0.5.10）里那串参数
    只列了 `--init_from=out/base_v2/last.pt --data-prefix v3_know …`，**没写 `--out_dir`**。
    而 `train.py:1022` 在 `init_from != 'resume'` 时会对 out_dir 调 `_backup_old_run()` ——
    warm start 用的正是 `<路径>.pt` 这种形式，**不是** 'resume'。
    照抄文档命令 + `configs/base_v2.yaml`（out_dir=out/base_v2）
    ⇒ 基座 61000 步的全部归档 ckpt（**含 last.pt 自己**）会被静默挪进
    `out/base_v2/old/`，新旧两个 run 的产物混在同一目录。
    文档拦不住手滑，这条断言可以。
    """
    cfg = load_yaml(cfg_path)
    name = os.path.basename(cfg_path)

    # (1) 权重来源：必须是 warm start（<路径>.pt），不能是 resume/scratch。
    init_from = cfg['init_from']
    assert init_from not in ('resume', 'scratch'), (
        f"{name}: init_from={init_from!r} —— 本阶段是 warm start；"
        f"用 resume 会把 base_v2 的 WSD 进度带进来（只剩 9000 步就退到 min_lr）")
    assert init_from.endswith('.pt'), f"{name}: init_from 必须是 <路径>.pt"

    # (2) ★ 核心：out_dir 不能是权重来源所在目录（_backup_old_run 会归档它）。
    out_dir = os.path.abspath(cfg['out_dir'])
    src_dir = os.path.abspath(os.path.dirname(init_from))
    assert out_dir != src_dir, (
        f"{name}: out_dir 与 init_from 所在目录相同（{cfg['out_dir']}）—— "
        f"warm start 会触发 _backup_old_run 把该目录下的 ckpt 全部挪进 old/")

    # (3) 打包：必须开打包，且必须是全域随机窗口（块对齐有 67~73% 覆盖漏洞）。
    assert cfg['use_doc_packing'] is True, f"{name}: 必须 use_doc_packing=True"
    assert cfg['pack_align'] is False, (
        f"{name}: pack_align 必须 False —— True 时 v3_know 有 72.91% 的 train token "
        f"永远进不了任何窗口（PROJECT_STATE §0.5.12）")

    # (4) loss masking：默认**必须关**（v3_lang 一个终止符都没有；v3_dlg 全 token 等权）。
    #     ★★ 例外（2026-09-14，人格层）：单流 `<resp>` 语料的**全部**监督信号都在
    #        `<resp>…<eos>` 区间里，关掉 masking = 零梯度 ⇒ 必须开。但例外**绑在
    #        `mask_mode` 上、不绑在文件名上**，并且要求语料前缀就是人格层 ——
    #        别的阶段写 `resp_span` 会在这里被拦下（那说明配错了语料：resp_span 只对
    #        单流格式成立，对 `A：/B：` 语料会给出全 False 的 mask ⇒ loss NaN）。
    mask_mode = cfg.get('mask_mode', 'eos_line')
    assert mask_mode in ('eos_line', 'resp_span'), (
        f"{name}: mask_mode={mask_mode!r} 不是已知模式（train.py 只认 eos_line/resp_span）")
    if mask_mode == 'resp_span':
        assert cfg['data_prefix'] == 'v3_persona', (
            f"{name}: mask_mode=resp_span 只对单流 <resp> 语料成立，"
            f"但 data_prefix={cfg['data_prefix']!r} —— 配错语料会让 mask 全 False ⇒ loss NaN")
        assert cfg['use_loss_masking'] is True, (
            f"{name}: resp_span 必须配 use_loss_masking=True（否则零梯度）")
    else:
        assert cfg['use_loss_masking'] is False, f"{name}: use_loss_masking 必须 False"

    # (5) 退火覆盖整段 + 语料前缀 + 步数与方案表一致。
    assert cfg['lr_decay_iters'] == cfg['max_iters'], (
        f"{name}: lr_decay_iters={cfg['lr_decay_iters']} != max_iters={cfg['max_iters']}")
    assert cfg['data_prefix'].startswith('v3_'), (
        f"{name}: data_prefix={cfg['data_prefix']!r} 不是 v3 阶段数据")
    assert name in V3_STAGE_STEPS, (
        f"{name}: 新增的 v3 阶段配置必须同时登记进 V3_STAGE_STEPS"
        f"（并更新 PROJECT_STATE §0.5.10 的方案表）")
    assert cfg['max_iters'] == V3_STAGE_STEPS[name], (
        f"{name}: max_iters={cfg['max_iters']} 与登记的 {V3_STAGE_STEPS[name]} 不符 —— "
        f"改步数必须同时改这里和 PROJECT_STATE §0.5.10 的方案表")
