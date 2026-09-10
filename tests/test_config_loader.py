"""`model/config_loader.py` 的测试。

## 为什么值得测
这个模块是**唯一**的配置入口，`train.py` 的 137 个全局变量全靠它覆盖。
它挂掉的表现是「配置没生效但训练照跑」—— 最坏的情况是白烧两天 GPU 才发现。

本项目已经在两条上栽过：
- `data_prefix` / `gradient_checkpointing` 定义在 `config_keys` 快照**之后**
  → YAML 里的值被静默改回默认。根因在 train.py 的书写顺序，
  但"类型/存在性检查"这类契约是 config_loader 的职责，值得钉死。
- `--set ndb_store=` 写出 YAML `None` → `TypeError`。这属于**预期行为**
  （类型不符就该报错），测试要把这个报错钉住，而不是让下一个人再踩。
"""
import sys
import textwrap

import pytest

from model.config_loader import _load_yaml_with_inheritance, load_config


def write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body), encoding='utf-8')
    return p


@pytest.fixture
def argv(monkeypatch):
    """替换 sys.argv，模拟命令行。"""
    def _set(*args):
        monkeypatch.setattr(sys, 'argv', ['train.py', *args])
    return _set


# ==========================================================================
# 1) YAML 加载与继承
# ==========================================================================
def test_plain_yaml(tmp_path):
    p = write(tmp_path, 'a.yaml', """
        n_layer: 4
        learning_rate: 0.001
        use_moe: true
    """)
    assert _load_yaml_with_inheritance(p) == {
        'n_layer': 4, 'learning_rate': 0.001, 'use_moe': True}


def test_empty_yaml_returns_empty_dict(tmp_path):
    p = write(tmp_path, 'empty.yaml', "")
    assert _load_yaml_with_inheritance(p) == {}


def test_child_overrides_parent(tmp_path):
    write(tmp_path, 'base.yaml', """
        n_layer: 12
        n_embd: 512
        dropout: 0.2
    """)
    child = write(tmp_path, 'child.yaml', """
        extends: base.yaml
        n_embd: 768
        batch_size: 4
    """)
    assert _load_yaml_with_inheritance(child) == {
        'n_layer': 12, 'n_embd': 768, 'dropout': 0.2, 'batch_size': 4}


def test_base_key_is_alias_for_extends(tmp_path):
    write(tmp_path, 'b.yaml', "n_layer: 6\n")
    child = write(tmp_path, 'c.yaml', "base: b.yaml\nn_head: 8\n")
    assert _load_yaml_with_inheritance(child) == {'n_layer': 6, 'n_head': 8}


def test_extends_and_base_are_stripped_from_result(tmp_path):
    """`extends`/`base` 不能泄漏成配置键 —— 否则会被注入成无意义的全局变量。"""
    write(tmp_path, 'b.yaml', "n_layer: 6\n")
    child = write(tmp_path, 'c.yaml', "extends: b.yaml\nbase: b.yaml\nn_head: 8\n")
    out = _load_yaml_with_inheritance(child)
    assert 'extends' not in out and 'base' not in out


def test_multilevel_inheritance(tmp_path):
    write(tmp_path, 'a.yaml', "n_layer: 1\nn_head: 2\nn_embd: 3\n")
    write(tmp_path, 'b.yaml', "extends: a.yaml\nn_head: 20\n")
    c = write(tmp_path, 'c.yaml', "extends: b.yaml\nn_embd: 300\n")
    assert _load_yaml_with_inheritance(c) == {'n_layer': 1, 'n_head': 20, 'n_embd': 300}


def test_inheritance_path_is_relative_to_child(tmp_path):
    """子配置在子目录里时，`extends: ../base.yaml` 要相对**子配置所在目录**解析。"""
    write(tmp_path, 'base.yaml', "n_layer: 9\n")
    sub = tmp_path / 'sub'
    sub.mkdir()
    child = sub / 'c.yaml'
    child.write_text("extends: ../base.yaml\nn_head: 8\n", encoding='utf-8')
    assert _load_yaml_with_inheritance(child) == {'n_layer': 9, 'n_head': 8}


def test_cycle_is_detected(tmp_path):
    """A extends B extends A 必须**报错**，不能无限递归到栈溢出。"""
    write(tmp_path, 'a.yaml', "extends: b.yaml\n")
    write(tmp_path, 'b.yaml', "extends: a.yaml\n")
    with pytest.raises(ValueError, match="循环"):
        _load_yaml_with_inheritance(tmp_path / 'a.yaml')


def test_self_cycle_is_detected(tmp_path):
    p = write(tmp_path, 'self.yaml', "extends: self.yaml\n")
    with pytest.raises(ValueError, match="循环"):
        _load_yaml_with_inheritance(p)


def test_non_mapping_yaml_raises(tmp_path):
    p = write(tmp_path, 'list.yaml', "- 1\n- 2\n")
    with pytest.raises(TypeError, match="必须是 YAML 映射"):
        _load_yaml_with_inheritance(p)


def test_non_string_extends_raises(tmp_path):
    p = write(tmp_path, 'bad.yaml', "extends: [a, b]\n")
    with pytest.raises(TypeError, match="必须是字符串路径"):
        _load_yaml_with_inheritance(p)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load_yaml_with_inheritance(tmp_path / 'nope.yaml')


# ==========================================================================
# 2) load_config：把配置应用到 globals()
# ==========================================================================
def test_applies_yaml_into_globals(tmp_path, argv):
    p = write(tmp_path, 'c.yaml', "n_layer: 4\nuse_moe: true\n")
    g = {'n_layer': 12, 'use_moe': False}
    argv(str(p))
    load_config(g)
    assert g['n_layer'] == 4 and g['use_moe'] is True


def test_cli_override_beats_yaml(tmp_path, argv):
    p = write(tmp_path, 'c.yaml', "n_layer: 4\n")
    g = {'n_layer': 12}
    argv(str(p), '--n_layer=8')
    load_config(g)
    assert g['n_layer'] == 8


def test_running_without_config_file_only_applies_cli(argv):
    g = {'n_layer': 12}
    argv('--n_layer=3')
    load_config(g)
    assert g['n_layer'] == 3


def test_type_mismatch_from_yaml_raises_with_key_name(tmp_path, argv):
    """YAML 写 `n_layer: '4'`（字符串）必须报错，错误信息里要有键名，方便定位。"""
    p = write(tmp_path, 'c.yaml', "n_layer: '4'\n")
    g = {'n_layer': 12}
    argv(str(p))
    with pytest.raises(TypeError, match="n_layer"):
        load_config(g)


def test_none_default_accepts_any_type(tmp_path, argv):
    """`g[k] is None` 时跳过类型检查 —— 这是 `kv_memory_layers` 这类
    Optional 参数唯一的可配置途径。改坏了它们就永远无法从 yaml 赋值。
    """
    p = write(tmp_path, 'c.yaml', "kv_memory_layers: 6\n")
    g = {'kv_memory_layers': None}
    argv(str(p))
    load_config(g)
    assert g['kv_memory_layers'] == 6


def test_none_into_non_none_global_is_rejected(tmp_path, argv):
    """默认值**不是** None 的键，不允许被设成 None。

    `None` 是"未设置"的哨兵，只对 `kv_memory_layers` 这类 Optional 参数开放。
    把 `n_layer` 或 `ndb_store` 这种变成 None 一定是写错了，必须报错。
    """
    p = write(tmp_path, 'c.yaml', "kv_memory_layers:\n")
    argv(str(p))
    with pytest.raises(TypeError, match="kv_memory_layers"):
        load_config({'kv_memory_layers': 3})


def test_none_default_stays_none(tmp_path, argv):
    p = write(tmp_path, 'c.yaml', "kv_memory_layers:\n")
    argv(str(p))
    g = {'kv_memory_layers': None}
    load_config(g)
    assert g['kv_memory_layers'] is None


def test_yaml_none_into_str_global_raises_loud(tmp_path, argv):
    """★ 真实事故：YAML 空值让全局拿到 None，而它是 str。

    正确行为是**报错**（本测试钉住），因为静默接受会让"关掉 NDB"变成
    "NDB 带着 None 路径继续跑"。正确写法是 `--ndb_store=""`（解析成空字符串）。
    """
    p = write(tmp_path, 'c.yaml', "ndb_store:\n")      # YAML 空值 → None
    argv(str(p))
    with pytest.raises(TypeError, match="ndb_store"):
        load_config({'ndb_store': ''})


def test_unknown_yaml_key_is_injected(tmp_path, argv):
    """未知键仍然注入（与旧 exec 方案行为一致），只是打一句 note。"""
    p = write(tmp_path, 'c.yaml', "totally_new_key: 42\n")
    g = {'n_layer': 12}
    argv(str(p))
    load_config(g)
    assert g['totally_new_key'] == 42


# ==========================================================================
# 3) 命令行覆盖的解析规则
# ==========================================================================
@pytest.mark.parametrize("raw,expect", [
    ('true', True), ('false', False), ('True', True), ('FALSE', False),
])
def test_cli_bool_parsing(argv, raw, expect):
    g = {'flag': not expect}
    argv(f'--flag={raw}')
    load_config(g)
    assert g['flag'] is expect


@pytest.mark.parametrize("raw,expect", [
    ('3', 3), ('3.5', 3.5), ('1e-4', 1e-4), ('-2', -2),
])
def test_cli_numeric_parsing(argv, raw, expect):
    g = {'x': 0 if isinstance(expect, int) else 0.0}
    argv(f'--x={raw}')
    load_config(g)
    assert g['x'] == expect


def test_cli_string_fallback_and_empty_string(argv):
    """解析不了就当字符串；**空字符串也要能赋值**（`--ndb_store=""` 就是这条路）。"""
    g = {'s': 'x', 'ndb_store': 'old.pt'}
    argv('--s=hello', '--ndb_store=""')
    load_config(g)
    assert g['s'] == 'hello'
    assert g['ndb_store'] == ''


def test_cli_hyphen_becomes_underscore(argv):
    """argparse 风格 `--byte-level` 要映射到 `byte_level` 全局。"""
    g = {'byte_level': False}
    argv('--byte-level=true')
    load_config(g)
    assert g['byte_level'] is True


def test_cli_unknown_key_raises(argv):
    argv('--totally_unknown=1')
    with pytest.raises(ValueError, match="Unknown config key"):
        load_config({'n_layer': 12})


def test_cli_type_mismatch_raises(argv):
    argv("--n_layer='4'")                       # 字符串塞进 int 全局
    with pytest.raises(TypeError, match="n_layer"):
        load_config({'n_layer': 12})


def test_cli_bare_flag_without_equals_is_treated_as_config_file(argv):
    """`--foo`（不带 =）不是合法配置文件名，必须报错而不是静默忽略。"""
    argv('--foo')
    with pytest.raises(AssertionError, match="不能以"):
        load_config({'n_layer': 12})


def test_int_into_float_global_is_rejected(argv):
    """int → float 也会被拦（`type(3)` is not `type(3.0)`）。

    这是**故意的严格**：`--temperature=1` 这种写法如果静默通过，
    YAML/CLI 的类型漂移就再也查不出来了。需要小数就写 `1.0`。
    """
    argv('--temperature=1')
    with pytest.raises(TypeError):
        load_config({'temperature': 1.0})
