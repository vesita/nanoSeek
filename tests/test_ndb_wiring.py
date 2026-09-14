"""NDB 接线的结构性回归测试（AST 静态检查 —— 不 import `train.py`，它一 import 就开训）。

钉住两条 2026-09-15 实测踩到的坑：

1. **派生开关 `use_ndb` 必须在 `load_config(globals())` 之后求值。**
   写在键定义处时 `ndb_slots` 还是默认值 0 ⇒ 永远算出 False ⇒ NDB **静默不启用**：
   冒烟里表现为"没有 `  NDB ` 挂载行、没有 `ndb.csv`、但训练照跑 20 步不报错"。
   这是「配置键必须定义在 `config_keys` 快照之前」那条铁律的**派生量版本**。

2. **`ndb.observe()` 只能出现在 `with ndb.write_enabled():` 里**，
   且不得落在 eval / val 路径上 —— 把验证集写进训练库就是 val 泄漏
   （`model/ngram_ndb.py:191` 的 docstring 明确警告）。

两条都是"判据能区分好坏"的：把代码改回出错的样子，它们会失败（见各自的负向对照注释）。
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PY = ROOT / 'training' / 'train.py'


def _tree():
    return ast.parse(TRAIN_PY.read_text(encoding='utf-8'))


def _call_linenos(tree, attr):
    """所有 `<something>.<attr>(...)` 调用的行号。"""
    return [n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == attr]


def test_use_ndb_is_computed_after_load_config():
    """`use_ndb` 必须在配置加载之后求值。

    负向对照：把 `use_ndb = ndb_slots > 0` 挪回 `ndb_*` 键定义处（`load_config` 之前），
    本测试失败 —— 那时实测的表现正是"训练正常但 NDB 完全没启用"。
    """
    tree = _tree()
    cfg_lines = [n.lineno for n in ast.walk(tree)
                 if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                 and isinstance(n.value.func, ast.Name) and n.value.func.id == 'load_config']
    ndb_lines = [n.lineno for n in ast.walk(tree)
                 if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'use_ndb' for t in n.targets)]
    assert cfg_lines, "train.py 里找不到 `load_config(globals())`"
    assert ndb_lines, "train.py 里找不到 `use_ndb` 的赋值"
    cfg_line, ndb_line = min(cfg_lines), min(ndb_lines)
    assert ndb_line > cfg_line, (
        f"`use_ndb` 在第 {ndb_line} 行求值，而 `load_config` 在第 {cfg_line} 行 —— "
        f"配置还没加载，`ndb_slots` 仍是默认值 0 ⇒ NDB 会静默不启用。")


def test_observe_only_inside_write_enabled():
    """`ndb.observe()` 的每一个调用点都必须在 `with ndb.write_enabled():` 的作用域内。

    负向对照：把一个 `ndb.observe(...)` 挪到 `with` 外面（或在 eval 里加一个），本测试失败。
    """
    tree = _tree()
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, ast.With):
            for item in node.items:
                ce = item.context_expr
                if (isinstance(ce, ast.Call) and isinstance(ce.func, ast.Attribute)
                        and ce.func.attr == 'write_enabled'):
                    end = max(getattr(n, 'lineno', node.lineno) for n in ast.walk(node))
                    spans.append((node.lineno, end))
    assert spans, "train.py 里没有 `with ndb.write_enabled():` —— 写路径断开了"
    calls = _call_linenos(tree, 'observe')
    assert calls, "train.py 里没有 `ndb.observe(...)` —— 写路径断开了"
    outside = [ln for ln in calls if not any(a <= ln <= b for a, b in spans)]
    assert not outside, (
        f"`ndb.observe()` 出现在 `write_enabled()` 之外（行 {outside}）—— "
        f"要么根本没写进表，要么写进了不该写的地方（eval/val = 验证集泄漏）。")


def test_write_and_read_paths_both_present():
    """读与写两条路径都必须接上 —— 这是「可读可写、且由模型自己决定」的最小判据。

    只接读（`ndb.read`）或只接写（`ndb.observe`）都是**漂移**：前者退化成离线库查询，
    后者永远学不会怎么用记忆。负向对照：注释掉任一条，本测试失败。
    """
    tree = _tree()
    assert _call_linenos(tree, 'read'), "train.py 里没有 `ndb.read(...)` —— 读路径断开了"
    assert _call_linenos(tree, 'observe'), "train.py 里没有 `ndb.observe(...)` —— 写路径断开了"
