"""训练主循环的**结构性**不变量：终止判据必须在优化器步**之前**。

## 为什么单独立一个文件

`training/train.py` 是**模块级脚本**（`import` 它就等于开训），跑不进 pytest。
所以这个文件用 **AST 读源码**，不导入它 —— 和 `tests/test_packing.py:60`、
`test_doc_packing_keys_defined_before_config_keys` 同一手法。

## 钉的是什么（2026-09-14 用户实测发现的 bug）

旧代码把终止判据放在循环**末尾**、`iter_num += 1` **之后**：

    [eval@k] → [前向/反向/optimizer.step()] → k += 1 → (k > max_iters 才 break)

⇒ 跑满 `max_iters` 之后**还会再走一次优化器步**。那一步：
  ① 不评估、不落盘（白算）；② 但**改了内存里的权重**，而 `last.pt` 是循环顶部
  `iter_num == max_iters` 时存的 ⇒ **盘上权重与内存权重差一步**。
实测：`--max_iters=3` 打印「训练完成：**4** 步」、tqdm 走到 `4it`（>100%）。

正确形状（终止判据在评估之后、优化器步之前，且必须用 `>=` 而不是 `>`）：

    [eval@k] → (k >= max_iters ? break) → [前向/反向/optimizer.step()] → k += 1

## 判据不是恒真的（本文件自带对照）

`analyze_loop()` 对源码文本做分析，`test_judge_distinguishes_old_and_new_shapes`
用**手写的旧形状 / 新形状片段**各跑一遍，要求旧形状被判违规、新形状通过。
没有这条对照，`assert term_idx < opt_idx` 这类断言可能是永真的空测试。
"""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PY = ROOT / 'training' / 'train.py'


def _is_iter_num_vs_max_iters(test):
    """`iter_num >= max_iters`（或 `max_iters <= iter_num`）→ 返回 'GtE'，否则 None。

    ★ 必须是**严格** `GtE`：写成 `>`（Gt）时，`iter_num == max_iters` 那一次不 break，
    于是又跑一步优化器 —— 正是要修的那个 bug 会悄悄回来。
    """
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        return None
    op, left, right = test.ops[0], test.left, test.comparators[0]
    if isinstance(left, ast.Name) and isinstance(right, ast.Name):
        if left.id == 'iter_num' and right.id == 'max_iters':
            return 'GtE' if isinstance(op, ast.GtE) else type(op).__name__
        if left.id == 'max_iters' and right.id == 'iter_num':
            return 'LtE' if isinstance(op, ast.LtE) else type(op).__name__
    return None


def _contains_break(node):
    return any(isinstance(n, ast.Break) for n in ast.walk(node))


def _first_stmt_index_containing(body, pred):
    for i, stmt in enumerate(body):
        if any(pred(n) for n in ast.walk(stmt)):
            return i
    return -1


def analyze_loop(source):
    """返回 `{term_idx, term_op, opt_idx, inc_idx, body_len}`；找不到的给 -1。"""
    tree = ast.parse(source)
    loop = None
    for node in ast.walk(tree):
        if isinstance(node, ast.While):
            loop = node
            break
    assert loop is not None, 'train.py 里找不到 `while` 主循环'
    body = loop.body

    term_idx, term_op = -1, None
    for i, stmt in enumerate(body):
        if isinstance(stmt, ast.If) and _contains_break(stmt):
            op = _is_iter_num_vs_max_iters(stmt.test)
            if op is not None:
                term_idx, term_op = i, op
                break

    opt_idx = _first_stmt_index_containing(
        body,
        lambda n: (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr == 'step'
                   and isinstance(n.func.value, ast.Name)
                   and n.func.value.id == 'scaler'))
    inc_idx = _first_stmt_index_containing(
        body,
        lambda n: (isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name)
                   and n.target.id == 'iter_num'))
    return {'term_idx': term_idx, 'term_op': term_op, 'opt_idx': opt_idx,
            'inc_idx': inc_idx, 'body_len': len(body)}


def check_loop(source):
    """返回违规说明列表（空 = 通过）。"""
    r = analyze_loop(source)
    bad = []
    if r['term_idx'] < 0:
        return [f"主循环里找不到 `iter_num` 与 `max_iters` 的终止判据（{r}）"]
    if r['term_op'] != 'GtE':
        bad.append(f"终止判据的算符是 {r['term_op']}，必须是 GtE（`>=`）—— "
                   f"`>` 会让 `iter_num == max_iters` 那一次不 break，又多跑一步优化器")
    if r['opt_idx'] < 0:
        bad.append('找不到优化器步（`scaler.step(...)`）—— 判据失效，别把它当成通过')
    if r['inc_idx'] < 0:
        bad.append('找不到 `iter_num += 1` —— 判据失效')
    if r['opt_idx'] >= 0 and r['inc_idx'] >= 0:
        if not (r['term_idx'] < r['opt_idx'] < r['inc_idx']):
            bad.append(
                f"顺序错了：termination@{r['term_idx']} / optimizer@{r['opt_idx']} / "
                f"iter_num+=1@{r['inc_idx']} —— 必须是 "
                f"termination < optimizer < increment（终止判据在优化器步**之前**）")
    return bad


def test_train_py_exists():
    assert TRAIN_PY.exists(), f'找不到 {TRAIN_PY}'


def test_termination_guard_is_before_the_optimizer_step():
    """★ 核心不变量：终止判据在优化器步之前，且用 `>=`。"""
    bad = check_loop(TRAIN_PY.read_text(encoding='utf-8'))
    assert not bad, '训练循环形状不对：\n  - ' + '\n  - '.join(bad)


# ---------------------------------------------------------------------------
# 已知答案对照：判据必须能区分「旧形状」与「新形状」
# ---------------------------------------------------------------------------
_OLD_SHAPE = '''
while True:
    lr = 0.1
    if iter_num % 100 == 0:
        evaluate()
    if iter_num == 0 and eval_only:
        break
    forward_backward()
    scaler.step(optimizer)
    iter_num += 1
    if iter_num > max_iters:
        break
'''

_NEW_SHAPE = '''
while True:
    lr = 0.1
    if iter_num % 100 == 0:
        evaluate()
    if iter_num == 0 and eval_only:
        break
    if iter_num >= max_iters:
        break
    forward_backward()
    scaler.step(optimizer)
    iter_num += 1
'''

# 形状对、但算符写成 `>` —— 多跑一步的 bug 会以这种方式悄悄回来
_GT_SHAPE = _NEW_SHAPE.replace('iter_num >= max_iters', 'iter_num > max_iters')


@pytest.mark.parametrize('src,should_pass', [
    (_NEW_SHAPE, True),
    (_OLD_SHAPE, False),
    (_GT_SHAPE, False),
], ids=['新形状-应通过', '旧形状-应判违规', '算符用Gt-应判违规'])
def test_judge_distinguishes_old_and_new_shapes(src, should_pass):
    """判据的**已知答案对照**：拿两份手写片段验证它真的能分辨好坏。"""
    bad = check_loop(src)
    assert (not bad) is should_pass, (
        f'判据结果与预期不符（should_pass={should_pass}）：{bad}')
