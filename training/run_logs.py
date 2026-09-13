"""运行产物的**续写**策略（纯决策函数 + 极薄的 I/O 包装，可离线单测）。

**为什么需要它**：`train.py` 原来无条件用 `'w'` 打开 `results.csv` / `ndb.csv`
⇒ **每次续训都把之前的指标历史清空**。2026-09-11 实测：暂停在 step 22000 后重启，
`results.csv` 立刻从 967 字节（22 行评估记录）变成 0 字节，且没有任何提示。

这类 bug 的特征和 `checkpoints.py` 里那个"只留最新 N 个"一样：
**不报错、不告警，只是让历史静默消失**，等到想画 loss 曲线时才发现前半段没了。

策略：
  * ``resuming=True`` 且文件已存在且非空 → **追加**，且**不重写表头**
    （否则表头会出现在文件中间，任何 `csv.reader` 都会把它当成一条数据行）
  * 其它情况（全新 run / 文件不存在 / 文件是空的）→ 新建并写表头

抽出来的理由与 `training/checkpoints.py` 一致：**写文件模式的判断写错就是数据丢失**，
而 `train.py` 是模块级脚本、import 即训练，没法直接单测它的内部逻辑。
"""

from __future__ import annotations

import csv
import os

__all__ = ['csv_open_mode', 'open_run_csv']


def csv_open_mode(resuming: bool, path: str) -> str:
    """返回该用 ``'a'`` 还是 ``'w'`` 打开 ``path``。

    - ``resuming`` 为假 ⇒ 一定 ``'w'``（全新 run，没东西可留）。
    - ``resuming`` 为真但文件不存在或是**空文件** ⇒ 仍用 ``'w'``：
      空文件没有表头可继承，追加会让新表头落在一个空文件里（等价于新建），
      但用 ``'w'`` 语义更明确，也能自愈「上次启动到一半被 SIGKILL、只留下 0 字节」的情况。
    - 其余 ⇒ ``'a'``。
    """
    if not resuming:
        return 'w'
    try:
        if os.path.getsize(path) > 0:
            return 'a'
    except OSError:
        # 文件不存在（或不可 stat）⇒ 当作新建
        return 'w'
    return 'w'


def open_run_csv(path: str, header: list[str], resuming: bool) -> tuple[object, 'csv._writer', bool]:
    """按续写策略打开一个 run CSV。

    返回 ``(文件句柄, csv.writer, 是否是追加模式)`` —— 把 ``appended`` 也返回，
    好让调用方打印一句"正在追加到已有 results.csv（N 行）"，
    **让续写这件事可见**（静默的策略等于没有策略）。
    """
    mode = csv_open_mode(resuming, path)
    appended = mode == 'a'
    fh = open(path, mode, newline='', encoding='utf-8')
    writer = csv.writer(fh)
    if not appended:
        writer.writerow(header)
    return fh, writer, appended


def count_csv_rows(path: str) -> int:
    """数一个 CSV 的数据行数（不含表头）；读不了就返回 0。仅用于打印提示。"""
    try:
        with open(path, newline='', encoding='utf-8') as f:
            return max(sum(1 for _ in f) - 1, 0)
    except OSError:
        return 0
