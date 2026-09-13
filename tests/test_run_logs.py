"""运行产物续写策略的单测（对应 `training/run_logs.py`）。

这块逻辑**决定用 'w' 还是 'a' 打开文件** —— 判断写错就是把指标历史静默清空。
2026-09-11 的真实事故：暂停在 step 22000 后重启，`results.csv` 从 967 字节
（22 行评估记录）变成 0 字节，没有任何提示。
"""

from __future__ import annotations

import os

from training.run_logs import count_csv_rows, csv_open_mode, open_run_csv

HEADER = ['step', 'train/loss', 'val/loss']


def _write(path, lines):
    # ★ lines 为空时要真的写出 0 字节，不能写成 '\n'
    #   （'\n'.join([]) + '\n' == '\n' 是 1 字节 —— 这个细节曾让本文件的两条
    #    "空文件" 测试失去了区分能力，实测抓到）
    with open(path, 'w', encoding='utf-8') as f:
        if lines:
            f.write('\n'.join(lines) + '\n')


# --------------------------------------------------------------------------
# csv_open_mode：纯决策
# --------------------------------------------------------------------------

def test_fresh_run_always_truncates(tmp_path):
    """全新 run 没东西可留，即使文件里有内容也该用 'w'。"""
    p = str(tmp_path / 'results.csv')
    _write(p, ['step,val', '1000,1.5'])
    assert csv_open_mode(resuming=False, path=p) == 'w'


def test_resume_appends_to_nonempty_file(tmp_path):
    p = str(tmp_path / 'results.csv')
    _write(p, ['step,val', '1000,1.5'])
    assert csv_open_mode(resuming=True, path=p) == 'a'


def test_resume_creates_when_missing(tmp_path):
    assert csv_open_mode(resuming=True, path=str(tmp_path / 'nope.csv')) == 'w'


def test_resume_creates_when_file_is_empty(tmp_path):
    """★ 自愈：上次启动到一半被 SIGKILL 会留下 0 字节文件。

    0 字节文件没有表头可继承，必须当成新建 —— 否则解出来的 CSV 没有表头，
    所有按列名读它的下游（画图、对比脚本）都会错位。
    """
    p = str(tmp_path / 'results.csv')
    _write(p, [])
    assert os.path.getsize(p) == 0
    assert csv_open_mode(resuming=True, path=p) == 'w'


# --------------------------------------------------------------------------
# open_run_csv：I/O 包装
# --------------------------------------------------------------------------

def test_open_run_csv_fresh_writes_header(tmp_path):
    p = str(tmp_path / 'results.csv')
    fh, _w, appended = open_run_csv(p, HEADER, resuming=False)
    fh.close()
    assert appended is False
    with open(p, encoding='utf-8') as f:
        assert f.readline().strip() == ','.join(HEADER)


def test_resume_keeps_old_rows_and_does_not_duplicate_header(tmp_path):
    """★ 核心回归：续训后旧行必须还在，且表头**只能出现一次**。

    表头重复是这类 bug 的典型次生灾害：表头行会变成一条 `step='step'` 的数据，
    任何 `float(row[0])` 都会炸，或者被静默跳过导致行数对不上。
    """
    p = str(tmp_path / 'results.csv')
    _write(p, [','.join(HEADER), '1000,1.9,2.0', '2000,1.8,1.95'])

    fh, w, appended = open_run_csv(p, HEADER, resuming=True)
    w.writerow([3000, 1.7, 1.9])
    fh.close()

    assert appended is True
    lines = open(p, encoding='utf-8').read().strip().split('\n')
    assert lines[0].strip() == ','.join(HEADER)          # 表头在且仅在开头
    assert sum(1 for ln in lines if ln.startswith('step')) == 1
    assert lines[1:] == ['1000,1.9,2.0', '2000,1.8,1.95', '3000,1.7,1.9']
    assert count_csv_rows(p) == 3


def test_open_run_csv_on_empty_file_writes_exactly_one_header(tmp_path):
    p = str(tmp_path / 'results.csv')
    _write(p, [])
    fh, _w, appended = open_run_csv(p, HEADER, resuming=True)
    fh.close()
    assert appended is False
    assert count_csv_rows(p) == 0
    with open(p, encoding='utf-8') as f:
        assert f.read().strip() == ','.join(HEADER)


# --------------------------------------------------------------------------
# count_csv_rows：只用于打印提示，不能因为它崩训练
# --------------------------------------------------------------------------

def test_count_csv_rows_excludes_header(tmp_path):
    p = str(tmp_path / 'results.csv')
    _write(p, [','.join(HEADER), '1000,1.9,2.0'])
    assert count_csv_rows(p) == 1


def test_count_csv_rows_missing_file_is_zero(tmp_path):
    assert count_csv_rows(str(tmp_path / 'nope.csv')) == 0
