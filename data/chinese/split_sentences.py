#!/usr/bin/env python3
"""数据分句器（dev-notes/49）—— **薄壳转发**。

★ 2026-09-14 起，真正的实现搬到了 **`training/segmentation.py`**（canonical）。
  本文件只负责两件事：
    1. 把仓库根加进 `sys.path`，然后**原样转出** `split_line` / `split_text` /
       `SENT_PAT` / `SUB_SEP` —— 这样 `prepare.py` 的裸模块名导入
       （`from split_sentences import split_text`）与它的测试**一行都不用改**；
    2. 保留独立 CLI（分句整份语料并写 `*_split.txt`）。

  为什么搬家：分句器是**纯函数模块**，按本项目规矩属于 `training/`；而且
  `training/dialogue_stream.py` 原先只能靠 `importlib` 按文件路径去 load 它
  （因为 `data/chinese/` 不是包），现在能正常 import。

原说明（仍然成立）：
  按句边界切分文本，保留标点，超长句二次切。
  - 中文句边界：。！？…（全角）；英文：.!?（半角句号带数字小数点保护）
  - 换行保持为轮次/段落分隔，`用户：/模型：` 行不拆
  - 超长句（>max_len 字）再按逗号/分号切，防整段一长句

用法（独立工具）：
    .venv/bin/python data/chinese/split_sentences.py --file data/chinese/lccc_dialogue.txt
（输出分句版到同目录 *_split.txt；prepare 集成时用 --split-sentences 开关）
"""
import os
import sys

# 把仓库根塞进 sys.path 才能 import training.*（脚本被直接跑时 sys.path[0] 是本目录）
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from training.segmentation import (  # noqa: E402,F401
    SENT_PAT,
    SPECIAL_RE,
    SUB_SEP,
    is_special_only,
    split_line,
    split_text,
)

__all__ = ['SENT_PAT', 'SUB_SEP', 'SPECIAL_RE', 'is_special_only', 'split_line', 'split_text']


def main(argv=None):
    """薄壳 CLI：走的是**同一套引擎**（`training.segmentation.run_cli`），
    只把默认输出后缀保留成历史的 `_split`。"""
    from training.segmentation import run_cli
    return run_cli(argv, default_suffix='_split')


if __name__ == "__main__":
    raise SystemExit(main())
