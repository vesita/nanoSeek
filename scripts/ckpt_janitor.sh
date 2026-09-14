#!/usr/bin/env bash
# ⛔️ 已退役（2026-09-11 晚）—— 本脚本拒绝运行，请用训练内置的保留策略。
#
# 退役理由（两条都是实测出来的）：
#   1. 归档保留策略**已经搬进训练内部**：training/checkpoints.py 的 steps_to_keep
#      （每 5000 步留一个 + 最新 2 个），在**每次归档落盘时**就地执行（train.py 存档点）。
#      外部看守不再是防线，只是一个**会被误当成防线的重复进程**（AGENTS.md 铁律 11）。
#   2. 本文件原来的启动说明写的是 `setsid nohup` —— 那恰是**铁律 0 明令禁止**的形态。
#      harness 每次工具调用都建一个 systemd scope，**会话重启会 SIGKILL 整条 cgroup**，
#      `setsid` 只脱离会话/tty、**脱离不了 cgroup**（2026-09-11 实测因此丢过一次训练，
#      日志无 traceback、无 OOM，只有 journalctl 一行 `Killed unit cgroup`）。
#      留一句"照抄就会踩坑"的命令，比没有这句更危险 —— 所以这里直接挡住。
#
# 现在该怎么做：
#   * 日常巡检（含幂等清理 + 打印保留清单）：bash scripts/watch.sh
#   * 手工清理一次（默认就是安全策略：每 5000 步留一个 + 最新 2 个）：
#       .venv/bin/python -m training.checkpoints out/base_v2            # 真删
#       .venv/bin/python -m training.checkpoints out/base_v2 --dry-run  # 只报告
#   * 真需要**外部**看守时（本项目的策略已内置，正常不需要）：
#       systemd-run --user --unit=<名> --collect \
#         --property=WorkingDirectory=/home/vesita/coding/my/nanoSeek \
#         /bin/bash -c '<命令> > out/<日志>.log 2>&1'
#
# 保留本文件只为历史追溯（退役记录见 dev-notes/83 §0.1）。

set -uo pipefail
cat >&2 <<'EOF'
[janitor] ⛔️ 本脚本已于 2026-09-11 退役，拒绝运行。

  原因：归档保留策略已内置在训练里（training/checkpoints.py，
        每次归档落盘就地稀疏化），外部看守只会让人误以为还有第二道防线；
        且本脚本原启动方式 `setsid nohup` 违反铁律 0（会被会话重启 SIGKILL）。

  请改用：
    bash scripts/watch.sh                              # 巡检（含幂等清理）
    .venv/bin/python -m training.checkpoints out/base_v2   # 手工清理一次

  详见 dev-notes/83 §0.1「守夜人已退役」与 AGENTS.md 铁律 0 / 11。
EOF
exit 64
