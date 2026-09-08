# pre-research — 模型外挂高速存储引擎 (预研, 不训练)

状态: 预研 P0。nanoSeek-100M 训练期间只做设计与代码骨架, **不启动任何训练**;
100M 训完后再拿 3M 小模型统一测试。AST 引擎 = 远期 (P3), 本期不动。

## 文件

| 文件 | 内容 | 可训练? |
|---|---|---|
| DESIGN.md | 设计 v4: KV+Vec 混合 / 指令位域 / 遗忘 / CPU-GPU 同步 / 梯度桥 / 路线 | - |
| db_engine/vdb.py | Python 参考实现 (f64, 数据无关, 确定性; KV精确/Vec topk/hybrid 三路) | 否 |
| db_engine/rust_core/ | Rust(PyO3) 内核骨架: fast O(1) + slow O(n log n) | 否 |
| interface/command_bridge.py | n-bit 指令直通层 (模型唯一可训练的桥接口) | 是 |

## 决策链 (浓缩, 详见 DESIGN.md §1)

Rust 内核(尝试期 Python) / f64 数组 / n-bit 指令直通层(2^n 组合) /
高速 O(1)+低速 O(n log n) / 每步 1写+K读 / NOP 旁路路由 / 遗忘=时间x活跃度
(引擎周期功能) / 梯度桥=GRPO 离散策略 (每 bit 逐位信用, SFT 热身用 STE)。

## 已验证 (自测, 只占 CPU 毫秒级, 非训练)

    $ .venv/bin/python db_engine/vdb.py
    vdb selftest ok: fast: 8 slow: 46 hits: 40
    $ .venv/bin/python interface/command_bridge.py
    bridge selftest ok: decoded (2, 80) logp (2, 8) | STE grad ok

## 统一测试 (100M 训完后, 用 3M 小模型)

- P1: 读出直通 + 规则热身 (mode='read') + 密集奖励, 冒烟 30/300 步
- P2: 完整指令 + KV/Vec 路由 + 遗忘 + 延迟奖励, 1500 步 A/B vs 无 vDB 基线
- P3: AST 引擎 (远期)
