# 预研设计 v2: 模型外挂高速存储引擎

> 状态: 预研设计 (P0)。不启动训练; 100M 训完后用 3M 小模型实测。AST 引擎列为远期 (P3)。

## 0. 一句话

给模型外挂一个**高性能、数据无关、不可训练**的存储/检索/遗忘引擎 (Rust 内核, 尝试期用 Python 参考实现),
模型经一个**可训练的 n-bit 指令直通层**与之交互 (只经 f64 数组)。
数据库不参与梯度; 只有指令桥接层用 GRPO 学习。

## 1. 已定决策

| 决策点 | 结论 |
|---|---|
| 核心存储引擎 | Rust (PyO3 绑定); 尝试期先 Python, 接口签名一致可替换 |
| 交互介质 | f64 数组 (numpy <-> Rust), 不用文字/JSON |
| 桥接形态 | n-bit 指令直通层: n 个二值节点 -> 2^n 指令组合; 其余维 = payload |
| 读写分区 | 高速区 O(1)哈希 / 低速区 O(n·log n)排序 |
| 访问预算 | 每步固定一次批量往返: 1 写 + K 读 (K=1/2/4/8 由指令位给出) |
| 路由/旁路 | op=00 NOP 直接跳过 (大多数步零成本, CPU-GPU 同步可控的关键) |
|遗忘 | 引擎周期功能: 时间衰减 x 活跃度; fast 满降级进 slow, slow 满淘汰 |
| 梯度桥 | GRPO 离散策略 (每 bit 独立 Bernoulli, 逐位信用); SFT 热身用 STE |
| AST 引擎 | 远期 P3, 单独立项 |

## 2. 为什么 (本项目笔记已实证)

| 已证痛点 | 来源 | 本方案解法 |
|---|---|---|
| 2.7M 真计算/真记忆 = 容量天花板 (81 条加法表都背不下) | 66-68 | 数据/知识不进权重, 进外部引擎 |
| 神经记忆训练 NaN / OOM / 3x 慢 | 45 | 引擎无 autograd; 指令输出有界(tanh/sigmoid) |
| 静态先验 = 伪胜利 (val 创新低但采样崩) | 20/B | 数据可用性由 RL 学出; 判据看采样 |
| 改/删已有通道 5 连败 | B | 只加旁路通道, 全零指令=NOP=默认无副作用 |
| 换词表=模型作废 | C | 存向量, 与词表无关 |
| loss 骗低 | A | 奖励桥以「读出被用上」为判据 |

## 3. 架构

    +--------------------------------------+
    |  模型主干 (GPU, transformer)           |   hidden 原样直通
    +-------+---------------------+--------+
            | 末位 hidden           ^ 解码向量 (加性接回, 类似记忆通道)
    +-------v---------------------+--------+
    |  指令桥接层 (Python 先, 可训练)        |
    |   旁路组合激活网络 -> n bit logits     |
    |   前向二值化 -> 2^n 指令; 反向 STE/GRPO |
    |   payload 头: key(32) + value(64)     |
    |   路由: NOP 则整步跳过 (无 CPU-GPU 往返)|
    +-------+---------------------+--------+
            | 每步一次批量 f64 round-trip (CPU)
    +-------v---------------------+--------+
    |  vdb-core (Rust, 不可训练, 确定性)     |
    |   fast: 开放寻址哈希      O(1)        |
    |   slow: 排序索引/连续扫描  O(n·log n)  |
    |   forget: 时间x活跃度 + fast->slow 降级 |
    +--------------------------------------+

## 4. 指令集设计 (桥接层核心, 用户方案)

n 个节点经旁路激活网络 -> sigmoid -> 阈值二值化 -> n bit -> 2^n 种指令组合。
推荐位域化 (n=6~8, 每段语义独立, GRPO 逐 bit 分解信用, 未见组合可泛化):

    [0:2] op     00=NOP旁路 01=写 10=读 11=写+K读
    [2]   region 0=KV(fast O(1)) 1=Vec(slow 精确topk/ANN)   <- v4 混合
    [3:5] K      读条数 1/2/4/8 (仅 Vec 读生效)
    [5:8] flags  bit5 读命中刷新活跃度 / bit6 覆盖写 / bit7 pin 防遗忘

要点:
- 2^n 是「参数最省的语义空间」: n=8 只花几千参数编码 256 条指令, 且逐位学习样本效率
  远高于 256 类 softmax —— 对 3M 小模型友好, 天然支持组合泛化。
- 全零 = NOP => 随机初始化时模型默认「不访问」, 与「多数 step 不需要查库」一致, 冷启动安全。
- n 上限 ~10: 超过后 2^n 采样稀疏, 小模型学不动。位域优于平面位。
- 热身用 STE (前向二值/反向直通) —— 「直通层」之名正合; GRPO 阶段无需穿过离散。
- 反面护栏 (dev-notes/45): payload 一律 tanh, 门控一律 sigmoid, 写强度 clamp, 防信号爆炸 NaN。

## 5. 存储分区

- 高速区 fast: 槽数小 (256~4096), 量化 key -> 哈希 -> 槽, 读/写 O(1)。热数据/工作记忆。
- 低速区 slow: 容量大 (1e5~1e7 -entry)。Python 参考: 连续 numpy 数组, 批量读 = 一次
  matmul (数学上 O(n) 但常数极小); Rust 版: sorted vec + 二分或 BTreeMap = O(log n),
  SIMD 批量扫描 O(n) 可选。冷数据/长尾。
- 路由指令位决定访问哪一区; 两区都是同一 payload 语义, 模型只管发指令。

## 6. 遗忘 (时间 x 活跃度, 引擎内部周期功能)

    score = exp(-dt / tau) * (1 + alpha * access_count)
    fast 满 -> score 最低者「降级」demote 进 slow (热 -> 冷)
    slow 满 -> score 最低者「淘汰」evict; 超过 ttl 直接回收
    被读到: access += 1, t_last 刷新; pin 位可保护关键条目

遗忘不占模型指令位 —— 它是引擎的后台周期功能 (用户原话: 「定期遗忘的功能」)。
可选进化 (P2): 「惊喜度门控写入」(Titans, dev-notes/58) —— 高预测误差的写才进 fast。

## 7. CPU-GPU 同步

1. 存储全在 CPU (f64), 模型在 GPU。
2. 每步只做一次批量往返: GPU->CPU 取指令/payload -> CPU 执行 -> 解码向量回 GPU。
3. NOP 指令 = 整步跳过 = 零同步成本 (路由旁路是同步开销可控的根本原因)。
4. 拷贝用 pinned memory + non_blocking 流水化; 注意 dev-notes/62 的 ROCm 教训
   (必要时 HSA_ENABLE_SDMA=0)。
5. 远期: fast 区整块常驻 GPU 显存 (小而热), slow 留 CPU 按需拉取。

## 8. 梯度桥 (GRPO, 用户选定)

- 策略: 每 bit 独立 Bernoulli(sigmoid(logit/T)); GRPO 组内 (G 条采样指令) 相对优势。
- 奖励维 (注册进现有插件化 reward 包, 见 57/63):
    r_mem_used   读出的向量被下一时刻主干用上 (相关度/困惑度改善) -> +
    r_mem_write  写过之后未来某刻能正确读出 -> + (延迟 credit, 配合课程)
    r_mem_nop    correctly 不发指令 (访问了却没受益 -> -) 抑制乱访问
    r_mem_forget 遗忘不伤性能 -> 微正
- 防reward-hacking (dev-notes/59/64 教训): 固定基座锚点 + 退化时通知我 KL 拉回, 复用
  grpo_char.py 现有的 ref_base 机制。
- warm-start: 先固定规则强制少量 read/write 课程让读出通路先有梯度, 再放开指令自由学习。

## 9. 与 nanoSeek 架构的接入原则

- 只增不改 (附录 B 教训): 解码向量作为一条新的加性通道接回残余流, 与 mHC/KV记忆通道同构。
- 指令来源: 主干末位 hidden (或 MLA latent); 默认关闭 (use_vdb=false) 行为与现在完全一致。
- 词表侧: <call>=134 <result>=135 等协议符仅 debug 用, 真实通道走张量, 不经文字。
- 尝试期只动 pre-research/ 目录; P1/P2 再接 model/config.py、grpo_char.py、reward/dimensions.py。

## 10. 已知风险与对策

| 风险 | 来源教训 | 对策 |
|---|---|---|
| 写信号爆炸 NaN | 45 | payload tanh / sigmoid / clamp |
| val 伪胜利 | 20/B, A | 判据 = 采样: 「读出信息是否真的影响输出」探针 |
| 同步开销吃掉收益 | 62 工程线 | NOP 默认 + 每步一次批量; 3x 慢就砍 (45 铁律) |
| 指令全乱发 | 59 奖励作弊 | r_mem_nop + 访问预算固定 + 组相对基线 |
| 2^n 学不动 | - | n<=8/bitfield/逐位 Bernoulli/规则热身 |
| 静态先验捷径 | 20 | fast 区不塞常量; 指令可用性由 RL 学出 |

## 11. 路线图

- **P0 (本期, 训练期间只做这个)**: 设计 + Python 参考实现 + Rust 骨架, 不训练。
- **P1 (100M 训完)**: 3M 小模型, 读出直通 + 规则热身 + 密集奖励, 冒烟 30/300 步,
  验数值稳定 + 「读出被用上」探针为正。
- **P2**: 完整位指令 (op/region/K/flags) + 分区 + 遗忘 + 延迟奖励, 1500 步 A/B
  vs 无 vDB 基线 (同 seed/同预算)。
- **P3 (远期)**: AST 引擎 —— 模型可调用语法树做结构化推理, 复用同一指令桥协议, 单独立项。

## 12. 目录

    pre-research/
      README.md                      本目录索引
      DESIGN.md                      本文档
      db_engine/vdb.py               Python 参考实现 (可 CPU 自测)
      db_engine/rust_core/           Rust(PyO3) 骨架: Cargo.toml + src/lib.rs + README
      interface/command_bridge.py    n-bit 指令直通层 (可训练, 不依赖 training/)

## 13. 修订 v4 — KV+Vec 混合存储（用户已定，取代 v3）

结论: KV+Vec **双索引同库、单一引擎**，比单用任一种都好 (业界 Milvus/Qdrant 的
metadata 过滤 + ANN 混合检索即此结构)。两索引指向同一份 value。

    写入:  一个条目 = [f64 向量; 前 n_sig 维兼作签名] -> fast(KV哈希) + slow(Vec) 双索引
    读取:  ① KV 精确读 (签名命中, O(1))          -> region bit=0, 工作记忆/刚写的数据
           ② Vec 相似读 (topk, 精确扫描或 ANN)    -> region bit=1, 联想召回/模糊查询
           ③ 混合读 read_hybrid (签名过滤 -> topk) -> 组合指令或后续指令位, P2 引入

指令位分工 (桥接层, 与 §4 位域一致):
    [2] region  0=KV(fast) 1=Vec(slow)      —— 用户提出的"组合"落点
    [3:5] K     仅 Vec 读生效 (topk 条数)
    KV 命中失败时: 桥接层可选"自动降级" (KV miss -> Vec 兜底, 一次往返两区皆查,
    不增加 CPU-GPU 同步); 该策略先固定规则, P2 再考虑可训练。

为什么这样分 (而不是纯 Vec / 纯 KV):
- 纯 Vec 做不了精确: "刚才存的这条"必须精确取回, 相似度 top-1 不保证是它。
- 纯 KV 做不了联想: 模型产出的本来就是连续向量, 精确位匹配几乎永不命中。
- 混合 = 各司其职: KV 管"工作记忆/刚存的事" (fast), Vec 管"联想召回" (slow)。

实现状态: db_engine/vdb.py 已升级 v4 并自测通过 (KV 精确 / Vec topk / hybrid 三路);
Rust 内核 (rust_core) 待写, 同一接口的 KV 哈希表 + 连续向量块 (sorted/二分可选 ANN)。
