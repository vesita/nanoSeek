# 82 · NDB 方案收敛：单一载具 `ngram_ndb`，以及"文档说 A、训练做 B"的漂移清除

> 日期：2026-09-15 · 状态：**已落地**（`c6664aa` + flush 优化）· 证据强度标注在每处
> 本文是**工程记录**；人格语料与人格设定在仓库外（`~/datasets/persona/`），本文不含其内容。

---

## 0. 一句话

训练侧的 NDB 从 `model/memory_cross_attn.py`（**只读**、库要**离线手动建**）换成
`model/ngram_ndb.py`（**读与写都由模型门控决定**、表在训练中**在线累积**），
并把只服务旧方案的那一整套实现、配置键、离线库与本地脚本**删除**。

---

## 1. 判据（用户原话，2026-09-15）

> 「所有只读、或者只写、或者需要手动建库的都是漂移」

> 「刚才那个能读能写的 ngram 才是应该采用的」

> 「这个就是此前测试好的」

⇒ 判据是**接口形态**，不是性能：谁决定读、谁决定写、表从哪里来。
`memory_cross_attn` 三条占两条（只读 + 手动建库），所以它不是"次优方案"，是**漂移**。

---

## 2. 漂移是怎么暴露的（事故链，值得单独记）

1. 我按 `PROJECT_STATE §0.5.14` 的 4 段计划建库、准备起 λ 闸门；
2. 用户问「现在接入的 ndb 是可读与可写的吗？」——查证发现：
   `train.py:754` **只** import `MemoryCrossAttention`，`NgramNDB` 在 `train.py` 里
   **一次都没出现**（只有离线探针 `scripts/ndb_online_ab.py` 用它）；
   `write_online` 的**非测试调用者是 0**。
3. 而 `AGENTS.md §1` 的门面表写着：
   **`model/ngram_ndb.py` = 可读可写的 NDB：读写策略都由模型自己学**，
   口径列直接标着「**本项目采用的**」。
4. ⇒ **文档说 A、训练做 B**。这比"少了一半功能"更严重：`AGENTS.md` 是下一个接手的人
   **唯一被 harness 自动注入**的文档，它一错，后面所有人都拿着错前提做决定。
5. 用户拍板：**修正文档不算修好，把实现也换成 A**。

**教训**：「采用的方案」这句话必须指向 `train.py` 里**真的被挂载**的那个类，
否则它就是一个会自我复制的错误。文档里的口径列要有**可执行的判据**
（本次补的两条 AST 断言就是），不能只靠人去核对。

---

## 3. 接线（`training/train.py`）

| 环节 | 做法 |
|---|---|
| 取隐藏态 | `h` = **`ln_f` 的输入**（`register_forward_pre_hook`），与探针 `scripts/ndb_online_ab.py:99` 一致；读门控与写门控都吃它 |
| 读 | `p_new = (1−g)·p_model + g·p_ng`（`g` 与多级混合 α 都由模型门控决定），把 loss 里的 **CE 项**换成 `CE(p_new)`，其余附加项（MoE 等）原样保留 |
| 写 | `loss.backward()` 之后，在 `with ndb.write_enabled():` 里 `ndb.observe(h.detach(), X, Y)`；每 `ndb_flush_every` 步、以及每次监控前 `flush()` |
| val 防泄漏 | `observe` / `write_enabled` **只在训练微步**出现；`ndb_eval` 结束后显式清 `_ndb_state['h']` |
| 监控 | `ndb.csv` 两臂配对：`base_off` / `mem` / `delta` / `write_gate_bias` / `coverage` |
| checkpoint | 只存门控参数（`ndb` 键）；**表不持久化**，每个 run 从空表在线重建 |

★ **梯度路径**（这点最容易看错）：`observe()` 自身是 `@torch.no_grad()`，
`write_gate` 的梯度**来自 `read()`**（`ngram_ndb.py:384` 的 `w_t` 进 `:422` 的 `read_gate`）。
判"写门控有没有在学"必须看 read 路径，光看 `observe` 会得出错误的"没梯度"结论。

---

## 4. 删除清单

**跟踪文件**：`model/memory_cross_attn.py`、`tests/test_memory_cross_attn.py`、
`configs/pilot_persona_lam{0,2}.yaml`（只改已删旋钮 `ndb_att_sim`）。
**配置键**：`ndb_store` / `ndb_heldout` / `ndb_layer` / `ndb_chunk` / `ndb_exclude_radius` /
`ndb_retr_noise` / `ndb_att_sim` / `ndb_dropout` / `ndb_gate_init`。
**本地（gitignored）**：`out/mem_store/`、`local/build_chunk_store.py` 及其余 10 个
建立在离线库上的 `local/*.py`。
**保留**：`dev-notes/79`、`dev-notes/80`、`analysis/NDB_cotrain_STATE_2026-09-10.md` ——
它们是**历史记录**，改写它们等于篡改证据；但它们描述的是已退役的方案，别照抄命令。

---

## 5. 实测数据

### 5.1 稳态代价（配对，`configs/base_v3_persona.yaml`，各 40 步，只差 `ndb_slots`）

| 臂 | 墙钟 | s/it | 吞吐 | 显存 |
|---|---:|---:|---:|---:|
| NDB 关 | 175s | 4.10 | 2426 t/s | 1.7G |
| NDB 开 | 187s | 4.42 | 2317 t/s | 1.9G |

⇒ **+7.8% 墙钟、+0.2G 显存**（单次配对，非多 seed）。
★ **反面教训**：先前 6 步冒烟显示 10.39 s/it，我差点把它当成 NDB 的代价 ——
那里面混着 `torch.compile` 首步、两次 NDB 监控、两次落表。
**单步耗时不配对测就等于没测**（同 §5.5 的措辞纪律）。

### 5.2 写门控确实在学（6 步冒烟，`out/last.pt` 的 `ndb` 键）

| 参数 | 初始化 | 6 步后 \|sum\| |
|---|---:|---:|
| `write_gate.weight` | **全 0** | **0.684384** |
| `write_gate.bias` | 1.0 | 0.998454 |
| `read_gate.weight` | 默认 | 11.419164 |
| `read_gate.bias` | −2.0 | 2.002133 |
| `level_weight` | 全 0 | 0.0（单级 ⇒ 梯度按对称性为零，预期） |

`write_gate.weight` 初始化为全零 ⇒ 非零就是**梯度经 `read()` 走通**的铁证。
覆盖率同时 `0.00% → 34.38% → 40.62%` ⇒ 表确实在线在长。

### 5.3 `flush()` 的 O(最大槽号) 缺陷（旧实现取自 git HEAD，同负载）

| 实现 | flush 耗时 | 峰值 RSS | flush 期间临时占用 |
|---|---:|---:|---:|
| 旧 | 1.948s | 5.33G | +3.76G（含 2.14G 的 `np.zeros` 临时数组） |
| 新 | **0.301s** | **3.74G** | +2.17G（3GB 表随机访问的按需换页） |

改法：`np.unique(..., return_inverse=True)` + `np.bincount`（数学等价，
`tests/test_ngram_ndb.py` 含"增量写 vs 离线建表**逐位对照**"全过）。
**它不是速度问题**（摊到 18000 步只有 ~2 分钟），是**2GB 临时尖峰**的 OOM 隐患，
且随 `slots` 线性放大（`2^30` ⇒ 8GB）。

---

## 6. 踩到的坑

### 6.1 派生量也必须在 `load_config()` 之后算

`use_ndb = ndb_slots > 0` 我写在 `ndb_*` 键定义处，而 `load_config(globals())` 在第 312 行 ——
那时 `ndb_slots` 还是默认值 0 ⇒ 永远 False ⇒ **NDB 静默不启用**。
表现极具欺骗性：训练照跑 20 步、exit 0、无 traceback，只是没有 `  NDB ` 挂载行、
没有 `ndb.csv`。

⇒ 这是「配置键必须定义在 `config_keys` 快照之前」那条铁律的**派生量版本**：
**要在配置基础上算出的东西，都得等配置加载完**。已由
`tests/test_ndb_wiring.py::test_use_ndb_is_computed_after_load_config` 用 AST 钉住，
并做了负向对照（把赋值挪回去 ⇒ 判据正确失败）。

### 6.2 "冒烟过了"必须具体到"我要的那条路真的被执行了"

旧的人格冒烟（00:10–00:29）`Result=success`，但 `out/base_v3_persona_train.log` 里
`NDB` 出现 **0 次** —— 那次 `ndb_store` 指向的文件不存在，NDB **根本没启用**。
真正第一次执行 NDB 时就崩了（`_ndb_hook` 的形状口径与 `use_mhc: false` 不符，
见 `d72612d`）。

⇒ 冒烟的验收判据要包含**被执行路径的特征输出**（挂载行 + 监控 CSV 存在 + 关键参数有变化），
不能只看 exit code。

---

## 7. 遗留（明确未验证 / 未做）

1. **Δ 目前是正的**：6 步冒烟里 `Δ = −0.0069 → +0.0333 → +0.0369`。冷表 + 读门控随机初值下
   这是**预期**的（注入了一个还不准的分布）；但"在线写的 NDB 到底能不能把 Δ 变负"
   **尚无实测**——离线探针的 −0.0738 是**另一套方案**的数字，**不可直接沿用**。
   ⚠ 这是本方案当前最大的未知，第一段训练的 `ndb.csv` 就是为回答它而记的。
2. **表不持久化**：每个 run 从空表重建 ⇒ 阶段之间没有"预热过的记忆"，
   每段开头的 Δ 都从冷启动开始。是否有必要做"跨阶段继承表"未定。
3. **覆盖率冷启动**：`observe_tokens` 提供了离线预热入口，我们**没用**（它会引入
   "手动准备数据"的味道）。短阶段（如 900 步的人格层）表可能一直偏冷。
4. **`read()` 的稠密分布**：只用得上 `p_new` 在**目标 token** 上的取值，
   现在却每 micro-batch 造 ~7 个 `(B,T,V)` 稠密张量 + 两次 CE。
   数学等价的"目标位"改法可以把它们全部消掉，但**上限只有那 7.8%**，暂不做。
5. **`_accumulate` 的 Python 循环**：`ngram_ndb.py:256` 每 token 一次 dict 更新
   （每步约 8000 次）。同样只值几个百分点，暂不做。

---

## 8. 纪律（新增/强化）

1. **文档里的"采用口径"必须指向 `train.py` 真的挂载的那个类**；
   能写成断言的，就不要只写成文字。
2. **派生开关在 `load_config()` 之后求值**。
3. **冒烟的判据要包含被执行路径的特征输出**，exit 0 不算通过。
4. **单步耗时/显存一律配对测**；含 `compile`、监控、落表的数字不能当稳态。
5. **换载具要删旧的**：只改文档不改实现 = 漂移；只删文档不删实现 = 漂移。
