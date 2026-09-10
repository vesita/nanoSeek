# 单元测试

```bash
uv run pytest                 # 全部（纯 CPU 也能跑；无卡机器自动跳过 GPU 用例）
uv run pytest -m 'not slow'   # 跳过十几秒以上的用例
uv run pytest tests/test_masking.py -q
uv run pytest -x -q           # 第一个失败就停
uv run ruff check             # lint 门禁（同样是 pytest 里的一条测试）
```

## 目录

| 文件 | 测什么 | 为什么值得测 |
|---|---|---|
| `test_masking.py` | **哪些 token 计入 loss** | 出错不报错、只让"loss 正常但学不会"。含一份朴素参考实现做随机对照 |
| `test_utils.py` | Newton-Schulz 正交化 / RoPE / RMSNorm / Sinkhorn / LSE 残差 | 数值原语；NS 是 Muon 的核心、占单步 21.7%。阈值全部按**实测**设定 |
| `test_optimizer.py` | Muon / MuonAdamW 更新规则、状态往返 | 叠了动量→Nesterov→正交化→权重衰减四层，写反不报错只是不收敛 |
| `test_model_smoke.py` | 前向形状、损失、梯度、共享权重、因果性、aux-free 副作用 | 核心模块，改动最频繁；参数量被算错过一次 |
| `test_schedules.py` | 学习率调度（cosine / WSD）、数据集文件名解析 | 抽取前的字面实现被拷进来做逐位回归；钉住文档里的 cosine-vs-WSD 表 |
| `test_config_loader.py` | YAML 继承、类型检查、命令行覆盖 | 唯一的配置入口；出错的表现是"配置没生效但训练照跑" |
| `test_prepare_split.py` | 数据切分（val 泄漏） | 本项目**代价最大**的一次事故（val 双标准 → 所有指标都是假的） |
| `test_ngram_hash.py` | 多项式哈希、槽位对齐、流式分块 | NDB 容量测量的工具；它算错会直接误导"NDB 值不值得做"的决策 |
| `test_ngram_delta.py` | Δ 插值的纯函数（门控 / kNN-LM 凸组合） | 一天内在这里犯了两个"只有跑起来才暴露"的错误 |
| `test_project_layout.py` | **配置布局不变量**（见下） | 把运维铁律变成"跑不过就红" |
| `test_lint.py` | 未定义名 / 重复定义的门禁 | `GATES`→`GATE_NAMES` 改名漏改，4 分钟评测跑完才在打印时崩 |

## 三条硬约定

1. **所有测试必须能在纯 CPU 上跑通。** 需要显卡的用例加 `@pytest.mark.gpu`，
   `conftest.py` 会自动 skip。
2. **不要 `import training/train.py`。** 它是模块级脚本，import 就会直接开训。
   要测的纯函数在 `training/schedules.py` / `training/masking.py`。
   `test_project_layout.py::test_no_test_imports_train_py` 会拦住你。
3. **不要依赖真实语料 / checkpoint。** 需要真实产物的用例要 `skipif` 文件不存在。

## `test_project_layout.py` 拦的是哪三件事

这三条都写过文档，但文档拦不住手滑；它们各自都真实发生过：

1. **配置文件不能放在它自己的 `out_dir` 里。**
   `train.py` 在 `init_from != 'resume'` 时调 `_backup_old_run(out_dir)`，
   把目录里**所有**文件挪进 `old/` —— 配置第一次启动就被挪走。
2. **每个配置键都必须在 `load_config(globals())` 之前定义。**
   `config_loader` 是 `if k in g: 覆盖 else: 注入`，所以定义在之后的键会先被
   注入、随后又被那句 `name = 默认值` **静默覆盖**（加载日志还会打印它，看起来生效了）。
   `data_prefix`、`gradient_checkpointing` 都栽在这上面。
3. **配置指向的数据文件必须存在。**
   `data_prefix: v2` 写错时训练会在启动时报错；但 `meta` 文件缺失只会让 vocab
   静默回退 —— 这条把它也拦住。

另外它还会拿 `PROJECT_STATE.md §5` 的那张表逐项对照 `configs/base_v2.yaml`，
防止"文档说 A、配置是 B"。

## 写测试时踩过的坑（都留在注释里了）

| 坑 | 表现 | 现在怎么防 |
|---|---|---|
| **空测试** | `targets=None` 时 forward 只返回最后一位，`l1[:, :-1]` 是空张量 → `allclose` 恒真 | 先断言形状非空，再比较（`test_causality_*`） |
| **空测试（对照）** | 错位写法与正确写法都能全对 → 对照测不出东西 | 显式断言"错误写法确实更差"（`test_off_by_one_placement_is_actually_worse`） |
| **期望写错当成 bug** | RoPE 的 Toeplitz 性质要求 q/k 是**同一向量**复制到各位置；用各自随机向量时性质不成立（代码没错） | 注释里写明前提 |
| **容差过紧** | float32 下 `abs=1e-9` 必然失败 | 按数据类型定量级（更新量 ~1e-2、p0 ~O(1) → `atol=2e-6`） |
| **误判成实现 bug** | `load_state_dict` 在 dtype 已匹配时**不拷贝**张量，两个优化器共享 buffer | 写成独立测试记录该行为（`test_load_state_dict_aliases_source_tensors`） |
| **测试数据不满足前提** | "检索全错"的场景里随机 token 有 3% 恰好撞对 | 关键场景用**确定性**构造而不是概率抽样 |

## 故意不覆盖的部分

- `training/train.py` 的训练循环本体：它是模块级脚本，测它等于起一次真训练。
  可测的纯逻辑已经抽到 `training/schedules.py`。
- `data/external/`（第三方语料仓库）、`out/`（训练产物）：已从 lint 里排除。
- CUDA/ROCm 专属路径：本机是 gfx1030 + ROCm 6.4（非官方支持），
  GPU 用例只做冒烟，不做数值断言。

## 加了新配置 / 新开关时

1. 在 `configs/` 新建 yaml（**放在 `out_dir` 之外**）。
2. 跑 `pytest tests/test_project_layout.py` —— 它会检查新键是否可覆盖。
3. 如果新开关会改变行为，加一条"它确实改变了行为"的测试。
   本项目出过空开关事故（`gradient_checkpointing` 被上游短路，两个 A/B 臂其实一样），
   见 `test_gradient_checkpointing_is_a_noop_under_aux_free_moe`。
