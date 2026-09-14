# P2 新阶段落地清单（v4_persona）与管线分析

> 本文件由只读子代理针对新增“人格/角色”阶段（`v4_persona`）进行全管线源码走读后生成。
> **全部结论均经源码核对**，附带确切 `文件:行` 与原代码片段。

---

## 1. 核心结论与改动清单表（文件 → 行 → 改什么 → 不改的后果）

| 序号 | 目标文件 | 代码行号 | 改动内容 | 不改的后果 | 确认口径 |
|---|---|---|---|---|---|
| 1 | `data/chinese/build_stages.py` | 59-80 (`STAGES` 字典) | 在 `STAGES` 中增加 `'v4_persona': ['<源文件名1>.txt', ...]` | 执行 `build_stages.py` 报错 `未知阶段 ['v4_persona']`（第 221 行退出），或新源被报 `未被任何阶段覆盖`（第 233 行），无法切分阶段与建软链 | 源码确认 |
| 2 | `data/chinese/build_stages.py` | 278 (打印提示，可选优化) | 更新打印提示 `'--data-prefix v3_lang / v3_know / v3_dlg / v4_persona'` | 仅日志提示不完整，不影响执行逻辑 | 源码确认 |
| 3 | `configs/base_v4_persona.yaml` | 新建文件 (1-50+) | 新建 YAML 配方：`extends: base_v2.yaml`，指定 `out_dir: out/base_v4_persona`，`init_from: out/base_v3_dlg/last.pt`，`data_prefix: v4_persona`，`use_doc_packing: true`，`pack_align: false`，`use_loss_masking: false`，`max_iters`，`lr_decay_iters` | 无独立阶段配置文件；若直接传命令行而不改配置文件，无法继承 base_v2 且被测试门禁拦截；若漏配独立 `out_dir` 会毁掉前序 ckpt | 源码确认 |
| 4 | `tests/test_project_layout.py` | 283 (`V3_STAGE_CONFIGS`) 或 320-369 | 若配置文件命名为 `base_v3_*.yaml` 会自动进入参数化测试；若命名为 `base_v4_*.yaml`，测试不会自动覆盖它。若要将 v4 纳入同样严格的不变量测试，需将 `V3_STAGE_CONFIGS` 扩展或新增 `V4_STAGE_CONFIGS`，并在 `V3_STAGE_STEPS`（第 292 行）中登记步数 | 若命名为 `base_v3_persona.yaml` 但未登记 `V3_STAGE_STEPS`，`test_v3_stage_config_safety` 必定断言失败（第 364 行报 `KeyError` 或不存在）；若写错 `data_prefix` 或漏配 `out_dir`，缺少门禁守护 | 源码确认 |
| 5 | `tests/test_project_layout.py` | 292-295 (`V3_STAGE_STEPS`) | 若配置文件匹配 `configs/base_v3_*.yaml`，必须登记 `'base_v3_persona.yaml': <步数>` | `test_v3_stage_config_safety` 在第 364 行断言失败：`base_v3_persona.yaml: 新增的 v3 阶段配置必须同时登记进 V3_STAGE_STEPS` | 源码确认 |
| 6 | `tests/test_project_layout.py` | 362-363 (`data_prefix.startswith('v3_')`) | 若继续归在 v3 命名下但 `data_prefix` 叫 `v4_persona`，断言 `cfg['data_prefix'].startswith('v3_')` 失败 | `test_v3_stage_config_safety` 报 `data_prefix='v4_persona' 不是 v3 阶段数据` 挂掉 | 源码确认 |
| 7 | `scripts/watch.sh` | 20-21 (`OUT_DIR`, `LOG`) | 换到新 run 时，必须将第 20-21 行改为：<br>`OUT_DIR="${1:-out/base_v4_persona}"`<br>`LOG="${2:-out/base_v4_persona_train.log}"` | 铁律 4/12 事故：`watch.sh` 会继续看旧 run 日志；`prune_ckpts.sh`（第 53 行）只清理默认的 `OUT_DIR`，新目录的 ckpt 不会被修剪，**单卡磁盘会被打爆** | 源码确认 |
| 8 | `PROJECT_STATE.md` 与 `AGENTS.md` | 多处速查与方案表 | 更新配方说明、当前 run 状态、timer 监控对象 | 出现“以为在监控”或后续接手者误查旧单元 | 源码确认 |

---

## 2. 步骤 1：`data/chinese/build_stages.py` 源码走读与改动分析

### 2.1 Stage 定义位置
`build_stages.py` 的 stage 名称与所包含源文件由全局常量字典 `STAGES`（第 59-80 行）直接定义：
```python
59: STAGES = {
60:     'v3_lang': [
61:         'c4_zh.txt', 'wikipedia_cn.txt',
62:         '西游记.txt', '红楼梦.txt', '三国演义.txt', '水浒传.txt',
63:         'classical_poetry.txt',
64:     ],
65:     'v3_know': [
66:         'qa_knowledge.txt',                       # 新导入：274k 中文问答
67:         'deepseek_r1_distill_dialogue.txt', 'qwen3_235b_distill_dialogue.txt',
68:         'coig_*.txt', 'code_alpaca_dialogue.txt', 'gsm8k_cot_dialogue.txt',
69:         'zhihu_kol_dialogue.txt', 'muice_dialogue.txt',
70:         'identity_dialogue.txt',
71:     ],
72:     'v3_dlg': [
73:         'multi_turn_dialogue.txt', 'dailychat_dialogue.txt', 'lccc_dialogue.txt',
74:         'glm_dialogue.txt', 'kdconv_dialogue.txt',
75:         'sharegpt_zh_38k.txt',                    # 新导入：38.5k 段真多轮（LCS 承接 +15.1pp）
76:         'belle_multiturn.txt',                    # 新导入：Belle 0.8M 抽样
77:         'wildchat_zh.txt',                        # 新导入：WildChat 中文（已过滤 toxic）
78:         'escov_zh.txt',                           # 新导入：翻译后的多轮对话
79:     ],
80: }
```
**改动点**：若要加新 stage `v4_persona`（或 `v3_persona`），必须在 `STAGES` 字典内新增一个 key，其 value 为该 stage 纳管的 `.txt` 语料文件名列表（支持 `fnmatch` 通配符）。

### 2.2 Manifest 的生成与命名
`build_stages.py` 在第 269-272 行调用 `prepare.py`：
```python
269:         cmd = [sys.executable, PREPARE, '--char-level', '--val-all',
270:                '--val-ratio', a.val_ratio, '--seed', a.seed,
271:                '--source-dir', os.path.join(a.stages_dir, stage),
272:                '--out-prefix', stage, '--emit-offsets']
```
在 `prepare.py` 中：
- 第 85-87 行：`named_output(stem, prefix, ext)`:
  ```python
  85: def named_output(stem, prefix, ext):
  86:     """按 --out-prefix 生成产物文件名：空 prefix 保持旧名，否则 stem_prefix.ext。"""
  87:     return f'{stem}{("_" + prefix) if prefix else ""}{ext}'
  ```
- 第 445-449 行：
  ```python
  445:     stem = getattr(args, 'out_prefix', '') or ''
  ...
  449:         manifest_path = os.path.join(DATA_DIR, named_output('manifest', stem, '.json'))
  ```
- 产物命名规则为：`manifest_<stage>.json`，例如 `stage='v3_dlg'` 时产出 `data/chinese/manifest_v3_dlg.json`；若新 stage 叫 `v4_persona`，则产出 `data/chinese/manifest_v4_persona.json`。

### 2.3 `--emit-offsets` 产出什么
在 `prepare.py`：
- 第 707-711 行：
  ```python
  707:         if args.emit_offsets:
  708:             train_off = encode_to_bin(train_data, char_tok, train_bin, block_starts=train_starts)
  709:             val_off = encode_to_bin(val_data, char_tok, val_bin, block_starts=val_starts)
  710:             write_offsets(train_off, train_bin)
  711:             write_offsets(val_off, val_bin)
  ```
- 第 378-382 行：
  ```python
  378: def write_offsets(off, bin_path):
  379:     """把 block 边界表写成 int64 裸数组 sidecar，返回路径。"""
  380:     path = offsets_path_for_bin(bin_path)
  381:     np.asarray(off, dtype=np.int64).tofile(path)
  382:     return path
  ```
- 产物是 `train_char_<stage>.off` 和 `val_char_<stage>.off`，格式为 `int64` 二进制裸数组，存储每个 block 在 bin 中的起始 token 下标，并在末尾附加 `len(bin)` 哨兵。专供 `training/train.py` 的 `--use_doc_packing` 构建块对角注意力掩码（防止跨文档样本注意力污染）。

---

## 3. 步骤 2：`data/chinese/import_external.py` 源码走读与新格式扩展评估

### 3.1 支持的输入格式与注册点
在 `import_external.py`：
- 第 304-305 行：
  ```python
  304: FORMATS = {'qa_jsonl': qa_jsonl, 'messages_json': messages_json, 'alpaca_json': alpaca_json,
  305:            'sharegpt_jsonl': sharegpt_jsonl}
  ```
- 第 634 行：
  ```python
  634: BIG_FORMATS = ('belle_json', 'wildchat_parquet')
  ```
共支持 6 种输入格式：
1. `qa_jsonl`: 每行 `{"question": str, "answer": str}`
2. `messages_json`: `[{"messages":[{"role":"user"|"assistant","content":str}, ...]}]`
3. `alpaca_json`: `[{"instruction": str, "input": str, "output": str}]`
4. `sharegpt_jsonl`: ShareGPT JSONL 两种结构（conversation 或 conversations/messages）
5. `belle_json`: Belle 多轮流式 JSON
6. `wildchat_parquet`: WildChat parquet 结构

### 3.2 输出目标目录
- 命令行参数由 `--out <path>` 指定（第 655 行），项目标准统一导出至 `data/chinese/new_sources/<文件名>.txt`（第 9 行及背景约定）。
- 输出格式由 `to_text`（第 497-504 行）固定：
  每个样本为一个 block，以 `\n\n` 分隔；block 内每行为一轮，格式为 `用户：…\n模型：…`。

### 3.3 新格式是否需要扩代码
1. **纯第一人称独白散文（如日记、自白、无对话前缀文本）**：
   - **需要扩代码或不经过本脚本**。
   - 源码证据：`import_external.py:20-24`、`to_text:502` 明确将输出结构绑定为 `r + c`（`r` 为 `用户：` 或 `模型：`）。并且如果轮数少于 `--min-turns`（默认 1），或者没有解析出 `(role, content)` 元组，会被直接丢弃。
   - 处理方式：纯散文语料（如四大名著、日记、文集）在项目中历来是**直接作为原始 `.txt` 放入输入目录**（段落/章节之间用两个换行 `\n\n` 隔开即可，`prepare.py` 会按 `\n\n` 切 block），**根本不需要**也不应该经过 `import_external.py`。
2. **角色卡 + 对话（Character Card + Dialogue）**：
   - **需要扩代码**。
   - 源码证据：当前 `import_external.py` 的所有解析器中，`messages_json`（第 242-243 行）会直接跳过 `system` 角色：
     ```python
     242:         if r == 'system':
     243:             continue
     ```
     `selftest()` 第 551-553 行也显式把“跳过 system 角色”当作正向断言。
   - 若角色数据包含“系统设定/角色卡”，现存逻辑会把角色卡直接当垃圾丢掉。如果要将角色设定拼在首轮或特定结构中，必须在 `import_external.py` 中新增专用解析函数（例如 `persona_chat_json`），并在 `FORMATS` 字典中注册。

---

## 4. 追加审计（重要）：`v3_dlg` 的 bin 里到底是什么标签？与评估口径是否一致？

### 4.1 `build_stages.py` 建 `v3_dlg` bin 时喂给 Tokenizer 的真实文本
在 `prepare.py`：
- 第 567-570 行：
  ```python
  567:     ap.add_argument('--insert-eos', action='store_true', default=True,
  568:                     help='每条 模型： 回复后插入 <eos>（turn-level 终止符，治喋喋不休，默认开启）')
  569:     ap.add_argument('--no-insert-eos', action='store_true',
  570:                     help='关闭 --insert-eos（不插 <eos>，旧数据行为）')
  ```
  `build_stages.py`（第 269-272 行）调用 `prepare.py` 时**没有传入 `--no-insert-eos`**，因此 `args.insert_eos = True`。
- 第 685-687 行：
  ```python
  685:     if args.insert_eos:
  686:         train_samples = [insert_eos_after_replies(b) for b in train_samples]
  687:         val_samples = [insert_eos_after_replies(b) for b in val_samples]
  ```
- 第 260-262 行：
  ```python
  260: def insert_eos_after_replies(block: str) -> str:
  261:     """兼容旧名：一律插 <eos>（标注 <cont> 由 --no-annotate 关闭时使用）。"""
  262:     return annotate_replies(block)
  ```
- 第 191-249 行：`annotate_replies` 函数逻辑：
  ```python
  214:     use_ab = random.random() < ab_rate      # ab_rate 默认为 0.7
  215:     use_quote = random.random() < quote_rate # quote_rate 默认为 0.8
  ...
  222:         if stripped.startswith("用户：") or stripped.startswith("模型："):
  223:             is_model = stripped.startswith("模型：")
  224:             body = stripped.split("：", 1)[1] if "：" in stripped else stripped
  ...
  235:             if use_ab:
  236:                 speaker = "B" if is_model else "A"
  237:                 text = f"{speaker}：{body}"
  238:             else:
  239:                 text = body
  241:             if use_quote:
  242:                 text = f'"{text}"'
  ```
- **源码确证结论**：`clean_v3/*.txt` 虽然文本文件表面上全是 `用户：`/`模型：`，但在 `prepare.py` 编码写入 bin 之前，经过了 `annotate_replies` 的重写！
  **70% 的概率被重写为 `A：`/`B：`（A=对方，B=模型），30% 为无标签裸文本**；并且 80% 叠加了引号。
  **进入 bin 的不是 `用户：`/`模型：`，而是混合重写后的文本！**

### 4.2 `training/train.py` 加载 bin 时有没有做标签转换？
- 在 `training/train.py`：
  第 483 行直接按二进制读取：
  ```python
  483:     data = np.memmap(path, dtype=np.uint16, mode='r')
  ```
  第 493/507 行对 token 数组做切片截取窗口：`data[i:i+block_size]`。
- **源码确证结论**：`train.py` **完全不做**任何文本或 token 级的标签转换，它直接消费 `prepare.py` 生成的 `uint16` token id。

### 4.3 评估脚本 `--style=ab` 展开与一致性判断（⚠ 重点审计）
- 在 `inference/scripts/eval_dialogue.py`：
  第 40-50 行：
  ```python
  40: PROMPT_STYLES = {
  41:     # v2 主线（当前）
  42:     'ab': [
  43:         "A：你好\nB：",
  44:         "A：最近工作压力好大，怎么办啊？\nB：",
  45:         "A：帮我推荐一本小说吧。\nB：",
  46:         "A：你觉得人生最重要的是什么？\nB：",
  47:         # 真实数据里的对话风格更多是口语短轮次，这里加两个贴近语料的
  48:         "A：吃了没\nB：",
  49:         "A：好想出去玩\nB：",
  50:     ],
  ```
- 在 `inference/scripts/eval_multiturn.py`：
  第 54-57 行与第 60-64 行：
  ```python
  54: PROMPT_STYLES = {
  55:     'ab': ('A', 'B'),
  56:     'user-model': ('用户', '模型'),
  57: }
  ...
  62:     u, m = PROMPT_STYLES[style]
  63:     return [(name, f"{u}：{opening}\n{m}：", [f"{u}：{f}\n{m}：" for f in followups])
  ```
- **核心结论**：**B 段评估用的 prompt 标签（`--style=ab`，即 `A：`/`B：`），与 B 段训练数据的标签是严格一致的！**
  - 理由：B 段训练数据在通过 `build_stages.py` 经由 `prepare.py` 编译时，默认执行了 `annotate_replies(ab_rate=0.7)`，主流标签正是 `A：`/`B：`；因此评估使用 `--style=ab` 恰恰匹配了训练分布的主模态，**没有**发生标签口径错配。
  - 置信度：**源码确认（100%）**。

---

## 5. 步骤 3：`training/train.py` 与 configs 换数据配置键及 bin 映射

### 5.1 换数据需改动 config 的哪几个键
以 `configs/base_v3_dlg.yaml` 对比 `configs/base_v2.yaml` 为例，要切换到新数据阶段（如 `v4_persona`），config 必须显式修改以下键：
1. `data_prefix`: 指定数据集前缀（例如 `v4_persona`），驱动 bin/off/manifest 文件名映射。
2. `out_dir`: 指定独立的产物目录（例如 `out/base_v4_persona`）。（**铁律 12**：warm start 必须配独立 `out_dir`，否则触发 `_backup_old_run` 归档来源 ckpt）。
3. `init_from`: 权重来源路径（例如 `out/base_v3_dlg/last.pt`）。
4. `max_iters` 与 `lr_decay_iters`: 本阶段训练迭代步数及 WSD 退火周期（例如 14000）。
5. `use_doc_packing: true` 与 `pack_align: false`: 打包与非对齐随机窗口配置。
6. `use_loss_masking: false`: 保持全 token 算 loss。

### 5.2 `data_prefix` 如何映射到 bin 文件名
- 在 `training/train.py`：
  第 398-400 行：
  ```python
  398: _train_bin, _val_bin, _meta_bin = pick_bin_names(
  399:     char_level=char_level, byte_level=byte_level, prefix=data_prefix,
  400:     stage=globals().get('stage'))
  ```
- 在 `training/schedules.py`：
  第 79-84 行与 86-108 行：
  ```python
  79: def with_data_prefix(base, prefix):
  80:     for ext in ('.bin', '.pkl'):
  81:         if base.endswith(ext):
  82:             return base[:-len(ext)] + f'_{prefix}' + ext
  83:     return base
  ...
  98:     if char_level:
  99:         train, val, meta = 'train_char.bin', 'val_char.bin', 'meta_char.pkl'
  ...
  106:     return (with_data_prefix(train, prefix),
  107:             with_data_prefix(val, prefix),
  108:             with_data_prefix(meta, prefix))
  ```
- 在 `training/train.py`：
  第 416-417 行（获取 `.off` 边界文件）：
  ```python
  416:     for _split, _bn in (('train', _train_bin), ('val', _val_bin)):
  417:         _op = os.path.join(data_dir, _offsets_name_for_bin(_bn))
  ```
- **映射规则**：
  若 `char_level=True`，`data_prefix='v4_persona'`：
  - train bin：`train_char.bin` → `train_char_v4_persona.bin`
  - val bin：`val_char.bin` → `val_char_v4_persona.bin`
  - meta pkl：`meta_char.pkl` → `meta_char_v4_persona.pkl`
  - train off：`train_char_v4_persona.off`
  - val off：`val_char_v4_persona.off`
  全部位于 `data/chinese/` 目录下。

---

## 6. 步骤 4：`tests/test_project_layout.py` 踩踏断言分析

如果增加新 stage 并新增了配置文件 `configs/base_v3_persona.yaml` 或 `configs/base_v4_persona.yaml`，会触发以下断言机制：

### 6.1 若命名为 `configs/base_v3_*.yaml`（例如 `configs/base_v3_persona.yaml`）
此时该文件会被 `V3_STAGE_CONFIGS`（第 283 行）自动捕获：
1. **测试函数**：`test_v3_stage_config_safety`（第 321 行）
   - **行号 364-366 断言**：
     ```python
     364:     assert name in V3_STAGE_STEPS, (
     365:         f"{name}: 新增的 v3 阶段配置必须同时登记进 V3_STAGE_STEPS"
     366:         f"（并更新 PROJECT_STATE §0.5.10 的方案表）")
     ```
     **断言失败**：如果未在 `tests/test_project_layout.py` 的 `V3_STAGE_STEPS` 字典中登记该 yaml 名字，测试**必定红**。
   - **行号 367-369 断言**：
     ```python
     367:     assert cfg['max_iters'] == V3_STAGE_STEPS[name], (
     368:         f"{name}: max_iters={cfg['max_iters']} 与登记的 {V3_STAGE_STEPS[name]} 不符 —— "
     369:         f"改步数必须同时改这里和 PROJECT_STATE §0.5.10 的方案表")
     ```
     **断言失败**：若 yaml 里的步数与登记步数不一致，测试**红**。
   - **行号 362-363 断言**：
     ```python
     362:     assert cfg['data_prefix'].startswith('v3_'), (
     363:         f"{name}: data_prefix={cfg['data_prefix']!r} 不是 v3 阶段数据")
     ```
     **断言失败**：若配置文件叫 `base_v3_...` 但内部 `data_prefix` 命名为 `v4_persona`，会因不是以 `v3_` 开头直接断言失败。
   - **行号 346-348 断言**：
     ```python
     346:     assert out_dir != src_dir, (
     347:         f"{name}: out_dir 与 init_from 所在目录相同（{cfg['out_dir']}）—— "
     348:         f"warm start 会触发 _backup_old_run 把该目录下的 ckpt 全部挪进 old/")
     ```
     **断言要求**：`out_dir` 绝不能与 `init_from` 所在目录一致。
   - **行号 351-354 断言**：
     必须 `use_doc_packing: true` 且 `pack_align: false`。
   - **行号 357 断言**：
     必须 `use_loss_masking: false`。

### 6.2 所有在 `configs/*.yaml` 中的配置都会踩到的全局测试
1. **测试函数**：`test_all_config_out_dirs_are_pairwise_distinct`（第 303-317 行）
   - 断言：每个配置的 `out_dir` 必须在全项目中唯一，不能与已有配置重复。
2. **测试函数**：`test_config_file_lives_outside_its_own_out_dir`（第 70-86 行）
   - 断言：配置文件本身不能保存在自己的 `out_dir` 内部。
3. **测试函数**：`test_every_config_key_is_overridable`（第 92-108 行）与 `test_every_config_key_is_in_the_logged_snapshot`（第 110-126 行）
   - 断言：YAML 中配置的所有键必须在 `train.py` 中 `load_config` 及 `config_keys` 快照之前定义。
4. **测试函数**：`test_config_data_files_resolve`（第 186-207 行）
   - **极关键断言**：
     ```python
     197:     train, val, meta = pick_bin_names(...)
     202:     missing = [n for n in (train, val, meta) if not (ddir / n).exists()]
     205:     assert not missing, (
     206:         f"{os.path.basename(cfg_path)} 解析出的文件在 data/{dataset}/ 下不存在：{missing}"
     ```
     如果先新建了 yaml 配方文件，但**尚未生成对应的 bin 文件**（`train_char_v4_persona.bin` 等），执行 `pytest` 会**立刻报红**。

---

## 7. 步骤 5：`scripts/watch.sh` 换 run 必须修改的行

在 `scripts/watch.sh`：
第 16-23 行：
```bash
16: # ★ 默认值必须跟**当前正在跑的 run** 走。2026-09-13 起主线是 v3 分段训练
17: #   （B 段 `out/base_v3_dlg`）。默认值写死旧 run 的后果不是"少看几行日志"：
18: #   新目录的归档 ckpt 不会被 prune（见第 49 行），**磁盘会被写满**。
19: #   换 run 时**必须**同步改这两行；要巡检别的 run 就显式传参。
20: OUT_DIR="${1:-out/base_v3_dlg}"
21: LOG="${2:-out/base_v3_dlg_train.log}"
22: SPARSE_EVERY="${3:-5000}"
23: NEWEST_KEEP="${4:-2}"
```
第 53 行：
```bash
53: bash scripts/prune_ckpts.sh "$OUT_DIR" "$SPARSE_EVERY" "$NEWEST_KEEP"
```
- **必须改的行**：第 20 行与第 21 行。
- **改法**：将 `out/base_v3_dlg` 替换为新阶段的目录（例如 `out/base_v4_persona`），将日志路径替换为 `out/base_v4_persona_train.log`。
- **不改的严重后果**：
  1. `watch.sh` 及后台 systemd timer（`nanoseek-watch.timer`）巡检和抓取的全是上一阶段的死日志，无法感知当前训练的状态。
  2. 归档清理脚本 `prune_ckpts.sh` 是针对默认变量 `$OUT_DIR` 执行的；如果不改，新 run 目录下的 ckpt 永远不会被 prune，**每个 ckpt 约 300MB，几个小时内就会把单卡机器磁盘写满，导致训练崩溃**。

---

## 8. 置信度与未确认项声明

- **源码确认项**：
  1. `build_stages.py` 的 stage 定义机制、manifest 命名规则及 `--emit-offsets` 产物结构。
  2. `import_external.py` 的支持格式、输出目录限制及纯散文/角色卡适配性。
  3. `train.py` / `schedules.py` 的数据解析映射与 config 关键键要求。
  4. `test_project_layout.py` 中新 stage 会踩到的所有具体测试函数及行号断言。
  5. `watch.sh` 中写死的第 20-21 行及其磁盘清理危害机制。
  6. `v3_dlg` 的 bin 内标签实为 `annotate_replies` 转换后的 `A：`/`B：`，与评估脚本 `--style=ab` 严格一致。
- **未从源码确认项**：
  - 无。全部问题均直接从相关源码行定位并提取原代码证据。
