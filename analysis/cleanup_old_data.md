# 旧派生数据清理报告

- 执行时间：2026-09-13 16:00 前后（本地）
- 项目根：`/home/vesita/coding/my/nanoSeek`
- 角色：执行者（删除清单由主 AI 给定；未扩大删除范围）
- 环境：`.venv/bin/python`（唯一 python）；**未使用 `uv run`，未用裸 `python`**
- 训练/冒烟：未启动任何训练、未 `pkill`、未碰 `/tmp/ns_smoke_a*`（巡检确认日志仍在，15:55）

---

## 0. 结论速览

| 项目 | 结果 |
|---|---|
| **实际释放总量** | **4,198,666,731 bytes = 3.91 GiB = 4.20 GB（十进制）** |
| A. 备份目录二进制 | 27 个文件，3,869,854,853 B（3.60 GiB） |
| B. v1 非 char 级 bin | 5 个文件，328,811,878 B（0.306 GiB） |
| C. `out/` 全树 | 0 个文件（脚本判定无可删派生物） |
| 保护集 sha256 前后 | **4/4 逐位相同** |
| 闸门 pytest | exit=0（前/后均绿） |
| 闸门 ruff | exit=0，`All checks passed!`（前/后均绿） |
| `git status` | 无任何 tracked 文件新增/删除/修改 |
| `git diff --stat tests/test_project_layout.py` | 前后逐字相同（`91 insertions` 未变） |
| 跳过项 | `data/chinese/train.bin`、`data/chinese/_hf_cache/`、`_discarded/`、`_backup_task17_*`（后两项本就不在清单） |

---

## 1. A. 旧 bin 备份目录里的二进制（§2A）

规则：每个目录**只删** `*.bin` / `*.pkl` / `*.npz` / `*.pt`；`.py/.json/.md/.txt/.log` 一律保留。
实际枚举（`find ... -type f \( -name '*.bin' -o -name '*.pkl' -o -name '*.npz' -o -name '*.pt' \)`）共 **27 个文件**，全部为 `.bin` / `.pkl`（无 `.npz` / `.pt`）。

执行时加了守卫：逐条校验「路径必须在 5 个允许目录内 + 扩展名必须在白名单 + 不在 `.venv`/`local/`/`datasets`」，27/27 通过，`bad=0`。

| 目录 | 删除文件数 | 释放 | 原 `du -sh` → 现 `du -sh` | 保留物 |
|---|---:|---:|---|---|
| `data/chinese/_backup_20260903_005043/` | 9 | 700,759,039 B | 1.1G → 386M | 12 个 `.txt` + `manifest.json`（13 个文件） |
| `data/chinese/_backup_20260907_212857/` | 9 | 903,891,279 B | 1.4G → 524M | 23 个 `.txt` + 2 个 tokenizer `.json` + `manifest.json`（26 个文件） |
| `data/chinese/_backup_barv_20260908_222540/` | 3 | 1,702,391,569 B | 1.6G → 8.0K | `manifest.json` |
| `data/chinese/_backup_cleanup_20260908_214617/` | 3 | 281,406,483 B | 269M → 8.0K | `manifest.json` |
| `data/chinese/_backup_prebuild_20260908_221130/` | 3 | 281,406,483 B | 269M → 0（目录已空） | 无（原本就只有 3 个二进制） |
| **合计** | **27** | **3,869,854,853 B（3.60 GiB）** | | |

明细（`大小(字节)  路径`）：

```
46         data/chinese/_backup_20260903_005043/meta_byte.pkl
85         data/chinese/_backup_20260903_005043/meta_char.pkl
66         data/chinese/_backup_20260903_005043/meta.pkl
75658480   data/chinese/_backup_20260903_005043/train.bin
275280570  data/chinese/_backup_20260903_005043/train_byte.bin
281349794  data/chinese/_backup_20260903_005043/train_char.bin
11105510   data/chinese/_backup_20260903_005043/val.bin
42425686   data/chinese/_backup_20260903_005043/val_byte.bin
14938802   data/chinese/_backup_20260903_005043/val_char.bin
46         data/chinese/_backup_20260907_212857/meta_byte.pkl
85         data/chinese/_backup_20260907_212857/meta_char.pkl
66         data/chinese/_backup_20260907_212857/meta.pkl
75658480   data/chinese/_backup_20260907_212857/train.bin
275280570  data/chinese/_backup_20260907_212857/train_byte.bin
484619942  data/chinese/_backup_20260907_212857/train_char.bin
11105510   data/chinese/_backup_20260907_212857/val.bin
42425686   data/chinese/_backup_20260907_212857/val_byte.bin
14800894   data/chinese/_backup_20260907_212857/val_char.bin
85         data/chinese/_backup_barv_20260908_222540/meta_char.pkl
1687594310 data/chinese/_backup_barv_20260908_222540/train_char.bin
14797174   data/chinese/_backup_barv_20260908_222540/val_char.bin
85         data/chinese/_backup_cleanup_20260908_214617/meta_char.pkl
266600624  data/chinese/_backup_cleanup_20260908_214617/train_char.bin
14805774   data/chinese/_backup_cleanup_20260908_214617/val_char.bin
85         data/chinese/_backup_prebuild_20260908_221130/meta_char.pkl
266600624  data/chinese/_backup_prebuild_20260908_221130/train_char.bin
14805774   data/chinese/_backup_prebuild_20260908_221130/val_char.bin
```

删空目录（`rmdir`，不是 `rm -rf`）：

- `data/chinese/_backup_prebuild_20260908_221127/` —— 清单里标为空，`find -mindepth 1` 确认 0 项 → `rmdir` 成功。
- `data/chinese/_backup_prebuild_20260908_221130/` —— **原本只有 3 个二进制**（删完后 `ls -A` = 0 项）→ `rmdir` 成功。这是清单未预料的“派生空目录”，无数据损失。
- `data/chinese/_hf_cache/` —— 清单说“空的”，但 `ls -A` 显示 **非空**（见 §9）⇒ **跳过，未 rmdir**。

---

## 2. B. v1 非 char 级 bin（§2B）—— 逐文件判定

核验命令（清单原文，逐文件跑）：

```bash
grep -rn --include='*.py' --include='*.sh' --include='*.yaml' -- "<文件名>" . \
  | grep -v '\.venv' | grep -v '_backup' | grep -v '^\./local/'
```

另加一道“真实读取调用”正则（更严）：

```bash
grep -rnE "(open|memmap|fromfile|np\.load|torch\.load|pickle\.load)\(.*['\"]<文件名>['\"]" \
  --include='*.py' . | grep -v '\.venv' | grep -v '_backup' | grep -v '^\./local/'
```

| 文件 | 原大小 | 引用命中数 | 直接读取调用 | 谁在读 | 结果 |
|---|---:|---:|---:|---|---|
| `data/chinese/train.bin` | 75,658,480 B | 74 | **2** | `model/probe.py:83`（`np.memmap`，路径 `data/<dataset>/train.bin`，`dataset` 来自 ckpt config；`configs/base_v2.yaml:16` = `chinese`），`inference/bench.py:42`（`np.memmap(os.path.join(data_dir,'train.bin'))`，`dataset` 可被 `load_config` 覆盖成 `chinese`） | **跳过** |
| `data/chinese/val.bin` | 11,105,510 B | 49 | 0 | 仅“名字/返回值/注释”，`data/shakespeare/prepare.py:30` 是**它自己目录下的写**（`tofile`） | 成功删除 |
| `data/chinese/train_byte.bin` | 275,280,570 B | 9 | 0 | 仅名字列表（`training/schedules.py:101` 返回值、`training/govern_dialogue_data.py:28` `GEN_FILES`、`tests/test_schedules.py`、`tests/test_project_layout.py` AST 扫描） | 成功删除 |
| `data/chinese/val_byte.bin` | 42,425,686 B | 7 | 0 | 同上（名字列表） | 成功删除 |
| `data/chinese/meta.pkl` | 66 B | 10 | 1 | 该 1 处是 `data/shakespeare_char/prepare.py:60` 的**写**（`open(...,'wb')`，另一个数据集目录）；`training/schedules.py:103` 返回值、`inference/scripts/package.py:100` 帮助字符串（且其 `CONVERT_PY=inference/scripts/convert.py` **文件不存在**，该脚本已失效）；`train.py:546` 读的是变量 `_meta_bin`（BPE 分支），且带 `os.path.exists` 兜底 | 成功删除 |
| `data/chinese/meta_byte.pkl` | 46 B | 7 | 0 | 仅名字列表 / 注释 | 成功删除 |

**B 小计：删 5 个，328,811,878 B（0.306 GiB）；跳过 1 个（`train.bin`）。**

判定说明：
- 清单的判定标准是「引用只能是字符串常量/名字列表，不能是真的 `open/memmap/fromfile` 在读这个文件」。`train.bin` 是唯一**在非 `local/`、非 `_backup/` 代码里存在真实 `np.memmap` 读路径**的文件（`model/probe.py` 的 `dataset` 取自 ckpt config，当前基座就是 `dataset: chinese`），故按规则**跳过**并在此写明“是 `model/probe.py:83` 与 `inference/bench.py:42` 在读”。
- 其余 5 个没有任何直接读取调用；`training/schedules.py::pick_bin_names` 的返回值属于清单**明确举例认可**的“字符串常量/名字列表”。
- **补充风险提示（供主 AI 决策）**：`training/train.py` 在 `char_level=false, byte_level=false`（模块级默认）时会经 `pick_bin_names`→`np.memmap(data_dir/_val_bin)` 读 `val.bin`，并在 `train.py:546` 读 `meta.pkl`；`byte_level=true` 时读 `*_byte.*`。但**当前没有任何配置走这两个分支**：`configs/*.yaml` 全部直接或 `extends: base_v2.yaml` 继承到 `char_level: true`（`grep -rn "char_level\|byte_level" configs/*.yaml` 只有 `base_v2.yaml:10:byte_level:false`、`:11:char_level:true`）。`tests/test_project_layout.py::test_config_data_files_resolve` 亦只覆盖 configs 解析出的（char 级）文件，故本次删除不触碰闸门。若日后要复活 BPE/字节模式，这 5 个需按 `prepare.py` 重建。

---

## 3. C. `out/` 全树（§2C）

用现成脚本，未手写 `rm`。

**dry-run**（输出存 `analysis/_cleanup_out_dryrun.txt`，exit=0）：

```
  .pt 删除          0.0 MB   （2 个保护目录的 .pt 10.19 GB 未动）
  .npz 缓存表       0.0 MB
  会话残留目录      0.0 MB
  ★ 合计释放        0.0 MB
  ★ 保留证据       35.0 MB  （全部 .log/.json/.csv/.png，一字不丢）
```

**`--apply`**（输出存 `analysis/_cleanup_out_apply.txt`，exit=0）：

```
清单已写入 out/CLEANUP_MANIFEST.md
✅ 已删除 0 个文件/目录
```

结果：`out/` 树内**已经没有**保护名单之外的可删派生物（`.pt`/`.npz`）。
现 `out/` 内 `.pt` 只剩两个保护目录：`out/base_v2` 15 个 / 8.9G，`out/nanoseek_100m` 2 个 / 1.4G（合计 10.19 GB，脚本报“未动”）。`out/CLEANUP_MANIFEST.md` 已由脚本写出（360 B）。

---

## 4. 保护集 sha256 前后对照（§4.1）

对且仅对指定的 4 个路径计算；删前存档到 `analysis/_cleanup_sha_before.txt`，删后 `analysis/_cleanup_sha_after.txt`。

| 文件 | 删前 sha256 | 删后 sha256 | 一致 |
|---|---|---|---|
| `data/chinese/train_char_v3_know.bin` | `858956d26916d57714eb8ac81e53ae34a859aa4be7c28ba358457b022b583f34` | 同左 | ✅ |
| `data/chinese/val_char_v2.bin` | `a724ee1be896ce33418fadf9bc1b6a4a88782d0b20848c42f89e4bb764ebff05` | 同左 | ✅ |
| `data/chinese/manifest_v3_dlg.json` | `5a8db1f576a9e814e75139807582b7e7792732864e16a6422bfbc374ba895e3f` | 同左 | ✅ |
| `out/base_v2/last.pt` | `1c46cac122f9d6b69877895db43bc58bff984f54c900676ae582b875285b0b5d` | 同左 | ✅ |

`diff analysis/_cleanup_sha_before.txt analysis/_cleanup_sha_after.txt` → 无差异（4/4 逐位相同）。

“文件仍在”（`ls -la`，删后）：

```
-rw-r--r-- 1 vesita vesita      6363  9月13日 15:34 data/chinese/manifest_v3_dlg.json
-rw-r--r-- 1 vesita vesita 975510814  9月13日 14:44 data/chinese/train_char_v3_know.bin
-rw-r--r-- 1 vesita vesita  18835890  9月10日 22:02 data/chinese/val_char_v2.bin
-rw-r--r-- 1 vesita vesita 635783601  9月13日 12:03 out/base_v2/last.pt
```

其余在用清单项亦逐一点名确认仍在（`ls data/chinese/*.bin data/chinese/*.pkl`）：
`train_char_v3_{lang,know,dlg}.bin`、`val_char_v3_{lang,know,dlg}.bin`、`train_char_v2.bin`、`val_char_v2.bin`、`train_char.bin`、`val_char.bin`、`meta_char{,_v2,_v3_lang,_v3_know,_v3_dlg}.pkl`、`tokenizer.json`、`char_tokenizer.json`、`char_tokenizer.json.bak_layout`、`manifest{,_v2,_v3_lang,_v3_know,_v3_dlg}.json` 全部在位。

---

## 5. `git status --short` 前后对比（§4.3）

- 被删的 32 个文件（A 27 + B 5）**全部是 gitignore / untracked**（`.gitignore:12-14,17-22,28,34`），因此 `git status` 中**不出现任何 ` D` 条目**：`git ls-files` 对这 32 个路径返回 0 条。
- 前后 `diff` 的唯一差异 = 我新落盘的证据文件（均在 `analysis/` 下，全部 `??` untracked）：

```
19a20,25
> ?? analysis/_cleanup_A_files.txt
> ?? analysis/_cleanup_df_after.txt
> ?? analysis/_cleanup_df_before.txt
> ?? analysis/_cleanup_gate_after.txt
> ?? analysis/_cleanup_gate_before.txt
> ?? analysis/_cleanup_git_after.txt
20a27,30
> ?? analysis/_cleanup_git_final.txt
> ?? analysis/_cleanup_gitdiff_after.txt
> ?? analysis/_cleanup_gitdiff_before.txt
> ?? analysis/_cleanup_out_apply.txt
21a32
> ?? analysis/_cleanup_sha_after.txt
```

即：**没有任何 tracked 文件被新增、删除或修改**；没有任何 `data/`、`out/`、`configs/`、`training/`、`tests/` 条目变化。`analysis/` 下自有 scratch 证据文件（`.txt`）按纪律保留未删。

- 既有改动 `configs/base_v2.yaml` / `configs/base_v3_{know,dlg}.yaml` / `tests/test_project_layout.py` 等仍是主 AI 的，未动。
- 主 AI 要求比对的：

```
$ diff analysis/_cleanup_gitdiff_before.txt analysis/_cleanup_gitdiff_after.txt
(无输出 —— IDENTICAL)
 tests/test_project_layout.py | 91 ++++++++++++++++++++++++++++++++++++++++++++
 1 file changed, 91 insertions(+)
```

删前删后**完全一致**，证明我没有改它。

---

## 6. 闸门（§4.4）

| 时点 | 命令 | 结果 |
|---|---|---|
| 删前 | `.venv/bin/python -m pytest -q -m 'not slow'` | `pytest_exit=0` |
| 删前 | `.venv/bin/python -m ruff check .` | `All checks passed!` / `ruff_exit=0` |
| 删后 | `.venv/bin/python -m pytest -q -m 'not slow'` | `pytest_exit=0` |
| 删后 | `.venv/bin/python -m ruff check .` | `All checks passed!` / `ruff_exit=0` |

删后 pytest 尾行（原样）：

```
........................................................................ [ 67%]
........................................................................ [ 84%]
..................................................................       [100%]
```

删后 ruff 尾行（原样）：

```
All checks passed!
```

⚠ 说明：`.venv` 的 pytest 是 **9.1.1**，`-q` 模式下**不打印 “N passed in Xs” 汇总行**（已用单文件 `pytest tests/test_schedules.py -q` 复现同样只有进度点）。故 pytest 的权威信号是 **exit code = 0**（前后均为 0，无 fail/error）。

---

## 7. `df -h /home` 前后（§4.5）

```
删前：/dev/nvme0n1p7  270G  196G   72G  74% /home
删后：/dev/nvme0n1p7  270G  194G   75G  73% /home
```

- 实际删除字节数（精确求和）：**4,198,666,731 B = 3.91 GiB = 4.20 GB（十进制）**
  - A：3,869,854,853 B（3.60 GiB）
  - B：328,811,878 B（0.306 GiB）
- 口径说明：`df -h` 只给 1G 分辨率（已用 196G→194G，可用 72G→75G）。按 GiB 计释放 3.91G，与 df 的「+3G 可用」同量级；差异来自 df 的四舍五入与期间其他进程的并发写盘（本次未启动任何训练，也未碰冒烟）。

---

## 8. 边界确认（绝对禁止项逐条自查）

| 禁止项 | 状态 |
|---|---|
| 不碰 `/home/vesita/datasets/NLP/` | ✅ 未触碰（删后仍 33 文件 / 3.8G） |
| 不碰 §1 在用清单 | ✅ 未删任何在用小文件；4 个指纹逐位相同 |
| 不删 `.py/.md/.json/.txt/.log/.csv` | ✅ 本次删除对象仅 `*.bin`/`*.pkl`；证据类文件全部保留 |
| 不 `rm -rf` 目录 | ✅ 仅按具体文件 `rm --`；空目录用 `rmdir`（2 个） |
| 不碰 `data/chinese/_discarded/` | ✅ 仍在，266M |
| 不改代码 / 不 `git add/commit/checkout/clean` | ✅ 仅运行 pytest/ruff（只读）与清理脚本 |
| 不启动训练 / 不 `pkill` / 不碰冒烟 | ✅ `/tmp/ns_smoke_a`、`/tmp/ns_smoke_a.log` 未动 |

---

## 9. 跳过的项与原因（§5）

| 跳过项 | 原因 | 证据 |
|---|---|---|
| `data/chinese/train.bin`（75,658,480 B） | 有**非 `local/`** 的真实读取：`model/probe.py:83` 用 `np.memmap` 读 `data/<dataset>/train.bin`，`dataset` 取自 ckpt config（当前 = `chinese`）；`inference/bench.py:42` 同样 `np.memmap`（`dataset` 可覆盖） | §2 表；`configs/base_v2.yaml:16 dataset: chinese` |
| `data/chinese/_hf_cache/` | 清单写“空的，直接 rmdir”，但**实际非空**：`_hf_cache/.cache/huggingface/` 下有 `CACHEDIR.TAG`、`.gitignore`、`download/wikipedia-cn-20230720-filtered.json.lock`、一个 `.incomplete` 下载碎片。既非空、其内容也不在 A 的扩展名白名单内 ⇒ 不做 `rm -rf`，整个跳过（`rmdir` 会因非空失败） | `ls -A data/chinese/_hf_cache` → `.cache` |
| `data/chinese/_discarded/`（266M） | 主 AI 明示保留（不可重建） | 仍在 |
| `data/chinese/_backup_task17_20260910_213459/`（32K） | 主 AI 明示保留 | 仍在 |
| 所有 `.py/.md/.json/.txt/.log/.csv` | §0.3 绝对禁止 | A 目录里 `.txt/.json/tokenizer.json` 全在 |

---

## 10. 异常 / 偏差记录

1. **`_hf_cache` 与清单描述不符**：清单说“空的”，实测非空（含一个未完成的 HF 下载与 lock）。未按清单 `rmdir`（会失败），改为**整项跳过**。
2. **`_backup_prebuild_20260908_221130` 删后变空**：该目录原本只含 3 个二进制，无 `.json` 证据，删完 `ls -A`=0 ⇒ 按纪律 4 用 `rmdir` 移除。
3. **`cleanup_out.py` 实删 0 个**：`out/` 里已无非保护 `.pt/.npz`，脚本 dry-run 与 `--apply` 均为 0.0 MB；`out/CLEANUP_MANIFEST.md` 已按要求写出。
4. **pytest 9.1.1 `-q` 无汇总行**：见 §6，用 exit code 作为判据。
5. **证据文件**：`analysis/_cleanup_*.txt` 为本次落盘的原始证据（sha/git/df/gate/A 文件清单），按“不删 `.txt`”纪律保留。

---

## 附：本次实际执行的删除命令形态

```bash
# A：从 analysis/_cleanup_A_files.txt 逐条读 大小\t路径，守卫校验后
rm -- "<每个具体文件>"          # 27 次
# B：
rm -- data/chinese/val.bin data/chinese/train_byte.bin data/chinese/val_byte.bin \
      data/chinese/meta.pkl data/chinese/meta_byte.pkl
# 空目录（仅 rmdir，无 rm -rf）：
rmdir data/chinese/_backup_prebuild_20260908_221127
rmdir data/chinese/_backup_prebuild_20260908_221130
# C：
.venv/bin/python scripts/cleanup_out.py            # dry-run
.venv/bin/python scripts/cleanup_out.py --apply
```
