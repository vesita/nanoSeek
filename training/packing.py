"""样本打包（document packing）的纯函数：边界表读取/校验、窗口 sample_id、块对角掩码。

## 为什么需要它

语料被 `data/chinese/prepare.py` 压成**一条扁平 token 流**（block 之间只有 `\\n\\n`），
而训练采样是**全域随机窗口**（`torch.randint(len(data) - block_size, ...)`）⇒ 一个
256-token 窗口经常横跨多个样本。模型侧唯一的"连续"是窗口内的因果注意力（窗口左边界
硬截断、无跨窗状态），于是同一窗口里的样本 A 会污染样本 B 的注意力。

项目里原本有一套边界阻断（`model/attention.py` 的 `win_causal & same_sample`），但它只接在
`_csa_forward`（需 `use_csa=true`）与 `_kv_memory_forward`（需 `use_kv_memory=true`）两条路径上；
主线 MLA 普通因果路径**根本不接收 is_eos** ⇒ `sample_boundary_reset` 是死旋钮。
而且 `<eos>` **不足以**当边界：非对话 block（c4_zh / wikipedia / 名著）整块没有 `<eos>`。

所以本模块消费 `prepare.py --emit-offsets` 产出的 `.off` sidecar：
**int64 一维数组，每个元素 = 一个 block 在 bin 里的起始 token 下标**，
末尾补一个 `len(bin)` 作为哨兵，便于 `searchsorted` 求区间。

## 接口一览（全部纯函数，CPU 可测）

| 函数 | 作用 |
|---|---|
| `offsets_name_for_bin(bin_name)` | `train_char_v2.bin` → `train_char_v2.off`（bin 的同名 sidecar） |
| `load_offsets(path, n_tokens)` | 读 `.off` 并校验；不合法**大声报错**，绝不静默退化 |
| `validate_offsets(off, n_tokens, path)` | 校验：`[0]==0`、末元素==n_tokens、严格递增 |
| `sample_id_in_window(off, starts, T, device)` | `(B,T)` 的窗口内样本 id（0 起），`searchsorted` 向量化，无 python 循环 |
| `boundary_mask(sample_id)` | `(B,T,T)` bool：`same_sample = sid[:,:,None] == sid[:,None,:]` |
| `causal_boundary_mask(sample_id)` | `boundary_mask & tril`，可直接与因果掩码相与 |
| `aligned_pack_starts(off, T, B, generator)` | 块对齐打包：贪心装完整 block 到 ≤T，窗口结束落在 block 边界 |
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = [
    'offsets_name_for_bin',
    'load_offsets',
    'validate_offsets',
    'sample_id_in_window',
    'boundary_mask',
    'causal_boundary_mask',
    'aligned_pack_starts',
    'aligned_window_starts',
    'unreachable_token_fraction',
    'reachable_fraction',
    'mask_cross_sample_labels',
]


def offsets_name_for_bin(bin_name):
    """由 bin 文件名派生 `.off` sidecar 名（同一目录、同前缀）。

    这是"用哪份边界表"的**唯一**派生点：`train_char_v2.bin` → `train_char_v2.off`，
    与 `pick_bin_names()` 产出的文件名严格同源，避免在 train.py 里另写一份文件名拼接。
    """
    if not bin_name.endswith('.bin'):
        raise ValueError(f'bin 文件名必须以 .bin 结尾，收到 {bin_name!r}')
    return bin_name[:-len('.bin')] + '.off'


def validate_offsets(off, n_tokens, path='<offsets>'):
    """校验边界表；任何不合法都抛 `ValueError`（调用方不许把它当"没边界"处理）。

    合法条件（与 `prepare.py --emit-offsets` 的产出契约一一对应）：
    * 一维 int 数组，至少 1 个元素（空 split 只有哨兵 `[0]`）；
    * `off[0] == 0` 且 `off[-1] == n_tokens`（末位是哨兵，等于 bin 的 token 数）；
    * 严格递增（block 非空 ⇒ 每段至少 1 个 token）。
    """
    off = np.asarray(off)
    if off.ndim != 1:
        raise ValueError(f'{path}: .off 必须是一维数组，实际 ndim={off.ndim}')
    if off.size == 0:
        raise ValueError(f'{path}: .off 为空文件（至少要有首 0 哨兵）')
    if off[0] != 0:
        raise ValueError(f'{path}: 首元素必须为 0，实际 {off[0]} —— .off 与 bin 未对齐或文件损坏')
    if off[-1] != n_tokens:
        raise ValueError(
            f'{path}: 末元素（哨兵）必须等于 bin 的 token 数 {n_tokens}，实际 {off[-1]}。'
            f'常见原因：.off 与 bin 不是同一次 prepare 的产物 / --out-prefix 混用 / bin 被替换。'
            f'请用 `prepare.py --emit-offsets` 重建数据。')
    if off.size >= 2:
        bad = np.flatnonzero(np.diff(off) <= 0)
        if bad.size:
            i = int(bad[0])
            raise ValueError(
                f'{path}: 必须严格递增；off[{i}]={off[i]} >= off[{i + 1}]={off[i + 1]}'
                f'（block 非空 ⇒ 每段至少 1 token；重复/回退说明 .off 损坏或平移）')


def load_offsets(path, n_tokens):
    """读 `.off`（int64 裸数组）并校验，返回 `np.ndarray`。

    `n_tokens` 必须由调用方从**对应的 bin**算出来（`os.path.getsize(bin)//2`），
    这样"边界表与 bin 对齐"这件事才有独立的对照物。
    """
    off = np.fromfile(path, dtype=np.int64)
    validate_offsets(off, n_tokens, path)
    return off


def sample_id_in_window(offsets, starts, block_size, device=None):
    """返回 `(B, T)` 的窗口内 sample_id（每个 token 属于窗口内第几个样本，从 0 起）。

    * `offsets`：`.off` 数组（numpy 或 torch，int64）；
    * `starts`：`(B,)` 窗口起点 token 下标；
    * `block_size`：窗口长度 T（= train.py 的 block_size）。

    实现：全局样本号 = `searchsorted(offsets, pos, right=True) - 1`，再按行减去首个样本号
    得到"窗口内第几个"。`searchsorted` 向量化，**无 python 循环**；`(B,T)` 次查询在
    1.5M 长度的边界表上是二分，代价可忽略。
    """
    off = offsets if torch.is_tensor(offsets) else torch.as_tensor(np.asarray(offsets))
    off = off.to(dtype=torch.long, device=device)
    if off.dim() != 1:
        raise ValueError(f'offsets 必须是一维，实际 {tuple(off.shape)}')
    st = torch.as_tensor(starts, dtype=torch.long, device=off.device).reshape(-1)
    pos = st.unsqueeze(1) + torch.arange(block_size, device=off.device, dtype=torch.long).unsqueeze(0)
    sid = torch.searchsorted(off, pos.reshape(-1), right=True).reshape(-1, block_size) - 1
    return sid - sid[:, :1]


def boundary_mask(sample_id):
    """`(B,T,T)` bool：同一样本内为 True。可直接与因果掩码相与。

    只做一次广播比较，不物化任何 float 中间量（内存 = B·T² 个 bool）。
    """
    if sample_id.dim() != 2:
        raise ValueError(f'sample_id 必须是 (B,T)，实际 {tuple(sample_id.shape)}')
    return sample_id.unsqueeze(2) == sample_id.unsqueeze(1)


def causal_boundary_mask(sample_id):
    """`boundary_mask(sample_id) & tril`：因果 + 同一样本，`(B,T,T)` bool。

    直接给手动注意力路径当 `~mask` 的输入；SDPA 路径可用 `mask.unsqueeze(1)` 广播到
    `(B,1,T,T)` 而不必展开 head 维。
    """
    T = sample_id.size(1)
    tril = torch.tril(torch.ones(T, T, device=sample_id.device, dtype=torch.bool))
    return boundary_mask(sample_id) & tril.unsqueeze(0)


def aligned_pack_starts(offsets, block_size, batch_size, generator=None):
    """块对齐打包的窗口起点：`(B,)` numpy int64。

    做法：随机挑一个 block `j`，从它开始**贪心**装完整 block（累计跨度 ≤ block_size）；
    若下一个完整 block 装不下就停。然后把窗口左移，使**结束位置恰好落在 block 边界**
    （`start = off[e] - block_size`）。于是每个窗口都装满 block_size 个真实 token，
    且末尾 block 完整不截断（窗口前端可能从前一个 block 的中途开始，由掩码保证不越界）。

    RNG 用 torch 全局流（可由 `generator` 指定）：续训恢复 torch RNG 时窗口序列可复现。
    返回的起点保证 `start + block_size <= n_tokens - 1`（y = x 错位 1 时不越界）。

    ★★ 2026-09-13 警告：这条路径有**结构性覆盖漏洞**，默认配置已不再使用它
    （`train.py` 的 `pack_align` 默认翻成 `False`）。窗口只有 T 长且必须**结束在 block 边界**
    ⇒ 位置 p 可达 ⟺ p 落在某个 block 边界前 T 个 token 内 ⇒ **比 T 长的块，前 L−T 个 token
    永远进不了任何窗口**；最后一个 block 整体不可达。实测不可达 train token 占比：
    v3_dlg 67.39% / v3_lang 67.05% / v3_know 72.91% / v2 70.67%
    （`unreachable_token_fraction` 可复算；审计见 `analysis/packing_audit.md`）。
    仅保留用于复现旧实验 / A-B 对照。
    """
    off = np.asarray(offsets, dtype=np.int64)
    n_blocks = off.size - 1
    if n_blocks < 2:
        raise ValueError(f'块对齐打包至少需要 2 个 block，实际 {n_blocks}')
    j = torch.randint(0, n_blocks - 1, (batch_size,), generator=generator).tolist()
    starts = np.empty(batch_size, dtype=np.int64)
    for b, jb in enumerate(j):
        jb = int(jb)
        e = jb + 1                      # 结束边界下标：off[e] = 第 jb 个 block 的结束
        while e < n_blocks - 1 and off[e + 1] - off[jb] <= block_size:
            e += 1                      # 下一个完整 block 还装得下 → 继续装
        start = int(off[e]) - block_size
        starts[b] = start if start > 0 else 0
    return starts


def aligned_window_starts(offsets, block_size):
    """`aligned_pack_starts` 可能产出的**全部**窗口起点（对每个合法 `j` 枚举一遍）。

    与 `aligned_pack_starts` 的贪心逐字等价，只是 `j` 取遍 `[0, n_blocks-2]` 而不是随机抽。
    用途：回答"按 `pack_align=True` 采样，哪些 token 有可能被采到"（见 `unreachable_token_fraction`）。
    返回 int64 `(n_blocks-1,)`，**非递减**（`e_j` 随 `j` 单调不减，且窗口锚在块尾）。
    空 / 单块 split 返回空数组（调用方自行决定是否视作"全部不可达"）。
    """
    off = np.asarray(offsets, dtype=np.int64)
    n_blocks = off.size - 1
    if n_blocks < 2:
        return np.empty(0, dtype=np.int64)
    j = np.arange(0, n_blocks - 1, dtype=np.int64)
    # 贪心的向量化等价：e_j = max e ∈ [j+1, n_blocks-1] 且 off[e]-off[j] <= T；
    # 连 off[j+1]-off[j] > T（超长块）也至少取 j+1（与原实现的 `e = jb + 1` 初值一致）。
    e = np.searchsorted(off, off[j] + block_size, side='right') - 1
    e = np.minimum(np.maximum(e, j + 1), n_blocks - 1)
    return np.maximum(off[e] - block_size, 0)


def unreachable_token_fraction(offsets, block_size, align=True):
    """**永远不可能被采样到的 token 占比**（0~1）。纯函数、只读 `.off`、不做蒙特卡洛。

    只统计"该采样模式下**全部合法窗口**的并集"覆盖不到的 token —— 与抽多少窗口无关。

    * `align=True`（`pack_align=True`，块对齐）：窗口 = `[max(off[e]-T,0), +T)`。
      ★ 结构性漏洞：比 T 长的块，前 L−T 个 token 永远进不了任何窗口（见
      `aligned_pack_starts` 的警告）。实测：v3_dlg 0.6739 / v3_lang 0.6705 /
      v3_know 0.7291 / v2 0.7067。
    * `align=False`（`pack_align=False`，当前默认）：起点 i ∈ `[0, n-T-1]` 全域随机，
      并集 = `[0, n-1)` ⇒ 只有最后一个 token 不可达（≈ 1/n，即 ≈ 0%）。
      若 `n <= T`（窗口都放不下）返回 1.0（该 split 无法采样）。
    """
    off = np.asarray(offsets, dtype=np.int64)
    if off.ndim != 1 or off.size < 2:
        return 1.0
    n_tokens = int(off[-1])
    block_size = int(block_size)
    if n_tokens <= 0 or block_size <= 0:
        return 0.0
    if not align:
        if n_tokens <= block_size:
            return 1.0
        return 1.0 - (n_tokens - 1) / n_tokens

    starts = aligned_window_starts(off, block_size)
    if starts.size == 0:
        return 1.0
    ends = starts + block_size                      # 两列都非递减
    # 合并排序区间求并集长度：新区间 ⟺ start > 之前所有 end 的最大值
    new = np.ones(starts.size, dtype=bool)
    new[1:] = starts[1:] > np.maximum.accumulate(ends)[:-1]
    idx = np.flatnonzero(new)
    grp_s = starts[idx]
    grp_e = np.empty_like(grp_s)
    if idx.size > 1:
        grp_e[:-1] = ends[idx[1:] - 1]              # 每组最后一个 end = 组内最大 end
    grp_e[-1] = ends[-1]
    covered = int((grp_e - grp_s).sum())
    return 1.0 - covered / n_tokens


def reachable_fraction(offsets, block_size, align=True):
    """**可达 token 占比** = `1 - unreachable_token_fraction(...)`（0~1）。

    名字与返回值同向（reachable = 可达）。`align=True` 时实测：
    v3_dlg 0.3261 / v3_lang 0.3295 / v3_know 0.2709 / v2 0.2933。
    """
    return 1.0 - unreachable_token_fraction(offsets, block_size, align=align)


def mask_cross_sample_labels(y, sample_id, ignore_index=-100):
    """把**跨样本**的 label 置为 ignore_index。

    ★ 契约（2026-09-13 修）：`sample_id` 必须是 `(B, T+1)`，比 `y` **多一格**。

    为什么：位置 `t` 的 label 是 token `t+1`，所以"label 是否跨样本"要比较
    `sample_id[t]` 与 `sample_id[t+1]`。旧实现只收到 `(B, T)`，最后一位 label
    （`t = T-1`）没有 `t+1` 可比 ⇒ **每个块对齐窗口固定漏 1 个跨样本 label**
    （实测 v3_dlg/v3_lang/v3_know/v2 的右端跨样本比例都是 **100.0000%**，
    占全部 label 的 0.3906%；见 `analysis/packing_audit.md`）。传入 T+1 格后，
    内部接缝与右端漏网一次覆盖。

    例：`y` 有 5 个 label、`sample_id = [0,0,0,0,0,1]` ⇒ 位置 4 的 label 被屏蔽
    （这正是旧契约漏掉的那一格）；`sample_id = [0,0,0,1,1,1]` ⇒ 位置 2 被屏蔽。

    `sample_id=None` 时原样返回（未启用打包 / 向后兼容）。
    传 `(B, T)` 会**大声抛 `ValueError`**，绝不静默漏掉最后一位。
    """
    if sample_id is None:
        return y
    if sample_id.dim() != 2 or sample_id.size(1) != y.size(1) + 1:
        raise ValueError(
            f'mask_cross_sample_labels 需要 (B, T+1) 的 sample_id（T = y.size(1) = {y.size(1)}），'
            f'实际 shape = {tuple(sample_id.shape)}。'
            f'传 (B,T) 会静默漏掉最后一个 label（右端跨样本）—— 见 docstring 与 '
            f'analysis/packing_audit.md。')
    cross = sample_id[:, 1:] != sample_id[:, :-1]     # (B, T)，True 处 = 该 label 跨样本
    y = y.clone()
    y[cross] = ignore_index
    return y
