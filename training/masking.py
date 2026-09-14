"""loss masking 纯函数：只对"模型回复行"计算 loss（chat 微调惯例，去标签版）。

dev-notes/61：去标签后数据不再有「用户：/模型：」，模型每条回复后统一插终止符
<eos>(收尾) 或 <cont>(待续/递回)。因此回复行的边界由终止符界定：

    mask 规则 = token 所在的"行"（到下一个换行 \n 之前）内存在 <eos>/<cont>
    终止符 → 该行整体计入 loss（含 B：/引号/正文/终止符本身）。

天然适配四种样式（"A：..." / A：... / "..." / 裸文本）：只有模型行后跟终止符，
用户/对方行之后没有 → 自动只 mask 模型行。纯向量化，无 Python 循环。
"""
import torch

__all__ = ['build_assistant_mask', 'build_resp_span_mask']


def build_assistant_mask(y, reply_ids, sep_ids):
    """返回 (B, T) bool mask：True = 计算 loss，False = 忽略。

    y: (B, T) int64 token id 张量。
    reply_ids: 可迭代的回复终止符 id。**以 tokenizer 为准，别硬编码**：
        当前 data/chinese/char_tokenizer.json 里 `<eos>`=128、`<cont>`=130
        （调用方用 `_cv['<eos>']` / `_cv['<cont>']` 取；旧词表曾是 117/119，已作废）。
    sep_ids: 忽略（保留参数以兼容调用方；换行边界自动检测）。
    """
    B, T = y.shape
    mask = torch.zeros(B, T, dtype=torch.bool, device=y.device)
    if T < 1:
        return mask

    # 1) 终止符位置 is_term
    is_term = torch.zeros(B, T, dtype=torch.bool, device=y.device)
    for r in reply_ids:
        if r is not None:
            is_term |= (y == r)

    # 2) 行分隔符：换行 \n（词表 id 0，字级布局区 1）。行 = 换行之间的跨距。
    nl = torch.tensor(0, dtype=y.dtype, device=y.device)
    is_nl = (y == nl)

    # 3) 每个 token 之后（含自身）最近的换行位置 next_nl；
    #    每个 token 之后（含自身）最近的终止符位置 next_term。
    idx = torch.arange(T, device=y.device).view(1, -1).expand(B, -1)
    big = torch.full((B, T), T, dtype=torch.long, device=y.device)
    nl_pos = torch.where(is_nl, idx, big).flip(1)
    term_pos = torch.where(is_term, idx, big).flip(1)
    next_nl, _ = torch.cummin(nl_pos, dim=1)
    next_nl = next_nl.flip(1)                 # 每个位置之后(含)最近换行；无则 T
    next_term, _ = torch.cummin(term_pos, dim=1)
    next_term = next_term.flip(1)             # 每个位置之后(含)最近终止符；无则 T

    # 4) token t 属于某模型回复行 ⟺ 该行内在 t 之后（含 t）有终止符，
    #    且该终止符出现在行内换行之前：
    #    next_term[t] < next_nl[t]
    #    （若 t 自身是终止符：next_term[t]=t < next_nl[t]（其后紧跟行尾\n）→ 计入）
    mask = (next_term < next_nl)

    # 5) 行尾换行本身不计入（换行 id 0 非回复内容）；但"空回复+终止符"保留。
    mask &= ~is_nl
    return mask


def build_resp_span_mask(y, resp_ids, eos_ids):
    """返回 (B, T) bool mask：**`<resp>` 之后 → 对应 `<eos>`（含）** 算 loss。

    这是「单流 + `<resp>`」格式（`training/dialogue_stream.py`）的**精确**规则
    （由 `DialogueStream.loss_token_spans()` 定义，此处是它的向量化实现）。

    与 `build_assistant_mask`（行内含终止符 ⇒ **整行**算 loss）的区别 —— 只有一处，
    但很关键：

        `<resp>你好。<eos>`   ← 模型行
        旧规则：整行算 loss ⇒ **含 `<resp>`**
        新规则：`<resp>` **之后**算 ⇒ 不含 `<resp>`

    `<resp>` 是 **harness 喂的**（模型不该生成它），所以旧规则会教模型"顺手输出 `<resp>`"。
    每轮只差 1 个 token，但方向是错的。`<topic>` 落在区间内、照常算 loss（模型要学它）✅

    判定（逐 token，三个条件同时成立）：
      1. **严格早于** t 的位置上存在 `<resp>`（`prev_resp >= 0`，用**右移一格**的严格版，
         不是"含自身"的 `cummax` 原值）；
      2. 且**自那个 `<resp>` 之后还没遇到 `<eos>`**（严格早于 t 的最后一个 `<eos>`
         比它更早：`prev_eos < prev_resp`）；
      3. 且 t 之后（含 t）存在 `<eos>`（`next_eos < T`）—— 没闭合的回复不算（防半截）。

    ★ 为什么必须是**严格版**：用"含自身"的 `prev_resp` 再加 `prev_resp < t` 这条，
      在**相邻两个 `<resp>`**（`<resp><resp>…<eos>`）上会给出不同答案 ——
      第二个 `<resp>` 处 `prev_resp == t`，被判为 False；而权威定义
      `DialogueStream.loss_token_spans` 是**顺序扫描**：第一个 `<resp>` 的扫描
      把第二个 `<resp>` 当**正文**吞进区间（它不是 `<eos>`），所以那个 token **算 loss**。
      实测确有分歧（`[140,140,128]`：向量化 `[F,F,T]` vs 权威 `[F,T,T]`）。
      真实流里 `append/commit` 不会产出相邻 `<resp>`，但"向量化实现 == 权威 span 定义"
      这条不变量必须**对所有输入**成立，否则它就不是同一个损失口径。已用
      `tests/test_masking.py::test_resp_span_consecutive_cue_matches_authority` 钉住。
    """
    B, T = y.shape
    mask = torch.zeros(B, T, dtype=torch.bool, device=y.device)
    if T < 1:
        return mask

    is_resp = torch.zeros(B, T, dtype=torch.bool, device=y.device)
    for r in resp_ids:
        if r is not None:
            is_resp |= (y == r)
    is_eos = torch.zeros(B, T, dtype=torch.bool, device=y.device)
    for e in eos_ids:
        if e is not None:
            is_eos |= (y == e)

    idx = torch.arange(T, device=y.device).view(1, -1).expand(B, -1)
    neg1 = torch.full((B, T), -1, dtype=torch.long, device=y.device)
    big = torch.full((B, T), T, dtype=torch.long, device=y.device)

    # 前缀：最后一个 <resp> / 最后一个 <eos>（都含自身）
    prev_resp_incl = torch.cummax(torch.where(is_resp, idx, neg1), dim=1).values
    prev_eos_incl = torch.cummax(torch.where(is_eos, idx, neg1), dim=1).values
    # **严格早于** t 的版本：把"含自身"的右移一格（见 docstring 的 ★）
    prev_resp = torch.cat([neg1[:, :1], prev_resp_incl[:, :-1]], dim=1)
    prev_eos = torch.cat([neg1[:, :1], prev_eos_incl[:, :-1]], dim=1)

    # 后缀：下一个 <eos>（含自身）—— flip→cummin→flip
    next_eos = torch.cummin(torch.where(is_eos, idx, big).flip(1), dim=1).values.flip(1)

    return (prev_resp >= 0) & (prev_eos < prev_resp) & (next_eos < T)