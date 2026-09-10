"""loss masking 纯函数：只对"模型回复行"计算 loss（chat 微调惯例，去标签版）。

dev-notes/61：去标签后数据不再有「用户：/模型：」，模型每条回复后统一插终止符
<eos>(收尾) 或 <cont>(待续/递回)。因此回复行的边界由终止符界定：

    mask 规则 = token 所在的"行"（到下一个换行 \n 之前）内存在 <eos>/<cont>
    终止符 → 该行整体计入 loss（含 B：/引号/正文/终止符本身）。

天然适配四种样式（"A：..." / A：... / "..." / 裸文本）：只有模型行后跟终止符，
用户/对方行之后没有 → 自动只 mask 模型行。纯向量化，无 Python 循环。
"""
import torch

__all__ = ['build_assistant_mask']


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