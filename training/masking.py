"""loss masking 纯函数：只对 assistant 回复区域计算 loss（chat 微调惯例）。

独立成模块：不依赖 train.py 全局变量（标记 id 作为参数传入），可单独测试
（见 training/check_loss_mask.py）。

实现是全向量化的（无 Python 循环）：用 cummax 恢复「最近 pair 起点」的
in_reply 状态机。已与旧版逐 token 扫描对拍 19200+ 样本完全一致（≈300× 加速）。
"""
import torch

__all__ = ['build_assistant_mask']


def _find_seq_starts(y, seq):
    """y: (B, T)；seq: 任意长度标记序列。返回 (B, T-1) bool：位置 i 是 seq 起点。

    长度 2 时与原实现（y[:,:-1]==s0 & y[:,1:]==s1）逐位等价；字节直入模式
    （dev-notes/48）的标记是 7 字节序列（如「模型：」的 UTF-8），需要通用匹配。
    """
    B, T = y.shape
    n = len(seq)
    starts = torch.zeros(B, T - 1, dtype=torch.bool, device=y.device)
    if T < n:
        return starts
    yw = y.unfold(1, n, 1)                       # (B, T-n+1, n)
    seq_t = torch.tensor(seq, dtype=y.dtype, device=y.device).view(1, 1, n)
    match = (yw == seq_t).all(dim=2)             # (B, T-n+1)
    starts[:, :T - n + 1] = match                # 起点 i ∈ [0, T-n]
    return starts


def build_assistant_mask(y, model_ids, user_ids, sep_ids):
    """返回 (B, T) bool mask：True = 计算 loss，False = 忽略。

    y: (B, T) int64 token id 张量。
    model_ids / user_ids / sep_ids: 标记 id 序列（BPE 版长度 2 如 [306, 228]；
    字节直入版长度 7 如 [0xe6, 0xa8, 0xa1, 0xe5, 0x9e, 0x8b, 0x3a]）。
    语义：从「模型：」对开始（含）到「用户：」对（含）或「\n\n」对（不含）
    之间的区域算 loss；模型/用户标记对本身也算（让模型学对话骨架）。

    注意：标记匹配不到时（如换了词表没改配置）所有位置都会被 mask 掉，
    loss 恒为 nan——训练循环的 NaN 防护会警告并跳过该步，不会静默训坏。
    """
    B, T = y.shape
    if T < 2:
        # 单 token 窗口不存在任何标记对，全部忽略
        return torch.zeros_like(y, dtype=torch.bool)
    is_model = _find_seq_starts(y, model_ids)    # (B, T-1)：seq 起点在 i
    is_user  = _find_seq_starts(y, user_ids)
    is_sep   = _find_seq_starts(y, sep_ids)
    starts = is_model | is_user | is_sep         # (B, T-1) 所有标记起点
    idx = torch.arange(T - 1, device=y.device)
    last = idx.unsqueeze(0).expand(B, -1).clone()
    last[~starts] = -1
    last_cum, _ = torch.cummax(last, dim=1)                  # 位置 i 处最近的 pair 起点
    # 自由位置（不属于任何 pair）的回复状态 = 最近一个「完全在它前面」的 pair 是模型对
    padded = torch.cat([torch.full((B, 2), -1, dtype=torch.long, device=y.device),
                        last_cum[:, :-1]], dim=1)            # (B, T): padded[t] = 最近 pair 起点 <= t-2
    has_prev = padded >= 0
    prev_model = torch.where(has_prev, is_model.gather(1, padded.clamp(min=0)),
                             torch.zeros((), dtype=torch.bool, device=y.device))
    mask = prev_model.clone()
    pair_mask = is_model | is_user                           # 模型/用户对的两个位置都算 loss（对话骨架）
    mask[:, :-1] = torch.where(starts, pair_mask, mask[:, :-1])
    mask[:, 1:] = torch.where(starts, pair_mask, mask[:, 1:])
    return mask
