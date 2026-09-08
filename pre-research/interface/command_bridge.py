"""n-bit 指令直通桥接层 (尝试期, 纯 torch, 不 import training/*)。

定位: 模型 <-> vDB 之间唯一可训练的部件 (决策链: GRPO 离散策略, 不穿过数据库)。
结构 (用户方案): 主干 hidden 直通 + 旁路小激活网络产 n bit -> 2^n 指令组合,
                 其余维做 payload; op=00 全零 = NOP 旁路 (默认零副作用)。

位域 (v4, KV+Vec 混合; bit0 与 region 共用 op[0], 读法见 decode()):
    [0:2] op     00=NOP 01=写 10=读 11=写+K读
    [2]   region 0=KV(fast, O(1)) 1=Vec(slow, 精确topk/ANN)
    [3:5] K      读条数 1/2/4/8   (仅 Vec 读生效)
    [5:8] flags  bit5 读命中刷活跃度 / bit6 覆盖写 / bit7 pin 防遗忘

安全护栏 (来自 dev-notes 教训):
    payload tanh / 门控 sigmoid / STE 直通 —— 防写信号爆炸 (45);
    默认 NOP —— 改通道五连败 (B) => 新通道默认无副作用。
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_engine.vdb import VDB  # noqa: E402

OP_NOP, OP_WRITE, OP_READ, OP_READK = 0, 1, 2, 3
K_TABLE = (1, 2, 4, 8)


def decode(bits: torch.Tensor):
    """bits: (B, 8) uint8 -> 指令字段张量。"""
    op = (bits[:, 1] << 1) | bits[:, 0]
    region = bits[:, 2]
    kbits = (bits[:, 4] << 1) | bits[:, 3]
    kvals = torch.tensor([K_TABLE[int(b)] for b in kbits.tolist()], device=bits.device)
    flags = bits[:, 5:8]
    return op, region, kvals, flags


class StraightThroughBin(torch.autograd.Function):
    """前向: bits = (p > 0.5); 反向: 梯度原路直通给 logits —— '直通层' 本义。"""

    @staticmethod
    def forward(ctx, logits):
        return (torch.sigmoid(logits) > 0.5).to(logits.dtype)

    @staticmethod
    def backward(ctx, g):
        return g


class CommandBridge(nn.Module):
    def __init__(self, n_embd, n_enc=64, d_key=32, d_val=64, n_sig=16,
                 fast_cap=1024, slow_cap=131072, n_bits=8):
        super().__init__()
        self.n_embd = n_embd
        # 可训练: 旁路组合激活网络 -> 指令位 logits
        self.enc = nn.Sequential(nn.Linear(n_embd, n_enc), nn.GELU(),
                                 nn.Linear(n_enc, n_bits))
        # 可训练: payload 头 (有界, 防 NaN — dev-notes/45)
        self.key_head = nn.Linear(n_embd, d_key)
        self.val_head = nn.Linear(n_embd, d_val)
        # 可训练: 读出解码 (加性接回主干, 与记忆通道同构)
        self.dec = nn.Linear(d_val, n_embd)
        # 不可训练引擎
        self.db = VDB(d_key=d_key, d_val=d_val, n_sig=n_sig,
                      fast_cap=fast_cap, slow_cap=slow_cap)

    def forward(self, hidden, t, mode='auto'):
        """hidden: (B, T, n_embd) 取末位 token 发指令; 返回 (decoded, log_probs, bits)。
        decoded 加性接回主干; log_probs (B, n_bits) 供 GRPO 逐位分解信用。"""
        last = hidden[:, -1, :] if hidden.dim() == 3 else hidden
        logits = self.enc(last)
        bits = StraightThroughBin.apply(logits)
        logp = (F.logsigmoid(logits) * bits
                + F.logsigmoid(-logits) * (1.0 - bits))     # 逐位 log-prob (STE 口径)
        key = torch.tanh(self.key_head(last))                # 有界 payload
        val = torch.tanh(self.val_head(last))
        op, region, kval, flags = decode(bits.long())
        out = self._exec(op, region, kval, flags, key, val, float(t), mode)
        return self.dec(out), logp, bits

    @torch.no_grad()
    def _exec(self, op, region, kval, flags, key, val, t, mode):
        outs = []
        for b in range(key.shape[0]):
            o = int(op[b]) if mode != 'read' else OP_READ
            if o == OP_NOP:
                outs.append(torch.zeros_like(val[b]))
                continue
            reg = int(region[b])
            if o in (OP_WRITE, OP_READK):
                self.db.write(key[b].cpu().numpy(), val[b].cpu().numpy(), t,
                              region=reg, overwrite=bool(flags[b, 1]))
                if o == OP_WRITE:
                    outs.append(torch.zeros_like(val[b]))
                    continue
            kk = int(kval[b]) if mode != 'read' else 1
            if reg == 0:
                r = self.db.read(key[b].cpu().numpy(), t, topk=kk, region=0)
            else:
                r = self.db.read(key[b].cpu().numpy(), t, topk=kk, region=1)
            outs.append(torch.from_numpy(r[0]).to(key.device) if r
                        else torch.zeros_like(val[b]))
        return torch.stack(outs)

    def periodic_forget(self, t):
        return self.db.forget(float(t))


if __name__ == '__main__':
    torch.manual_seed(0)
    br = CommandBridge(n_embd=80)
    h = torch.randn(2, 8, 80)
    dec, lp, bits = br(h, t=1.0, mode='read')   # 规则热身: 强制读
    assert dec.shape == (2, 80) and lp.shape == (2, 8)
    dec.sum().backward()                          # STE 梯度全通
    g = br.enc[0].weight.grad
    assert g is not None and torch.isfinite(g).all()
    print('bridge selftest ok: decoded', tuple(dec.shape), 'logp', tuple(lp.shape),
          '| STE grad ok')
