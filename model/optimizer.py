"""Muon 优化器（DeepSeek-V4）与 MuonAdamW 组合。"""
import torch

from .utils import zeropower_via_newtonschulz


class Muon(torch.optim.Optimizer):
    """Muon 优化器（DeepSeek-V4 / Llama 4 的核心优化器）。

    更新规则（对每个矩阵参数 W）：
        1. 动量  m = 0.95·m + g
        2. 正交化 Q = NewtonSchulz(m)   ← 与 AdamW 的本质区别
        3. 权重衰减插值：g = (1-wd)·Q + wd·W
        4. W -= lr·g

    一维参数（bias、norm 权重）没有「方向」可言，退化为纯动量更新。
    使用标准的 state 机制，checkpoint 里能正常 save/load。
    """

    def __init__(self, params, lr, momentum=0.95, nesterov=True, ns_steps=10,
                 orthogonalization_fn=None):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)
        self.orthogonalization_fn = orthogonalization_fn or zeropower_via_newtonschulz

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            wd = group.get('weight_decay', 0.0)
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if nesterov:
                    # Nesterov 加速：在动量基础上再看一眼当前梯度
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                if g.ndim >= 2:
                    # Muon 的核心：只对矩阵参数做正交化
                    g = self.orthogonalization_fn(g, steps=ns_steps)
                    # 权重衰减：正交化的方向 + 掺一点原参数做收缩
                    g = (1 - wd) * g + wd * p.data
                p.data.add_(g, alpha=-lr)
        return loss


class MuonAdamW:
    """Muon + AdamW 组合优化器（V4 风格）。

    矩阵参数（除 embedding/lm_head）用 Muon 做正交化更新；embedding / lm_head /
    1D 参数（norm/bias）用 AdamW——它们没有矩阵结构，正交化没有意义。

    对 train.py 而言行为像单个优化器：支持 param_groups / step / zero_grad /
    state_dict / load_state_dict，checkpoint 里正常保存。
    """

    def __init__(self, muon, adamw):
        self.muon = muon
        self.adamw = adamw
        self._step_supports_amp_scaling = True  # 兼容 GradScaler（float16 训练）

    @property
    def param_groups(self):
        return self.muon.param_groups + self.adamw.param_groups

    def step(self, closure=None):
        self.muon.step(closure)
        self.adamw.step(closure)

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {'muon': self.muon.state_dict(), 'adamw': self.adamw.state_dict()}

    def load_state_dict(self, sd):
        self.muon.load_state_dict(sd['muon'])
        self.adamw.load_state_dict(sd['adamw'])