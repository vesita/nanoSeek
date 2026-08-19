"""GPT：模型主体（嵌入 / 层堆 / 输出头 / MTP / 优化器配置）。"""
import inspect
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from .config import GPTConfig
from .utils import RMSNorm
from .block import Block, MTPModule
from .optimizer import Muon, MuonAdamW


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config, layer_idx=i) for i in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))
        # 显式记忆 token（拓扑实验）：K×n_embd 可学习嵌入，forward 里拼到序列前、
        # 参与每一层注意力+FFN、过完所有层后剥离。是模型可写入/检索的跨 token 长程工作区。
        # 用 nn.Parameter 直接挂在 GPT 上（不进 ModuleDict，避免干扰 wte/lm_head 的 key 结构）。
        if config.n_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.zeros(config.n_memory_tokens, config.n_embd))
        # 用 RoPE 时，位置信息由 attention 内部注入，不需要可学习的位置编码表
        if not config.use_rope:
            self.transformer['wpe'] = nn.Embedding(config.block_size, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # MTP 模块（可选）：额外的"小 transformer 层"，训练时预测更远的 token
        if config.use_mtp:
            self.mtp_modules = nn.ModuleList([MTPModule(config) for _ in range(config.n_mtp)])
            self.mtp_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # 使用 torch.compile() 做权重共享（weight tying）时会产生一些警告：
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # 不完全确定这是什么原因，目前看是无害的。TODO 调查一下
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
        if config.use_mtp:
            # V4：MTP 输出头和主模型共享 lm_head 权重（节省参数，DeepSeek-V3/V4 都这么做）
            self.mtp_head.weight = self.lm_head.weight

        # 初始化所有权重
        self.apply(self._init_weights)
        # 按照 GPT-2 论文，对残差投影应用特殊缩放的初始化
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        # memory_tokens 是裸 Parameter，_init_weights（apply 遍历子模块）不会碰到它，手动初始化。
        # 同嵌入标准：N(0, 0.02)。K=16 时 16×80=1,280 参数，全参与训练。
        if config.n_memory_tokens > 0:
            torch.nn.init.normal_(self.memory_tokens, mean=0.0, std=0.02)

        # 报告参数量
        print("参数量：%.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        返回模型中的参数量。
        按非嵌入参数计数（默认）时，会减去位置嵌入。
        token 嵌入本来也应减去，但由于参数共享，这些参数实际上
        被用作最后一层的权重，所以我们要把它们包含进来。
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # 用 RoPE 时没有 wpe，跳过（cos/sin 是 buffer，本来就不算参数）
            if hasattr(self.transformer, 'wpe'):
                n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"无法前向传播长度为 {t} 的序列，block size 只有 {self.config.block_size}"

        # 前向传播 GPT 模型本身
        tok_emb = self.transformer.wte(idx) # 形状为 (b, t, n_embd) 的 token 嵌入
        if self.config.use_rope:
            # RoPE：位置信息在 attention 内部注入，这里只需要 token embedding
            x = self.transformer.drop(tok_emb)
        else:
            pos = torch.arange(0, t, dtype=torch.long, device=device) # 形状 (t)
            pos_emb = self.transformer.wpe(pos) # 形状为 (t, n_embd) 的位置嵌入
            x = self.transformer.drop(tok_emb + pos_emb)
        # 显式记忆 token：拼到序列前，作为每一层的全局工作区（真实 token 的相对位置不变）。
        # RoPE 表已扩到 block_size+K，前缀位置 0..K-1 正常旋转。
        if self.config.n_memory_tokens > 0:
            mem = self.memory_tokens.unsqueeze(0).expand(b, -1, -1)  # (b, K, n_embd)
            x = torch.cat([mem, x], dim=1)                            # (b, K+t, n_embd)
        if self.config.use_mhc:
            # mHC：4 个残差流从同一个嵌入出发（在流维扩展）
            x = x.unsqueeze(2).expand(b, x.size(1), self.config.hc_mult, self.config.n_embd)
        for block in self.transformer.h:
            x = block(x)
        if self.config.use_mhc:
            # 4 流均值回到 1 流，再给 ln_f / lm_head（V4 用可学习合并，这里用均值简化）
            x = x.mean(dim=2)
        if self.config.n_memory_tokens > 0:
            # 剥离记忆 token 前缀：它们不该喂给 lm_head/MTP（会生成无意义 token）
            x = x[:, self.config.n_memory_tokens:, :]
        x = self.transformer.ln_f(x)

        if targets is not None:
            # 如果给了目标 targets，就同时计算损失
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
            if self.config.use_moe:
                # 把各层 MoE 的负载均衡辅助损失加到总损失上
                moe_loss = torch.zeros(1, device=x.device, dtype=x.dtype)
                for block in self.transformer.h:
                    aux = block.get_moe_aux_loss()
                    if aux is not None:
                        moe_loss = moe_loss + aux
                loss = loss + moe_loss
            if self.config.use_mtp:
                # 多 token 预测：额外预测 t+2、t+3...，按权重加进总损失
                loss = loss + self.config.mtp_weight * self._compute_mtp_loss(x, targets)
            if self.config.use_lightning_indexer:
                # Lightning Indexer 辅助损失：让 indexer 的选块分布逼近真实注意力分布
                # （权重 0.01，作为辅助信号，不喧宾夺主）
                idx_loss = torch.zeros(1, device=x.device, dtype=x.dtype)
                for block in self.transformer.h:
                    aux = block.get_indexer_loss()
                    if aux is not None:
                        idx_loss = idx_loss + aux
                loss = loss + 0.01 * idx_loss
        else:
            # 推理时的小优化：只对最后一个位置前向传播 lm_head
            logits = self.lm_head(x[:, [-1], :]) # 注意：用列表 [-1] 来保留时间维度
            loss = None

        return logits, loss

    def _compute_mtp_loss(self, x, targets):
        """MTP 损失：第 k 个模块用「位置 t 的隐藏状态 + 目标 t+k+1 的嵌入」预测 t+k+2。
        x: (B, T, n_embd) 主模型 ln_f 的输出；targets: (B, T) 训练目标（即 t+1 的正确答案）。
        """
        B, T, C = x.shape
        if T < self.config.n_mtp + 2:
            return torch.zeros(1, device=x.device, dtype=x.dtype)
        mtp_loss = torch.zeros(1, device=x.device, dtype=x.dtype)
        h = x  # 当前"预备预测"的隐藏状态序列
        for k in range(self.config.n_mtp):
            # off：h[j] 当前对应要预测 targets[j+off]（主模型 off=1，预测 t+1）
            if k == 0:
                hidden = h[:, :-2]      # 主模型输出预测的是 t+1，要再前进两步到 t+2
                off = 1
            else:
                hidden = h[:, :-1]      # 上级模块输出预测的是 t+k+1，再前进一步即可
                off = k + 1
            length = hidden.shape[1]
            # 对 embedding 查找用 clamp 替换 -100（loss masking 屏蔽位），
            # 这些位置在 cross_entropy 中仍被 ignore_index=-100 忽略。
            safe_targets = targets.clamp(min=0)
            next_emb = self.transformer.wte(safe_targets[:, off : off+length])       # (B, len, C)
            mtp_targets = targets[:, off+1 : off+1+length]                      # (B, len)
            h = self.mtp_modules[k](hidden, next_emb)                           # (B, len, C)
            logits = self.mtp_head(h)
            mtp_loss = mtp_loss + F.cross_entropy(
                logits.view(-1, logits.size(-1)), mtp_targets.reshape(-1), ignore_index=-100)
        return mtp_loss / self.config.n_mtp

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # 从所有候选参数开始
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # 过滤掉那些不需要梯度的参数
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        if self.config.use_muon:
            # V4：矩阵参数（除 embedding/lm_head）走 Muon；嵌入/输出头/1D 参数用 AdamW 保护。
            # 注意 Muon 不依赖 beta2（没有二阶矩），所以 betas 参数被忽略。
            muon_params, adamw_decay, adamw_nodecay = [], [], []
            for n, p in param_dict.items():
                if n.startswith('transformer.wte') or n.startswith('lm_head'):
                    adamw_decay.append(p)   # 嵌入/输出头无矩阵结构，正交化无意义
                elif p.dim() < 2:
                    adamw_nodecay.append(p) # norm/bias
                else:
                    muon_params.append(p)   # 其余矩阵参数（attention/FFN/router）
            muon = Muon([{'params': muon_params, 'weight_decay': weight_decay}],
                        lr=learning_rate, momentum=self.config.muon_momentum,
                        ns_steps=self.config.muon_ns_steps)
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            extra_args = dict(fused=True) if fused_available and device_type == 'cuda' else {}
            adamw = torch.optim.AdamW(
                [{'params': adamw_decay, 'weight_decay': weight_decay},
                 {'params': adamw_nodecay, 'weight_decay': 0.0}],
                lr=learning_rate, betas=betas, **extra_args)
            return MuonAdamW(muon, adamw)

        # 创建优化器分组。任何 2D 参数都会做权重衰减，其余不做。
        # 即 matmul 和嵌入中的所有权重张量做衰减，所有 bias 和 layernorm 参数不做。
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        # 创建 AdamW 优化器，如果可用就使用 fused 版本
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ 估算模型算力利用率（MFU），以 A100 bfloat16 峰值 FLOPS 为单位 """
        # 首先估算每次迭代我们要做的 flops 数。
        # 参考 PaLM 论文附录 B：https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # 用 A100 bfloat16 峰值 flops 的比例来表示我们的 flops 吞吐量
        flops_achieved = flops_per_iter * (1.0/dt) # 每秒
        flops_promised = 312e12 # A100 GPU bfloat16 峰值 flops 是 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        输入一个条件序列 idx（形状为 (b,t) 的 LongTensor），并连续生成 max_new_tokens 次，
        每次把预测结果喂回模型。
        做这个之前，你很可能需要把模型切换到 model.eval() 模式。
        """
        for _ in range(max_new_tokens):
            # 如果序列上下文太长，必须在 block_size 处裁剪
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # 前向传播模型，得到序列中各位置对应的 logits
            logits, _ = self(idx_cond)
            # 取出最后一步的 logits，并按期望的温度缩放
            logits = logits[:, -1, :] / temperature
            # 可选：把 logits 裁剪到只保留 top k 个选项
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # 应用 softmax 把 logits 转成（归一化的）概率
            probs = F.softmax(logits, dim=-1)
            # 从分布中采样
            idx_next = torch.multinomial(probs, num_samples=1)
            # 把采样出的索引追加到序列末尾并继续
            idx = torch.cat((idx, idx_next), dim=1)

        return idx