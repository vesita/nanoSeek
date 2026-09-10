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


def _attn_head_spec(name, p, nh):
    """GLM-5 Muon Split：判断参数是不是「按头拼接」的注意力投影，返回 (n_heads, head_first)。

    head_first=True  : 权重行方向按头拼接（q/k/v 投影，形状 (H·d, n)）
    head_first=False : 权重列方向按头拼接（输出投影，形状 (n, H·d)）
    返回 None 表示不是注意力投影（整块正交化即可）。

    注意 c_proj 名字冲突：SwiGLU 的下投影也叫 c_proj（(n_embd, hidden)，非方形）。
    注意力输出投影恒为 (n_embd, n_embd) 方形且在 attn 子模块下，两个条件联合判定。
    """
    if name.endswith('c_qkv_csa.weight') or name.endswith('c_attn.weight'):
        return (3 * nh, True)          # 融合 QKV / 标准 c_attn：行 = [q; k; v]，各 H·d 行
    if name.endswith('c_proj.weight') and '.attn.' in name and p.shape[0] == p.shape[1]:
        return (nh, False)             # 注意力输出投影（方形）；SwiGLU 下投影非方形 → 跳过
    if name.endswith(('q_proj.weight', 'k_up.weight', 'v_up.weight',
                      'q_proj_csa.weight', 'k_proj_csa.weight', 'v_proj_csa.weight')):
        return (nh, True)              # MLA q/k_up/v_up、非融合 CSA q/k/v：行 = 头
    return None


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config, layer_idx=i) for i in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))
        if config.byte_level:
            # 字节直入 + 3:1 聚合（dev-notes/48，Mamba-Byte 思想）：无 BPE 分词。
            # byte_emb：0-256 字节嵌入；byte_agg：3 字节 → 1 token（可学习软聚合，
            # 等价于"可微 BPE"）；byte_unagg：1 token → 预测下一组的 3 字节。
            self.byte_emb = nn.Embedding(config.vocab_size, config.n_embd)
            self.byte_agg = nn.Linear(3 * config.n_embd, config.n_embd, bias=False)
            self.byte_unagg = nn.Linear(config.n_embd, 3 * config.vocab_size, bias=False)
            self.lm_head = None
        elif config.factorized_emb_dim > 0:
            # 因式分解嵌入（ALBERT 思想）：词表先投影到 factorized_emb_dim 低秩空间，
            # 再由 emb_proj 升维到 n_embd；输出头由 head_down 降维后复用 wte.weight 做 weight tying。
            # 极大削减静态词表参数（省 60~70% 嵌入参数），将参数预算重投到 Transformer 深度。
            self.transformer['wte'] = nn.Embedding(config.vocab_size, config.factorized_emb_dim)
            self.emb_proj = nn.Linear(config.factorized_emb_dim, config.n_embd, bias=False)
            self.head_down = nn.Linear(config.n_embd, config.factorized_emb_dim, bias=False)
            self.lm_head = None
        else:
            self.transformer['wte'] = nn.Embedding(config.vocab_size, config.n_embd)
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # 显式记忆 token（拓扑实验）：K×n_embd 可学习嵌入，forward 里拼到序列前、
        # 参与每一层注意力+FFN、过完所有层后剥离。是模型可写入/检索的跨 token 长程工作区。
        # 用 nn.Parameter 直接挂在 GPT 上（不进 ModuleDict，避免干扰 wte/lm_head 的 key 结构）。
        if config.n_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.zeros(config.n_memory_tokens, config.n_embd))
        # 用 RoPE 时，位置信息由 attention 内部注入，不需要可学习的位置编码表
        if not config.use_rope:
            self.transformer['wpe'] = nn.Embedding(config.block_size, config.n_embd)
        # MTP 模块（可选）：额外的"小 transformer 层"，训练时预测更远的 token
        if config.use_mtp:
            self.mtp_modules = nn.ModuleList([MTPModule(config) for _ in range(config.n_mtp)])
            if config.factorized_emb_dim == 0 and not config.byte_level:
                self.mtp_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # 使用 torch.compile() 做权重共享（weight tying）时会产生一些警告：
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        if not config.byte_level and config.factorized_emb_dim == 0:
            self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
        if config.use_mtp and config.factorized_emb_dim == 0 and not config.byte_level:
            # V4：MTP 输出头和主模型共享 lm_head 权重（节省参数，DeepSeek-V3/V4 都这么做）
            if self.lm_head is not None:
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

    def get_token_emb(self, idx):
        if self.config.factorized_emb_dim > 0:
            return self.emb_proj(self.transformer.wte(idx))
        return self.transformer.wte(idx)

    def compute_logits(self, x):
        if self.config.factorized_emb_dim > 0:
            return F.linear(self.head_down(x), self.transformer.wte.weight)
        return self.lm_head(x)

    def forward(self, idx, targets=None, rope_offset=0):
        """rope_offset（dev-notes/46 推理状态续传）：输入序列的全局起始位置。
        窗口截断推理时传窗口起点的绝对位置，RoPE 保持绝对坐标（默认 0 = 训练/全序列）。"""
        device = idx.device
        b, t = idx.size()
        if self.config.byte_level:
            # 字节直入（dev-notes/48）：输入 UTF-8 字节流，3:1 聚合后进 transformer
            return self._forward_byte(idx, targets, rope_offset)
        assert t <= self.config.block_size, f"无法前向传播长度为 {t} 的序列，block size 只有 {self.config.block_size}"

        # 前向传播 GPT 模型本身
        tok_emb = self.get_token_emb(idx) # 形状为 (b, t, n_embd) 的 token 嵌入
        if self.config.use_rope:
            # RoPE：位置信息在 attention 内部注入，这里只需要 token embedding
            x = self.transformer.drop(tok_emb)
        else:
            pos = torch.arange(0, t, dtype=torch.long, device=device) # 形状 (t)
            pos_emb = self.transformer.wpe(pos) # 形状为 (t, n_embd) 的位置嵌入
            x = self.transformer.drop(tok_emb + pos_emb)
        # RoPE 表已扩到 block_size+K，前缀位置 0..K-1 正常旋转。
        if self.config.n_memory_tokens > 0:
            mem = self.memory_tokens.unsqueeze(0).expand(b, -1, -1)  # (b, K, n_embd)
            x = torch.cat([mem, x], dim=1)                            # (b, K+t, n_embd)
        if self.config.use_mhc:
            # mHC：4 个残差流从同一个嵌入出发（在流维扩展）
            x = x.unsqueeze(2).expand(b, x.size(1), self.config.hc_mult, self.config.n_embd)
        is_eos = (idx == getattr(self.config, 'eos_token_id', 0)) if (not self.config.byte_level and getattr(self.config, 'sample_boundary_reset', True)) else None
        # 梯度检查点：aux-free MoE 的 router_bias 在 forward 内就地更新（no_grad 副作用），
        # 与 checkpoint 的 recompute 重算不兼容（重算时 bias 已变 → 路由结果不一致 → CheckpointError）。
        # 因此 use_moe + use_aux_free_balance 时自动降级为不检查点（MoE 稀疏激活显存本就不高，
        # 100M 模型在 8GB 卡上靠 batch_size 控制即可，无需强制检查点）。
        use_ckpt = getattr(self.config, 'gradient_checkpointing', False) and self.training
        if use_ckpt and self.config.use_moe and getattr(self.config, 'use_aux_free_balance', False):
            use_ckpt = False
        for block in self.transformer.h:
            if use_ckpt:
                x = torch.utils.checkpoint.checkpoint(block, x, rope_offset, is_eos, use_reentrant=False)
            else:
                x = block(x, rope_offset=rope_offset, is_eos=is_eos)
        if self.config.use_mhc:
            # 4 流均值回到 1 流，再给 ln_f / lm_head（V4 用可学习合并，这里用均值简化）
            x = x.mean(dim=2)
        if self.config.n_memory_tokens > 0:
            # 剥离记忆 token 前缀：它们不该喂给 lm_head/MTP（会生成无意义 token）
            x = x[:, self.config.n_memory_tokens:, :]
        x = self.transformer.ln_f(x)

        if targets is not None:
            # 如果给了目标 targets，就同时计算损失
            logits = self.compute_logits(x)
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
                loss = loss + self.config.mtp_weight * self._compute_mtp_loss(x, targets, is_eos=is_eos)
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
            logits = self.compute_logits(x[:, [-1], :]) # 注意：用列表 [-1] 来保留时间维度
            loss = None

        return logits, loss

    def _forward_byte(self, idx, targets, rope_offset=0):
        """字节直入 + 3:1 可学习聚合（dev-notes/48，Mamba-Byte 思想）。

        idx: (B, 3T) 字节 id（0-256，256=<eos>）；byte_agg 把 3 字节 → 1 token
        （网络学"哪些字节组合成语义单位"，等价可微软 BPE，无话术固化）。
        输出 logits (B, T, 3, V)：位置 t 预测**下一组**的 3 字节（错位对齐）。
        目标 targets: (B, 3T) 字节（-100 已按回复区 mask）→ 组目标 (B, T, 3)，
        组内任一字节被 mask → 整组忽略（跨 用户/模型 边界的组保守跳过）。
        """
        device = idx.device
        b, tb = idx.size()
        assert tb % 3 == 0, "字节直入要求输入长度为 3 的倍数（3 字节/组）"
        T = tb // 3
        emb = self.byte_emb(idx)                                   # (B, 3T, n_embd)
        x = emb.view(b, T, 3 * self.config.n_embd)
        x = self.byte_agg(x)                                       # (B, T, n_embd)
        x = self.transformer.drop(x)
        if self.config.use_mhc:
            # mHC：4 个残差流（与主 forward 一致）
            x = x.unsqueeze(2).expand(b, x.size(1), self.config.hc_mult, self.config.n_embd)
        for block in self.transformer.h:
            x = block(x, rope_offset=rope_offset)
        if self.config.use_mhc:
            x = x.mean(dim=2)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.byte_unagg(x).view(b, T, 3, self.config.vocab_size)
            # 组 g 的目标 = 字节 3(g+1)..3(g+1)+2 = targets[3g+2:3g+5]
            # （targets 是错位 1 的字节流）；组 0..254 有效，组 255（最后）无目标。
            tgt = targets[:, 2:2 + 3 * (T - 1)].view(b, T - 1, 3)   # (B, T-1, 3)
            tgt = torch.cat([tgt, torch.full(
                (b, 1, 3), -100, dtype=tgt.dtype, device=device)], dim=1)
            tgt = torch.where((tgt == -100).any(dim=2, keepdim=True),
                              torch.full_like(tgt, -100), tgt)
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size),
                                   tgt.view(-1), ignore_index=-100)
            if self.config.use_moe:
                moe_loss = torch.zeros(1, device=x.device, dtype=x.dtype)
                for block in self.transformer.h:
                    aux = block.get_moe_aux_loss()
                    if aux is not None:
                        moe_loss = moe_loss + aux
                loss = loss + moe_loss
            if self.config.use_lightning_indexer:
                idx_loss = torch.zeros(1, device=x.device, dtype=x.dtype)
                for block in self.transformer.h:
                    aux = block.get_indexer_loss()
                    if aux is not None:
                        idx_loss = idx_loss + aux
                loss = loss + 0.01 * idx_loss
        else:
            logits = self.byte_unagg(x[:, [-1], :]).view(b, 1, 3, self.config.vocab_size)
            loss = None
        return logits, loss

    def get_memory_state(self):
        """推理状态续传（dev-notes/46）：收集各记忆层的联想状态末态。
        key = 层索引；未启用记忆或未前向过的层不包含。"""
        states = {}
        for i, blk in enumerate(self.transformer.h):
            a = blk.attn
            s = a.get_mem_state() if getattr(a, 'kv_memory_enabled', False) else None
            if s is not None:
                states[i] = s
        return states

    def set_memory_state(self, states):
        """注入续传状态：下次前向各记忆层从缓存状态继续递推（替代恒 0）。
        states=None 或空 dict = 清除所有注入（记忆从零开始）。"""
        if not states:
            for i, blk in enumerate(self.transformer.h):
                if getattr(blk.attn, 'kv_memory_enabled', False):
                    blk.attn.set_mem_state(None)
            return
        for i, s in states.items():
            self.transformer.h[i].attn.set_mem_state(s)

    def _compute_mtp_loss(self, x, targets, is_eos=None):
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
            next_emb = self.get_token_emb(safe_targets[:, off : off+length])       # (B, len, C)
            mtp_targets = targets[:, off+1 : off+1+length]                      # (B, len)
            is_eos_mtp = is_eos[:, off : off+length] if is_eos is not None else None
            h = self.mtp_modules[k](hidden, next_emb, is_eos=is_eos_mtp)       # (B, len, C)
            logits = self.compute_logits(h)
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
            split_heads = {}   # GLM-5 Muon Split：id(p) → (n_heads, head_first)
            for n, p in param_dict.items():
                if (n.startswith('transformer.wte') or n.startswith('lm_head')
                        or n.startswith('byte_emb') or n.startswith('byte_unagg')
                        or 'neural_db' in n):
                    adamw_decay.append(p)   # 嵌入/输出头/神经数据库稀疏槽位走 AdamW，不进 Muon 正交化
                elif p.dim() < 2:
                    adamw_nodecay.append(p) # norm/bias
                elif p.dim() > 2:
                    # 3D+ 参数（如 kv 记忆的 mem_persist (nh,l,l)）不是矩阵，整块
                    # 正交化无意义且会触发 NS 的 ndim==2 断言 → 走 AdamW 带衰减。
                    adamw_decay.append(p)
                else:
                    muon_params.append(p)   # 其余 2D 矩阵参数（attention/FFN/router）
                    if self.config.muon_split:
                        spec = _attn_head_spec(n, p, self.config.n_head)
                        if spec is not None:
                            split_heads[id(p)] = spec
            muon = Muon([{'params': muon_params, 'weight_decay': weight_decay,
                          'lr_ratio': self.config.muon_lr_scale}],
                        lr=learning_rate * self.config.muon_lr_scale,
                        momentum=self.config.muon_momentum,
                        ns_steps=self.config.muon_ns_steps,
                        ns_aggressive=getattr(self.config, 'muon_ns_aggressive', 0),
                        split_heads=split_heads)
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            extra_args = dict(fused=True) if fused_available and device_type == 'cuda' else {}
            adamw = torch.optim.AdamW(
                [{'params': adamw_decay, 'weight_decay': weight_decay, 'lr_ratio': 1.0},
                 {'params': adamw_nodecay, 'weight_decay': 0.0, 'lr_ratio': 1.0}],
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
        """ 估算模型算力利用率（MFU），按实际设备的 bf16 峰值 FLOPS 计算 """
        # 首先估算每次迭代我们要做的 flops 数。
        # 参考 PaLM 论文附录 B：https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt)  # 每秒
        # 动态取当前设备峰值算力：无法从 torch 直接读时回退 A100 312 TFLOPS。
        # 消费级显卡（如 RX 6600 ~22 TFLOPS fp32 / 加倍 fp16）下，硬编码 A100 会把 MFU
        # 算成几个百分点、失去参考意义——这里按设备属性估算（AMD 无官方 bf16 峰值时按 fp32×2）。
        try:
            import torch
            props = torch.cuda.get_device_properties(0)
            # 优先用设备上报的 fp32 峰值（tensor core bf16 通常约为 fp32 的 2~4 倍，取 2 保守）
            # AMD ROCm 下可用 props 无直接 bf16 峰值，故用 fp32×2 近似。
            fp32_tflops = getattr(props, 'multi_processor_count', 0) * getattr(props, 'clock_rate', 0) * 2 / 1e6
            flops_promised = fp32_tflops * 2 * 1e12 if fp32_tflops > 0 else 312e12
        except Exception:
            flops_promised = 312e12
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