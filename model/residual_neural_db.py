# -*- coding: utf-8 -*-
"""残差神经数据库 (ResidualNeuralDB) —— 全量优化版。

设计理念（继承 logits 层 / 隐藏层注入的教训）：
  注入位置二选一（logits 层 `forward_and_correct` 或 `transformer.h[-2]` 隐藏层）；
  写入为「高惊讶错题本」；所有内部记忆为纯 no_grad（零优化器状态）。

本文件在早期版本基础上落地下列数学/算法优化（全部可开关，默认向后兼容）：

  P1' (value_type="gradient")  value = 目标嵌入 − E_p[嵌入] = CE 负梯度方向（一阶最优修正），
                               而非旧的 argmax 残差（过冲、离散）。
  P2' (share_io_proj=True)     读写共用同一 key 投影，消除「写时槽 ≠ 读时槽」的地址漂移。
  P3' (entropy_gate=True)      注入门控乘以检索置信 f(1−H(top_k))，检索锐利才注入。
  P5' (require_support=True)   写槽累加支持度 slot_support，读时低支持(不可靠)槽被置信缩压。
  P6' (value_dim=value_dim)    库值可提到 n_embd(512) 直存残差，消除 512→128 压缩瓶颈。
  P7' (inject_mode="knn")      读改成 kNN-LM 分布插值：λ·p_LM + (1−λ)·p_kNN（更稳、更可解释）。
  P4'                           冻结基座对照由训练脚本 `--freeze_base` 实现。

第二轮框架改进（默认等价旧版，需显式开启）:

  P8'  values_init="zeros"      未写入槽返回 0（旧版 randn → 稀疏检索退化为噪声平均）。
  P9'  write_mode="delta"       软 top-m delta 规则：v_i += η·w_i·(target − Σ_j w_j v_j)，
                                邻近槽共同承担误差，值场平滑可泛化（旧版 top-1 EMA 覆盖）。
  P10' diff_addr=True           读聚合权重对 query 可微 → key 投影真正被训练
                                （旧版 retrieve 整体 no_grad，寻址永远是随机投影）。
  P11' surprise_quantile>0      自适应惊讶分位：只写 batch 内 top-q% 的 CE token
                                （字级绝对阈值 3.0 会命中 ~56% token，等于不筛）。
  P13' conf_mode="coverage"     覆盖率置信 = Σw·1[support>0]，替代熵门控
                                （τ=0.1 下 softmax 熵恒近 0，熵门控无信息）。

默认参数等价于早期版本（value_dim=None→key_dim, share_io_proj=False, entropy_gate=False,
require_support=False, value_type="argmax", inject_mode="residual"）。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.product_key_memory import ProductKeyMemory


class ResidualNeuralDB(nn.Module):
    def __init__(
        self,
        n_embd=512,
        vocab_size=8192,
        key_dim=128,
        sub_keys=1024,      # 1024^2 = 1M 槽
        top_k=16,
        surprise_threshold=3.0,
        temperature=0.1,
        write_lr=0.2,
        device=None,
        dtype=torch.bfloat16,
        # --- P6: value 维度（None→key_dim；n_embd→直存残差，无压缩瓶颈）---
        value_dim=None,
        # --- P2': 读写共用 key 投影 ---
        share_io_proj=False,
        # --- P3': 熵门控（检索置信）---
        entropy_gate=False,
        entropy_exp=1.0,
        # --- P5': 支持度门控（可复制/可靠记忆才被信任）---
        require_support=False,
        support_floor=1.0,
        support_scale=2.0,
        support_cap=20.0,
        # --- P1': value 类型（argmax 残差 | gradient 残差）---
        value_type="argmax",
        # --- P7': 注入模式（residual 残差偏置 | knn 分布插值）---
        inject_mode="residual",
        # --- 第二轮框架改进（P8'/P9'/P10'/P11'/P12'/P13'）---
        values_init="randn",          # P8': randn | zeros
        write_mode="ema",             # P9': ema | delta
        write_top_m=1,                # P9': 写入扩散的近邻槽数
        err_clip=0.0,                 # P9': 单条误差范数上限
        value_clip=0.0,               # P9': 槽值范数上限
        usage_init="ones",            # P12': ones | zeros
        diff_addr=False,              # P10': 可微寻址
        surprise_quantile=0.0,        # P11': >0 时按 batch 内分位筛选惊讶 token
        conf_mode="entropy",          # P13': entropy | coverage
        renorm_written=False,         # P8'': 零初始化下权重只在已写槽重归一化
        trainable_codebook=False,     # P15': 码本可训练
    ):
        super().__init__()
        self.n_embd = n_embd
        self.vocab_size = vocab_size
        self.key_dim = key_dim
        self.surprise_threshold = surprise_threshold
        self.dtype = dtype
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.value_dim = value_dim or key_dim
        self.share_io_proj = share_io_proj
        self.entropy_gate = entropy_gate
        self.entropy_exp = entropy_exp
        self.value_type = value_type
        self.inject_mode = inject_mode
        self.values_init = values_init
        self.write_mode = write_mode
        self.write_top_m = int(write_top_m)
        self.err_clip = err_clip
        self.value_clip = value_clip
        self.usage_init = usage_init
        self.diff_addr = diff_addr
        self.surprise_quantile = surprise_quantile
        self.conf_mode = conf_mode
        self.renorm_written = renorm_written

        # 1. 外部 no_grad 记忆库
        self.memory = ProductKeyMemory(
            key_dim=key_dim,
            sub_keys=sub_keys,
            top_k=top_k,
            value_dim=self.value_dim,
            write_lr=write_lr,
            temperature=temperature,
            device=self.device,
            dtype=dtype,
            use_support=require_support,
            support_floor=support_floor,
            support_scale=support_scale,
            support_cap=support_cap,
            values_init=values_init,       # P8'
            write_mode=write_mode,         # P9'
            write_top_m=write_top_m,       # P9'
            err_clip=err_clip,             # P9'
            value_clip=value_clip,         # P9'
            usage_init=usage_init,         # P12'
            trainable_codebook=trainable_codebook,   # P15'
        )

        # 2. 读/写接口（可共享，P2'）
        if share_io_proj:
            self.key_proj = nn.Linear(n_embd, key_dim, bias=False)
            self.key_norm = nn.LayerNorm(key_dim)
        else:
            self.q_proj = nn.Linear(n_embd, key_dim, bias=False)
            self.q_norm = nn.LayerNorm(key_dim)
            self.k_proj = nn.Linear(n_embd, key_dim, bias=False)
            self.k_norm = nn.LayerNorm(key_dim)

        # 读注入门控（冷启动偏负，默认少干扰）：residual 模式为注入强度，knn 模式为 λ
        self.read_gate = nn.Linear(n_embd, 1, bias=True)
        nn.init.constant_(self.read_gate.bias, -2.0)

        # 读解码器：memory value (value_dim) -> logits (vocab)
        self.logits_proj = nn.Linear(self.value_dim, vocab_size, bias=False)
        # 隐藏注入路径：value_dim -> n_embd（value_dim==n_embd 时置 None，直存不加投影）
        if self.value_dim == n_embd:
            self.hidden_proj = None
        else:
            self.hidden_proj = nn.Linear(self.value_dim, n_embd, bias=False)

        # 写编码器：n_embd 残差 -> value_dim（value_dim==n_embd 时置 None，直存）
        if self.value_dim == n_embd:
            self.target_enc = None
        else:
            self.target_enc = nn.Linear(n_embd, self.value_dim, bias=False)

        # P2: 写入时排除无信息目标
        self.ignorable_ids = frozenset({128, 129, 130, -100})

        self.to(self.device).to(dtype)
        # 统计缓冲保持 f32：bf16 累加精度不足（support 步进 0.2~1.0，上限 20）
        self.memory.slot_usage = self.memory.slot_usage.float()
        self.memory.slot_support = self.memory.slot_support.float()

    # ------------------------------------------------------------------ 读写投影一致
    def _query(self, h_flat):
        """读 query：与写 key 在同一地址空间（share_io_proj）或独立（默认）。"""
        if self.share_io_proj:
            return self.key_norm(self.key_proj(h_flat)).float()
        return self.q_norm(self.q_proj(h_flat)).float()

    def _key(self, h_flat):
        """写 key：读/写保持一致时与 _query 相同。"""
        if self.share_io_proj:
            return self.key_norm(self.key_proj(h_flat)).float()
        return self.k_norm(self.k_proj(h_flat)).float()

    # ------------------------------------------------------------------ 检索（含 P3'/P5'）
    def _retrieve(self, h, conf=True):
        B, T, _ = h.shape
        h_flat = h.reshape(B * T, self.n_embd)
        q = self._query(h_flat)
        mem_vals, slot_ids, weights, support, coverage = self.memory.retrieve(
            q, return_support=True, diff_addr=self.diff_addr,
            renorm_written=self.renorm_written)   # (N,value_dim),(N,k),(N,k),(N,k),(N,1)
        gate_conf = torch.ones(weights.size(0), 1, device=weights.device)
        if conf and self.conf_mode == "coverage":
            # P13': 覆盖率置信（已写槽权重质量）——零初始化下这是有信息的信任信号
            gate_conf = coverage
        elif conf and self.entropy_gate:
            # 检索熵 → 置信：低熵(锐利) → 置信高；高熵(模糊) → 置信低
            H = -(weights * torch.log(weights + 1e-12)).sum(-1)
            H_norm = H / math.log(weights.size(-1))
            gate_conf = ((1.0 - H_norm).clamp(0, 1) ** self.entropy_exp).reshape(-1, 1)
        return mem_vals, slot_ids, weights, gate_conf

    # ------------------------------------------------------------------ 读：logits 残差注入
    def read_logits(self, h, logits):
        B, T, _ = h.shape
        mem_vals, _slot_ids, _weights, conf = self._retrieve(h)
        delta = self.logits_proj(mem_vals).reshape(B, T, self.vocab_size)
        gate = torch.sigmoid(self.read_gate(h.reshape(B * T, self.n_embd))).reshape(B, T, 1) * conf.reshape(B, T, 1)
        corrected = logits + (gate * delta).to(logits.dtype)
        return corrected, {"gate_mean": gate.mean().item(), "delta_norm": delta.norm(dim=-1).mean().item()}

    # ------------------------------------------------------------------ 读：kNN-LM 分布插值 (P7')
    def read_knn(self, h, logits):
        B, T, _ = h.shape
        N = B * T
        h_flat = h.reshape(N, self.n_embd)
        _mem_vals, slot_ids, weights = self.memory.retrieve(self._query(h_flat))  # (N,k)
        K = slot_ids.size(1)
        tok = self.memory.token_ids[slot_ids]                      # (N,k)，未写槽 = -1
        valid = (tok >= 0).float()
        p_knn = torch.zeros(N, self.vocab_size, device=slot_ids.device, dtype=torch.float32)
        p_knn.scatter_add_(1, tok.clamp_min(0), weights.float() * valid)   # (N,vocab)
        p_knn = p_knn / p_knn.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        p_model = F.softmax(logits.float(), dim=-1).reshape(N, self.vocab_size)
        lam = torch.sigmoid(self.read_gate(h_flat)).reshape(-1, 1)  # 每 token λ ∈ [0,1]
        p_corr = lam * p_model + (1 - lam) * p_knn
        corrected = torch.log(p_corr + 1e-9).reshape(B, T, self.vocab_size).to(logits.dtype)
        return corrected, {"gate_mean": lam.mean().item(), "delta_norm": 0.0}

    # ------------------------------------------------------------------ 读：隐藏层注入（供 hidden 脚本）
    def read_hidden(self, h):
        B, T, _ = h.shape
        h_flat = h.reshape(B * T, self.n_embd)
        mem_vals, _slot_ids, _weights, conf = self._retrieve(h)
        if self.hidden_proj is not None:
            delta = self.hidden_proj(mem_vals).reshape(B, T, self.n_embd)
        else:
            delta = mem_vals.reshape(B, T, self.n_embd)
        gate = torch.sigmoid(self.read_gate(h_flat)).reshape(B, T, 1) * conf.reshape(B, T, 1)
        return delta, gate

    # ------------------------------------------------------------------ 写：高惊讶错题本（P1'/P2'/P5'/P7' 共用）
    def write_from_logits(self, h, logits, targets, token_embeddings):
        """写库（no_grad）。返回 (n_surprises, info)。"""
        with torch.no_grad():
            B, T = h.shape[0], h.shape[1]
            losses = F.cross_entropy(
                logits.reshape(-1, self.vocab_size), targets.reshape(-1), reduction="none"
            ).reshape(B, T).float()
            # P2: 排除无信息目标
            ignorable = sorted(self.ignorable_ids)
            is_info = ~torch.isin(targets, torch.tensor(ignorable, device=targets.device))
            # P11': 自适应惊讶分位（>0 时只写 batch 内最惊讶的 q 比例）；否则退回绝对阈值
            if self.surprise_quantile > 0:
                lv = losses[is_info]
                if lv.numel() == 0:
                    return 0, {"surprise_written": 0}
                thr = float(torch.quantile(lv, 1.0 - self.surprise_quantile))
            else:
                thr = self.surprise_threshold
            surprise_mask = (losses > thr) & is_info
            n_surprises = int(surprise_mask.sum().item())
            if n_surprises == 0:
                return n_surprises, {"surprise_written": 0}

            h_surp = h[surprise_mask]                        # (N, n_embd)
            k_surp = self._key(h_surp.reshape(-1, self.n_embd))
            toks = targets[surprise_mask]                    # (N,)
            target_emb = F.embedding(toks, token_embeddings)

            # P1': value 类型
            if self.value_type == "gradient":
                p = F.softmax(logits[surprise_mask].float(), dim=-1)          # (N,vocab)
                e_exp = torch.einsum("nv,vk->nk", p, token_embeddings.float())  # (N,n_embd)
                residual = target_emb.float() - e_exp
            else:  # argmax
                pred_tokens = logits.argmax(dim=-1)[surprise_mask]
                pred_emb = F.embedding(pred_tokens, token_embeddings)
                residual = target_emb.float() - pred_emb.float()

            if self.target_enc is not None:
                val = self.target_enc(residual.to(self.target_enc.weight.dtype))
            else:
                val = residual.to(self.memory.values.dtype)

            sw = torch.clamp((losses[surprise_mask] - thr) / 2.0, 0.2, 1.0)
            # P9'/P7': 写入模式(delta/ema)与目标槽数(top_m)由 memory 决定
            self.memory.write(
                k_surp, val, weight=sw,
                top_m=self.write_top_m, mode=self.write_mode,
                token_ids=(toks if self.inject_mode == "knn" else None),
            )
            return n_surprises, {"surprise_written": n_surprises}

    # ------------------------------------------------------------------ 主入口：logits 层前向纠偏
    def forward_and_correct(self, h, logits, targets=None, token_embeddings=None):
        B, T, _ = h.shape
        if self.inject_mode == "knn":
            corrected, info = self.read_knn(h, logits)
        else:
            corrected, info = self.read_logits(h, logits)

        if self.training and targets is not None and token_embeddings is not None:
            n_sur, w_info = self.write_from_logits(h, logits, targets, token_embeddings)
            info["surprise_written"] = n_sur
        else:
            info["surprise_written"] = 0
        return corrected, info
