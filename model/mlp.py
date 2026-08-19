"""SwiGLU 前馈网络 + MoE 混合专家。"""
import torch
import torch.nn as nn
from torch.nn import functional as F


class SwiGLU(nn.Module):
    """SwiGLU：门控前馈网络，DeepSeek/LLaMA 的标准 FFN。
    输出 = SiLU(x W1) ⊙ (x W2)，比单一路径的 GELU MLP 表达能力更强。
    参数量对比：
      MLP    : W(embd,4embd) + W(4embd,embd) = 8·embd²
      SwiGLU : W(embd,h) + W(embd,h) + W(h,embd) = 3·h·embd
    取 h = 8/3·embd 时两者参数量持平 —— 这就是 LLaMA 用 hidden = 8/3·n_embd 的来历。
    """

    def __init__(self, config, hidden_scale=8 / 3):
        super().__init__()
        self.config = config  # 保存 config，forward 里可能要用到钳制等技巧
        hidden = int(hidden_scale * config.n_embd)  # 默认 ≈ 2.67·n_embd；hidden_scale=3 时参数量≈注意力层（补注意力槽）
        self.c_fc   = nn.Linear(config.n_embd, hidden, bias=config.bias)  # 值分支
        self.c_fc2  = nn.Linear(config.n_embd, hidden, bias=config.bias)  # 门控分支
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = F.silu(self.c_fc(x)) * self.c_fc2(x)  # SiLU(xW1) ⊙ (xW2)
        if self.config.swiglu_clamp > 0:
            # V4 稳定性技巧：钳制门控输出，从源头压制异常值。
            # 注意是在 c_proj 之前钳——异常值就是在这个门控乘积里产生的。
            x = x.clamp(-self.config.swiglu_clamp, self.config.swiglu_clamp)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class MoE(nn.Module):
    """MoE：混合专家（DeepSeek-V3 的核心创新）。

    关键思想：每个 token 只路由到 top-k 个专家（而不是所有专家都算一遍）。
    于是参数量随专家数线性增长，但每个 token 的计算量保持不变——
    这就是「用更多参数、同一份算力」换更强的模型。

    还需要一个「负载均衡辅助损失」：迫使 token 均匀分布到各专家。
    否则路由器会陷入"赢家通吃"，少数专家承担绝大部分计算，
    其它专家几乎不被激活、学不到东西（类似专家"饿死"）。
    """

    def __init__(self, config, use_hash=False):
        super().__init__()
        self.n_experts = config.n_experts
        self.n_top_k = config.n_top_k
        self.moe_aux_weight = config.moe_aux_weight
        self.router_z_loss_weight = config.z_loss_weight
        self.use_shared_expert = config.use_shared_expert
        self.use_aux_free_balance = config.use_aux_free_balance
        self.use_sqrtsoftplus = config.use_sqrtsoftplus
        self.route_scale = config.route_scale
        # V4 Hash 路由：浅层用 hash(token_id)%n_experts 确定性分配，不学习。
        # 设计动机：浅层特征是简单语法/常见搭配，确定性路由稳定且零算力；
        # 深层语义复杂才需要学习型路由。use_hash 由 Block 按 layer_idx 传入。
        self.use_hash = use_hash
        # 路由器：给每个 token 在每个专家上打一个分（hash 模式下仍保留，作为后续层复用）
        self.router = nn.Linear(config.n_embd, config.n_experts, bias=False)
        # 专家：每个专家是一份完整的 SwiGLU FFN
        self.experts = nn.ModuleList([SwiGLU(config) for _ in range(config.n_experts)])
        # V4 共享专家：始终激活，捕获所有 token 的共性特征（语法、常见搭配）
        if self.use_shared_expert:
            self.shared_expert = SwiGLU(config)
        # 本次前向累积的辅助损失，forward 后由 GPT 取走并清零
        self.aux_loss = torch.tensor(0.0)
        self.z_loss = torch.tensor(0.0)
        if self.use_aux_free_balance:
            # aux-free 偏置修正：每个专家一个 bias，加到路由 logits 上影响 top-k 选择。
            # 不参与梯度（requires_grad=False），每步根据负载偏差用 balance_factor 更新，
            # 过载专家降 bias、欠载升 bias → 自然均衡。这是 V4 替代 Switch aux loss 的做法。
            self.register_buffer('router_bias', torch.zeros(config.n_experts))
            self.balance_factor = config.balance_factor

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)          # (B*T, C)，把 batch 和序列压平逐个 token 路由
        N = x_flat.shape[0]

        if self.use_hash:
            # --- V4 Hash 路由：确定性分配，不学习 ---
            # Knuth 乘法哈希：用 token 嵌入的第一个标量作签名，均匀映射到各专家。
            # 用 k 个不同的乘子生成 k 个不重复的专家槽位。
            hash_input = x_flat[:, 0].long()  # 用输入第一维当 token 签名（token 无关）
            # 生成 n_top_k 个不同的哈希：hash + i*prime 再 % n_experts
            top_k_indices = torch.stack([
                ((hash_input * (2654435761 + 97 * i)) >> 16) % self.n_experts
                for i in range(self.n_top_k)
            ], dim=-1)  # (N, k)
            top_k_probs = x_flat.new_full((N, self.n_top_k), 1.0 / self.n_top_k)  # 均匀权重
            self.aux_loss = torch.tensor(0.0, device=x_flat.device, dtype=x_flat.dtype)
            self.z_loss = torch.tensor(0.0, device=x_flat.device, dtype=x_flat.dtype)
            # 跳过负载均衡：哈希本身均匀分配，天然均衡
        else:
            # 路由打分：给每个 token 在每个专家上打一个分
            router_logits = self.router(x_flat)               # (N, n_experts)
            # aux-free 偏置修正：bias 加到 logits 上影响 top-k 选择（bias 不参与梯度）
            if self.use_aux_free_balance:
                router_logits = router_logits + self.router_bias
            # Router Z-Loss：惩罚 logsumexp(router_logits) 的平方，防止路由 logits 过大。
            if self.router_z_loss_weight > 0:
                self.z_loss = self.router_z_loss_weight * (
                    torch.logsumexp(router_logits, dim=-1).square().mean()
                )
            else:
                self.z_loss = torch.tensor(0.0, device=router_logits.device, dtype=router_logits.dtype)
            # 软概率：Switch aux loss 用（aux-free 模式下不参与损失）
            router_probs = F.softmax(router_logits, dim=-1)   # (N, n_experts)
            if self.use_sqrtsoftplus:
                # V4 打分：√softplus(logits) * route_scale，直接用于选择 + 归一化
                scores = torch.sqrt(F.softplus(router_logits)) * self.route_scale
                top_k_scores, top_k_indices = scores.topk(self.n_top_k, dim=-1)
                top_k_probs = top_k_scores / (top_k_scores.sum(dim=-1, keepdim=True) + 1e-6)
            else:
                # 基线：softmax 全专家 → top-k → 重归一化
                top_k_probs, top_k_indices = router_probs.topk(self.n_top_k, dim=-1)  # 各 (N, k)
                top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

            # 负载均衡：f_i = 第 i 个专家实际接到的 token 比例
            one_hot = F.one_hot(top_k_indices, self.n_experts).float()   # (N, k, n_experts)
            f_i = one_hot.sum(dim=(0, 1)) / (N * self.n_top_k)           # (n_experts,)
            if self.use_aux_free_balance:
                # V4 aux-free：无辅助损失，改为按负载偏差更新 bias（无梯度，决定性推动均衡）。
                # 过载专家（f_i > 均值）bias 下降 → 更难被选中；欠载专家 bias 上升 → 更容易被选中。
                self.aux_loss = torch.tensor(0.0)
                with torch.no_grad():
                    self.router_bias.add_((f_i - 1.0 / self.n_experts).sign() * self.balance_factor)
            else:
                # Switch Transformer 辅助损失：P_i = 路由器给第 i 个专家的平均概率。
                # 两者都高意味着该专家又热门又常被选，均衡时 sum(f_i * P_i) 取最小。
                P_i = router_probs.mean(dim=0)                           # (n_experts,)
                self.aux_loss = self.moe_aux_weight * self.n_experts * (f_i * P_i).sum()

        # V4 共享专家：所有 token 都过一遍共享专家（捕获共性），再叠加路由专家的差异化输出
        output = self.shared_expert(x_flat) if self.use_shared_expert else x_flat.new_zeros(N, C)
        # 逐个专家计算：把路由到它的 token 收集起来算 FFN，再按门控权重放回原处
        for i in range(self.n_experts):
            expert_mask = (top_k_indices == i)             # (N, k) 该专家被选中的位置
            token_ids, slot_ids = expert_mask.nonzero(as_tuple=True)
            if token_ids.numel() == 0:
                continue
            weights = top_k_probs[token_ids, slot_ids]     # 这些 token 在该专家上的门控权重
            expert_in = x_flat[token_ids]                  # (num, C)
            expert_out = self.experts[i](expert_in)        # (num, C)
            output[token_ids] += expert_out * weights.unsqueeze(-1)

        return output.view(B, T, C)

    def get_aux_loss(self):
        """取出本次前向累积的辅助损失（负载均衡 + Router Z-Loss）并清零。"""
        loss = self.aux_loss + self.z_loss
        self.aux_loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
        self.z_loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
        return loss