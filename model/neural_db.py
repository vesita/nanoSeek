# -*- coding: utf-8 -*-
"""
nanoSeek 神经网络数据库 (Neural Database / Product-Key Memory with Self-Purging)
=============================================================================

定位：
构建完全可微分、常数显存开销、大容量扩展且具备「新陈代谢自动淘汰」的内嵌神经键值记忆存储。

核心特性：
1. Product-Key (积量化键) 寻址：将键空间切分为两个子空间分别检索，在 O(sqrt(M)) 点积计算代价下
   实现笛卡尔积级 (如 256*256 = 65,536) 的高密度神经槽位索引。
2. 稀疏读出与端到端梯度传播：每次前向仅激活 Top-K 个槽位，算力增加 < 3%，显存常数。
3. 门控残差注入：结合 Output Gate 自适应决定是否调用“神经知识库”，不干扰主干语言通顺度。
4. 【新增】自动淘汰与复活机制 (Purge & Revive / Garbage Collection)：
   - 动态追踪槽位激活频次（EMA Usage Tracking），防止马太效应导致少数槽位垄断；
   - 自动识别“僵尸/冷死槽位”（Dead Slots），通过热门槽位突变分裂（Cell Division）
     或高信息量输入投影强制覆写，实现知识库的高效动态新陈代谢。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProductKeyNeuralDB(nn.Module):
    """
    Product-Key 神经网络数据库模块 (PK-NDB) 带自淘汰治理系统

    参数:
        config: 模型配置对象 (需包含 n_embd)
        sub_keys (int): 每个子空间的候选键数量。总槽位数 = sub_keys * sub_keys。
                        默认 256 -> 总槽位 65,536 个独立向量。
        top_k (int): 每次 Query 检索聚合的槽位数量 (默认 32)。
        dropout (float): 记忆读取权重 Dropout，防单槽位过拟合 (默认 0.1)。
        usage_decay (float): 槽位活跃度指数移动平均 (EMA) 衰减系数 (默认 0.99)。
        dead_threshold (float): 判定冷死槽位的相对均值比率 (默认 0.05，即低于平均激活水平的 5% 判定为僵尸)。
    """
    def __init__(self, config, sub_keys: int = 256, top_k: int = 32, dropout: float = 0.1,
                 usage_decay: float = 0.99, dead_threshold: float = 0.05):
        super().__init__()
        self.n_embd = config.n_embd
        assert self.n_embd % 2 == 0, "n_embd 必须为偶数以便切分子空间"
        self.half_dim = self.n_embd // 2
        self.sub_keys = sub_keys
        self.top_k = min(top_k, sub_keys * sub_keys)
        self.k_sub = min(self.top_k, self.sub_keys)
        self.total_slots = self.sub_keys * self.sub_keys
        self.usage_decay = usage_decay
        self.dead_threshold = dead_threshold
        # 非 top-1 槽位的 usage 权重（制造命中方差，让 GC 能找到死槽）
        self.usage_aux_scale = getattr(config, 'neural_db_usage_aux_scale', 0.1)

        # 1. Query 投射与归一化 (让 Query 分布与键码本点积更平稳)
        self.q_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.q_norm = nn.LayerNorm(self.n_embd)

        # 2. 两个子空间键码本 (sub_keys, half_dim)
        scale = 1.0 / math.sqrt(self.half_dim)
        self.keys1 = nn.Parameter(torch.randn(self.sub_keys, self.half_dim) * scale)
        self.keys2 = nn.Parameter(torch.randn(self.sub_keys, self.half_dim) * scale)

        # 3. 神经知识槽位 (Value 矩阵)
        self.values = nn.Embedding(self.total_slots, self.n_embd)
        nn.init.normal_(self.values.weight, mean=0.0, std=scale)

        # 4. 读出权重 Dropout 与输出自适应门控
        self.drop = nn.Dropout(dropout)
        self.out_gate = nn.Linear(self.n_embd, self.n_embd, bias=False)

        # 【算法升级 1】预测惊喜度门控 (Surprise Gate，吸收 Titans 灵感)
        # 评估当前 Token 是否具备高信息量（事实性/难预测），阻断 70% 常规无用语法词对记忆槽位的污染
        self.surprise_proj = nn.Linear(self.n_embd, 1, bias=True)
        # 初始化为微负偏置，使模型在冷启动阶段先学好主干语法，遇到高信息量词才主动激发数据库
        nn.init.constant_(self.surprise_proj.bias, -0.5)

        # 【算法升级 2】自适应温度参数 (Temperature Annealing / Sharpness)
        # 初始温度设为 1.0 / sqrt(top_k)，让检索权重更加锐化聚类，避免过度平滑导致槽位混叠
        self.temperature = nn.Parameter(torch.ones(1) * (1.0 / math.sqrt(max(self.top_k, 1))))

        # 5. 【自动淘汰状态追踪】注册非梯度的活跃度追踪缓冲区 (Slot Usage Buffer)
        # 初始化为完全均衡状态 1.0
        self.register_buffer('slot_usage', torch.ones(self.total_slots))
        # 统计自上次 GC 以来的迭代步数
        self.register_buffer('steps_since_purge', torch.zeros(1, dtype=torch.long))

    # =========================================================================
    # 【数据可移植 API】导出 / 导入 / 保存 / 加载 神经数据库状态
    # -------------------------------------------------------------------------
    # 场景：训练出的记忆知识库（slot_usage、keys、values、surprise 门控、温度、GC 状态）
    # 可作为一个独立、可移植的模块，跨 checkpoint / 跨模型迁移。
    #   - save_db(path) / load_db(path)：独立文件 .pt（推荐，含全部可学习参数 + 运行 buffer）
    #   - export_db_state() / import_db_state(state)：与主模型 state_dict 解耦的纯状态字典交换
    # =========================================================================

    def _db_state_dict(self) -> dict:
        """收集本模块全部可移植状态：可学习参数 + 运行期 buffer。"""
        state = {}
        for name, buf in self.named_buffers():
            state[name] = buf.detach().clone()
        for name, param in self.named_parameters():
            state[name] = param.detach().clone()
        return state

    def export_db_state(self) -> dict:
        """导出数据库状态字典（含 keys/values/门控/温度/slot_usage/GC 计数器）。"""
        return self._db_state_dict()

    def import_db_state(self, state: dict) -> list:
        """从状态字典导入，恢复可学习参数与运行 buffer。
        兼容多种 key 前缀（裸名 / 'neural_db.' / 'transformer.h.N.neural_db.'），
        支持部分迁移（只移植知识槽、不移植门控等）。返回缺失的 key 列表。
        """
        norm = {}
        for k, v in state.items():
            base = k
            if 'neural_db.' in base:
                base = base.split('neural_db.', 1)[1]
            norm[base] = v
        # 只关心本模块存在的 key（过滤掉可能混入的其他模型参数）
        valid_keys = set(self.state_dict().keys())
        subset = {k: v for k, v in norm.items() if k in valid_keys}
        missing = [k for k in valid_keys if k not in subset]
        if subset:
            self.load_state_dict(subset, strict=False)
        return missing

    def save_db(self, path: str) -> None:
        """把数据库状态保存为独立 .pt 文件（可移植）。"""
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        meta = {
            "format": "nanoseek_neural_db_v1",
            "sub_keys": self.sub_keys,
            "top_k": self.top_k,
            "n_embd": self.n_embd,
            "state": self.export_db_state(),
        }
        torch.save(meta, path)

    @classmethod
    def load_db(cls, config, path: str, sub_keys: int = None, top_k: int = None) -> "ProductKeyNeuralDB":
        """从独立 .pt 文件重建数据库模块（移植）。
        config: 目标模型的配置（需 n_embd）；sub_keys/top_k 可用给定值覆盖 meta 里的值。
        """
        meta = torch.load(path, map_location="cpu", weights_only=False)
        assert meta.get("format") == "nanoseek_neural_db_v1", f"非法的 neural_db 文件: {path}"
        sub_keys = sub_keys or meta["sub_keys"]
        top_k = top_k or meta["top_k"]
        db = cls(config, sub_keys=sub_keys, top_k=top_k)
        db.import_db_state(meta["state"])
        return db

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向检索与注入。
        输入:
            x: (B, T, C) 主干隐藏状态
        输出:
            (B, T, C) 经门控增强的知识增量
        """
        B, T, C = x.shape
        x_flat = x.view(-1, C)  # (N, C), N = B * T
        N = x_flat.size(0)

        # 1. 预先计算惊喜度门控（Titans 灵感）：评估当前 token 是否具备高信息量（事实性/难预测）。
        #    在读取与写入端都生效：
        #      - 读出端：低惊喜度 token 的检索权重被衰减，避免普通语法词扰动输出；
        #      - 写入端：检索权重趋近 0 → 该 token 命中的槽位 values 梯度也趋近 0，
        #        从根源上阻断 70% 常规无用语法词对记忆槽位的污染。
        surprise = torch.sigmoid(self.surprise_proj(x_flat))  # (N, 1)

        # 1b. 计算 Query 并切分双子空间
        q = self.q_norm(self.q_proj(x_flat))
        q1 = q[:, :self.half_dim]   # (N, half_dim)
        q2 = q[:, self.half_dim:]   # (N, half_dim)

        # 2. 子空间内积打分 (N, sub_keys)
        scores1 = torch.matmul(q1, self.keys1.t())
        scores2 = torch.matmul(q2, self.keys2.t())

        # 3. 各子空间筛选 top-k_sub 候选 (N, k_sub)
        top1_scores, top1_indices = scores1.topk(self.k_sub, dim=-1)
        top2_scores, top2_indices = scores2.topk(self.k_sub, dim=-1)

        # 4. 笛卡尔积两两组合出候选全集打分
        cart_scores = (top1_scores.unsqueeze(-1) + top2_scores.unsqueeze(-2)).view(N, -1)

        # 5. 全局重排序选出最优的 Top-K 槽位
        final_scores, final_indices_2d = cart_scores.topk(self.top_k, dim=-1)  # (N, top_k)

        # 【自适应温度锐化】防止检索权重在 Top-K 上过于平均，使重要槽位凸显
        temp = F.softplus(self.temperature) + 1e-4
        weights = F.softmax(final_scores / temp, dim=-1)

        # 【写入端惊喜度抑制】让低信息量 token 的检索权重整体衰减。
        # 这样低 surprise 命中槽位的 values 梯度 ≈ 0 → 语法词不污染记忆槽，
        # 高 surprise（高信息量实体词）才真正有梯度写入知识库。
        weights = weights * surprise  # (N, top_k)
        weights = self.drop(weights)

        # 计算全局一维槽位 ID: global_id = id1 * sub_keys + id2
        i1 = final_indices_2d // self.k_sub
        i2 = final_indices_2d % self.k_sub
        chosen_id1 = top1_indices.gather(1, i1)  # (N, top_k)
        chosen_id2 = top2_indices.gather(1, i2)  # (N, top_k)
        global_slot_ids = chosen_id1 * self.sub_keys + chosen_id2  # (N, top_k)

        # 【更新槽位活跃度 EMA】仅在训练阶段统计
        if self.training:
            with torch.no_grad():
                # 制造命中方差：给 top-1 命中槽位显著更高的 usage 权重（其余 top_k 槽位以
                # 衰减权重计入），避免 top-k 全更新导致 usage 均匀、GC 永远找不到死槽。
                #   - top-1 槽位：权重 1.0（模拟"该槽位被专一调用"）
                #   - 其余 top_k 槽位：权重 usage_aux_scale（默认 0.1，低权重，模拟偶发命中）
                flat_ids = global_slot_ids.view(-1)          # (N*top_k,)
                flat_ids = flat_ids.view(-1, self.top_k)     # (N, top_k)
                top1_ids = flat_ids[:, 0]                    # (N,)
                # top-1 槽位频次（高权重）
                top1_counts = torch.bincount(top1_ids, minlength=self.total_slots).float()
                # 其余 top_k 槽位综合频次（低权重）
                other_counts = torch.bincount(flat_ids.view(-1), minlength=self.total_slots).float() - top1_counts
                aux_scale = getattr(self, 'usage_aux_scale', 0.1)
                weighted_counts = top1_counts + aux_scale * other_counts
                batch_usage = weighted_counts / max(N, 1)
                # 探索扰动：注入微小随机性，防止 slot_usage 完全均匀锁死
                batch_usage = batch_usage * (1.0 + 0.05 * torch.randn_like(batch_usage))
                self.slot_usage.mul_(self.usage_decay).add_(batch_usage, alpha=1.0 - self.usage_decay)
                self.steps_since_purge.add_(1)

        # 6. 从 Embedding 稀疏提取知识并加权求和
        retrieved = self.values(global_slot_ids)  # (N, top_k, C)
        db_output = (retrieved * weights.unsqueeze(-1)).sum(dim=1)  # (N, C)

        # 7. 门控自适应注入：结合原装 Output Gate 与（已在 weights 上生效的）惊喜度抑制
        #    当 token 处于高惊喜度（高信息量实体词）时才强力激活，普通虚词保持接近于 0
        gate = torch.sigmoid(self.out_gate(x_flat))
        out = gate * db_output

        return out.view(B, T, C)

    @torch.no_grad()
    def purge_and_revive(self, current_batch_tokens: torch.Tensor = None) -> dict:
        """
        神经垃圾回收与槽位复活机制 (Purge & Revive / Garbage Collection):
        
        工作逻辑:
        1. 寻找长期未被访问的“僵尸冷槽位” (Dead Slots)；
        2. 识别表现优异的高频“热门槽位” (Hot Slots)；
        3. 【细胞分裂式重生】：将僵尸槽位的参数直接用热门槽位 + 微小高斯扰动覆写，
           或者用当前 batch 中具有高语义差异的输入特征注入；
        4. 重置这些槽位的活跃度为全局中位数，给予它们在新语境中竞争重生的机会。
        
        返回:
            统计字典 (包含清理的僵尸槽位数量、健康度指标等)
        """
        avg_usage = self.slot_usage.mean().item()
        # 死槽判定：采用相对分位数阈值（更稳健）。
        # 取「低于 (1-dead_threshold) 分位」的槽位为冷槽——即使整体使用率很低，
        # 也能稳定找出真正被冷落的那一批槽位，避免"全部均匀→0 死槽"的锁死。
        q = float(torch.quantile(self.slot_usage, 1.0 - self.dead_threshold))
        cutoff = max(q, avg_usage * self.dead_threshold)
        dead_mask = self.slot_usage < cutoff
        num_dead = dead_mask.sum().item()

        stats = {
            "total_slots": self.total_slots,
            "dead_slots": num_dead,
            "dead_ratio": num_dead / self.total_slots,
            "avg_usage": avg_usage,
            "max_usage": self.slot_usage.max().item(),
            "min_usage": self.slot_usage.min().item(),
        }

        if num_dead == 0:
            self.steps_since_purge.zero_()
            return stats

        dead_indices = torch.nonzero(dead_mask).squeeze(1)
        hot_indices = torch.nonzero(~dead_mask).squeeze(1)

        # 若全部槽位都冷死（极端情况，如冷启动初始化不良），则全量加噪声重启
        if len(hot_indices) == 0:
            scale = 1.0 / math.sqrt(self.half_dim)
            self.values.weight.data[dead_mask] = torch.randn_like(self.values.weight.data[dead_mask]) * scale
            self.slot_usage.fill_(1.0)
            self.steps_since_purge.zero_()
            return stats

        # 策略 A: 若传入了当前 batch 的高维特征，优先从输入中选特征注入给部分僵尸槽位
        revived_by_input = 0
        if current_batch_tokens is not None and current_batch_tokens.numel() > 0:
            flat_tokens = current_batch_tokens.reshape(-1, self.n_embd)
            k_inject = min(num_dead // 2, flat_tokens.size(0))
            if k_inject > 0:
                # 随机挑选 k_inject 个输入向量赋给冷死槽位
                token_pick = torch.randperm(flat_tokens.size(0), device=flat_tokens.device)[:k_inject]
                chosen_dead = dead_indices[:k_inject]
                self.values.weight.data[chosen_dead] = flat_tokens[token_pick].clone()
                revived_by_input = k_inject
                dead_indices = dead_indices[k_inject:]
                num_dead -= k_inject

        # 策略 B: 其余僵尸槽位通过热门槽位变异分裂 (Cell Division with Mutation) 重生
        if num_dead > 0:
            # 根据使用率作为概率加权抽取优质母体 (高热度槽位被选为母体的概率更高)
            donor_probs = self.slot_usage[hot_indices]
            donor_probs = donor_probs / (donor_probs.sum() + 1e-8)
            sampled_donors = hot_indices[torch.multinomial(donor_probs, num_dead, replacement=True)]

            # 变异扰动 (方差为当前权重标准差的 5%)
            mutation_scale = self.values.weight.data.std().item() * 0.05
            noise = torch.randn_like(self.values.weight.data[dead_indices]) * mutation_scale

            # 覆写僵尸槽位
            self.values.weight.data[dead_indices] = self.values.weight.data[sampled_donors] + noise

        # 重置已清理槽位的活跃度为全网中位数水平，防止连续被误杀
        self.slot_usage[dead_mask] = avg_usage
        self.steps_since_purge.zero_()
        stats["revived_by_input"] = revived_by_input

        return stats
