#!/usr/bin/env python3
"""GRPO 更新器 (Updater): 从 grpo_char.py 抽出的优势合成与策略更新。

职责单一: 输入组内奖励 R 与候选 logprob/hidden → 输出
  (advantages 组相对归一化 + winsorize 门控 + 退化解剔除) →
  (policy loss + SFT 基座 KL + EOS quiet loss)。

与采样 (Sampler) 和奖励评估 (RewardEngine) 完全解耦。
"""
import torch
import torch.nn.functional as F


class GrpoUpdater:
    def __init__(self, model, eos_id, cont_id=None,
                 beta_kl=0.6, clip_eps=0.2, quiet_scale=0.0, topk=200,
                 max_new_tokens=55):
        self.model = model
        self.eos_id = eos_id
        self.cont_id = cont_id
        self.beta_kl = beta_kl
        self.clip_eps = clip_eps
        self.quiet_scale = quiet_scale

    # ------------------------------------------------------------------
    # 组相对优势归一化 (GRPO 核心: 同组比较、降采样偏差)
    # ------------------------------------------------------------------
    @staticmethod
    def normalize_advantages(rewards):
        r = torch.tensor(rewards, dtype=torch.float32)
        mean = r.mean()
        std = r.std() + 1e-8
        return (r - mean) / std

    # ------------------------------------------------------------------
    # 优势 winsorize 门控 (四分位距离群截断, 抑制单条离群主导梯度)
    # ------------------------------------------------------------------
    @staticmethod
    def winsorize_advantages(adv, iqr_k=1.5):
        q1 = torch.quantile(adv, 0.25)
        q3 = torch.quantile(adv, 0.75)
        iqr = q3 - q1 + 1e-8
        lo, hi = q1 - iqr_k * iqr, q3 + iqr_k * iqr
        return torch.clamp(adv, lo, hi)

    # ------------------------------------------------------------------
    # 一整轮补丁 (backbone 策略梯度) 更新
    # ------------------------------------------------------------------
    def update_step(self, advantages, new_logprobs, reflogprobs, mask,
                    quiet_losses, optimizer,
                    old_logprobs=None):
        """单组策略更新 (无需重算前向, logprob/quiet 已在外部采样循环算好)。

        Args:
          advantages   : (G,) 组内优势 (已在外部归一化 + winsorize, 退化时全 0)。
          new_logprobs : (G, L) 当前策略逐 token logprob (带梯度)。
          reflogprobs  : (G, L) SFT 基座参考逐 token logprob (KL 目标, 无梯度)。
          mask         : (G, L) 有效 token 掩码。
          quiet_losses : (G,) EOS 位置 L1 能量可用标量 (捕获隐藏状态时计算)。
          optimizer    : 优化器 (本方法内 zero_grad + backward + step)。
          old_logprobs : 可选, 重要性采样基线 (得 ratio=exp(new-old)); 传 None 时
                         退化为朴素优势加权 `-adv*lp` 语义 (与 v1 基线一致)。
        """
        adv = advantages
        adv_exp = adv.unsqueeze(-1)
        if old_logprobs is None:
            # 朴素优势加权: policy_loss = -adv * lp (v1 基线语义, 便于对照)
            pol_loss = -(adv_exp * new_logprobs * mask.float()).sum() / mask.sum().clamp(min=1)
        else:
            # 重要性采样比 + PPO 裁剪
            ratio = torch.exp(new_logprobs - old_logprobs)
            pol1 = ratio * adv_exp
            pol2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_exp
            pol_loss = -(torch.min(pol1, pol2) * mask.float()).sum() / mask.sum().clamp(min=1)

        # KL 到 SFT 基座: 正确实现 = 策略分布到参考分布的 KL 散度。
        #   F.kl_div(input, target, log_target=True) 逐点 = target*(log(target)-input)
        #   = ref_logp*(ref_logp - policy_logp)。策略 == 参考时严格 0, 越偏离越正,
        #   且有界下界 → 拉回基座 (与重构前 F.kl_div(curr, ref) 语义一致)。
        #   注意绝不能写回裸均值 (ref_logp - policy_logp): 它无下界, 策略越自信值越负,
        #   优化器最小化 total 时反而驱动"模式坍缩" (rl_cont 系列全组坍缩根因, 日志里
        #   KL 一项恒为负 = 证据)。degenerate 时 advantages 归零, 这里就是拉回基座的主力。
        kl_tensor = F.kl_div(new_logprobs, reflogprobs, log_target=True, reduction='none')
        kl_loss = (kl_tensor * mask.float()).sum() / mask.sum().clamp(min=1)

        if isinstance(quiet_losses, (float, int)):
            quiet = torch.tensor(float(quiet_losses))
        elif isinstance(quiet_losses, list):
            # quiet_losses 可能混有 CPU/CUDA 张量 (compute_quiet_loss_from_hidden 的
            # 早退分支返回 CPU 0.0), stack 前统一归一到与 advantages 同设备。
            qr = [q for q in quiet_losses]
            if qr:
                dev = advantages.device
                quiet = torch.stack([q.to(dev) for q in qr]).mean()
            else:
                quiet = torch.tensor(0.0, device=advantages.device)
        else:
            quiet = quiet_losses.mean()

        total = pol_loss + self.beta_kl * kl_loss + self.quiet_scale * quiet
        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()
        return pol_loss.item(), kl_loss.item(), quiet.item()


__all__ = ["GrpoUpdater"]