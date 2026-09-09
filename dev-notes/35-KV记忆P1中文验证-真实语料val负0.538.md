# 35-KV记忆P1中文验证：真实语料上 val −0.538，记忆通道主力化

对象：`out/obs_zh_mem/best.pt` vs `out/obs_zh_base/best.pt`
数据：中文语料首次构建，7 源（shareai/zhuangxialie/dailychat/muice/multi_turn/zhihu_kol + 本地 DSH 292 交换），`--task-ratio 0.25 --insert-eos`（dev-notes/27 配方），train 60.8M / val 5.6M token，8000 词表；同种子 1337、同 400 步（0.1 epoch，无过拟合）。

## 结果
| 指标 | 基线（CSA+HCA） | 记忆（CSA+KV记忆） |
|---|---|---|
| **best val @400** | 4.9415 | **4.4033（−0.538）** |
| train loss @400 | 5.24 | 4.76（学得更快） |
| 长程通道贡献 | glob 0.09-0.14（最弱） | **mem 1.59-2.40（主力，2-4× 窗路径）** |
| 每步耗时 | 0.6s（compile） | 2.2s（eager） |

## 结论
1. **真实中文数据上记忆再次胜出、差距更大**（−0.538 vs synth 的 −0.24）：可学习遗忘/写入状态 > HCA 静态平均摘要，两数据集均成立。
2. **记忆通道被重度使用**：mem 输出范数 1.59-2.40，是原 HCA 的 10-20 倍且高于窗路径——主力长程通道，非辅助。
3. **400 步=0.1 epoch 无过拟合**：两条 val 曲线全程下降，对比干净。
4. **工程代价**：顺序扫描（256 步×6 层）与 torch.compile 不兼容 → eager 2.2s/步；下一步 chunk 并行（GLA）恢复 compile/提速。

## 数据构建备忘
`download_dialogue.py` 7 源下载成功；`train_tokenizer.py` 8000 词表（标记 id 与 masking 一致：[306,228] 模型：、[308,228] 用户：、[177,177] \n\n）；`prepare.py --task-ratio 0.25 --insert-eos`（val 来自 DIALOGUE_FILES 白名单 90/10）。
坑：DIALOGUE_FILES 白名单只有旧 5 源，shareai/zhuangxialie/dsh 不在其中 → val=0；补下 dailychat/muice/multi_turn/zhihu_kol 后正常。

## 待办（按优先级）
1. chunk 并行训练（恢复 compile，2.2s/步 → ~0.8s/步）
2. P2：Delta 擦写律 A/B（先擦冲突联想再写入）
3. P3：跨轮次持久化 + recall 验证（不重传上下文的流式推理）——记忆机制核心卖点
