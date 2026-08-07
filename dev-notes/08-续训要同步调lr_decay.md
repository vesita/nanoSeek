# 08 · 续训要同步调 lr_decay_iters

## 现象
`--init_from=resume` 续训，loss 基本不动，训练"白跑"。

## 根因
学习率余弦调度：`get_lr(it)` 在 `it > lr_decay_iters` 后返回 `min_lr`。
续训时 `iter_num` 已经 = 上次的 max_iters，如果 `lr_decay_iters` 还是旧值，
那续训一开始 `it > lr_decay_iters`，**学习率卡死在 min_lr**，几乎不学习。

## 修复
续训时把 `lr_decay_iters` 和 `max_iters` **同步拉长**，让余弦调度重新展开：

```sh
# 错：lr_decay_iters 还是 6000，续训从 iter 6000 开始 → LR 已是 min_lr
uv run python training/train.py config/xxx.yaml --init_from=resume --max_iters=12000

# 对：max_iters 和 lr_decay_iters 一起拉长 → 余弦从 max_lr 重新衰减
uv run python training/train.py config/xxx.yaml --init_from=resume --max_iters=12000 --lr_decay_iters=12000
```

## 怎么避免
- **resume 不只是加步数，要重设完整的学习率计划**（warmup 起点、decay 终点都要核对）。
- 检查方法：看训练输出的 lr 列，resume 后 lr 应该在重新走一遍余弦（从高到低），
  而不是一开始就是 min_lr。
- 同理，任何依赖全局步数的调度（warmup / 退火 / 循环 LR）在续训时都要显式重置。
