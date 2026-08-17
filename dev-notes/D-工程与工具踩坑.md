# 主题 D · 工程与工具踩坑

> 吸收笔记：[01](01-重训后要重新部署.md) [02](02-生成循环要加EOS停止.md) [03](03-prompt格式要和训练数据一致.md)
> [08](08-续训要同步调lr_decay.md) [09](09-matplotlib中文字体.md) [10](10-torch-compile良性警告.md) [15](15-Windows交叉编译用gnu不用msvc.md)

## 部署链路

1. **重训后必须重新部署**（01）：`convert.py`/`package.py` 做的是**快照**，新权重不会自动进 runtime。
   排查"模型效果不对"先查 `runtime/model.safetensors` 时间戳。习惯：训练完 → 手动过一遍部署链路。
2. **任何自回归生成循环必须有停止条件**（02）：EOS 或长度上限至少一个。从 tokenizer 取 `<eos>` id，
   采样后命中即 break（不输出、不进上下文）。训练数据里 `<eos>` 就是为"这句话说完了"设计的（注意：
   本项目当前 prepare.py **未插入 EOS**——模型分布里没有结束概念，是"喋喋不休"的根因之一，见主题 A）。
3. **Windows 交叉编译用 gnu 目标**（15）：zig 不内置 MSVC 头（`new.h`），`-msvc` ABI 一碰 C++ 依赖就缺头；
   `x86_64-pc-windows-gnu` 一次通过（package.py 已默认）。工具链：`cargo-zigbuild` + `uv pip install ziglang`。

## 训练工程

4. **prompt 模板必须和训练数据逐字符一致**（03）：全角 `：` vs 半角 `:` 是不同 token；`用户：…\n模型：`
   后紧跟答案无空格。写推理端前先 `grep "模型：" 数据 | head` 确认真实格式，排查时逐位对比 token 序列。
5. **续训要同步拉长 lr_decay_iters**（08）：resume 时 `iter_num` 已在上次 max_iters，若 `lr_decay_iters`
   不跟着拉长，LR 卡死在 min_lr → 训练"白跑"。`--max_iters` 和 `--lr_decay_iters` 一起改；看训练输出 lr 列
   确认余弦在重新走。

## 工具类坑

6. **matplotlib 中文三件套**（09）：① `font_manager.findfont` 找 CJK 字体（Noto Sans CJK SC 等，找不到
   try/except 回退）② 设 `plt.rcParams['font.family']` ③ `axes.unicode_minus=False`。验证：把 UserWarning
   当错误抛，有 glyph 警告即失败。`training/compare_ab.py` / `train.py` 的画图函数已内置这套。
7. **torch.compile 的 inductor 良性警告**（10）：RTX 5060 只有 30 SM（<68），`max_autotune_gemm` 用不了，
   打印后退回默认 matmul —— 纯良性。这是 logging 不是 warnings，用自定义 `logging.Filter` 定向静音
   （`train.py` 已内置），别把整个 logger 调 ERROR。

## 排查顺序建议

模型输出不对：先按 03（prompt 逐字符）→ 01（部署快照）→ 06（数据/训练/部署三层排查，见主题 A）的顺序走。
采样崩：先数据分布（主题 C）→ 再采样器交叉验证（inference/sample.py）→ 最后才怀疑架构。