"""loss mask 向量化实现的等价性校验。

build_assistant_mask 已从逐 token Python 扫描改为全向量化（cummax 状态机，
见 training/masking.py）。本脚本用「冻结的旧实现」作为 ground truth，在随机
数据 + 真实 batch 上对拍，防止向量化重构引入语义漂移。用法：

    uv run --no-sync python training/check_loss_mask.py

通过时输出：全部一致（N 样本），并打印性能对比。
"""
import sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from training.masking import build_assistant_mask  # 现役向量化实现

M = [306, 228]; U = [308, 228]; S = [177, 177]  # train.py 默认标记配置


def reference_mask(ids_1d):
    """冻结的旧实现（逐 token 扫描），作为 ground truth。"""
    mask = torch.zeros(len(ids_1d), dtype=torch.bool, device=ids_1d.device)
    i = 0
    in_reply = False
    while i < len(ids_1d):
        tok = ids_1d[i].item()
        if i + 1 < len(ids_1d):
            nxt = ids_1d[i + 1].item()
            if tok == 306 and nxt == 228:   # 模型：
                mask[i] = True; mask[i + 1] = True; in_reply = True; i += 2; continue
            if tok == 308 and nxt == 228:   # 用户：
                mask[i] = True; mask[i + 1] = True; in_reply = False; i += 2; continue
            if tok == 177 and nxt == 177:   # \n\n
                in_reply = False; i += 2; continue
        if in_reply:
            mask[i] = True
        i += 1
    return mask


def random_batch(B, T, vocab):
    return torch.randint(0, vocab, (B, T))


def real_batch(path, B, T):
    data = np.memmap(path, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - T, (B,))
    return torch.stack([torch.from_numpy((data[i:i + T]).astype(np.int64)) for i in ix])


def check(y, tag):
    got = build_assistant_mask(y, M, U, S)
    for b in range(y.shape[0]):
        ref = reference_mask(y[b])
        assert torch.equal(got[b], ref), (
            f"{tag} sample {b} 不一致！"
            f"diff @ { (got[b] != ref).nonzero().flatten()[:10].tolist() }"
        )


def main():
    torch.manual_seed(1337)
    # 1) 随机数据：覆盖各种标记分布（含窗口内第一个事件是 user/sep 的情况）
    for trial in range(200):
        check(random_batch(64, 256, 8000), f"随机数据 trial {trial}")
    print("✓ 随机数据 200 batch × 64 样本：全部一致")

    # 2) 真实数据：synth（含完整对话结构）
    synth = os.path.join(os.path.dirname(__file__), '..', 'data', 'synth', 'train.bin')
    if os.path.exists(synth):
        for trial in range(50):
            check(real_batch(synth, 64, 256), f"synth 数据 trial {trial}")
        print("✓ synth 数据 50 batch × 64 样本：全部一致")
    else:
        print("（无 data/synth，跳过真实数据检查）")

    # 3) 边界：T=1 / T=2 最小窗口
    check(random_batch(4, 1, 8000), "T=1 边界")
    check(random_batch(4, 2, 8000), "T=2 边界")
    print("✓ 边界情况（T=1 / T=2）：全部一致")

    # 4) 性能对比
    y = random_batch(64, 256, 8000)
    import time
    for f, name in [(lambda: [reference_mask(y[i]) for i in range(64)], '旧实现(逐token)'),
                    (lambda: build_assistant_mask(y, M, U, S), '向量化')]:
        f(); t0 = time.perf_counter()
        for _ in range(100): f()
        print(f"{name}: {(time.perf_counter() - t0) / 100 * 1000:.2f} ms/batch")

    print("\n全部通过 ✅")


if __name__ == '__main__':
    main()
