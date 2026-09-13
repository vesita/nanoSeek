"""样本打包（document packing）+ 块对角注意力掩码的测试。

## 这个任务是干什么的

语料是一条扁平 token 流（block 之间只有 `\\n\\n`），而采样是全域随机窗口 ⇒
一个 256-token 窗口经常横跨多个样本，窗口内因果注意力会让样本 A 污染样本 B。
本项目原来的边界阻断只接在 CSA / KV 记忆两条路径上（都要显式开开关），
主线 MLA 普通因果路径根本不接收 `is_eos` ⇒ `sample_boundary_reset` 是死旋钮；
而且 `<eos>` 不足以当边界（c4_zh/wikipedia/名著整块没有 `<eos>`）。

修法：`prepare.py --emit-offsets` 产出与 bin 逐 token 对齐的 block 边界表 `.off`，
训练时算出 `sample_id` 交给模型，让「同一样本」与因果掩码相与。

## 四组测试

1. `packing.py` 纯函数：校验、sample_id、块对角掩码、块对齐窗口；
2. **污染判据（决定性）**：同一窗口里扰动样本 A，样本 B 的 logits 必须**逐位不变**；
   关掉掩码后同一扰动必须改变 B 的 logits（证明这个测试真能检出污染，不是橡皮图章）；
3. **对齐判据**：`prepare.py --emit-offsets` 的 `.off` 与 bin 对齐（手工逐位核对 +
   切区间解码回原 block），以及**负向对照**（把 `.off` 平移 1 个 token 必须报错）；
4. **向后兼容**：`sample_id=None` 时行为与旧实现逐位一致（`is_eos` 仍被 MLA 路径忽略）。
"""
import ast
import json
import os
import pathlib
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from model.config import GPTConfig
from model.gpt import GPT
from training.packing import (
    aligned_pack_starts,
    aligned_window_starts,
    boundary_mask,
    causal_boundary_mask,
    load_offsets,
    mask_cross_sample_labels,
    offsets_name_for_bin,
    reachable_fraction,
    sample_id_in_window,
    unreachable_token_fraction,
    validate_offsets,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
DC = ROOT / 'data' / 'chinese'


def _module_literal(name):
    """取 `train.py` 里模块级赋值 `name = <literal>` 的字面值（不 import 那个脚本）。

    与 `test_doc_packing_keys_defined_before_config_keys` 同一手法（AST 读源码）。
    """
    tree = ast.parse((ROOT / 'training' / 'train.py').read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f'train.py 里找不到模块级赋值 {name}')


@pytest.fixture(scope='module')
def prep(load_module_from_path):
    """加载 prepare.py（与 tests/test_prepare_source_dir.py 同一手法）。"""
    if not (DC / 'prepare.py').exists():
        pytest.skip('data/chinese/prepare.py 不存在')
    if str(DC) not in sys.path:
        sys.path.insert(0, str(DC))
    return load_module_from_path(DC / 'prepare.py', 'prepare_offsets_under_test')


@pytest.fixture(scope='module')
def char_tok(prep):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(DC / 'char_tokenizer.json'))


# ==========================================================================
# 1) packing.py 纯函数
# ==========================================================================
def test_offsets_name_for_bin():
    assert offsets_name_for_bin('train_char_v2.bin') == 'train_char_v2.off'
    assert offsets_name_for_bin('pretrain.bin') == 'pretrain.off'
    assert offsets_name_for_bin('val.bin') == 'val.off'
    with pytest.raises(ValueError):
        offsets_name_for_bin('train_char_v2.bin.bak')


def test_offsets_name_matches_prepare_derivation(prep):
    """train.py 与 prepare.py 两处派生规则必须一致（否则训练找不到 sidecar）。"""
    for name in ('train_char_v2.bin', 'val_char_v2.bin', 'pretrain.bin', 'train.bin'):
        assert offsets_name_for_bin(name) == os.path.basename(prep.offsets_path_for_bin(name))


@pytest.mark.parametrize('off,n_tokens,ok', [
    (np.array([0, 5, 10]), 10, True),
    (np.array([0]), 0, True),                     # 空 split：只有哨兵
    (np.array([1, 5, 10]), 10, False),            # 首元素非 0（整体平移 +1）
    (np.array([0, 5, 9]), 10, False),             # 末元素 != n_tokens（整体平移 -1）
    (np.array([0, 5, 5, 10]), 10, False),         # 非严格递增（重复）
    (np.array([0, 5, 3, 10]), 10, False),         # 回退
    (np.array([0, 5]), 10, False),                # 哨兵缺失
    (np.array([[0, 5]]), 5, False),               # 非一维
])
def test_validate_offsets(off, n_tokens, ok):
    if ok:
        validate_offsets(off, n_tokens)
    else:
        with pytest.raises(ValueError):
            validate_offsets(off, n_tokens)


def test_load_offsets_roundtrip(tmp_path):
    p = tmp_path / 'x.off'
    np.array([0, 3, 7, 12], dtype=np.int64).tofile(p)
    off = load_offsets(str(p), 12)
    assert off.dtype == np.int64
    assert off.tolist() == [0, 3, 7, 12]
    # 负向对照：同一份 .off 配一个 token 数不匹配的 bin 必须报错
    with pytest.raises(ValueError, match='哨兵'):
        load_offsets(str(p), 13)


def test_sample_id_in_window_matches_bruteforce():
    off = np.array([0, 5, 12, 20, 40], dtype=np.int64)
    starts = np.array([0, 3, 12, 30], dtype=np.int64)
    sid = sample_id_in_window(off, starts, 8, device='cpu')
    assert sid.shape == (4, 8)
    assert (sid[:, 0] == 0).all(), '窗口内第一个 token 的样本号必须是 0'
    for b, s in enumerate(starts.tolist()):
        base = int(np.searchsorted(off, s, side='right') - 1)
        row = sid[b].tolist()
        assert row == sorted(row), '同一窗口内样本号必须单调不减'
        for t, v in enumerate(row):
            expect = int(np.searchsorted(off, s + t, side='right') - 1) - base
            assert v == expect, f'start={s} t={t}: {v} != {expect}'


def test_sample_id_jumps_exactly_at_boundaries():
    off = np.array([0, 4, 9, 15], dtype=np.int64)
    sid = sample_id_in_window(off, np.array([0]), 15, device='cpu')[0].tolist()
    assert sid[:4] == [0] * 4
    assert sid[4:9] == [1] * 5
    assert sid[9:15] == [2] * 6


def test_boundary_mask_shape_and_semantics():
    sid = torch.tensor([[0, 0, 1, 1]])
    m = boundary_mask(sid)
    assert m.shape == (1, 4, 4) and m.dtype == torch.bool
    assert m[0, 0, 1] and m[0, 2, 3]
    assert not m[0, 0, 2] and not m[0, 2, 0]
    assert m[0].diagonal().all(), '同一样本自己必须可见（配合因果用）'
    cm = causal_boundary_mask(sid)
    assert cm[0, 1, 0] and not cm[0, 0, 1], '因果上三角必须被排除'
    assert not cm[0, 2, 0], '跨样本即使因果可见也必须被排除'


def test_aligned_pack_starts_properties():
    # 小块（长度 3~7）→ 一个 16-token 窗口能装多个完整 block
    lens = [3, 5, 7, 4, 6, 3, 8, 5, 4, 6, 7, 3]
    off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    n_tokens = int(off[-1])
    T, B = 16, 64
    g = torch.Generator().manual_seed(0)
    starts = aligned_pack_starts(off, T, B, generator=g)
    assert starts.shape == (B,) and starts.dtype == np.int64
    off_set = set(off.tolist())
    for s in starts.tolist():
        assert s >= 0
        assert s + T <= n_tokens - 1, 'y 错位 1 不能越界'
        # 窗口结束落在 block 边界（只有被 0 截断的极端情况例外）
        assert (s + T) in off_set or s == 0, f'start={s} 的窗口末尾不落在 block 边界'
    # 同 seed 可复现
    g2 = torch.Generator().manual_seed(0)
    assert np.array_equal(starts, aligned_pack_starts(off, T, B, generator=g2))
    # RNG 消耗不同 seed 会给出不同窗口（不是常数）
    g3 = torch.Generator().manual_seed(1)
    assert not np.array_equal(starts, aligned_pack_starts(off, T, B, generator=g3))


# ==========================================================================
# 2) ★ 污染判据（决定性）
# ==========================================================================
def _doc_model(use_mla=True, use_attn_sink=True, manual=False, **over):
    """构造一个**无状态**的小模型（关掉 MoE/MTP/mHC），保证两次前向逐位可比。

    注意：MoE 的 aux-free router bias 会在 forward 里原地更新（no_grad 副作用），
    用它做逐位对比会自己制造差异 —— 这里刻意关掉。
    """
    kw = dict(block_size=64, vocab_size=128, n_layer=2, n_head=2, n_embd=32,
              dropout=0.0, bias=False, use_rope=True, rope_theta=10000.0,
              use_mla=use_mla, kv_lora_rank=16, qk_rope_head_dim=8,
              use_moe=False, use_mtp=False, use_mhc=False,
              use_attn_sink=use_attn_sink, use_qk_norm=False,
              char_level=True, eos_token_id=128)
    kw.update(over)
    torch.manual_seed(1234)
    m = GPT(GPTConfig(**kw)).eval()
    if manual:
        for blk in m.transformer.h:
            blk.attn.flash = False   # 强制走手动 attention 分支
    return m


def _two_sample_batch(T=32, split=16, seed=3):
    """一个窗口装两个样本：位置 [0,split) 是 A，[split,T) 是 B。"""
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 100, (1, T), generator=g)
    y = torch.randint(0, 100, (1, T), generator=g)
    sid = torch.zeros(1, T, dtype=torch.long)
    sid[:, split:] = 1
    return x, y, sid


@pytest.mark.parametrize('use_mla', [True, False])
@pytest.mark.parametrize('manual', [False, True])
def test_packing_blocks_cross_sample_contamination(use_mla, manual):
    """★ 决定性判据：扰动 A，B 位置的 logits 必须**逐位不变**。

    同一测试内必须同时验证反向对照：关掉掩码（sample_id=None）后扰动 A
    **必须**改变 B 的 logits —— 否则这条测试可能在"扰动根本没起作用"时空过。
    """
    m = _doc_model(use_mla=use_mla, manual=manual)
    x, y, sid = _two_sample_batch()
    x_pert = x.clone()
    x_pert[:, :16] = (x_pert[:, :16] + 37) % 100      # 只扰动样本 A

    with torch.no_grad():
        logits_same = m(x, y, sample_id=sid)[0]
        logits_pert = m(x_pert, y, sample_id=sid)[0]
        logits_off_same = m(x, y)[0]
        logits_off_pert = m(x_pert, y)[0]

    # 打包生效：B 段逐位不变
    assert torch.equal(logits_same[:, 16:], logits_pert[:, 16:]), \
        '打包没有阻断跨样本污染：扰动 A 改变了 B 的 logits'
    # A 段当然会变（证明扰动确实进了模型）
    assert not torch.equal(logits_same[:, :16], logits_pert[:, :16])
    # 反向对照：关掉掩码后 B 段必须变（证明这条测试能检出污染）
    assert not torch.equal(logits_off_same[:, 16:], logits_off_pert[:, 16:]), \
        '关掉掩码后扰动 A 竟然不影响 B —— 这个测试是橡皮图章，不能作为判据'
    # 而且关掉掩码时确实不等于打包版（说明掩码真的在改变计算）
    assert not torch.equal(logits_same[:, 16:], logits_off_same[:, 16:])


def test_packing_blocks_with_attn_sink_float_mask():
    """sink 分支走的是显式 float 掩码（B,nh,T,T+1），必须同样被覆盖。"""
    m = _doc_model(use_mla=True, use_attn_sink=True)
    x, y, sid = _two_sample_batch()
    x_pert = x.clone()
    x_pert[:, :16] = (x_pert[:, :16] + 11) % 100
    with torch.no_grad():
        a = m(x, y, sample_id=sid)[0][:, 16:]
        b = m(x_pert, y, sample_id=sid)[0][:, 16:]
        c = m(x, y)[0][:, 16:]
        d = m(x_pert, y)[0][:, 16:]
    assert torch.equal(a, b)
    assert not torch.equal(c, d)


def test_packing_three_samples_no_leak_between_any_pair():
    """三样本窗口：扰动中间样本，前后两个样本的 logits 都必须不变。"""
    T, cuts = 48, [0, 12, 26, 48]
    sid = torch.zeros(1, T, dtype=torch.long)
    sid[:, cuts[1]:cuts[2]] = 1
    sid[:, cuts[2]:] = 2
    m = _doc_model(use_mla=True)
    g = torch.Generator().manual_seed(7)
    x = torch.randint(0, 100, (1, T), generator=g)
    y = torch.randint(0, 100, (1, T), generator=g)
    x_pert = x.clone()
    x_pert[:, cuts[1]:cuts[2]] = (x_pert[:, cuts[1]:cuts[2]] + 5) % 100
    with torch.no_grad():
        a, b = m(x, y, sample_id=sid)[0], m(x_pert, y, sample_id=sid)[0]
    assert torch.equal(a[:, :cuts[1]], b[:, :cuts[1]])
    assert torch.equal(a[:, cuts[2]:], b[:, cuts[2]:])
    assert not torch.equal(a[:, cuts[1]:cuts[2]], b[:, cuts[1]:cuts[2]])


# ==========================================================================
# 3) 向后兼容（sample_id=None 与旧实现逐位一致）
# ==========================================================================
@pytest.mark.parametrize('use_mla', [True, False])
def test_is_eos_is_still_ignored_on_standard_path(use_mla):
    """旧实现里 MLA/标准路径**不接收** is_eos（sample_boundary_reset 是死旋钮）。

    改动后必须保持：只传 is_eos（不传 sample_id）时，该层输出与什么都不传逐位一致 ——
    否则 base_v2 的既有语义会被这次改动偷偷改掉。
    """
    from model.attention import CausalSelfAttention
    cfg = GPTConfig(block_size=64, vocab_size=128, n_layer=1, n_head=2, n_embd=32,
                    dropout=0.0, bias=False, use_rope=True, rope_theta=10000.0,
                    use_mla=use_mla, kv_lora_rank=16, qk_rope_head_dim=8,
                    use_attn_sink=True, use_qk_norm=False)
    torch.manual_seed(1234)
    attn = CausalSelfAttention(cfg).eval()
    x = torch.randn(1, 24, 32, generator=torch.Generator().manual_seed(0))
    is_eos = torch.zeros(1, 24, dtype=torch.bool)
    is_eos[:, 12] = True
    with torch.no_grad():
        a = attn(x)                       # 旧调用方式（不传 is_eos/sample_id）
        b = attn(x, is_eos=is_eos)        # 旧 gpt.py 会传的那条路
        c = attn(x, is_eos=is_eos, sample_id=None)
    assert torch.equal(a, b), 'is_eos 不该在 MLA/标准路径生效（那会改变 base_v2 行为）'
    assert torch.equal(a, c)


def test_no_sample_id_ignores_offsets_entirely():
    """不传 sample_id 时，模型对"边界表/样本划分"零依赖：同样输入同样输出。"""
    m = _doc_model(use_mla=True)
    x, y, _ = _two_sample_batch()
    with torch.no_grad():
        a = m(x, y)[0]
        b = m(x, y)[0]
    assert torch.equal(a, b)


def test_doc_packing_keys_defined_before_config_keys():
    """铁律 8：新配置键必须定义在 `config_keys` 快照**之前**，否则被静默改回默认。"""
    tree = ast.parse((ROOT / 'training' / 'train.py').read_text(encoding='utf-8'))
    names, marker_line = [], None
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    if t.id == 'config_keys':
                        marker_line = node.lineno
                    names.append(t.id)
        if marker_line is not None:
            break
    assert marker_line is not None
    for key in ('use_doc_packing', 'pack_align'):
        assert key in names, (
            f'{key} 必须定义在 config_keys 快照（train.py:{marker_line}）之前，'
            f'否则 CLI/YAML 设了会被静默改回默认')


# ==========================================================================
# 4) ★ 对齐判据：prepare.py --emit-offsets
# ==========================================================================
# 3 个 block，其中第 3 个非对话（**没有 <eos>**，正是"<eos> 不足以当边界"的现场）
TINY = {
    'a.txt': ('用户：你好吗\n模型：我很好，你呢？\n\n'
              '用户：讲个故事\n模型：从前有座山\n\n'
              '山里有座庙，庙里有个老和尚'),
    'b.txt': ('这是一段没有对话标签的百科文本，整块都没有终止符。\n\n'
              '第二段也一样，用来验证 .off 对非对话 block 同样成立。'),
}


def _write_corpus(root, mapping=None):
    root.mkdir(parents=True, exist_ok=True)
    for name, text in (mapping or TINY).items():
        (root / name).write_text(text, encoding='utf-8')
    return root


def _args_ns(prep, argv):
    """用 parse_args 拿真实 Namespace，再按 main 的解析补上 source_dir 语义。"""
    a = prep.parse_args(argv)
    if a.no_insert_eos:
        a.insert_eos = False
    return a


def _replay_pipeline(prep, src, args):
    """按 main() 的完全相同顺序重建 train/val 的样本列表（含标注与随机种子）。

    这样测试能拿到"prepare 实际写进 bin 的那些 block"，用来手算期望的 .off。
    """
    train_samples, val_samples = [], []
    for fn in sorted(os.listdir(src)):
        if not fn.endswith('.txt'):
            continue
        with open(os.path.join(src, fn), 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        blocks = [b.strip() for b in text.split('\n\n') if b.strip()]
        src_ratio = None
        for spec in args.source_ratio:
            name, ratio = spec.split('=', 1)
            if fn.startswith(name):
                src_ratio = float(ratio)
        tr, va = prep.split_one_source(fn, blocks, args, src_ratio)
        train_samples += tr
        val_samples += va
    if args.seed is not None:
        random.seed(args.seed)
    if args.insert_eos:
        train_samples = [prep.insert_eos_after_replies(b) for b in train_samples]
        val_samples = [prep.insert_eos_after_replies(b) for b in val_samples]
    return train_samples, val_samples


def _expected_offsets(tok, samples):
    """独立手算：逐 block 编码的 token 数 + 每个 '\\n\\n' 分隔符的 token 数。

    off[i] = 第 i 个 block 的**起始** token 下标 ⇒ 累加 block i 的长度后再加分隔符。
    """
    off = [0]
    pos = 0
    sep = len(tok.encode('\n\n').ids)
    for i, b in enumerate(samples):
        pos += len(tok.encode(b).ids)
        if i < len(samples) - 1:
            pos += sep
        off.append(pos)
    return off


def _run_prepare(prep, tmp_path, monkeypatch, src, out, argv):
    monkeypatch.setattr(prep, 'DATA_DIR', str(out))
    prep.main(argv)


def test_emit_offsets_aligns_with_bin_and_decodes_back(prep, char_tok, tmp_path, monkeypatch):
    """★ 对齐判据：.off 与 bin 逐位对齐 + 用 .off 切出的区间解码回原 block。"""
    src = _write_corpus(tmp_path / 'corpus')
    out = tmp_path / 'products'
    out.mkdir()
    argv = ['--char-level', '--source-dir', str(src), '--out-prefix', 'x',
            '--seed', '7', '--emit-offsets']
    _run_prepare(prep, tmp_path, monkeypatch, src, out, argv)

    args = _args_ns(prep, argv)
    train_samples, val_samples = _replay_pipeline(prep, str(src), args)
    assert len(train_samples) == 5 and len(val_samples) == 0

    train_bin = out / 'train_char_x.bin'
    tokens = np.fromfile(train_bin, dtype=np.uint16)
    off = np.fromfile(out / 'train_char_x.off', dtype=np.int64)

    expect = _expected_offsets(char_tok, train_samples)
    assert off.tolist() == expect, f'.off 与手工 token 长度不一致：{off.tolist()} != {expect}'
    assert len(off) == len(train_samples) + 1
    assert off[0] == 0 and off[-1] == len(tokens) == os.path.getsize(train_bin) // 2

    # 性质：每个区间解码后 == 原 block（分隔符 '\n\n' 归到前一个区间尾部）。
    # skip_special_tokens=False：<eos>/<cont> 是标注的一部分，默认会被 decode 丢掉。
    for i, blk in enumerate(train_samples):
        seg = tokens[off[i]:off[i + 1]]
        text = char_tok.decode(seg.tolist(), skip_special_tokens=False)
        expect_text = blk if i == len(train_samples) - 1 else blk + '\n\n'
        assert text == expect_text, f'block {i} 解码不一致：{text!r} != {expect_text!r}'

    # 关键点：至少有一个 block 没有任何 <eos>/<cont>，.off 依旧成立
    eos_ids = {tok_id for sym in ('<eos>', '<cont>')
               if (tok_id := char_tok.token_to_id(sym)) is not None}
    assert any(not (set(tokens[off[i]:off[i + 1]].tolist()) & eos_ids)
               for i in range(len(train_samples))), '测试语料必须含无终止符 block'

    # 空 val：只有哨兵 [0]，且合法
    val_off = np.fromfile(out / 'val_char_x.off', dtype=np.int64)
    assert val_off.tolist() == [0]
    validate_offsets(val_off, 0)

    # manifest 登记（文件名 + sha256 + 元素数）
    man = json.loads((out / 'manifest_x.json').read_text(encoding='utf-8'))
    assert man['prepare_args']['emit_offsets'] is True
    art = man['artifacts']['train_char_x.off']
    assert art['elements'] == len(off)
    assert len(art['sha256']) == 64


def test_emit_offsets_does_not_change_bin_bytes(prep, tmp_path, monkeypatch):
    """默认关 = 旧行为逐字节不变；开了 --emit-offsets 也不许改 bin（只多一个 sidecar）。"""
    src = _write_corpus(tmp_path / 'corpus')
    out = tmp_path / 'products'
    out.mkdir()
    base = ['--char-level', '--source-dir', str(src), '--seed', '7']
    _run_prepare(prep, tmp_path, monkeypatch, src, out, base + ['--out-prefix', 'n1'])
    _run_prepare(prep, tmp_path, monkeypatch, src, out,
                 base + ['--out-prefix', 'n2', '--emit-offsets'])
    a = (out / 'train_char_n1.bin').read_bytes()
    b = (out / 'train_char_n2.bin').read_bytes()
    assert a == b, '--emit-offsets 不许改变 bin 的字节'
    tokens = np.fromfile(out / 'train_char_n2.bin', dtype=np.uint16)
    off = np.fromfile(out / 'train_char_n2.off', dtype=np.int64)
    assert off[-1] == len(tokens) and off[0] == 0
    # manifest 记录开关状态 + artifacts 登记 sidecar
    man = json.loads((out / 'manifest_n2.json').read_text(encoding='utf-8'))
    assert man['prepare_args']['emit_offsets'] is True
    assert 'train_char_n2.off' in man['artifacts']


def test_offsets_valid_on_nonempty_val_split(prep, char_tok, tmp_path, monkeypatch):
    """val-all 切分下 val 也非空，两份 .off 都要与各自 bin 对齐。"""
    src = _write_corpus(tmp_path / 'corpus')
    out = tmp_path / 'products'
    out.mkdir()
    argv = ['--char-level', '--source-dir', str(src), '--out-prefix', 'v',
            '--val-all', '--val-ratio', '0.4', '--seed', '3', '--emit-offsets']
    _run_prepare(prep, tmp_path, monkeypatch, src, out, argv)
    args = _args_ns(prep, argv)
    tr, va = _replay_pipeline(prep, str(src), args)
    assert va, 'val-ratio=0.4 必须切出非空 val，否则本测试没覆盖到'
    for split, samples in (('train', tr), ('val', va)):
        bin_name = f'{split}_char_v.bin'
        n = os.path.getsize(out / bin_name) // 2
        off = load_offsets(str(out / f'{split}_char_v.off'), n)
        assert off.tolist() == _expected_offsets(char_tok, samples)


def test_negative_control_shifted_offsets_are_rejected(prep, char_tok, tmp_path, monkeypatch):
    """★ 负向对照：故意把 .off 平移 1 个 token（以及重复项）必须报错，不许静默通过。"""
    src = _write_corpus(tmp_path / 'corpus')
    out = tmp_path / 'products'
    out.mkdir()
    argv = ['--char-level', '--source-dir', str(src), '--out-prefix', 's',
            '--seed', '7', '--emit-offsets']
    _run_prepare(prep, tmp_path, monkeypatch, src, out, argv)
    n = os.path.getsize(out / 'train_char_s.bin') // 2
    good = np.fromfile(out / 'train_char_s.off', dtype=np.int64)
    assert good[-1] == n

    # 平移 +1：首元素变成 1（与 .off 契约首 0 冲突）
    p = out / 'shift_p1.off'
    (good + 1).astype(np.int64).tofile(p)
    with pytest.raises(ValueError, match='首元素'):
        load_offsets(str(p), n)

    # 平移 -1：末位哨兵变成 n-1 ≠ bin token 数
    p = out / 'shift_m1.off'
    (good - 1).astype(np.int64).tofile(p)
    with pytest.raises(ValueError, match='首元素|哨兵'):
        load_offsets(str(p), n)

    # 只动末位哨兵 → 必须报"哨兵"错（.off 与 bin 错位的典型现场）
    p = out / 'bad_sentinel.off'
    bad = good.copy()
    bad[-1] += 1
    bad.tofile(p)
    with pytest.raises(ValueError, match='哨兵'):
        load_offsets(str(p), n)

    # 缺哨兵（截断一个元素）
    p = out / 'truncated.off'
    good[:-1].tofile(p)
    with pytest.raises(ValueError, match='哨兵'):
        load_offsets(str(p), n)

    # 重复元素 → 非严格递增
    dup = good.copy()
    dup[1] = dup[0]
    p = out / 'dup.off'
    dup.tofile(p)
    with pytest.raises(ValueError, match='递增'):
        load_offsets(str(p), n)


def test_emit_offsets_rejected_for_byte_level(prep, tmp_path, monkeypatch):
    """byte 模式没实现边界对齐 → 必须显式报错，而不是产出一个错的 .off。"""
    src = _write_corpus(tmp_path / 'corpus')
    out = tmp_path / 'products'
    out.mkdir()
    monkeypatch.setattr(prep, 'DATA_DIR', str(out))
    with pytest.raises(SystemExit, match='emit-offsets'):
        prep.main(['--byte-level', '--source-dir', str(src), '--out-prefix', 'b',
                   '--emit-offsets'])


# ==========================================================================
# 5) 边界 label 泄漏（掩码管注意力，管不到 y 的错位）
# ==========================================================================
# 位置 t 的 label 是 token t+1。若 t 与 t+1 不同样本，这条标签就是"用 A 的结尾
# 预测 B 的开头"——纯噪声，而且正是打包要消除的跨样本污染。
# ★ 2026-09-13 修：sample_id 必须给 (B, T+1)；旧契约 (B,T) 会让最后一个 label
#   （t = T-1，块对齐窗口下**必然**跨样本）漏网。见 analysis/packing_audit.md §3。
def test_mask_cross_sample_labels_known_answer_t_plus_1():
    import torch
    y = torch.tensor([[10, 11, 12, 13, 14]])
    # 内部接缝：t=2 处跨样本（sid[2]!=sid[3]）→ 屏蔽位置 2
    out = mask_cross_sample_labels(y, torch.tensor([[0, 0, 0, 1, 1, 1]]))
    assert out.tolist() == [[10, 11, -100, 13, 14]]
    # ★ 右端：x 全属样本 0，y[4]（= 第 6 格）属样本 1 → 只屏蔽最后一位。
    #   这正是旧 (B,T) 契约结构上看不到的那一格。
    out_right = mask_cross_sample_labels(y, torch.tensor([[0, 0, 0, 0, 0, 1]]))
    assert out_right.tolist() == [[10, 11, 12, 13, -100]]
    # 三样本窗口：t=0/2/4 三处 label 跨样本（含右端 t=4）
    out3 = mask_cross_sample_labels(y, torch.tensor([[0, 1, 1, 2, 2, 3]]))
    assert out3.tolist() == [[-100, 11, -100, 13, -100]]
    # ★ 反向对照：同一段样本（sid 全同）时**一个都不许屏蔽** ——
    #   防止实现退化成"无差别屏蔽一半 label"这种看似有信号其实错的写法
    same = mask_cross_sample_labels(y, torch.zeros(1, 6, dtype=torch.long))
    assert same.tolist() == y.tolist()


def test_mask_cross_sample_labels_rejects_t_length_loudly():
    """★ 旧契约 (B,T) 必须**大声报错**，不许静默漏掉最后一位（那是原 bug 的根源）。"""
    import torch
    y = torch.tensor([[1, 2, 3, 4, 5]])
    with pytest.raises(ValueError, match=r'T\+1'):
        mask_cross_sample_labels(y, torch.zeros_like(y))
    with pytest.raises(ValueError):
        mask_cross_sample_labels(y, torch.zeros(1, 7, dtype=torch.long))


def test_mask_cross_sample_labels_none_is_noop():
    import torch
    y = torch.tensor([[1, 2, 3]])
    assert mask_cross_sample_labels(y, None) is y


def test_boundary_labels_never_leak_across_samples():
    """不变量：打包窗口里，任何**跨样本接缝**处的 label 都必须是 ignore_index。"""
    import torch
    sid_ext = torch.tensor([[0, 0, 0, 0, 1, 1, 2, 2, 2]])   # y 有 8 个 label
    y = torch.arange(8).reshape(1, -1) + 100
    out = mask_cross_sample_labels(y, sid_ext)
    cross = sid_ext[:, 1:] != sid_ext[:, :-1]              # (1, 8)
    assert bool((out[cross] == -100).all())
    assert bool((out[~cross] != -100).all())


def test_real_offsets_no_cross_sample_label_left_both_sampling_modes():
    """★ 真实 `.off` 上：T+1 掩码后，**两种采样口径**剩下的跨样本 label 都必须是 0。

    同时给出反向对照：把 sample_id 退回旧的 T 长视图（模拟修复前的行为）时，
    对齐口径必须**报出非 0**（每窗口恰好 1 个）—— 证明这个判据不是恒真的空测试。
    只读 `.off`（不需要 bin）：跨样本与否只取决于样本号，与 token 内容无关。
    """
    import torch
    offp = DC / 'train_char_v3_dlg.off'
    if not offp.exists():
        pytest.skip('data/chinese/train_char_v3_dlg.off 不存在')
    off = np.fromfile(offp, dtype=np.int64)
    T, B = 256, 512
    rng = np.random.default_rng(0)
    starts_random = rng.integers(0, int(off[-1]) - T - 1, size=B).astype(np.int64)
    starts_aligned = aligned_pack_starts(off, T, B, generator=torch.Generator().manual_seed(0))
    y = torch.zeros(B, T, dtype=torch.long)
    left = {}
    for tag, starts in (('random', starts_random), ('aligned', starts_aligned)):
        sid_new = sample_id_in_window(off, starts, T + 1)
        sid_old = sample_id_in_window(off, starts, T)
        # 真值：每个 label 是否跨样本
        truth = sid_new[:, 1:] != sid_new[:, :-1]
        new_masked = mask_cross_sample_labels(y, sid_new)
        left[tag] = int((truth & (new_masked != -100)).sum())
        old = y.clone()
        old[:, :-1][sid_old[:, 1:] != sid_old[:, :-1]] = -100      # 模拟修复前
        if tag == 'aligned':
            old_left = int((truth & (old != -100)).sum())
    assert left['random'] == 0, f'随机口径仍有 {left["random"]} 个跨样本 label 漏网'
    assert left['aligned'] == 0, f'对齐口径仍有 {left["aligned"]} 个跨样本 label 漏网'
    # ★ 反向对照：修复前行为在 aligned 口径下漏掉接近整整 B 个（每窗口 1 个）
    assert old_left > 0.9 * B, f'反向对照失效：旧 T 视图只漏了 {old_left}/{B}（应对齐口径≈每窗 1 个）'


# ==========================================================================
# 6) 采样覆盖：块对齐模式的**结构性漏洞**（2026-09-13 审计）
# ==========================================================================
def test_aligned_window_starts_matches_sampled_function():
    """枚举式 `aligned_window_starts` 必须覆盖真实 `aligned_pack_starts` 能抽到的一切。"""
    import torch
    lens = [3, 1000, 40, 5, 600, 7, 300, 4, 9, 2]        # 短块 + 超长块混合
    off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    T = 64
    enumerated = set(aligned_window_starts(off, T).tolist())
    g = torch.Generator().manual_seed(0)
    sampled = set(aligned_pack_starts(off, T, 4096, generator=g).tolist())
    assert sampled, '采样必须是非空的，否则这条测试什么也没测'
    assert sampled <= enumerated, f'真实函数抽到了枚举集之外的 start：{sorted(sampled - enumerated)}'


def test_unreachable_token_fraction_known_answer():
    """★ 已知答案小例（期望值手算得出，直接写死在测试里）。"""
    # 块长 1000 / 256，T=256：唯一窗口 = [1000-256, 1000) = [744,1000)
    # ⇒ 744 个 token 不可达 + 后一个块 1000..1255 也采不到 → 1000/1256
    off = np.array([0, 1000, 1256], dtype=np.int64)
    assert abs(unreachable_token_fraction(off, 256, align=True) - 1000 / 1256) < 1e-12
    assert abs(reachable_fraction(off, 256, align=True) - 256 / 1256) < 1e-12
    assert abs(unreachable_token_fraction(off, 256, align=True)
               + reachable_fraction(off, 256, align=True) - 1.0) < 1e-12
    # 块长 300 / 300，T=256：唯一窗口 = [44,300) ⇒ 344/600 不可达
    off2 = np.array([0, 300, 600], dtype=np.int64)
    assert abs(unreachable_token_fraction(off2, 256, align=True) - 344 / 600) < 1e-12
    # 全部块都短：窗口能盖满整段 ⇒ 0 个不可达
    off3 = np.array([0, 100, 200, 256], dtype=np.int64)
    assert unreachable_token_fraction(off3, 256, align=True) == 0.0
    # 全域随机窗口口径：并集 = [0, n-1) ⇒ 只有最后一个 token 不可达
    off4 = np.array([0, 100, 200, 400], dtype=np.int64)
    assert abs(unreachable_token_fraction(off4, 256, align=False) - 1 / 400) < 1e-12


def test_unreachable_token_fraction_real_offsets_pins_known_bad():
    """★ 把"已知的坏"钉在真实数据上：aligned ≈ 67.4%（v3_dlg），random ≈ 0%。"""
    offp = DC / 'train_char_v3_dlg.off'
    if not offp.exists():
        pytest.skip('data/chinese/train_char_v3_dlg.off 不存在')
    off = np.fromfile(offp, dtype=np.int64)
    u_aligned = unreachable_token_fraction(off, 256, align=True)
    u_random = unreachable_token_fraction(off, 256, align=False)
    # 2026-09-13 审计实测 76,650,229 / 113,734,729 = 0.67394
    assert abs(u_aligned - 0.67394) < 0.001, f'aligned 口径不可达比例变了：{u_aligned}'
    assert u_random < 1e-6, f'随机窗口口径必须≈0，实际 {u_random}'
    # ★ 反向对照：判据必须能把两种口径分开（否则是恒真的空测试）
    assert u_aligned - u_random > 0.5


def test_pack_align_default_is_false_pinned():
    """★ 铁律级：`train.py` 的 `pack_align` 默认必须是 `False`（随机窗口口径）。

    块对齐模式有**结构性覆盖漏洞**：窗口锚在块尾 ⇒ 比 T 长的块，前 L−T 个 token
    永远进不了任何窗口。实测不可达 train token：v3_dlg 67.39% / v3_lang 67.05% /
    v3_know 72.91% / v2 70.67%（见 analysis/packing_audit.md §6）。
    这条测试把"默认走随机窗口"钉住，防止改回 True 而静默丢掉 2/3 语料。
    """
    assert _module_literal('pack_align') is False
    assert _module_literal('use_doc_packing') is False   # 打包默认仍关
