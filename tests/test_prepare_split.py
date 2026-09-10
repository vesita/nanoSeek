"""`data/chinese/prepare.py` 数据切分逻辑的测试。

## 为什么值得测（这是本项目代价最大的一次事故）
旧 `prepare.py` 里 `DIALOGUE_FILES` 只硬编码了 5 个文件名，目录里却有 13+ 个
`*_dialogue.txt`：
- 名单内 → 10% 进 val；名单外（含 573MB 的 deepseek、267MB 的 qwen3）→ 1% 进 val
- 结果 val 把对话样本放大约 5 倍 → **train/val 测的根本不是同一个任务**
- 后果：终止符密度 val/train = 3.15×、有效 token 占比 6.29% / 27.39%、
  bigram CE 差 −0.12 nats —— 所有指标都是假的，还据此做过多次错误决策

修法是加 `--val-all`：所有来源统一 `--val-ratio`。下面的
`test_val_all_gives_every_source_the_same_ratio` 就是那次修复的回归测试。

## 另一层保障
`test_shipped_v2_manifest_matches_this_logic` 把**已验收的数据产物**
（`manifest_v2.json` 的逐源计数）与这段代码绑起来：只要切分逻辑被改动，
产物与代码就对不上，测试立刻红。
"""
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
DC = ROOT / 'data' / 'chinese'


@pytest.fixture(scope='module')
def prep(load_module_from_path):
    """加载 prepare.py。

    它用 `from split_sentences import split_text` 这种**裸模块名**导入，
    所以必须先把 data/chinese 塞进 sys.path（脚本直接跑时由 cwd 隐式提供）。
    """
    if not (DC / 'prepare.py').exists():
        pytest.skip("data/chinese/prepare.py 不存在")
    if str(DC) not in sys.path:
        sys.path.insert(0, str(DC))
    return load_module_from_path(DC / 'prepare.py', 'prepare_under_test')


def args(val_all=False, val_ratio=0.01, task_ratio=1.0):
    return SimpleNamespace(val_all=val_all, val_ratio=val_ratio, task_ratio=task_ratio)


def blocks_of(n, tag='b'):
    return [f'{tag}{i}' for i in range(n)]


# 既在旧 DIALOGUE_FILES 名单里的、也有不在里面的，覆盖两条分支
IN_LEGACY_LIST = 'dailychat_dialogue.txt'
NOT_IN_LEGACY_LIST = 'deepseek_dialogue.txt'      # 573MB 那个，当年被漏掉的
PLAIN = 'c4_zh.txt'


# ==========================================================================
# 1) ★ --val-all：所有来源同一个比例（事故回归）
# ==========================================================================
@pytest.mark.parametrize("fn", [IN_LEGACY_LIST, NOT_IN_LEGACY_LIST, PLAIN, 'wikipedia.txt'])
@pytest.mark.parametrize("n,ratio", [(1000, 0.1), (10000, 0.01), (250, 0.04)])
def test_val_all_gives_every_source_the_same_ratio(prep, fn, n, ratio):
    """不管来源是"对话文件"还是普通文本，val 比例必须**完全一致**。"""
    tr, va = prep.split_one_source(fn, blocks_of(n), args(True, ratio), None)
    assert len(va) == int(n * ratio), f"{fn}: val {len(va)} != int({n}*{ratio})"
    assert len(tr) == n - len(va)


def test_val_all_fix_removes_the_5x_distortion(prep):
    """把"旧行为"与"新行为"并排断言 —— 事故的本质就是两者的差异。

    旧行为下名单内 10%、名单外 1%，差 10 倍；`--val-all` 之后两者相等。
    """
    n, ratio = 10000, 0.01
    legacy_in = len(prep.split_one_source(IN_LEGACY_LIST, blocks_of(n), args(False, ratio), None)[1])
    legacy_out = len(prep.split_one_source(NOT_IN_LEGACY_LIST, blocks_of(n), args(False, ratio), None)[1])
    assert legacy_in == 1000 and legacy_out == 0, "旧行为：名单内 10%、名单外全进 train"
    assert legacy_in != legacy_out, "旧行为的偏差必须被这条测试显式记录"

    new_in = len(prep.split_one_source(IN_LEGACY_LIST, blocks_of(n), args(True, ratio), None)[1])
    new_out = len(prep.split_one_source(NOT_IN_LEGACY_LIST, blocks_of(n), args(True, ratio), None)[1])
    assert new_in == new_out == int(n * ratio), "修复后两者必须相等"


def test_val_all_val_composition_ratio_matches_train(prep):
    """更一般的性质：val 占比与 val_ratio 的偏差不超过一个 block。"""
    for n, ratio in [(1000, 0.1), (777, 0.03), (5000, 0.01)]:
        _, va = prep.split_one_source(PLAIN, blocks_of(n), args(True, ratio), None)
        assert abs(len(va) / n - ratio) <= 1.0 / n + 1e-12


def test_val_all_small_source_can_contribute_zero_val(prep):
    """小源可以贡献 0 条 val（`int(n*ratio)` 不取 max(1,·)）—— 这是刻意行为。"""
    n = 50
    _, va = prep.split_one_source(PLAIN, blocks_of(n), args(True, 0.01), None)
    assert len(va) == 0


# ==========================================================================
# 2) 旧模式（不传 --val-all）：保证向后兼容没被破坏
# ==========================================================================
def test_legacy_dialogue_is_90_10(prep):
    tr, va = prep.split_one_source(IN_LEGACY_LIST, blocks_of(1000), args(False), None)
    assert len(tr) == 900 and len(va) == 100


def test_legacy_non_dialogue_all_goes_to_train(prep):
    tr, va = prep.split_one_source(PLAIN, blocks_of(1000), args(False, task_ratio=1.0), None)
    assert len(tr) == 1000 and len(va) == 0


def test_legacy_task_ratio_zero_drops_task_data(prep):
    """task_ratio=0：任务/指令样本全部剔除（当年为了治"碎片拼贴"）。"""
    tr, va = prep.split_one_source(PLAIN, blocks_of(1000), args(False, task_ratio=0.0), None)
    assert tr == [] and va == []


def test_legacy_task_ratio_subsamples(prep):
    tr, va = prep.split_one_source(PLAIN, blocks_of(1000), args(False, task_ratio=0.2), None)
    assert len(tr) == 200 and va == []


# ==========================================================================
# 3) --source-ratio：只降采样 train，val 保持全量
# ==========================================================================
def test_src_ratio_truncates_train_only(prep):
    tr, va = prep.split_one_source(PLAIN, blocks_of(1000), args(False, task_ratio=1.0), 0.3)
    assert len(tr) == 300 and va == []


def test_src_ratio_does_not_change_val(prep):
    """★ 契约：`--source-ratio` 承诺"val.bin 不变，跨实验可比"。

    所以同一来源、同一 val_ratio 下，val 必须与有没有 src_ratio 完全无关。
    """
    base = prep.split_one_source(PLAIN, blocks_of(1000), args(True, 0.1), None)[1]
    with_ratio = prep.split_one_source(PLAIN, blocks_of(1000), args(True, 0.1), 0.5)[1]
    assert base == with_ratio, "src_ratio 不该影响 val 切分"


def test_src_ratio_applies_after_val_split(prep):
    n = 1000
    tr, va = prep.split_one_source(PLAIN, blocks_of(n), args(True, 0.1), 0.5)
    assert len(va) == 100
    assert len(tr) == int((n - 100) * 0.5)          # 在 train 侧（900）再砍一半


def test_src_ratio_never_yields_empty_train(prep):
    """`max(1, ...)` 保护：极端小比例也不能把 train 砍成 0（会炸后续编码）。"""
    tr, _ = prep.split_one_source(PLAIN, blocks_of(1000), args(False, task_ratio=1.0), 1e-9)
    assert len(tr) >= 1


# ==========================================================================
# 4) 不丢块 / 不重复 / 可复现
# ==========================================================================
@pytest.mark.parametrize("val_all,task_ratio", [
    (True, 1.0), (False, 1.0),
])
def test_no_block_is_lost_or_duplicated(prep, val_all, task_ratio):
    """**不做降采样**时，train + val 必须恰好是原集合（不丢不重）。

    切分 bug 最隐蔽的形态就是"某段数据既不在 train 也不在 val"——
    loss 曲线一切正常，只是模型永远学不到那部分。

    注意这里只覆盖 task_ratio=1.0（不丢数据）的两种模式；task_ratio<1
    是**故意**丢数据的，由 `test_legacy_task_ratio_subsamples` 单独覆盖。
    """
    src = blocks_of(1000)
    tr, va = prep.split_one_source(PLAIN, list(src),
                                   args(val_all, 0.1, task_ratio), None)
    assert sorted(tr + va) == sorted(src)
    assert len(set(tr) & set(va)) == 0, "train 与 val 有交集 = val 泄漏"


def test_same_input_gives_same_split(prep):
    """同 seed 派生规则必须逐位可复现 —— 否则数据集无法复现。"""
    a = prep.split_one_source(PLAIN, blocks_of(500), args(True, 0.1), None)
    b = prep.split_one_source(PLAIN, blocks_of(500), args(True, 0.1), None)
    assert a == b


def test_different_sources_get_different_permutations(prep):
    """seed 按文件名派生 → 各源排列独立（否则各源会以同一模式切片，引入系统性偏差）。"""
    a = prep.split_one_source('a.txt', blocks_of(500), args(True, 0.1), None)[1]
    b = prep.split_one_source('b.txt', blocks_of(500), args(True, 0.1), None)[1]
    assert a != b


def test_blocks_list_is_shuffled_in_place(prep):
    """记录一个**副作用**：`blocks` 会被原地 shuffle。

    当前调用方每次都新建列表且不复用，所以无害；但这是维护陷阱，钉住它，
    以后有人想复用同一个列表时会立刻看到这条测试的说明。
    """
    src = blocks_of(200)
    before = list(src)
    prep.split_one_source(PLAIN, src, args(True, 0.1), None)
    assert src != before, "如果这里不再原地打乱，就把这条测试改成断言无副作用"


# ==========================================================================
# 5) ★ 把已验收的数据产物与这段代码绑起来
# ==========================================================================
def test_shipped_v2_manifest_matches_this_logic():
    """`manifest_v2.json` 的逐源 val 计数必须与 `int(blocks*val_ratio)` 完全一致。

    这条测试的价值：数据产物是**已经验收通过**的（终止符密度 0.97×、
    bigram CE 差 +0.027）。只要切分逻辑被改动而没重新生成产物，
    产物与代码就对不上 —— 这个测试立刻红，避免"代码说 A、手里数据是 B"。
    """
    man = DC / 'manifest_v2.json'
    if not man.exists():
        pytest.skip("manifest_v2.json 不存在（换机器/未生成 v2 数据）")
    with open(man, encoding='utf-8') as f:
        m = json.load(f)

    a = m['prepare_args']
    assert a['val_all'] is True, "v2 必须是用 --val-all 生成的"
    assert a['source_ratio'] == [], "v2 没有做逐源降采样（全来源全量）"
    ratio = a['val_ratio']

    srcs = m['source_breakdown']
    assert len(srcs) == 25, f"来源数 {len(srcs)} 与记录不符"
    bad = []
    for s in srcs:
        expect_val = int(s['blocks'] * ratio)
        if s['val_blocks'] != expect_val or s['train_blocks'] != s['blocks'] - expect_val:
            bad.append((s['file'], s['blocks'], s['val_blocks'], expect_val))
    assert not bad, f"以下来源的切分与代码逻辑不符（文件, 总数, 实际val, 期望val）：{bad}"


def test_shipped_v2_has_no_dialogue_double_standard():
    """v2 里**每个**来源的 val 占比都要贴近 val_ratio，不允许有 10 倍的离群。

    旧数据集正是栽在这里（对话源 10%、其余 1%）。这条是"产物层面"的体检。
    """
    man = DC / 'manifest_v2.json'
    if not man.exists():
        pytest.skip("manifest_v2.json 不存在")
    with open(man, encoding='utf-8') as f:
        m = json.load(f)
    ratio = m['prepare_args']['val_ratio']
    worst = 0.0
    for s in m['source_breakdown']:
        if s['blocks'] < 100:            # 小源 int() 截断主导，不参与比较
            continue
        worst = max(worst, abs(s['val_blocks'] / s['blocks'] - ratio))
    assert worst < 0.002, f"最大 val 占比偏差 {worst:.4f}，说明存在双标准"


# ==========================================================================
# 6) named_output
# ==========================================================================
@pytest.mark.parametrize("stem,prefix,ext,expect", [
    ('train_char', '', '.bin', 'train_char.bin'),
    ('train_char', 'v2', '.bin', 'train_char_v2.bin'),
    ('val_char', 'v2', '.bin', 'val_char_v2.bin'),
    ('meta_char', 'v2', '.pkl', 'meta_char_v2.pkl'),
    ('manifest', 'v2', '.json', 'manifest_v2.json'),
    ('manifest', '', '.json', 'manifest.json'),
])
def test_named_output(prep, stem, prefix, ext, expect):
    assert prep.named_output(stem, prefix, ext) == expect
