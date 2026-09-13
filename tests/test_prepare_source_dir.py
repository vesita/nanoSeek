"""`data/chinese/prepare.py` 的 `--source-dir` 测试。

契约（任务书 B 项）：
* **扫描输入**用 `--source-dir`，默认 = `DATA_DIR`（即旧行为，一字不变）；
* **产物**（bin / manifest / pkl）仍写在 `DATA_DIR`；
* tokenizer 路径不受影响；
* `--source-dir` 进 manifest 的 `prepare_args`（可反查构建用了哪份语料）。

测试手法：把模块级 `DATA_DIR` monkeypatch 到临时目录，这样 `main()` 的产物写进
tmp 而**不会污染仓库里的 data/chinese**。tokenizer 路径是 import 时算好的独立常量，
正好用来验证"不受影响"。语料用 `--char-level`（走 `char_tokenizer.json`），
几行文本即可端到端跑完。
"""
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
DC = ROOT / 'data' / 'chinese'

TINY = {
    'a.txt': '用户：你好\n模型：你好，有什么可以帮你\n\n用户：再问\n模型：好的',
    'b.txt': '用户：讲个故事\n模型：从前有座山，山里有座庙',
}


@pytest.fixture(scope='module')
def prep(load_module_from_path):
    if not (DC / 'prepare.py').exists():
        pytest.skip('data/chinese/prepare.py 不存在')
    if str(DC) not in sys.path:
        sys.path.insert(0, str(DC))
    return load_module_from_path(DC / 'prepare.py', 'prepare_srcdir_under_test')


def write_corpus(root: pathlib.Path, mapping=None):
    root.mkdir(parents=True, exist_ok=True)
    for name, text in (mapping or TINY).items():
        (root / name).write_text(text, encoding='utf-8')
    return root


def run_main(prep, extra):
    prep.main(['--char-level', '--val-all', '--val-ratio', '0.5',
               '--seed', '1', *extra])


# ==========================================================================
# 1) 参数本身
# ==========================================================================
def test_source_dir_flag_is_recorded(prep):
    args = prep.parse_args(['--source-dir', '/tmp/somewhere'])
    assert args.source_dir == '/tmp/somewhere'


def test_source_dir_default_is_none_then_resolved_to_data_dir(prep):
    """默认不传 ⇒ `None`；`main` 里解析成 `DATA_DIR`（旧行为）。"""
    assert prep.parse_args([]).source_dir is None
    assert prep.DATA_DIR == str(DC), 'DATA_DIR 必须还是 data/chinese'


# ==========================================================================
# 2) 默认行为不变：不传 --source-dir 时扫 DATA_DIR，产物也写 DATA_DIR
# ==========================================================================
def test_default_scans_data_dir_and_writes_products_there(prep, tmp_path, monkeypatch):
    work = write_corpus(tmp_path / 'work')
    monkeypatch.setattr(prep, 'DATA_DIR', str(work))
    run_main(prep, ['--out-prefix', 'd'])

    assert (work / 'train_char_d.bin').exists()
    assert (work / 'val_char_d.bin').exists()
    assert (work / 'meta_char_d.pkl').exists()
    man = json.loads((work / 'manifest_d.json').read_text(encoding='utf-8'))
    assert sorted(s['file'] for s in man['source_breakdown']) == ['a.txt', 'b.txt']
    assert sorted(s['file'] for s in man['source_files']) == ['a.txt', 'b.txt']
    assert man['prepare_args']['source_dir'] == str(work)


# ==========================================================================
# 3) --source-dir：只扫指定目录，产物还是写 DATA_DIR
# ==========================================================================
def test_source_dir_scans_only_that_dir_and_writes_products_to_data_dir(
        prep, tmp_path, monkeypatch):
    src = write_corpus(tmp_path / 'corpus')
    out = tmp_path / 'products'
    out.mkdir()
    (out / 'should_be_ignored.txt').write_text('用户：x\n模型：y', encoding='utf-8')
    monkeypatch.setattr(prep, 'DATA_DIR', str(out))

    run_main(prep, ['--source-dir', str(src), '--out-prefix', 's'])

    # 产物落在 DATA_DIR（=out），不在 src
    assert (out / 'train_char_s.bin').exists()
    assert not (src / 'train_char_s.bin').exists()
    man = json.loads((out / 'manifest_s.json').read_text(encoding='utf-8'))
    assert sorted(s['file'] for s in man['source_breakdown']) == ['a.txt', 'b.txt']
    assert sorted(s['file'] for s in man['source_files']) == ['a.txt', 'b.txt']
    assert 'should_be_ignored.txt' not in [s['file'] for s in man['source_files']]
    assert man['prepare_args']['source_dir'] == str(src)


def test_source_dir_is_used_for_pretrain_mode_too(prep, tmp_path, monkeypatch):
    """`--pretrain` 分支同样改用 source_dir（否则新参数在一条支线上是死旋钮）。"""
    src = write_corpus(tmp_path / 'corpus')
    out = tmp_path / 'products'
    out.mkdir()
    (out / 'ignored.txt').write_text('不该被读到的内容', encoding='utf-8')
    monkeypatch.setattr(prep, 'DATA_DIR', str(out))
    run_main(prep, ['--source-dir', str(src), '--pretrain', '--out-prefix', 'p'])
    man = json.loads((out / 'manifest_p.json').read_text(encoding='utf-8'))
    assert sorted(s['file'] for s in man['source_files']) == ['a.txt', 'b.txt']


# ==========================================================================
# 4) ★ 默认 == 显式指定 DATA_DIR（同一份语料必须产出逐字节相同的 bin）
# ==========================================================================
def test_default_and_explicit_source_dir_are_byte_identical(prep, tmp_path, monkeypatch):
    """旧命令（不传 --source-dir）与显式 `--source-dir <DATA_DIR>` 必须等价 ——
    这是"默认行为不变"能拿出的最硬证据（不是看代码，是比字节）。"""
    work = write_corpus(tmp_path / 'work')
    out = tmp_path / 'products'
    out.mkdir()

    monkeypatch.setattr(prep, 'DATA_DIR', str(work))
    run_main(prep, ['--out-prefix', 'p1'])
    monkeypatch.setattr(prep, 'DATA_DIR', str(out))
    run_main(prep, ['--source-dir', str(work), '--out-prefix', 'p2'])

    for name in ('train_char', 'val_char'):
        a = (work / f'{name}_p1.bin').read_bytes()
        b = (out / f'{name}_p2.bin').read_bytes()
        assert a == b, f'{name}: 默认扫描与显式 --source-dir 产物不一致'
    m1 = json.loads((work / 'manifest_p1.json').read_text(encoding='utf-8'))
    m2 = json.loads((out / 'manifest_p2.json').read_text(encoding='utf-8'))
    assert [s['file'] for s in m1['source_files']] == [s['file'] for s in m2['source_files']]
    assert m2['prepare_args']['source_dir'] == str(work)
    assert m1['prepare_args']['source_dir'] == str(work)
