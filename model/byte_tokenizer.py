"""字节直入 tokenizer（dev-notes/48，Mamba-Byte 思想）：无 BPE 分词。

- 词表 257：0-255 = UTF-8 字节，256 = <eos>（turn-level 终止符）
- encode(text) → 对象带 .ids（兼容 tokenizers 接口：tok.encode(x).ids）
- decode(ids) → str（UTF-8 容错；字节可能跨 token 切半，errors='replace' 自愈）
- 无 OOV、无分词错误、话术片段不会像 BPE 那样固化进词表
"""
from types import SimpleNamespace

EOS_ID = 256
VOCAB_SIZE = 257


class ByteTokenizer:
    """极简字节编解码，接口对齐 tokenizers.Tokenizer 的调用面。"""

    vocab_size = VOCAB_SIZE

    def encode(self, text, add_special_tokens=True):
        ids = []
        parts = text.split("<eos>")
        for i, part in enumerate(parts):
            ids.extend(part.encode("utf-8"))          # 0-255
            if i < len(parts) - 1:
                ids.append(EOS_ID)
        return SimpleNamespace(ids=ids)

    def decode(self, ids, skip_special_tokens=True, errors="replace"):
        raw = bytes(t for t in ids if 0 <= t < 256)
        return raw.decode("utf-8", errors=errors)

    def token_to_id(self, tok):
        return EOS_ID if tok == "<eos>" else None
