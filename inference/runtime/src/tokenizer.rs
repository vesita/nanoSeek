//! BPE 分词器：用 tokenizers crate 加载 tokenizer.json。
//! 这个文件由 Python 端 data/chinese/train_tokenizer.py 训出、convert.py 复制过来，
//! Python（tokenizers 库）和 Rust（tokenizers crate）共用同一份，编解码完全一致。
use anyhow::Result;

pub struct BpeTokenizer {
    tok: tokenizers::Tokenizer,
}

impl BpeTokenizer {
    /// 从 tokenizer.json 加载完整分词管线（BPE 模型 + ByteLevel 预分词）。
    /// tokenizers crate 的错误是 Box<dyn Error>，`?` 转不了 anyhow，用 map_err 包装。
    pub fn load(path: &str) -> Result<Self> {
        let tok = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("加载 tokenizer.json 失败: {e}"))?;
        Ok(Self { tok })
    }

    /// 字符串 → token id 列表（add_special_tokens=false，和训练编码一致）。
    pub fn encode(&self, s: &str) -> Result<Vec<u32>> {
        self.tok
            .encode(s, false)
            .map(|enc| enc.get_ids().to_vec())
            .map_err(|e| anyhow::anyhow!("BPE 编码失败: {e}"))
    }

    /// token id 列表 → 字符串（非流式一次性解码）。
    /// 流式输出请用 stream() + StreamingDecoder。
    #[allow(dead_code)]
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tok
            .decode(ids, false)
            .map_err(|e| anyhow::anyhow!("BPE 解码失败: {e}"))
    }

    /// `<eos>` token 的 id（train_tokenizer.py 里特殊 token 固定为 0~3：pad/unk/bos/eos）。
    /// 生成时遇到 eos 应立即停止。返回 None 表示这份 tokenizer.json 里没有 <eos>。
    pub fn eos_id(&self) -> Option<u32> {
        self.tok.token_to_id("<eos>")
    }

    /// 创建流式解码器（逐 token 输出，处理跨 token 的多字节字符）。
    pub fn stream(&self) -> StreamingDecoder {
        StreamingDecoder::new(self.tok.clone())
    }
}

/// 流式解码器：逐 token 喂入，输出"确认完整"的文本片段。
///
/// ByteLevel 可能把一个多字节汉字拆成多个 byte token，跨 token 单独解码会
/// 得到半个字符（tokenizers 产出替换符 U+FFFD）。这里维护一个尾巴缓冲，
/// 等字符拼完整再输出，避免逐 token 打印出现乱码。
pub struct StreamingDecoder {
    tok: tokenizers::Tokenizer,
    tail: Vec<u32>, // 尚未确认完整的尾巴 token
}

impl StreamingDecoder {
    fn new(tok: tokenizers::Tokenizer) -> Self {
        Self { tok, tail: Vec::new() }
    }

    /// 喂入一个新 token，返回可以输出的新文本（None = 还在缓冲等字符拼完整）。
    pub fn push(&mut self, id: u32) -> Result<Option<String>> {
        self.tail.push(id);
        let s = self
            .tok
            .decode(&self.tail, false)
            .map_err(|e| anyhow::anyhow!("流式解码失败: {e}"))?;
        // 出现替换字符说明尾巴里有不完整的多字节字符，继续等；
        // 留个兜底：尾巴过长也强制输出，避免遇到真实的 U+FFFD 卡死。
        if s.contains('\u{FFFD}') && self.tail.len() < 32 {
            return Ok(None);
        }
        self.tail.clear();
        Ok(Some(s))
    }

    /// 结束：flush 剩余缓冲（若有不完整字符，交给 tokenizers 容错处理）。
    pub fn finish(&mut self) -> Result<String> {
        let s = self
            .tok
            .decode(&self.tail, false)
            .map_err(|e| anyhow::anyhow!("流式解码失败: {e}"))?;
        self.tail.clear();
        Ok(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Python/Rust 分词器一致性：这些 id 由 Python 端 tokenizers 库（同版本 0.22）
    /// 对同一 tokenizer.json 编码得到，Rust 端必须逐位一致。
    /// 需要先跑 convert.py 把 tokenizer.json 复制到 runtime 目录（数据产物，git 忽略）。
    #[test]
    fn parity_with_python() {
        let path = "tokenizer.json";
        if !std::path::Path::new(path).exists() {
            eprintln!("跳过：缺 {path}（先跑 convert.py 复制 tokenizer.json）");
            return;
        }
        let tok = BpeTokenizer::load(path).expect("tokenizer 加载失败");
        assert_eq!(tok.encode("你好世界").unwrap(), vec![4285, 8957, 27833]);
        assert_eq!(tok.encode("悟空").unwrap(), vec![24060, 210, 20474]);
        assert_eq!(tok.decode(&[4285, 8957, 27833]).unwrap(), "你好世界");
    }

    /// 流式解码：逐 token 喂入，最终还原必须和一次性解码一致。
    #[test]
    fn streaming_roundtrip() {
        let path = "tokenizer.json";
        if !std::path::Path::new(path).exists() {
            eprintln!("跳过：缺 {path}（先跑 convert.py 复制 tokenizer.json）");
            return;
        }
        let tok = BpeTokenizer::load(path).expect("tokenizer 加载失败");
        let text = "用户：你好，请问有什么可以帮您？\n模型：我想查看订单状态。";
        let ids = tok.encode(text).unwrap();
        let mut dec = tok.stream();
        let mut out = String::new();
        for &id in &ids {
            if let Some(chunk) = dec.push(id).unwrap() {
                out.push_str(&chunk);
            }
        }
        out.push_str(&dec.finish().unwrap());
        assert_eq!(out, text, "流式还原失败");
    }
}
