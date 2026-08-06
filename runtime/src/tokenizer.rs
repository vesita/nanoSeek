//! 字符级分词器：从 vocab.json 加载 stoi/itos 映射（与 PyTorch 端 meta.pkl 对应）。
use std::collections::HashMap;

use anyhow::Result;

pub struct CharTokenizer {
    stoi: HashMap<String, u32>,
    itos: HashMap<u32, String>,
}

impl CharTokenizer {
    /// 从 vocab.json 加载（{"stoi": {char: id}, "itos": {id_str: char}}）。
    pub fn load(path: &str) -> Result<Self> {
        let raw: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path)?)?;
        let mut stoi = HashMap::new();
        for (ch, id) in raw["stoi"].as_object().unwrap() {
            stoi.insert(ch.clone(), id.as_u64().unwrap() as u32);
        }
        let mut itos = HashMap::new();
        for (id, ch) in raw["itos"].as_object().unwrap() {
            itos.insert(id.parse::<u32>()?, ch.as_str().unwrap().to_string());
        }
        Ok(Self { stoi, itos })
    }

    /// 字符串 → token id 列表。遇到词表外的字符退回 id 0（和训练时的行为一致）。
    pub fn encode(&self, s: &str) -> Vec<u32> {
        s.chars()
            .map(|c| self.stoi.get(&c.to_string()).copied().unwrap_or(0))
            .collect()
    }

    /// token id 列表 → 字符串。未知 id 跳过。
    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter_map(|i| self.itos.get(i))
            .flat_map(|s| s.chars())
            .collect()
    }
}
