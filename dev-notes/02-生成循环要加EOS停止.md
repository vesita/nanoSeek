# 02 · 生成循环要加 EOS 停止

## 现象
输入对话，模型一直输出换行/垃圾，永远不自己停。

## 根因
`generate_stream` 的循环是 `for _ in 0..n` **无脑生成满 `max_new_tokens` 个 token**，
完全没有检查 `<eos>` token。模型就算预测了结束符，也不会停下来。

## 修复
从 tokenizer 取 `<eos>` 的 id，采样后立刻检查，命中就 break（eos 不输出、不进上下文）：

```rust
// tokenizer.rs
pub fn eos_id(&self) -> Option<u32> {
    self.tok.token_to_id("<eos>")
}

// model.rs generate_stream 循环里
let next = sample(&logits, temperature, top_k, rng)?;
if Some(next) == eos_id {
    break; // <eos>：立即停止
}
on_token(next);
tokens.push(next);
new_tokens.push(next);
```

## 怎么避免
**任何自回归生成循环都必须有停止条件**：要么 EOS，要么长度上限，二者至少一个。
RLHF / 对话模型尤其要检查 eos —— 训练数据里 `<eos>` 就是为"这句话说完了"设计的，
生成端不尊重它就是又聋又瞎。
