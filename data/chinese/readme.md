# 中文语料与对话数据集 (data/chinese)

本目录包含 nanoSeek 中文语言建模与对话微调所用的数据下载、分词器训练和二进制数据生成脚本。

> **注意**：所有生成的文本与二进制数据文件（`*.txt`, `*.bin`, `*.json`, `*.pkl` 等）均已被 `.gitignore` 忽略，不提交到代码仓库。请按照下方流程在本地生成。

---

## 1. 数据来源地址

### 对话语料（来自 [魔搭社区 ModelScope](https://modelscope.cn/)）

通过 `download_dialogue.py` 自动流式下载与清洗：

1. **shareAI-Llama3-DPO-zh-en-emoji**
   - 来源：`shareai/shareAI-Llama3-DPO-zh-en-emoji`
   - 链接：[https://modelscope.cn/datasets/shareai/shareAI-Llama3-DPO-zh-en-emoji](https://modelscope.cn/datasets/shareai/shareAI-Llama3-DPO-zh-en-emoji)
   - 描述：中文单轮高质量问答（QA），包含日常、科普、emoji 等内容。

2. **Llama3-Chinese-Dataset**
   - 来源：`zhuangxialie/Llama3-Chinese-Dataset`
   - 链接：[https://modelscope.cn/datasets/zhuangxialie/Llama3-Chinese-Dataset](https://modelscope.cn/datasets/zhuangxialie/Llama3-Chinese-Dataset)
   - 描述：中文多轮对话数据集，涵盖广泛的问答与通用任务。

3. **Muice-Dataset**
   - 来源：`Moemuu/Muice-Dataset`
   - 链接：[https://modelscope.cn/datasets/Moemuu/Muice-Dataset](https://modelscope.cn/datasets/Moemuu/Muice-Dataset)
   - 描述：中文多轮闲聊语料，具有丰富的上下文连贯性。

4. **dailychat**
   - 来源：`yyy6778/dailychat`
   - 链接：[https://modelscope.cn/datasets/yyy6778/dailychat](https://modelscope.cn/datasets/yyy6778/dailychat)
   - 描述：中文单轮日常口语化对话语料。

5. **Multi-turn-dialogue**
   - 来源：`justgo10000/Multi-turn-dialogue`
   - 链接：[https://modelscope.cn/datasets/justgo10000/Multi-turn-dialogue](https://modelscope.cn/datasets/justgo10000/Multi-turn-dialogue)
   - 描述：多轮对话数据集，含角色与 prompt/chosen 结构。

6. **Zhihu-KOL**
   - 来源：`OmniData/Zhihu-KOL`
   - 链接：[https://modelscope.cn/datasets/OmniData/Zhihu-KOL](https://modelscope.cn/datasets/OmniData/Zhihu-KOL)
   - 描述：知乎高质量问答生活问答精选。

### 经典文学语料（四大名著）

通过 `prepare.py --with-books` 自动下载：

- 来源仓库：[tennessine/corpus](https://github.com/tennessine/corpus)
- 国内镜像：`https://cdn.jsdelivr.net/gh/tennessine/corpus@master/{enc}.txt`
- 包含书目：《西游记》、《红楼梦》、《三国演义》、《水浒传》

---

## 2. 数据准备流程

从项目根目录执行以下步骤：

### 步骤 1：下载语料
```bash
# 从 ModelScope 下载并清洗中文对话语料为 *.txt
uv run python data/chinese/download_dialogue.py
```

### 步骤 2：训练 BPE 分词器
```bash
# 使用已下载语料训练 BPE 分词器，输出 data/chinese/tokenizer.json（默认 8000 词表）
uv run python data/chinese/train_tokenizer.py
```

### 步骤 3：数据编码（生成二进制数据）

#### 选项 A：对话 / SFT 微调模式（默认）
```bash
# 将语料编码为 token ids，生成 train.bin / val.bin / meta.pkl / manifest.json
# 默认启用 turn-level <eos> 插入
uv run python data/chinese/prepare.py
```

#### 选项 B：预训练模式（两阶段训练 Stage 1）
```bash
# 预训练无掩码语料生成：按原始文本 90/10 切分，生成 pretrain.bin / val.bin
uv run python data/chinese/prepare.py --pretrain
```
