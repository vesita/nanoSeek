# 数据来源与许可清单（Data Provenance & Licenses）

> 本文档记录 nanoSeek 项目训练/验证所使用的**全部数据来源、许可证、用途与处理方式**。
> 训练数据与第三方代码的许可合规是开源项目的底线，请在任何数据增删后同步更新本文档。
> 最后更新：2026-09-06

---

## 1. 总览

当前字符级训练集（`data/chinese/train_char.bin`）：**197.2M tokens**（988,948 → 1,212,308 样本）
字符级词表：`char_tokenizer.json`（四区稀疏分区 v3，8192 = 2^13，`<eos>`=128）

数据按来源分三类：

| 类别 | 说明 | 许可风险 |
| :--- | :--- | :--- |
| **A. 项目自研合成数据** | 数学进位链 CoT / 代码算法 / 工具调用 / 百科问答 | 无（MIT，项目自有） |
| **B. 开源许可数据** | 来自 GitHub/ModelScope，许可证明确 | 低（MIT / CC BY-SA 等） |
| **C. 需查证/版权敏感数据** | 早期抓取的对话/知乎语料 | 高（研究用途，商用前需逐项核查） |

---

## 2. A 类：项目自研合成数据（无第三方版权，MIT）

| 文件 | 生成脚本 | 内容 | 样本量 | 许可证 |
| :--- | :--- | :--- | :--- | :--- |
| `math_cot_dialogue.txt` | `data/scripts/synthetic_generator.py` | 四则运算/代数/几何 CoT | 60,000 | MIT（项目自有） |
| `math_cot_v2_dialogue.txt` | `data/scripts/gen_math_cot_v2.py` | **逐位进位链 CoT**（对齐 nano_arith 范式） | 80,000 | MIT（项目自有） |
| `code_syntax_dialogue.txt` | `data/scripts/generate_multisource_data.py` | Python 基础语法模板 | 8,000 | MIT（项目自有） |
| `code_algo_v2_dialogue.txt` | `data/scripts/gen_code_v2.py` | 10 大类真实算法 + 复杂度分析 | 40,000 | MIT（项目自有） |
| `agent_tools_dialogue.txt` | `data/scripts/synthetic_generator.py` | 天气/沙盒/检索工具调用闭环 | 30,000 | MIT（项目自有） |
| `baike_qa_dialogue.txt` | `data/scripts/synthetic_generator.py` | 百科通识问答 | 15,000 | MIT（项目自有） |

---

## 3. B 类：开源许可数据（许可证明确）

### 3.1 代码与算法

| 来源 | 仓库/数据集 | 许可证 | 用途 | 样本量 |
| :--- | :--- | :--- | :--- | :--- |
| TheAlgorithms | [`TheAlgorithms/Python`](https://github.com/TheAlgorithms/Python) | **MIT** | 代码语法/算法（待接入） | ~2,000+ 文件 |
| doocs | [`doocs/leetcode`](https://github.com/doocs/leetcode) | **CC BY-SA 4.0** | 中文题解 + 代码（待接入） | 3.6 万文件 |
| codefuse-ai | [`CodeExercise-Python-27k`](https://modelscope.cn/datasets/codefuse-ai/CodeExercise-Python-27k) | **CC BY-NC-SA 4.0 ⚠️ 非商用** | Python 代码 | 14,909 |
| OpenAI | [`GSM8K`](https://github.com/openai/grade-school-math) | **MIT** | 数学应用题 CoT | 7,473 |

> ⚠️ **CodeExercise-Python-27k 是 CC BY-NC-SA 4.0（非商用）**：
> 若 nanoSeek 未来商用，必须**剔除** `code_alpaca_dialogue.txt` 对应数据并重新编码。

### 3.2 中文文本

| 来源 | 仓库/数据集 | 许可证 | 用途 | 样本量 |
| :--- | :--- | :--- | :--- | :--- |
| chinese-poetry | [`chinese-poetry/chinese-poetry`](https://github.com/chinese-poetry/chinese-poetry) | **MIT** | 古诗词/文化语料（待接入） | ~30 万首 |
| m-a-p | [`COIG-CQIA`](https://modelscope.cn/datasets/m-a-p/COIG-CQIA) | **需查证**（README 未明确） | 知乎/百科/逻辑/代码问答 | 44,031 |

> ⚠️ **COIG-CQIA 的许可证在仓库 README 中标注为 "More Information Needed"**：
> 其知乎子集含真实知乎回答（`metadata` 含 `qid/aid`），**版权归属复杂**，
> 商用前需逐子集核查。当前仅用于研究/学习。

---

## 4. C 类：需查证 / 版权敏感数据（早期抓取）

> 以下数据通过 `data/chinese/download_*.py` 在早期阶段抓取，**来源许可未完全核实**。
> 它们对模型的**口语语感**贡献巨大，但**商用前必须逐项核查或替换为干净替代源**。

| 文件 | 来源 | 疑似许可证 | 用途 | 风险 |
| :--- | :--- | :--- | :--- | :--- |
| `lccc_dialogue.txt` | [`silver/lccc`](https://huggingface.co/datasets/silver/lccc)（微博对话） | 需查证（研究用途） | 日常闲聊 | 中 |
| `kdconv_dialogue.txt` | [`thu-coai/KdConv`](https://huggingface.co/datasets/thu-coai/KdConv)（清华） | 需查证（研究用途） | 知识对话 | 中 |
| `zhihu_kol_dialogue.txt` | 知乎问答精选 | 需查证（知乎版权） | 长文本问答 | **高** |
| `multi_turn_dialogue.txt` | 心理倾听多轮对话 | 需查证 | 共情对话 | 中 |
| `glm_dialogue.txt` | GLM 对话 | 需查证 | 通用多轮 | 中 |
| `muice_dialogue.txt` | 二次元角色对话 | 需查证 | 闲聊 | 低 |
| `dailychat_dialogue.txt` | 日常口语问答 | 需查证 | 口语问答 | 低 |
| `identity_dialogue.txt` | 项目自研（nanoSeek 人设） | MIT（项目自有） | 自我认知 | 无 |

---

## 5. 处理方式与合规约定

1. **统一格式**：所有文本转成「用户：/模型：」对话或纯文本，空行分隔，经 `prepare.py --char-level` 编码；
2. **溯源**：`data/chinese/manifest.json` 记录每个源文件的 SHA-256 哈希、大小与时间戳，训练 checkpoint 关联该哈希，实现「模型 → 数据版本」可追溯；
3. **商用前必做**：
   - 剔除 CC BY-NC 数据（`code_alpaca_dialogue.txt`）；
   - 逐项核查 C 类数据，替换为可商用替代源（如 WuDao、BELLE 商用子集）；
   - 重新跑 `prepare.py` 并更新 manifest 哈希。

---

## 6. 项目许可证

- 项目代码：**MIT License**（继承自 [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) 基础，详见根目录 `LICENSE`）；
- 第三方代码/数据的独立许可证见根目录 `THIRD_PARTY_NOTICES.md`。
