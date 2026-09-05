# 第三方软件与数据许可声明（Third-Party Notices）

nanoSeek 项目使用了以下第三方开源代码与数据。本文件记录各方的版权归属与许可证，
以履行相应许可证（MIT / CC BY-SA / CC BY-NC-SA 等）的**署名与声明义务**。

---

## 1. 代码基础

### nanoGPT
- **来源**：https://github.com/karpathy/nanoGPT
- **作者**：Andrej Karpathy
- **许可证**：MIT License
- **说明**：nanoSeek 的模型主体（`model/`）、训练循环（`training/train.py`）等
  架构骨架源自 nanoGPT，并在此之上进行了大量扩展（MoE / CSA / mHC / MLA / Muon 等）。
- **版权声明**：

```
MIT License
Copyright (c) 2022 Andrej Karpathy
```

---

## 2. 第三方代码仓库（已克隆至 `data/external/`）

### TheAlgorithms/Python
- **来源**：https://github.com/TheAlgorithms/Python
- **许可证**：MIT License
- **版权**：Copyright (c) 2016-2022 TheAlgorithms and contributors
- **用途**：代码算法语料（补充模型的代码语法与算法能力）

### doocs/leetcode
- **来源**：https://github.com/doocs/leetcode
- **许可证**：Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
- **用途**：LeetCode 题解与代码语料
- **义务**：署名（doocs） + 相同方式共享（衍生数据需以 CC BY-SA 4.0 分发）

### chinese-poetry
- **来源**：https://github.com/chinese-poetry/chinese-poetry
- **许可证**：MIT License
- **版权**：Copyright (c) 2016 JackeyGao
- **用途**：古诗词与文化语料

---

## 3. 第三方数据集

### GSM8K（数学应用题）
- **来源**：https://github.com/openai/grade-school-math
- **作者**：OpenAI（Cobbe et al., 2021）
- **许可证**：MIT License
- **用途**：数学应用题思维链（CoT）训练

### CodeExercise-Python-27k
- **来源**：https://modelscope.cn/datasets/codefuse-ai/CodeExercise-Python-27k
- **版权**：Copyright 2023 Ant Group
- **许可证**：**CC BY-NC-SA 4.0（非商用）**
- **用途**：Python 代码语料
- **⚠️ 义务**：仅限非商业用途；衍生使用需署名 + 相同方式共享。商用前必须剔除。

### COIG-CQIA
- **来源**：https://modelscope.cn/datasets/m-a-p/COIG-CQIA
- **作者**：m-a-p（BAAI）
- **许可证**：仓库 README 标注 "More Information Needed"（**尚未明确**）
- **用途**：知乎/百科/逻辑/代码高质量问答
- **⚠️ 义务**：含真实知乎回答，版权归属复杂，商用前需逐子集核查许可。

### LCCC（微博对话）
- **来源**：https://huggingface.co/datasets/silver/lccc
- **许可证**：需查证（研究用途）
- **用途**：日常闲聊语料

### KdConv（知识驱动对话）
- **来源**：https://huggingface.co/datasets/thu-coai/KdConv
- **作者**：清华大学（THU-COAI）
- **许可证**：需查证（研究用途）
- **用途**：知识驱动多轮对话

---

## 4. 合成数据（无第三方版权）

以下数据由 nanoSeek 项目自行生成，无第三方版权，随项目以 MIT 分发：

- 数学进位链 CoT（`gen_math_cot_v2.py`、`synthetic_generator.py`）
- 代码算法模板（`gen_code_v2.py`、`generate_multisource_data.py`）
- 工具调用闭环（`synthetic_generator.py`）
- 百科通识问答（`synthetic_generator.py`）

---

## 5. 合规提示

1. **研究/学习用途**：当前项目定位为学习与实验，C 类数据（微博/知乎/对话）暂可保留；
2. **商用发布前**：
   - 删除 CC BY-NC-SA 数据（CodeExercise）；
   - 逐项核查 C 类数据许可，或替换为可商用替代源；
   - 保留本文档与 `DATA_SOURCES.md` 作为合规留痕。
3. **衍生数据分发**：若分发基于 doocs/leetcode 的衍生语料，需遵循 CC BY-SA 4.0 的相同方式共享条款。
