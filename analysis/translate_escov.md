# ESConv 英文共情对话 → 中文 翻译管线报告

> 任务：把 `adafny123/visual_noval_atri` 的 `fine-tune_dst.json`（1300 段英文多轮共情对话）
> **逐条消息**翻成中文，产出 `escov_zh.json` / `escov_zh.txt`，并**证明管线是对的**。
> 证据标注：**【实测】** = 本机跑出来的数；**【推断】** = 由实测外推。
> 全流程脚本：`scripts/translate_escov.py`（bench / canary / run / repass / assemble / validate 六个子命令）。

---

## 0. 一句话结论

用 **Helsinki-NLP/opus-mt-en-zh**（MarianMT，torch CPU）逐条消息翻译，配合**三处必须的修正**
（transformers 钉 4.46.3、`no_repeat_ngram_size=4`、CPU 只用 4 线程）后：

- **全量 38365 条消息 109.75 分钟（1 h 50 min）跑完**，5.83 msg/s；
- **结构 1300/1300 完全对齐，0 条违反**，`.txt` 1300 个 block 结构零错；
- **4 个负向对照全部报错**（校验器不是橡皮图章）；
- **中文串扰对照 3/3 被破坏** ⇒ 该管线**只能用于纯英文语料**。

ModelScope 的 `damo/nlp_csanmt_translation_en2zh`（候选 A）**译质明显更高，但实测慢 11 倍
（ETA 15.9 h）**，按 2 h 时间门槛弃用。

---

## 1. 输入与产物

| 项 | 值 |
|---|---|
| 输入 | `/home/vesita/datasets/NLP/escov_en_1300_multiturn.json`（父 agent 搬迁后路径）|
| 规模 | 1300 段 / **38365 条消息** / 3192606 字符（均值 83.2 字符/条，最长 912）|
| 角色分布 | user 19989 / assistant 18376 |
| 消息数/段 | 中位 27，最少 16，最多 120 |
| 原始语言 | 100% 英文（仅 221 条含非 ASCII，是 `’` 这类弯引号）|
| 产物 1 | `data/chinese/new_dialogue/escov_zh.json`（4590033 B）—— 与输入**逐条对齐**，仅 `content` 换中文 |
| 产物 2 | `data/chinese/new_dialogue/escov_zh.txt`（2969068 B）—— 项目语料格式，块内 `用户：`/`模型：`，块间空行 |
| 产物 3 | 本报告 |

输入**无内嵌换行、无空 content、无首尾空白**【实测】，因此按行写 `.txt` 不会破格式；
产出后又用 `validate` 复核了这一点（见 §5）。

---

## 2. 环境（**严格不碰项目 `.venv`**）

硬约束要求训练环境唯一且不被污染，因此**另建独立 venv**：

| | 项目训练 `.venv` | 翻译 `~/.venvs/mt` |
|---|---|---|
| Python | 项目自带 | **3.12.14**（`uv venv --python 3.12` 拉独立 CPython；系统只有 3.14，torch 无 3.14 轮子）|
| 关键包 | torch 2.9.1+rocm6.4 / modelscope 1.39.1 / pyarrow / pandas | torch **2.14.0+cpu** / transformers **4.46.3** / sentencepiece / sacremoses / modelscope / tensorflow-cpu 2.21.0 |
| 本次是否 `pip install` | **没有，一个包都没装**【实测】| 全部装在这里 |
| 安装器 | — | `uv pip install --python ~/.venvs/mt/bin/python …`（显式指定目标解释器，避免误落到项目 venv）|

计算设备：**CPU**（12 核，torch 限 4 线程）。GPU（gfx1030）本次未用于翻译：
torch 装的是 CPU 轮子；候选 A 是 TensorFlow 模型，而 **gfx1030 没有可用的 ROCm-TF**。

---

## 3. 模型选型：A vs B

### 3.1 候选 A —— `damo/nlp_csanmt_translation_en2zh`（**弃用：太慢**）

- 结构：CSANMT，hidden 1024 / **encoder 24 层** / decoder 6 层 / vocab 50000 / beam 4（读其 `configuration.json`）【实测】
- **单 checkpoint 7.88 GB**，约 13 min 下完 @10 MB/s【实测】
- modelscope 1.39.1 在这个干净 venv 里**依赖是残缺的**：依次缺 `addict`、`datasets`、`PIL`、
  `simplejson`、`jieba`、`subword_nmt`。它们都在 modelscope 的 `nlp`/`framework` extra 里，
  而**装全量 `nlp` extra 会把 protobuf 钉到 `<3.21.0`、直接弄坏 TF 2.21** ⇒ 只能逐个补最小依赖集【实测】
- **译质明显好于 B**，短句尤其稳（12 条分层抽样）：

| 英文 | A 输出 | B 输出（对照）|
|---|---|---|
| `Hello` | **你好** | 你好（经复读折叠）|
| `I could try. It mostly gets to me at the end of the day` | 我能理解. | 我可以尝试,大部分都是在一天结束时 |
| `Probably not. I was with the same company for a long time and I consistently...` | 可能不会。我在同一家公司工作了很长时间，并且每年都会获得奖金 | 可能不会吧,我和同一家公司 在一起很久了,我每年都有奖金 |

- **速度（决定性）**：load 13.9 s，逐条推理 **均值 1.50 s/条 → 0.67 msg/s → 全量 ETA 15.9 h**【实测】
  （分档：5 字符 1.36 s；41 字符 0.92 s；93 字符 1.68 s；133 字符 2.04 s；132 字符 2.76 s）

> **弃用理由**：比 B 慢 **11 倍**（15.9 h vs 1.84 h），远超 2 h 门槛。
> 另注：其 config 写 `'device': 'cuda'`，但本机只能跑 TF-CPU。

### 3.2 候选 B —— `Helsinki-NLP/opus-mt-en-zh`（**选用**）

MarianMT（约 77M 参数），transformers + sentencepiece + sacremoses，**torch CPU**。

**三个必须的修正，全部实测得出（本报告最有价值的部分）：**

| # | 坑 | 现象（实测）| 修正 |
|---|---|---|---|
| 1 | **transformers 5.x 退化成不复读机** | 5.17.0 下生成**永不吐 EOS**：`Hello` → 200 个 `哈`；`How are you?` → `你好 你好 你好…`（跑到 max_new_tokens 才停）| **钉 `transformers==4.46.3`**（tokenizers 随之降到 0.20.3）。换回后同一输入正常 |
| 2 | **必须 `no_repeat_ngram_size=4`** | 不加时短输入循环到上限；加上后 13 条测试里 `>80 字符` 的条数 **2 → 0**，同时耗时 17.2 s → **1.2 s（快 14 倍）** | generate 时传 `no_repeat_ngram_size=4` |
| 3 | **CPU 线程数不是越多越好** | 同一批输入：4 线程 3.91 s，8 线程 4.39 s，**12 线程 15.19 s（慢 4 倍）** | `torch.set_num_threads(4)` |

最终解码参数：`num_beams=4, no_repeat_ngram_size=4, do_sample=False, batch=32, threads=4`，
`max_new_tokens = min(512, max(64, 3×本批最长源 token 数))`，并设 `USE_TF=0` 防止
transformers 把 TensorFlow（只为候选 A 装的）拖进 Marian 路径。

**复读折叠后处理**（`collapse_degeneration`）：B 对**无标点超短句**仍会产出重复串，
用一组**保守**规则折叠到不动点（整串周期重复 / 首部立即重复且尾巴纯拉丁 / 所有空白 token 相同 / 单 token 占绝对多数）。
保守性设计：`不是不是，我是说…` 这类**合法强调不会被动**（要求整串是重复、或残留尾巴是纯拉丁/标点）。
效果：裸 `Hello` 的原始输出 `你好 你好 你好 你好` → 折叠成 **`你好`**（已知答案对照因此 PASS）。

### 3.3 选型结论

| | A（CSANMT）| B（opus-mt-en-zh）|
|---|---|---|
| 译质 | **更高**（短句明显更稳）| 够用，长句偶有漏译 |
| 速度 | 0.67 msg/s（ETA 15.9 h）| **5.83 msg/s（实测 109.75 min）** |
| 依赖 | TF + 7.88 GB ckpt + 残缺依赖 | torch CPU + ~300 MB |
| 采用 | ✗（超时）| **✓** |

---

## 4. 速度与 ETA

### 4.1 任务要求的「先跑 50 段测速」

```
50 段 → 1551 条消息，271.7 s  →  5.71 msg/s  →  全量 ETA 112.0 min (1.87 h)   【实测】
输出长度：均值 22.5 字符 / 中位 16 / 最大 249 / 打到上限的条数 0
复读折叠触发 30/1551
```

### 4.2 ★ 探针同构核查（否则 ETA 会算错）

前 50 段**比全库短**：均值 76.0 字符/条 vs 全库 83.2（比值 **0.91**），四分位 33/58/100 vs 33/66/111。
⇒ 直接外推会**低估** ETA，按字符比校正约 **123 min ≈ 2.05 h**。
因此额外做了一次**分层抽样**（每隔 13 段取 1 段，14 段 / 407 条，均值 80.6 字符）重测【实测】。

### 4.3 解码配置扫描（分层样本，模型只加载一次）

| 配置 | 速度 | 全量 ETA | 与 beams=4 的逐条完全一致率 |
|---|---|---|---|
| `beams=1` | **26.03 msg/s** | 24.6 min | 27.3% |
| `beams=2` | 12.26 msg/s | 52.1 min | 49.9% |
| `beams=4` ← 采用 | 6.92 msg/s | **92.4 min** | — |
| `batch=64`（beams=4）| 4.90 msg/s | 130.4 min | 与 batch=32（4.98）**无增益** |

**选 beams=4 而不是 beams=1**：逐条对照显示 beam=4 明显更好——
`是什么让你的工作压力很大?`（b4）vs `你的工作是什么让你压力大?`（b1）；
b1 还会**丢主语/丢从句**（`帮助客户改善财政状况?` 丢掉 "Do you"；`近乎你目前的工作吗?` 把 salary 译成「工作」）。
beams=4 的 ETA 在门槛内，故不拿质量换速度。

### 4.4 全量实跑

| 项 | 值 |
|---|---|
| 启动方式 | `systemd-run --user --unit=escov-translate`（项目铁律 0：长跑走 systemd 用户单元，会话重启不会杀掉）|
| 起止 | **13:30:53 → 15:20:38**，wall **109.75 min（1 h 49 m 45 s）** |
| 吞吐 | **38365 条 / 109.75 min = 5.83 msg/s** |
| 退出码 | 0 |
| 断点续跑 | 每条消息一行 `{"i":…, "zh":…}` 的 jsonl，每批 `flush + fsync`；重启时按「最长连续已完成前缀」对齐到**批边界**后续跑（批边界固定 ⇒ 续跑结果与一次跑完逐字一致）|

ETA 预测 92~123 min，**实际 109.75 min**，落在预测区间内（样本长度差是主因）。

---

## 5. 结构校验 + 负向对照

校验函数 `check_dialogue(src, zh)` 逐条断言：**消息数相同 ∧ role 序列逐位相同 ∧ 无空 content**。

**正向结果（最终产物上跑）**【实测】：

```
[validate] dialogues=1300 ok=1300 violating=0
[validate] txt blocks=1300 (expect 1300)
[validate] txt blocks with wrong line/label structure: 0
[validate] contents=38365 with-newline=0 empty=0 len min/mean/max=1/24.1/264
[validate] contents containing CJK: 38347/38365 = 100.0%
```

**★ 负向对照（证明校验不是橡皮图章）**——故意制造 4 种破坏，全部必须被抓到：

| 负向对照 | 期望 | 实际（在最终产物上跑）|
|---|---|---|
| 交换相邻两条不同 role 的消息（必然改变 role 序列）| 报错 | **CAUGHT** — `role[0] 'assistant' != 'user'`, `role[1] 'user' != 'assistant'` |
| 随机打乱一段对话的消息顺序 | 报错 | **CAUGHT** — `role[2] 'assistant' != 'user'` |
| 删掉一条消息 | 报错 | **CAUGHT** — `message count 36 != 37` |
| 把某条 content 掏成空白 | 报错 | **CAUGHT** — `empty content[2]` |

`all_caught = True`【实测】。校验器在 4 种破坏下都报错 ⇒ 不是恒真断言。

### 5.1 第一次 assemble 抓到 1 条真缺陷，已修

第一次 assemble 报 `ok=1299/1300 violating=1`：**对话 195 第 8 条 content 为空**
（全局 index 6150，源文 `Maple Syrup Urine Disease` —— **无标点裸名词短语**，解码器直接吐 EOS）。

已修，且**修在管线里**（不是手改数据）：`MarianBackend._retry_empty` —— 空输出时**补一个句号重译**
（Marian 训练语料都是带标点的整句），并在 `run` 模式增加「完成后再扫一遍空输出并补译」。
修复后 `still empty: []`。

> ⚠ **残留问题要如实说**：补句号后得到 `麻黄素尿病。`（结构合法，但**译错**——
> Maple Syrup Urine Disease 的标准译名是 **枫糖尿症**，模型显然不认识这个罕见病名）。
> 这 **1 条（0.0026%）** 是**已知错误译**，我选择保留管线的自动结果、不复写人工答案，但在此标注。

---

## 6. 已知答案对照（5 条事先知道译法的短句）

| 英文 | 模型输出 | 可接受集合 | 判定 |
|---|---|---|---|
| `Hello` | 你好 | 你好 / 您好 / 嗨 / 哈罗 / 你好。 | **PASS** |
| `How are you?` | 你好吗? | 你好吗 / 你还好吗 / 你怎么样 | **PASS** |
| `Thank you.` | 谢谢 | 谢谢 / 谢谢你 / 感谢你 | **PASS** |
| `Good morning.` | 早上好,你好吗? | 早上好 / 早安 | **FAIL** |
| `I love you.` | 我爱你 | 我爱你 | **PASS** |

**4/5 PASS**【实测】。

关于唯一的 FAIL：`早上好,你好吗?` 其实**不算错译**，是我的可接受集合写窄了（只收纯问候，没收「问候+反问」）。
**我保留 FAIL 不改判据**，以免落进「改标准把绿刷出来」的窠臼。

---

## 7. 确定性

| 检验 | 结果 |
|---|---|
| 同 40 条输入翻两遍（两次独立 `generate`）| **40/40 逐字符完全一致**，0 条不同【实测】|
| 批组合敏感性（同样 40 条：1 大批 vs 32 一小批）| **0 条不同**【实测】|

⇒ 满足「固定住采样随机性」的要求：`do_sample=False` + beam search，无采样噪声。
（唯一残留不确定性来自**批内最长序列**决定 `max_new_tokens`；批边界固定且可复现，续跑不改变结果。）

---

## 8. ★ 中文串扰对照（3 条本来就是中文的句子）

**结论：该管线会严重破坏中文输入，绝不能用于混合语料。**【实测】

| 中文输入 | 模型输出 | 是否原样返回 |
|---|---|---|
| 我今天心情不太好，有点焦虑。 | `~ , ~ ~ 的, ~` | ✗ 完全被破坏 |
| 谢谢你一直陪着我，听我说这些。 | `~ , ~ ~ 的, ~` | ✗ 完全被破坏（**与上一条输出完全相同**，说明是退化的兜底模式）|
| 你能告诉我该怎么办吗？ | `{\fn黑体\fs22\bord1\shad0\3aHBE\4aH00\fscx67\fscy66\2cHFFFFFF\3cH808080}什么?` | ✗ 吐出了 **ASS 字幕样式标记** |

3/3 被破坏，0/3 原样返回。
⇒ **本管线只可用于纯英文语料**（本输入恰好 100% 英文，所以安全）；
若将来要翻中英混杂数据，**必须先做语种分流**，只把英文片段送进来。

---

## 9. 质量抽检（6 段中英对照，原文粘贴供人工判断）

抽检点在 1300 段里等距取 6 段（index 0 / 260 / 520 / 780 / 1040 / 1299），每段贴前 6 条。

```
########## 抽检 1/6  (对话 index=0, 27 条消息) ##########
[用户] EN: Hello
[用户] ZH: 你好
[模型] EN: Hello, what would you like to talk about?
[模型] ZH: 你好,你想谈什么?
[用户] EN: I am having a lot of anxiety about quitting my current job. It is too stressful but pays well
[用户] ZH: 我对辞掉目前的工作感到非常焦虑,这压力太大,但报酬不错。
[模型] EN: What makes your job stressful for you?
[模型] ZH: 是什么让你的工作压力很大?
[用户] EN: I have to deal with many people in hard financial situations and it is upsetting
[用户] ZH: 我不得不面对许多财政困难的人,这令人心烦
[模型] EN: Do you help your clients to make it to a better financial situation?
[模型] ZH: 您是否帮助客户改善财务状况?

########## 抽检 2/6  (对话 index=260, 38 条消息) ##########
[模型] EN: Hello
[模型] ZH: 你好
[用户] EN: Hello there
[用户] ZH: 你好
[模型] EN: How may i help you today?
[模型] ZH: 今天有什么需要帮忙的吗?
[用户] EN: I'm really panicking. I just lost my job and I don't know how I'm going to pay my rent.
[用户] ZH: 我真的慌了 我只是失业了 我不知道要怎么付房租
[模型] EN: Don't panic everything is going to be okay i understand how you are feeling
[模型] ZH: 不要惊慌 一切都会好起来的 我明白你的感受
[模型] EN: Did you lose your job because of the pandemic?
[模型] ZH: 你因为传染病而失业吗?

########## 抽检 3/6  (对话 index=520, 20 条消息) ##########
[用户] EN: Hello
[用户] ZH: 你好
[模型] EN: Hi
[模型] ZH: 嗨
[模型] EN: how are you? how it's going on?
[模型] ZH: 你好吗?
[用户] EN: I've been felling quite deppressed recently, just wanted to ask for some mental support
[用户] ZH: 我最近很沮丧,想寻求心理支持
[模型] EN: why are you feeling in that way? tell me what it's happening?
[模型] ZH: 你为什么有这种感觉 告诉我到底发生了什么?
[用户] EN: I both failed my final exam last week and my dog that I've owned for 10 years died 3 days ago
[用户] ZH: 我上周的期末考试都失败了 我养了10年的狗 三天前死了

########## 抽检 4/6  (对话 index=780, 27 条消息) ##########
[模型] EN: Hi, how is your day?
[模型] ZH: 嗨,你今天过得怎么样?
[用户] EN: Not good. I was told today that my contract is not being renewed once it ends this month. I am very anxious about finding another job during pandemic.
[用户] ZH: 不好,我今天被告知,我的合同在本月到期后不再续签,我非常担心在大流行病期间另找工作。
[模型] EN: I see, It's been a difficult year for all of us. But I think it would be possible to negotiate
[模型] ZH: 我知道,这年对我们所有人来说都是一个艰难的一年, 但我认为谈判是有可能的
[用户] EN: But I tried negotiating. They are down-sizing the entire sales department. What else can I do?
[用户] ZH: 但我尝试了谈判,他们正在缩小整个销售部门的规模,我还能做什么?
[模型] EN: Hmmm, I can see that this is stressful for you
[模型] ZH: 嗯,我看得出来这对你来说压力很大
[模型] EN: Did you already search for a new job?
[模型] ZH: 你已经在找新工作了吗?

########## 抽检 5/6  (对话 index=1040, 28 条消息) ##########
[模型] EN: Hello. How are you doing?
[模型] ZH: 你好,你好吗?
[用户] EN: I used to love my boyfriend, but now I can't stand him. Why has this happened and how to break up with him without hurting his feelings much?
[用户] ZH: 我曾经爱过我的男朋友 但现在我受不了他了 为什么发生这种事?
[模型] EN: What changed to bring you to the point you can't stand him?
[模型] ZH: 是什么让你变得无法忍受他?
[用户] EN: I just find him extremely annoying. How he walks, talks, breathes even... I should have walked away from him earlier.
[用户] ZH: 我只是觉得他很烦人 他走路、说话、呼吸的方式...
[用户] EN: He is very respectful, committed, sweet and loving, but I just don't feel anything towards him anymore.
[用户] ZH: 他非常尊重, 忠诚,甜蜜和爱, 但我只是不再对他有任何感觉了。
[模型] EN: That sounds like a really difficult and uncomfortable situation to be in.
[模型] ZH: 这听起来像 一个非常困难 和不舒服的情况 进入。

########## 抽检 6/6  (对话 index=1299, 22 条消息) ##########
[模型] EN: Hi, can i help today? Please, tell me about yourself?
[模型] ZH: 嗨,今天我能帮忙吗?
[用户] EN: hi
[用户] ZH: hii hi hi 喜 喜
[用户] EN: i'm nereida
[用户] ZH: 我是尼瑞达
[模型] EN: Hi, do have any issues that you would like to share with me today?
[模型] ZH: 嗨,你今天有什么问题想和我谈谈吗?
[用户] EN: i am in disputed mod with my friends
[用户] ZH: 我和我的朋友们在有争议的模式中
[模型] EN: I am sorry to hear that. So, you and your friends are not seeing eye to eye? Do you mind telling me what the issue is that yo are not agreeing on?
[模型] ZH: 我很遗憾听到这个消息。所以你和你的朋友们没有亲眼看到吗?你介意告诉我,你不同意什么问题吗?
```

### 主观评价：哪类句子翻得差

我另外随机抽读 28 条（中英对照），结论如下。

**翻得好的：简单陈述句、单个疑问句、问候、短情绪句。**
`I like to read` → `我喜欢看书`；`Are you getting enough sleep lately?` → `你最近睡得好吗?`；
`Hi, what can I do to help you?` → `嗨,有什么我能帮你的吗?`；`take care!` → `保重!`；
`I think that is an excellent idea! That is a good way to end this once and for all.`
→ `我认为这是一个极好的主意!这是一劳永逸地结束这一局面的好办法。`（长句也能整句保住）
连源文本的**拼写错误与粘连**都能扛：`What canI do to help?` → `我能帮什么忙?`。

**翻得差的（按严重度排序，全是实际抓到的例子）：**

| 类型 | 例子 | 问题 |
|---|---|---|
| ★ **整句/整从句丢失** | `Is there someone else you could go to about this? Besides whoever you've already attempted to discuss it with?` → `除了你已经尝试过和谁讨论过之外?` | **主句整句消失**，只剩状语残片 |
| 同上 | `do you have any hobby? that you can use to wipe away time` → `你有爱好吗?` | 第二个从句整段丢失 |
| 同上 | 抽检 5：`Why has this happened and how to break up with him without hurting his feelings much?` → `为什么发生这种事?` | **后半整句丢失** |
| ★ **极性译反** | `you are welcome.` → **`不欢迎你`** | 「不客气」译成「不欢迎你」，语义反向（全量实测见 §10，命中率 1.0%）|
| **逐词硬译 / 不地道** | `you seem like a very nice and sweet person` → `你看起来像一个非常好和可爱的人` | 英文语序直搬 |
| 同上 | `(You have to hit Quit and fill the questionarie)` → `(你必须打退出,填满问题)` | `打退出`、`填满问题` 都不是中文说法 |
| 同上 | 抽检 5：`That sounds like a really difficult and uncomfortable situation to be in.` → `这听起来像 一个非常困难 和不舒服的情况 进入。` | 词序、空格、`进入` 都错 |
| 同上 | 抽检 6：`are not seeing eye to eye?` → `没有亲眼看到吗?` | 习语按字面翻了 |
| **凭空加内容** | `Good day!` → `日安! 再见!` | 多出一个「再见!」，源文没有 |
| 同上 | `Sorry, I can't come over the situation.` → `抱歉,我不能过来,…` | `come over the situation`（走出困境）译成物理意义的「过来」|
| **罕见但完全崩坏** | 抽检 6：`hi` → `hii hi hi 喜 喜` | 裸 `hi` 的原始退化输出，**没被折叠规则覆盖**（token 不完全相同）|
| **语气/搭配生硬** | `yes. thank you` → `是。谢谢` | 「是。」不成话，应为「是的」 |
| 同上 | `pandemic` → `传染病` / `大流行病` | 应为「疫情」；不统一 |

**规律**：句子**越长、从句越多**，B 越容易**丢内容**；**口语/俚语/缩写/拼写错误**（`GF`、`snice`、`canI`）
是重灾区；**固定客套话**（`you are welcome`）可能出致命错。
A（CSANMT）在这些点上明显更稳（见 §3.1 对照），这是本次选型的已知代价。

---

## 10. 附加量化（缺陷率，全部实测）

| 指标 | 数值 |
|---|---|
| 结构违反 | **0 / 1300 段**（0 / 38365 条）|
| 空 content（初次 assemble 后）| 1 条（0.0026%），已由管线补译修复；修后 0 |
| 复读折叠命中 | 单遍触发 **558/38365 = 1.45%**；幂等 repass 再改 **79 条（0.21%）**；合计 **637 = 1.66%** |
| 输出全无 CJK 的条数 | **18/38365 = 0.047%**（全是非词/错拼输入被原样透传：`HII`→`HII`、`yup`→`yup`、`so`→`so so, so, so`、`or 911`→`911`）|
| 字符长度 | 源 3192606 → 中文 924873，比值 **0.290**（均值 83.2 → 24.1 字符）|
| 无句末标点的源消息 | **141/407 = 34.6%**（分层抽样）|
| `you are welcome` / `you're welcome`（207 条）| 正确（不客气/不用客气/别客气）**130 = 62.8%**；**极性译反（不欢迎）2 = 1.0%**；改写或省略 75 = 36.2% |

关于 34.6% 无句末标点：我**试过**「给无标点输入补一个句号再翻」，结果**好坏参半**
（`hello` → `哈罗。`、`Sure` → `当然可以。` 变好；`No` → `没有 No.` 变差），因此**否决**该改动，保持原文直译。
（这条纪律与 `AGENTS.md §5.9` 一致：判据接线前先把会被翻转的样本 dump 出来读一遍。）

---

## 11. 缺陷与局限（不粉饰）

1. **长句丢内容**是本管线最普遍的质量问题（见 §9 主观评价），不是偶发。
2. **无标点超短句最脆弱**：B 对 `Hello` / `Hi` / `hello` 会先产出重复串，靠
   `collapse_degeneration` 折叠后才可用。折叠前后都**不是模型真的在翻译**。
   抽检 6 的 `hi` → `hii hi hi 喜 喜` 就是折叠**没能覆盖**的残留。
3. **折叠规则是启发式**，1.66% 的消息被它改动过。它**不是模型能力**，是补丁。
4. **中文串扰**：见 §8，管线不能碰中文。
5. **已知错误译**：`Maple Syrup Urine Disease` → `麻黄素尿病`（应为 `枫糖尿症`），1 条。
6. **A 的译质更高**：若时间充裕（能接受 16 h），A 更好；或做「短句走 A、长句走 B」的混合，
   但那超出本任务「二选一」的范围，本次未做。
7. **未做人工逐条审校**：质量结论基于 6 段抽检 + 28 条随机抽读 + 5 条已知答案 + 12 条 A/B 对照，
   **不是**全量人工评估。

---

## 12. 复现命令

```bash
# 独立环境（绝不碰项目 .venv）
uv venv ~/.venvs/mt --python 3.12
uv pip install --python ~/.venvs/mt/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python ~/.venvs/mt/bin/python "transformers==4.46.3" sentencepiece sacremoses modelscope

cd /home/vesita/coding/my/nanoSeek
P=~/.venvs/mt/bin/python
$P scripts/translate_escov.py bench    --dialogues 50      # 测速 + ETA
$P scripts/translate_escov.py canary                      # 已知答案/确定性/中文串扰/负向对照
systemd-run --user --unit=escov-translate $P scripts/translate_escov.py run   # 全量（可断点续跑，含空输出补译）
$P scripts/translate_escov.py repass                      # 幂等复读折叠（纯文本变换）
$P scripts/translate_escov.py assemble                    # json + txt + 结构校验
$P scripts/translate_escov.py validate                    # 最终产物校验 + 负向对照
```
