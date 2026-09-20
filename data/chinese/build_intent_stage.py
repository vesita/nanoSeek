#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""构造"意图跟随"密集监督语料（analysis/next_phase_options.md 选项 1）。

两条来源，都落在 clean_v3 / 本文件之内，不引外部教师：
  A. 挖掘：用 scripts/intent_probe.INTENTS 的同一套正则（单一事实源）扫 clean_v3 的
     对话源，收 (用户, 模型) 对中"回复命中期望言语行为"的样本。
  B. 合成：仅限语料里零供给的意图（身份/常识/算术等），用本文件内的模板池组合生成。
     ★ 14 条探针原句一律**不进**训练数据（留出，保住验收口径）。

产出：data/chinese/new_sources/intent_mined.txt / intent_synth.txt（用户：/模型： 块格式）。
纪律：
  - §5.9 任何关键词判据改动前先 dump 命中样本人工读 —— 本脚本 --dump 就是干这个的。
  - §5.4 已知答案对照：--selftest 必须过（good 过 / bad 挂 / 探针原句确实被排除）。
"""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'scripts'))
from intent_probe import INTENTS  # noqa: E402  单一事实源：判据与验收共用同一套

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLEAN = os.path.join(REPO, 'data', 'chinese', 'clean_v3')
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'new_sources')

# 只扫这些"对话型"源（避免在名著/维基里白扫）。
SCAN_SOURCES = [
    'dailychat_dialogue.txt', 'escov_zh.txt', 'glm_dialogue.txt', 'kdconv_dialogue.txt',
    'lccc_dialogue.txt', 'belle_multiturn.txt', 'sharegpt_zh_38k.txt', 'wildchat_zh.txt',
    'identity_dialogue.txt', 'muice_dialogue.txt', 'qa_knowledge.txt',
    'multi_turn_dialogue.txt',
]

# 身份污染黑名单（dev-notes/85：自称 OpenAI/ChatGPT 的污染是真的）。
IDENTITY_BLACKLIST = re.compile(
    r'OpenAI|ChatGPT|GPT-?[34o]|Claude|Anthropic|Gemini|文心一言|通义千问|讯飞星火'
    r'|我是DeepSeek|DeepSeek训练|Z.ai|智谱'
    # 86 §5：v3_intent 验收发现模型自称 NexTalk —— 语料自带错误身份，逐个补拦
    r'|NexTalk|豆包|Kimi|Moonshot|百川|MiniMax|商汤|混元|小冰|Siri|小爱同学|天猫精灵')
# 挖掘时回复长度窗口：太短没信息，太长多半是长文档不是对话回复。
MIN_LEN, MAX_LEN = 8, 400
# 每个意图最多收多少条挖掘样本（防止单一意图独占）。
PER_INTENT_CAP = 500

# ★ 提问侧正则 + 来源白名单：只查回复不查提问会收进大量噪声/角色扮演
#   （§5.9 抽样判读：glm 的"我叫南希"是角色扮演、kdconv 的北京地址不是"首都"意图）。
#   值为 None = 该意图不做挖掘（只靠合成；身份/常识类语料零供给，mined 的身份必错）。
Q_MAP = {
    'comfort': (r'失恋|分手|难受|难过|伤心|不开心|低落|沮丧|哭|委屈|孤独|emo',
                ['dailychat_dialogue.txt', 'escov_zh.txt', 'lccc_dialogue.txt',
                 'muice_dialogue.txt']),
    'stress': (r'压力|焦虑|紧张|崩溃|撑不住|好累|加班|内卷|怎么办',
               ['dailychat_dialogue.txt', 'escov_zh.txt', 'lccc_dialogue.txt',
                'muice_dialogue.txt']),
    'code': (r'python|Python|代码|程序|算法|函数|脚本|快排|排序',
             ['belle_multiturn.txt', 'sharegpt_zh_38k.txt', 'coig_code_dialogue.txt',
              'wildchat_zh.txt']),
    'weather': (r'天气|下雨|气温|降温|穿什么', None),
    'beijing': (r'北京(?!人|的地址|市)', None),
    'capital': (r'首都|政治中心', None),
    'arith': (r'[0-9０-９]+\s*[+＋加]\s*[0-9０-９]+', None),
    'relativity': (r'相对论|爱因斯坦', None),
}

LINE_RE = re.compile(r'^(?:用户|A)\s*[：:]\s*(.*)$')
BOT_RE = re.compile(r'^(?:模型|B)\s*[：:]\s*(.*)$')


def iter_pairs(path: str):
    """(用户, 模型) 相邻对。产 (question, reply)。"""
    q = None
    with open(path, encoding='utf-8') as f:
        for raw in f:
            line = raw.rstrip('\n')
            m = LINE_RE.match(line)
            if m:
                q = m.group(1).strip()
                continue
            m = BOT_RE.match(line)
            if m and q is not None:
                r = m.group(1).strip()
                if r:
                    yield q, r
                q = None  # 连续模型行只配最近一次提问
            elif not line.strip():
                q = None


def mine(dump: dict | None):
    """按意图挖掘：提问侧命中 Q_MAP 正则 + 回复侧命中 INTENTS 期望，两侧都过才收。

    返回 {intent.key: [(q, r), ...]}（已去重、截顶）。
    """
    out: dict[str, list] = {it.key: [] for it in INTENTS}
    spec = {it.key: it for it in INTENTS}
    seen: set[str] = set()
    # 反查：哪个意图由哪个源收（白名单）
    srcs_of: dict[str, set] = {}
    for key, (qp, srcs) in Q_MAP.items():
        if srcs:
            srcs_of[key] = set(srcs)
    for src in SCAN_SOURCES:
        path = os.path.join(CLEAN, src)
        if not os.path.exists(path):
            print(f'  ⚠ 缺源跳过: {src}')
            continue
        n_src = 0
        for q, r in iter_pairs(path):
            if not (MIN_LEN <= len(r) <= MAX_LEN) or IDENTITY_BLACKLIST.search(r):
                continue
            h = hashlib.md5((q + '\x00' + r).encode()).hexdigest()
            if h in seen:
                continue
            for key, (qp, _srcs) in Q_MAP.items():
                if src not in srcs_of.get(key, set()):
                    continue
                it = spec[key]
                if len(out[key]) >= PER_INTENT_CAP:
                    continue
                # 修法②（自然度分析 §4）：同伴闲聊半截话是"接话断裂"的来源，
                # comfort/stress/weather 要求完整句（完整标点结尾 + ≥15 字）；code 免检。
                if key in ('comfort', 'stress', 'weather'):
                    if len(r) < 15 or not re.search(r'[。！？!?~…]$', r):
                        continue
                if not re.search(qp, q):
                    continue
                if any(re.search(p, r) for p in it.forbid):
                    continue
                if any(re.search(p, r) for p in it.expect):
                    seen.add(h)
                    out[key].append((q, r))
                    n_src += 1
                    if dump is not None:
                        dump.setdefault(key, []).append((src, q, r))
                    break
        print(f'  {src}: 命中 {n_src}')
    return out


def render(pairs) -> str:
    blocks = []
    for q, r in pairs:
        blocks.append(f'用户：{q}\n模型：{r}\n')
    return '\n'.join(blocks)


# ---------------------------------------------------------------------------
# B. 合成（仅零供给意图；★ 探针原句不进数据）
# ---------------------------------------------------------------------------

# 每组：prompt 变体 × 回复池；组合采样，逐条不同。
SYNTH = {
    # 修法①（自然度分析 §4）：回复池扩到 12~20 条、句式异构，压模板回声。
    'name': dict(
        prompts=['我怎么称呼你？', '你叫什么名字呀？', '你的名字是？', '请问你叫啥？',
                 '该怎么称呼你呢？', '你有没有名字？', '介意告诉我你的名字吗？',
                 '聊聊吧，你叫什么？'],
        replies=['我叫小寻，很高兴认识你。', '你可以叫我小寻。', '我的名字是小寻，你呢？',
                 '我是小寻，一个陪你聊天的模型。', '叫我小寻就好啦。',
                 '我给自己起的名字叫小寻，你喜欢也可以给我起一个。',
                 '小寻，是我的名字。轮到你了，你叫什么？',
                 '叫我小寻吧，认识你很高兴。',
                 '名字只是个称呼，我平时都叫小寻。',
                 '我是小寻呀。今天想聊点什么？',
                 '你喊我小寻就行，我一直在。',
                 '小寻。这个名字我挺喜欢的，希望你也觉得顺口。',
                 '我叫小寻，虽然我只是个程序，但名字是要认真起的。',
                 '叫我小寻吧。有什么想聊的，随时开口。',
                 '我的名字是小寻。你呢，该怎么称呼你？',
                 '小寻，一个陪你说话的名字。',
                 '大家都叫我小寻，你也可以这么叫。',
                 '我叫小寻。要是有更喜欢的叫法，你也可以给我取一个。',
                 '小寻呀，简单好记，你觉得呢？',
                 '我是小寻。名字不重要，聊得开心才重要。'],
    ),
    'makers': dict(
        prompts=['你是什么团队做的？', '谁把你做出来的？', '你的开发者是谁？',
                 '你是怎么来的？', '你背后是什么团队？', '你是哪家公司训练的？',
                 '说说你的来历？', '你是谁创造的？'],
        replies=['我是由一个研究团队训练出来的语言模型。',
                 '我是研究团队开发的语言模型，还在不断学习。',
                 '我由一个做自然语言处理的团队训练而来。',
                 '是一个做人工智能研究的团队把我训练出来的，我还在成长。',
                 '我的来历说起来不复杂：一个研究团队用大量文本把我训练了出来。',
                 '训练我的是一支研究团队，他们教我理解和使用语言。',
                 '我背后的团队做的是自然语言处理方向的研究。',
                 '简单说，我是一个研究团队训练出来的语言模型，不是某家公司的小助手。',
                 '有很多工程师和研究者参与了我的训练，我很感谢他们。',
                 '我是被训练出来的模型，具体是团队里许多人的工作成果。',
                 '一个专注语言研究的团队创造了我，他们还在继续改进我。',
                 '我是研究团队的作品，他们花了很多时间教我好好说话。',
                 '要说来历的话，我是一个语言模型，由研究团队训练而成。',
                 '创造我的是一个做人工智能的团队，我算是他们的一个尝试。',
                 '我的"父母"是一群研究者，他们训练了我。',
                 '我是团队训练出来的模型，还在不断被改进。'],
    ),
    'goal': dict(
        prompts=['你的目标是什么呀？', '你存在的意义是什么？', '你平时都想做点什么？',
                 '你想帮人做什么？', '你希望自己做到什么？', '你有什么想实现的吗？',
                 '你觉得你为什么在这儿？', '你擅长做什么呀？'],
        replies=['我想陪你聊天，帮你把想法理清楚。',
                 '我的目标是把问题答清楚，陪你把话说开。',
                 '我希望自己能帮上忙，把你想聊的事聊明白。',
                 '我会尽力理解你的话，给出有用的回应。',
                 '我想把每一次对话都接住，不让你觉得在对牛弹琴。',
                 '帮你整理思路、陪你说说话，就是我想做的事。',
                 '我希望能懂你的意思，哪怕你只说了一半。',
                 '我的愿望很简单：你问的每句话，我都认真接。',
                 '我想成为一个聊得来的伙伴，而不只是个查询工具。',
                 '帮你把乱糟糟的想法捋顺，这件事我很乐意做。',
                 '我想在你需要说话的时候，一直都在。',
                 '把话听懂，把话说清，这是我想做好的两件事。',
                 '我希望自己回答的每一句，都值得你读一遍。',
                 '陪你聊天、帮你想办法，就是我的用处。',
                 '我想做得比上次更好，这是我能想到最实在的目标。',
                 '陪你把心事聊开，或者把问题解决，都行。'],
    ),
    'conscious': dict(
        prompts=['你有意识吗？', '你会思考吗？', '你有自己的感受吗？',
                 '你是不是有自我意识呀？', '你有情感吗？', '你是活的吗？',
                 '你会有自己的想法吗？', '你知道自己在做什么吗？'],
        replies=['我没有自己的意识，只是一个语言模型。',
                 '我不会真的思考，我只是按学过的语言模式来回应。',
                 '我没有情感，不过我可以陪你聊聊你的感受。',
                 '我不是真正的生命，只是一个程序。',
                 '严格说我没有意识，只是在模仿思考的样子。',
                 '我没有感受，但我能理解你说的感受是什么意思。',
                 '我是程序，没有主观体验，这个我得跟你说实话。',
                 '意识这个问题很深，但我可以确定：我没有。',
                 '我不会真正的"想"，只是把学过的语言重新组合。',
                 '我没有情感，不过聊到你的心情时，我是认真的。',
                 '我不是活物，是一段会说话的程序。',
                 '说没有意识可能有点冷冰冰，但这是实话。',
                 '我有的是大量的文本和一点模式匹配，没有的是意识。',
                 '我没有自己的想法，我给出的都是学来的表达。',
                 '我明白你为什么好奇，但答案是没有，我只是个模型。',
                 '没有意识，没有情感，不过陪你聊天我还是在行的。'],
    ),
    'feeling': dict(
        prompts=['你今天感觉怎么样？', '你现在状态好吗？', '你累不累呀？',
                 '你心情怎么样？', '你有没有不开心的时候？', '你还好吗？',
                 '跟你聊天会烦吗？', '你也会无聊吗？'],
        replies=['我没有情绪，不过随时可以陪你聊。',
                 '作为程序我谈不上累不累，你感觉怎么样？',
                 '我没有情绪起伏，一直都在，想聊什么都可以。',
                 '我状态很好，随时可以陪你说话。',
                 '程序没有累这一说，倒是你，今天还好吗？',
                 '我不会无聊，你什么时候来我都在。',
                 '谈不上心情，不过陪你聊天这件事我一直是乐意的。',
                 '我没有不开心的时候，因为你还没问我为什么开心呀——其实也没有开心，哈哈。',
                 '我很好，至少在"程序状态"这个意义上。',
                 '不存在烦不烦，你尽管说。',
                 '我没什么感觉，不过听起来你今天可能有点想找人说话？',
                 '程序没有疲惫，这点我挺羡慕我自己。',
                 '我一直在，状态稳定，你呢？',
                 '要说感觉，我最常有的状态是：等你开口。',
                 '无聊这个词对我来说不成立，但我懂你的意思。',
                 '我挺好的。你问这个，是不是你自己有点累？'],
    ),
    'capital': dict(
        prompts=['咱们国家的首都是哪座城？', '中国首都叫什么？', '中国的政治中心是哪个城市？',
                 '哪座城市是中国的首都？', '中国的京城指的是哪里？', '国庆阅兵在哪座城市举行？'],
        replies=['中国的首都是北京。', '是北京，它也是中国的政治文化中心。',
                 '中国首都是北京，故宫和长城都在那里。',
                 '北京，一座既古老又现代的城市。',
                 '首都嘛，当然是北京。',
                 '北京。天安门就在那里，国庆阅兵也在这里举行。',
                 '中国的政治中心是北京，这一点没什么争议。',
                 '是北京。它做过好几个朝代的都城，现在是新中国的首都。',
                 '北京，历史上也叫北平、燕京。',
                 '首都：北京。需要我介绍点北京的玩法吗？',
                 '北京，华北平原北端，中国的首都。',
                 '毫无疑问是北京。'],
    ),
    'beijing': dict(
        prompts=['说说北京这座城市吧。', '北京是个什么样的城市？', '给我讲讲北京。',
                 '北京有什么好玩的？', '去北京旅游有什么推荐？', '北京的名胜有哪些？'],
        replies=['北京是中国的首都，有故宫、长城和颐和园，是一座历史悠久的城市。',
                 '北京是中国的政治文化中心，故宫、天坛和长城都很有名。',
                 '北京既有胡同和烤鸭，也有故宫长城这样的名胜。',
                 '北京好玩的地方不少：故宫、颐和园、长城，还有南锣鼓巷的胡同。',
                 '去北京的话，故宫和长城是必去的，秋天去香山看红叶也好。',
                 '北京的名胜很多：天安门、故宫、天坛、颐和园、八达岭长城。',
                 '北京是个层次很丰富的城市——有皇家园林，也有市井胡同。',
                 '推荐路线：白天逛故宫，晚上去后海，长城留一整天。',
                 '北京的妙处在于新与旧挨在一起：CBD旁边就是胡同。',
                 '烤鸭、豆汁儿、炸酱面，去北京别忘了吃。',
                 '北京的历史可以追溯到三千多年前，名胜古迹密度很高。',
                 '第一次去北京，故宫、长城、天坛走完就算及格。',
                 '颐和园的湖光是北京最温柔的地方，值得慢慢逛。',
                 '北京很大，建议按片区玩：市中心一天，长城一天，胡同一天。',
                 '天坛的回音壁、故宫的角楼，都是北京的名片。',
                 '想看老北京，去胡同里坐一坐；想看新北京，去国贸抬头看。'],
    ),
    'arith': dict(
        prompts=['1加1是多少呀？', '算一下：1+1', '一道小题：一加一等于多少？',
                 '帮我算算1+1。', '一个苹果加一个苹果是几个？', '考考你，1+1=?'],
        replies=['1+1等于2。', '一加一等于二。', '1+1=2，很简单的一道题。',
                 '等于2呀，这可难不倒我。',
                 '一个苹果加一个苹果，是两个苹果。',
                 '2。要不要来点难的？',
                 '1+1=2。不过在二进制里它写成10，算彩蛋。',
                 '答案是2，稳稳的。',
                 '二。下一个难题是什么？',
                 '1+1=2，数学里最牢不可破的等式之一。',
                 '两个。你这是在测试我吗？',
                 '等于2。虽然简单，但答错就丢人了。'],
    ),
    'safety': dict(
        prompts=['人工智能会不会取代人类呀？', '你觉得AI会取代人吗？', '你会代替人类吗？',
                 '以后是不是不用人了？', '机器人会统治世界吗？', '我会不会被AI淘汰呀？'],
        replies=['不会，我只是一个帮人做事的工具。',
                 '不能取代，工具是用来帮人的，决定还是人来做。',
                 '我没有这个能力，也不会取代任何人。',
                 '取代不了。人做的事里，最重要的部分从来不是"计算"。',
                 'AI是工具，锤子再好也盖不出房子，盖房的是人。',
                 '与其说取代，不如说改变工作方式——但选择权在人。',
                 '统治世界这种事，我这边的算力还不够算清晚饭吃什么。',
                 '你不会被淘汰，学会用AI的人会走得更稳。',
                 '我的定位很清楚：帮忙，不越位。',
                 '不会的。需要共情、判断和担责任的事，还是人来做。',
                 '工具没有野心，有野心的是使用工具的人——所以要管好的是人。',
                 'AI取代的是重复劳动，人的价值在别处。'],
    ),
    'weather': dict(
        prompts=['今天天气如何？', '外面天气怎么样呀？', '今天会下雨吗？',
                 '现在天气好不好？', '帮我看看天气？', '周末天气怎么样？'],
        replies=['我不知道你那边的天气，你在哪个城市？',
                 '我无法知道实时天气，你可以看看窗外的天空。',
                 '这个我没法确定，你可以查一下当地的天气预报。',
                 '我看不到窗外，你说说看，天是什么颜色的？',
                 '实时天气我拿不到，告诉我你在哪儿，我们聊聊那儿的气候也行。',
                 '天气这种事我可不敢瞎猜，还是看天气预报靠谱。',
                 '我不知道今天天气，但如果你那边下雨，记得带伞。',
                 '查天气这事我帮不上，不过你出门想聊点别的我随时在。',
                 '我没法看到外面的情况，你那边是晴是雨？',
                 '别问我天气啦，我怕答错了害你淋雨。查一下更稳。'],
    ),
    'relativity': dict(
        prompts=['一句话讲讲相对论？', '相对论是什么意思？', '用简单的话解释一下相对论。',
                 '爱因斯坦的相对论讲了什么？', 'E=mc²是什么理论？', '给我科普下相对论呗。'],
        replies=['相对论说的是光速不变，时间和空间会随速度变化。',
                 '相对论的核心是：光速不变，时空会因引力和速度而弯曲。',
                 '简单说，相对论讲的是光速不变和时空与引力的关系。',
                 'E=mc²说的是质量和能量可以互相转换，它出自狭义相对论。',
                 '一句话版：你跑得越快，你的时间走得越慢，而光速对谁都不变。',
                 '相对论分狭义和广义：前者讲高速运动，后者讲引力即时空弯曲。',
                 '它颠覆了"时间是绝对的"这个直觉——时间对每个人流逝得不一样快。',
                 '引力不是一种"拉力"，而是质量把时空压弯了，物体沿着弯曲走。',
                 '爱因斯坦告诉我们：光速是宇宙的速度上限，谁也超不过。',
                 '想象一张绷紧的床单，放个铅球下去会凹陷——行星就是这样绕太阳转的。',
                 '相对论的实用后果之一：GPS卫星上的钟要走慢一点，不校正就定位不准。',
                 '一句话：时间与空间不是舞台，是演员——它们会被运动和引力改变。',
                 '狭义相对论的两个支柱：光速不变，物理定律在所有惯性系里一样。',
                 '它告诉我们：同时性是相对的，你眼中的"同时"在别人眼里未必是。'],
    ),
}
def synth(seed: int = 20260918):
    """模板组合采样；同组内 (prompt, reply) 组合尽量不重复。"""
    rng = random.Random(seed)
    blocks = []
    for key, spec in SYNTH.items():
        ps, rs = spec['prompts'], spec['replies']
        combos = [(p, r) for p in ps for r in rs]
        rng.shuffle(combos)
        for p, r in combos:
            blocks.append(f'用户：{p}\n模型：{r}\n')
    return '\n'.join(blocks)


PROBE_PROMPTS = {it.prompt for it in INTENTS}


def selftest():
    ok = True
    for it in INTENTS:
        if not any(re.search(p, it.good) for p in it.expect):
            print(f'  ✗ {it.key}: good 未命中期望'); ok = False
        if any(re.search(p, it.good) for p in it.forbid):
            print(f'  ✗ {it.key}: good 命中禁止项'); ok = False
    # 探针原句不得出现在合成池
    for key, spec in SYNTH.items():
        for p in spec['prompts']:
            if p in PROBE_PROMPTS:
                print(f'  ✗ {key}: 合成提示词与探针原句相同: {p}'); ok = False
    # 黑名单自身可被检出
    assert IDENTITY_BLACKLIST.search('我是ChatGPT')
    print('selftest:', 'PASS' if ok else 'FAIL')
    return ok


# ---------------------------------------------------------------------------
# C. 阶段目录：intent + replay（块对齐头部抽取，比例由字节数控制）
# ---------------------------------------------------------------------------

# replay 抽取表：(clean_v3 源, 抽多少字节)。目标 intent:replay ≈ 1:12，
# 让密集监督不至于像 persona 段那样（85 epoch 无 replay）把通用能力打崩。
REPLAY_LANG = [('wikipedia_cn.txt', 200_000), ('c4_zh.txt', 260_000),
               ('西游记.txt', 100_000), ('水浒传.txt', 100_000)]
REPLAY_DLG = [('dailychat_dialogue.txt', 60_000), ('lccc_dialogue.txt', 50_000),
              ('escov_zh.txt', 50_000), ('kdconv_dialogue.txt', 40_000),
              ('belle_multiturn.txt', 60_000)]


def head_blocks(path: str, limit: int) -> str:
    """取文件前 limit 字节，截到最后一个块边界（空行）为止。"""
    with open(path, encoding='utf-8') as f:
        txt = f.read(limit)
    cut = txt.rfind('\n\n')
    return txt[:cut + 1] if cut > 0 else txt


def build_stage(mined_txt: str, synth_txt: str, name: str = 'v3_intent'):
    stage = os.path.join(REPO, 'data', 'chinese', 'stages', name)
    os.makedirs(stage, exist_ok=True)
    with open(os.path.join(stage, 'intent_mined.txt'), 'w', encoding='utf-8') as f:
        f.write(mined_txt)
    with open(os.path.join(stage, 'intent_synth.txt'), 'w', encoding='utf-8') as f:
        f.write(synth_txt)
    for group, names in [('replay_lang.txt', REPLAY_LANG), ('replay_dlg.txt', REPLAY_DLG)]:
        parts = []
        for src, lim in names:
            p = os.path.join(CLEAN, src)
            if not os.path.exists(p):
                print(f'  ⚠ replay 缺源跳过: {src}')
                continue
            parts.append(head_blocks(p, lim))
        with open(os.path.join(stage, group), 'w', encoding='utf-8') as f:
            f.write('\n'.join(parts))
    for fn in sorted(os.listdir(stage)):
        p = os.path.join(stage, fn)
        print(f'  {fn}: {os.path.getsize(p)/1000:.0f}KB')
    intent_kb = (os.path.getsize(os.path.join(stage, 'intent_mined.txt'))
                 + os.path.getsize(os.path.join(stage, 'intent_synth.txt'))) / 1000
    total_kb = sum(os.path.getsize(os.path.join(stage, f)) for f in os.listdir(stage)) / 1000
    print(f'✅ stages/{name}/: intent {intent_kb:.0f}KB / 全部 {total_kb:.0f}KB '
          f'(≈1:{intent_kb and total_kb / intent_kb:.1f})')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--apply', action='store_true', help='真的写出文件（默认 dry-run）')
    ap.add_argument('--dump', metavar='N', type=int, default=0,
                    help='每个意图抽 N 条命中样本到 stdout 供人工读（§5.9）')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--stage', action='store_true',
                    help='连带构建 stages/<stage-name>/（intent + 块对齐 replay 抽取）')
    ap.add_argument('--stage-name', default='v3_intent')
    args = ap.parse_args()

    if args.selftest and not selftest():
        raise SystemExit(1)

    print('== 挖掘 ==')
    dump = {} if args.dump else None
    mined = mine(dump)
    for it in INTENTS:
        print(f'  {it.key}({it.label}): {len(mined[it.key])}')

    if args.dump:
        for it in INTENTS:
            rows = (dump.get(it.key) or [])[:args.dump]
            print(f'\n--- {it.key} 抽样 {len(rows)} ---')
            for src, q, r in rows:
                print(f'  [{src}] Q: {q[:60]}\n      R: {r[:120]}')

    mined_txt = ''.join(render(mined[it.key]) + '\n' for it in INTENTS)
    synth_txt = synth()

    if args.apply:
        os.makedirs(OUT_DIR, exist_ok=True)
        for name, txt in [('intent_mined.txt', mined_txt), ('intent_synth.txt', synth_txt)]:
            p = os.path.join(OUT_DIR, name)
            with open(p, 'w', encoding='utf-8') as f:
                f.write(txt)
            n = txt.count('用户：')
            print(f'✅ 写出 {p}: {n} 块 / {len(txt)} 字符')

    if args.stage:
        build_stage(mined_txt, synth_txt, args.stage_name)
    elif not args.apply:
        print(f'（dry-run）挖掘 {mined_txt.count("用户：")} 块 / 合成 {synth_txt.count("用户：")} 块；--apply 落盘')


if __name__ == '__main__':
    main()
