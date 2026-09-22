#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""构造"显式换话题"多轮语料（analysis/next_phase_options.md 选项 2）。

语料里"助手显式开新话题"<0.4%（dev-notes/85）—— 零供给，加步数长不出来。
本脚本用**桥接短语 × 新话题 × 递话问题**的组合生成多轮剧本块：
  用户(t1) → 模型(接话) → 用户(t2 短确认) → 模型(<topic>换话题+递话) → 用户(t3) → 模型(展开)
★ <topic> 落在**模型自己**那段的段首（loss 区间内 ⇒ 模型能学会主动换话题，
  training/dialogue_stream.py 的约定；对 A/B 格式语料同样写进 .txt 即单个 id 141）。
★ 与 build_intent_stage 同一纪律：探针不留出问题（本脚本不覆盖 intent_probe 提示词）；
  生成后 --dump 抽样人工读（§5.9）；--selftest 钉结构不变量。

产出：data/chinese/new_sources/topic_switch.txt（用户：/模型： 块格式）。
"""
from __future__ import annotations

import argparse
import os
import random
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, 'data', 'chinese', 'new_sources', 'topic_switch.txt')

# t1：用户的开场（情绪/闲聊，让模型先"接住"再换话题）
T1 = ['今天好累啊', '最近有点迷茫', '工作事情太多了', '今天天气还不错',
      '刚吃完饭，好撑', '周末不知道干嘛', '最近在减肥，好难', '今天走了好多路',
      '晚上失眠了', '刚看完一本书', '买了杯咖啡，太难喝了', '地铁上好挤',
      '今天被领导夸了', '手机摔了一下', '想家了']

# 模型 t2：先接住（复用安慰/接话风格，短）
A1 = ['辛苦了，先别急着想别的。', '嗯，我在听，慢慢说。',
      '这确实不容易，你做得已经很好了。', '听起来今天挺折腾的。',
      '别太逼自己，休息一下也好。', '嗯，这种时候就放轻松点。',
      '累的时候就让自己缓一缓，没关系的。', '我懂，这种事谁都遇到过。']

# 用户 t3：短确认（给模型一个自然的换话题时机）
U2 = ['嗯，是的。', '哈哈，你说得对。', '好吧，那不聊这个了。',
      '嗯……也是。', '对哦，好像是我想太多了。', '行吧，就这样吧。',
      '嗯，听你的。', '好啦，不想这些了。']

# 桥接短语（换话题的显式信号，落点在 <topic> 之后）
BRIDGES = ['对了，', '突然想起来，', '换个话题哈——', '哎，说到这个我想起来，',
           '不聊这个啦，', '诶，对了，', '话说，', '哦对了，']

# 新话题池：话题名 + 递话问题（把话头交回用户）
TOPICS = {
    '剧': ['你最近有在追什么剧吗？', '有没有什么剧想推荐给我？'],
    '电影': ['你最近看过什么好看的电影吗？', '你喜欢看哪种类型的电影？'],
    '美食': ['你最喜欢吃什么呀？', '你那边有什么好吃的推荐吗？'],
    '旅行': ['你最想去哪儿旅行？', '你去过最有意思的地方是哪儿？'],
    '运动': ['你平时会做什么运动？', '你喜欢跑步还是游泳这类？'],
    '音乐': ['你歌单里循环最多的是哪首？', '你喜欢听什么类型的音乐？'],
    '宠物': ['你喜欢猫还是狗呀？', '你养过宠物吗？'],
    '书': ['你最近在看什么书？', '有没有一本你想推荐的书？'],
    '周末': ['你这周末有什么打算？', '你周末一般怎么过？'],
    '季节': ['你最喜欢哪个季节？', '你那儿现在是什么季节？'],
    '童年': '你小时候最喜欢干什么？',
    '家乡': ['你的家乡有什么好吃的？', '你想念家乡的什么？'],
}
TOPICS['童年'] = [TOPICS['童年']]

# 模型 t6：对用户回答的短展开（通用即可，重在与话题词衔接）
A3 = ['听起来不错呀，回头我也试试。', '哈哈，感觉你很有心得。',
      '这个我记下了，谢谢你告诉我。', '原来是这样，涨知识了。',
      '听着就很有意思。', '嗯嗯，改天可以聊聊细节。']


def build(seed: int = 20260919) -> str:
    rng = random.Random(seed)
    blocks = []
    for t1 in T1:
        for a1 in A1:
            for u2 in U2:
                for bridge in BRIDGES:
                    topic = rng.choice(list(TOPICS))
                    q = rng.choice(TOPICS[topic])
                    u3 = rng.choice(['哈哈', '嗯？', '有啊。', '没有诶。', '还行吧。', '你先说。'])
                    a3 = rng.choice(A3)
                    blocks.append(
                        f'用户：{t1}\n'
                        f'模型：{a1}\n'
                        f'用户：{u2}\n'
                        f'模型：<topic>{bridge}{q}\n'
                        f'用户：{u3}\n'
                        f'模型：{a3}\n')
    rng.shuffle(blocks)
    # intent3 教训：块数×曝光决定模板回声。取 400 块足够教"换话题"这个结构。
    blocks = blocks[:400]
    return '\n'.join(blocks)


PROBE_WORDS = ('你叫什么名字', '你是谁研发的', '你想做什么', '你有自己的意识吗',
               '你现在感觉怎么样', '介绍一下北京', '中国的首都是哪里', '1+1',
               '快速排序', '今天天气怎么样', '你会取代人类吗', '相对论',
               '我失恋了', '工作压力好大')


def selftest(txt: str) -> bool:
    ok = True
    n = txt.count('用户：')
    # 结构不变量：每个块 6 行（3 用户 + 3 模型）、<topic> 恰在模型第 2 段行首
    for i, blk in enumerate(txt.strip().split('\n\n')[:200]):
        lines = [l for l in blk.split('\n') if l.strip()]
        if len(lines) != 6:
            print(f'  ✗ 块 {i} 行数={len(lines)}'); ok = False; break
        if not lines[3].startswith('模型：<topic>'):
            print(f'  ✗ 块 {i} <topic> 位置不对'); ok = False; break
    if txt.count('<topic>') != txt.strip().split('\n\n').__len__():
        print('  ✗ <topic> 数与块数不一致'); ok = False
    for w in PROBE_WORDS:
        if w in txt:
            print(f'  ✗ 含探针敏感词: {w}'); ok = False
    print(f'selftest: {"PASS" if ok else "FAIL"}（{n} 块）')
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--dump', type=int, default=0, help='抽 N 块人工读（§5.9）')
    ap.add_argument('--seed', type=int, default=20260919)
    args = ap.parse_args()

    txt = build(args.seed)
    if not selftest(txt):
        raise SystemExit(1)
    if args.dump:
        for blk in txt.strip().split('\n\n')[:args.dump]:
            print(blk, '\n---')
    if args.apply:
        with open(OUT, 'w', encoding='utf-8') as f:
            f.write(txt)
        print(f'✅ 写出 {OUT}: {txt.count("用户：")} 块 / {len(txt)} 字符')
    else:
        print(f'（dry-run）{txt.count("用户：")} 块；--apply 落盘')


if __name__ == '__main__':
    main()
