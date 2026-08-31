#!/usr/bin/env bash
# 对话自然度实测：用 chat.py（真实交互管线：完整上下文累积 + stop_on_turn/stop_on_eos/
# clip_at_sentence，与终端体验一致）批量驱动 6 个 A/B 臂 × 4 组对话。
#   3 个多轮场景（天气/心情/写诗，各 4 轮） + 6 个通用开场（单轮）
# 每组对话写一个文件（头部注释 = 该组用户输入序列），供自然度分析。
# 用法：bash training/run_chat_naturalness.sh [--max-new-tokens 200]
set -u
cd "$(dirname "$0")/.." || exit 1

MAX_NEW=${1:-200}
OUT=out/chat_nat
mkdir -p "$OUT"

ARMS="arm0:out/char_fact24_300
arm1:out/ab_char_gate_300
arm2:out/ab_char_moe8_300
arm3:out/ab_char_mtp2_300
arm4:out/ab_char_all_300
arm5:out/ab_char_all6_1500"

# 3 个多轮场景（与 eval_char_ab_comprehensive 的 MULTITURN_SCENARIOS 一致，便于对照）
s1="你好，今天天气怎么样？
那适合出门跑步吗？
如果下雨该带什么？
谢谢你的建议！
exit
"
s2="我今天心情有点低落。
工作上遇到了点挫折。
感觉大家都不理解我。
听你这么说我好多了。
exit
"
s3="你能帮我写一首关于秋天的短诗吗？
再加点落叶的意象呢？
韵脚能不能押得更工整一点？
太棒了，给这首诗起个标题吧。
exit
"
# 6 个通用开场（与 GENERAL_PROMPTS 一致）
gen="你好！
请介绍一下你自己。
如何学好人工智能？
什么是量子计算？
推荐两本好看的历史小说。
为什么天空是蓝色的？
exit
"

run_one() { # $1=短名 $2=out_dir $3=场景名 $4=prompt序列 $5=输出文件
  {
    echo "===== $1 ($2) — $3 | chat.py --seed 42 --max-new-tokens $MAX_NEW ====="
    echo "----- 用户输入序列（管道无回显，对照此列表）:"
    printf '%s' "$4" | grep -v '^exit$' | sed 's/^/  Q: /'
    echo "----- 对话实录:"
  } > "$5"
  printf '%b' "$4" | env HSA_OVERRIDE_GFX_VERSION=10.3.0 \
    .venv/bin/python inference/scripts/chat.py \
      --out_dir "$2" --seed 42 --max-new-tokens "$MAX_NEW" >> "$5" 2>&1
  echo "  已生成: $5"
}

for a in $ARMS; do
  name=${a%%:*}; dir=${a##*:}
  [ -f "$dir/best.pt" ] || { echo "⚠ 跳过 $name：无 $dir/best.pt"; continue; }
  echo "▶ $name ($dir)"
  run_one "$name" "$dir" "多轮场景1-天气" "$s1" "$OUT/$name-s1.txt"
  run_one "$name" "$dir" "多轮场景2-心情" "$s2" "$OUT/$name-s2.txt"
  run_one "$name" "$dir" "多轮场景3-写诗" "$s3" "$OUT/$name-s3.txt"
  run_one "$name" "$dir" "通用开场x6" "$gen" "$OUT/$name-gen.txt"
done
echo "✅ 全部完成 → $OUT/"
