#!/usr/bin/env python3
"""nanoSeek 1500 轮渐进式强化学习训练总控流水线 (1500-Round Curriculum Master)

总预算：1500 轮 (严控不超过 1500 轮)，划分为 15 个阶段 (100 轮/次)。

课程体系架构：
┌─────────┬────────────┬────────────────────────────────────────────────────────┐
│ 阶段    │ 轮数范围   │ 核心训练任务与攻坚目标                                 │
├─────────┼────────────┼────────────────────────────────────────────────────────┤
│ Phase 1 │ 001 - 300  │ 【语言纯度与利落收尾】GRPO 指数奖励 + EOS 静默正则    │
│ Phase 2 │ 301 - 700  │ 【逐句记忆倒逼】单句输入 + KV 联想记忆跨轮实体精准召回 │
│ Phase 3 │ 701 - 1200 │ 【双模型自博弈】Alice vs Bob 真实交替对话互训与认知融合│
│ Phase 4 │ 1201 - 1500│ 【极限抗扰与泛化】复杂意图切换 + 学习率余弦退火收敛    │
└─────────┴────────────┴────────────────────────────────────────────────────────┘

用法：
    # 运行指定单个 100 轮阶段 (如第 1 阶段 1-100 轮)
    .venv/bin/python training/rl/run_curriculum.py --stage 1

    # 连续运行所有阶段 (1 到 15 阶段，总共 1500 轮，每 100 轮自动归档)
    .venv/bin/python training/rl/run_curriculum.py --all
"""
import argparse
import os
import sys
import subprocess
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

CURRICULUM_STAGES = [
    # Phase 1: 语言纯度与 EOS 收尾 (1-300 轮)
    {"stage": 1, "rounds": 100, "phase": 1, "task": "grpo", "lr": "2e-5", "desc": "基础表达去模板化与 EOS 静默强化"},
    {"stage": 2, "rounds": 100, "phase": 1, "task": "grpo", "lr": "2e-5", "desc": "常识百科与精准问答去冗余强化"},
    {"stage": 3, "rounds": 100, "phase": 1, "task": "grpo", "lr": "1.5e-5", "desc": "短句自然度与 3-gram 重复率压制"},

    # Phase 2: 逐句输入记忆倒逼强化 (301-700 轮)
    {"stage": 4, "rounds": 100, "phase": 2, "task": "memory", "lr": "3e-5", "desc": "基础身份实体(姓名/城市/职业)单句跨轮召回"},
    {"stage": 5, "rounds": 100, "phase": 2, "task": "memory", "lr": "3e-5", "desc": "用户个性化偏好与日常习惯跨轮记忆"},
    {"stage": 6, "rounds": 100, "phase": 2, "task": "memory", "lr": "2e-5", "desc": "突发事件与上下文情绪长程留存"},
    {"stage": 7, "rounds": 100, "phase": 2, "task": "memory", "lr": "2e-5", "desc": "复合实体联想与多跳信息单句检索"},

    # Phase 3: 双模型自博弈与交替互训 (701-1200 轮)
    {"stage": 8, "rounds": 100, "phase": 3, "task": "selfplay", "lr": "2e-5", "desc": "Alice vs Bob 日常生活主题交替对聊"},
    {"stage": 9, "rounds": 100, "phase": 3, "task": "selfplay", "lr": "2e-5", "desc": "知识与技术探讨主题对抗质检与进化"},
    {"stage": 10, "rounds": 100, "phase": 3, "task": "selfplay", "lr": "1.5e-5", "desc": "情感倾听与适度共情交替互动"},
    {"stage": 11, "rounds": 100, "phase": 3, "task": "selfplay", "lr": "1.5e-5", "desc": "开放式辩论与逻辑澄清对聊"},
    {"stage": 12, "rounds": 100, "phase": 3, "task": "selfplay", "lr": "1.2e-5", "desc": "自由主题长程自博弈与 Model Soup 融合"},

    # Phase 4: 极限抗扰与全局收敛 (1201-1500 轮)
    {"stage": 13, "rounds": 100, "phase": 4, "task": "selfplay", "lr": "1e-5", "desc": "跨领域生僻问题与抗偏题微调"},
    {"stage": 14, "rounds": 100, "phase": 4, "task": "memory", "lr": "8e-6", "desc": "终极单句记忆鲁棒性退火"},
    {"stage": 15, "rounds": 100, "phase": 4, "task": "grpo", "lr": "5e-6", "desc": "全场景小学习率平滑收敛封顶"},
]


def run_stage(cfg, initial_ckpt, base_out="out/curriculum"):
    stage_id = cfg["stage"]
    out_dir = os.path.join(base_out, f"stage_{stage_id:02d}_{stage_id*100}")
    os.makedirs(out_dir, exist_ok=True)

    # 确定上一个阶段的检查点作为输入
    if stage_id == 1:
        ckpt_in = initial_ckpt
    else:
        prev_dir = os.path.join(base_out, f"stage_{stage_id-1:02d}_{(stage_id-1)*100}")
        ckpt_in = os.path.join(prev_dir, "best.pt")
        if not os.path.exists(ckpt_in):
            ckpt_in = initial_ckpt

    print("=" * 70)
    print(f"🚀 开始执行 Stage {stage_id:02d}/15 (累计第 {(stage_id-1)*100 + 1} - {stage_id*100} 轮)")
    print(f"  • 所属阶段: Phase {cfg['phase']} | 任务类型: {cfg['task'].upper()}")
    print(f"  • 训练内容: {cfg['desc']}")
    print(f"  • 初始检查点: {ckpt_in}")
    print(f"  • 学习率: {cfg['lr']} | 输出目录: {out_dir}")
    print("=" * 70)

    env = os.environ.copy()
    env["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
    t0 = time.time()

    if cfg["task"] == "grpo":
        cmd = [
            sys.executable, "training/rl/grpo_char.py",
            "--ckpt", ckpt_in,
            "--out", out_dir,
            "--steps", str(cfg["rounds"]),
            "--lr", cfg["lr"],
            "--group_size", "4",
            "--tau", "1.5"
        ]
    elif cfg["task"] == "memory":
        cmd = [
            sys.executable, "training/rl/memory_rl.py",
            "--ckpt", ckpt_in,
            "--out", out_dir,
            "--steps", str(cfg["rounds"]),
            "--lr", cfg["lr"],
            "--group_size", "4",
            "--tau", "1.5"
        ]
    elif cfg["task"] == "selfplay":
        cmd = [
            sys.executable, "training/rl/dual_chat_selfplay.py",
            "--ckpt", ckpt_in,
            "--out", out_dir,
            "--rounds", str(cfg["rounds"]),
            "--lr", cfg["lr"],
            "--tau", "1.5"
        ]
    else:
        raise ValueError(f"Unknown task: {cfg['task']}")

    res = subprocess.run(cmd, env=env)
    dt = time.time() - t0
    if res.returncode != 0:
        print(f"❌ Stage {stage_id} 执行失败 (退出码 {res.returncode})")
        return False

    print(f"✅ Stage {stage_id} 完成！耗时: {dt:.1f} 秒 | 检查点已保存至 {out_dir}/best.pt\n")
    return True


def main():
    ap = argparse.ArgumentParser(description="nanoSeek 1500 轮训练总控流水线")
    ap.add_argument("--stage", type=int, default=None, help="执行指定单阶段 (1-15)")
    ap.add_argument("--all", action="store_true", help="连续运行全部 15 个阶段 (1500 轮)")
    ap.add_argument("--ckpt", default="out/rl_grpo_2epoch/best.pt", help="起始检查点")
    ap.add_argument("--out", default="out/curriculum", help="归档总目录")
    args = ap.parse_args()

    if args.stage is not None:
        cfg = next((c for c in CURRICULUM_STAGES if c["stage"] == args.stage), None)
        if not cfg:
            print(f"错误: 未找到 Stage {args.stage}，有效范围 1-15")
            return
        run_stage(cfg, args.ckpt, base_out=args.out)
    elif args.all:
        print(f"🌟 启动 nanoSeek 1500 轮全量课程训练 (15 个阶段，每阶段 100 轮) 🌟\n")
        for cfg in CURRICULUM_STAGES:
            ok = run_stage(cfg, args.ckpt, base_out=args.out)
            if not ok:
                print("流水线中断！")
                break
        print("\n🎉 1500 轮全课程训练顺利收官！")
    else:
        print("请指定 --stage <1-15> 运行单个阶段，或使用 --all 运行全部 1500 轮。")
        print("\n可用阶段概览：")
        for c in CURRICULUM_STAGES:
            print(f"  Stage {c['stage']:02d}: Phase {c['phase']} [{c['task']:8s}] {c['desc']} ({c['rounds']} 轮)")


if __name__ == "__main__":
    main()
