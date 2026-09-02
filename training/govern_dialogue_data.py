#!/usr/bin/env python3
"""对话语料治理：保留日常对话，剔除任务/噪音/话术固化源。

治理决策（基于 2026-09-03 全量分类核查 + 用户确认的推荐分档）：
  - 剔除 (train+val):  zhuangxialie (任务/NLP 碎片拼贴, 38% 污染源, dev-notes 21/29)
                        dsh          (多轮度 0.09 = CI 命令回显, 非对话噪音)
                        shareai      (多轮度 0.08 = 单轮谜题/问答任务)
  - 降权 train 侧:     multi_turn    (445字/条、咨询词99%、d1=0.0005 = 极端话术固化,
                        保留 15% 作为"倾听/共情"能力来源)
  - 全量保留:          lccc(真实短闲聊) glm(mid对话) zhihu_kol(长篇连贯)
                        muice  kdconv  identity  dailychat(优质小样本)

工作流:
  1. 备份原始 bin/manifest/txt 到 data/chinese/_backup_<ts>/   (先拍快照, 可回滚)
  2. 把剔除源移入 data/chinese/_discarded/  (prepare.py 遍历 *.txt, 移走即不参与)
  3. 用 --source-ratio multi_turn=0.15 + --char-level 重跑 prepare.py
  4. 打印治理前后 token/样本对比表 + 写治理报告
"""
import os, shutil, sys, subprocess, time, json, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根
DATA = os.path.join(ROOT, "data", "chinese")

DISCARD = ["zhuangxialie_dialogue.txt", "dsh_dialogue.txt", "shareai_dialogue.txt"]
DOWNWEIGHT = {"multi_turn_dialogue.txt": 0.15}   # train 侧降权比例
# 其余 *_dialogue.txt 全量保留

GEN_FILES = ["train.bin", "train_char.bin", "train_byte.bin",
             "val.bin", "val_char.bin", "val_byte.bin",
             "manifest.json", "meta.pkl", "meta_char.pkl", "meta_byte.pkl"]

def load_manifest_counts():
    p = os.path.join(DATA, "manifest.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        m = json.load(f)
    c = m.get("counts", {})
    return {"samples": c.get("train_samples", 0), "tokens": c.get("train_tokens", 0),
            "chars": c.get("train_chars", 0)}

def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    bdir = os.path.join(DATA, f"_backup_{ts}")
    ddir = os.path.join(DATA, "_discarded")
    os.makedirs(bdir, exist_ok=True)
    os.makedirs(ddir, exist_ok=True)

    before = load_manifest_counts()

    # 1) 备份生成的 bin/manifest/meta
    for fn in GEN_FILES:
        p = os.path.join(DATA, fn)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(bdir, fn))
    # 备份全部 txt (快照到本次 _backup_<ts>, 时间戳隔离, 可多代共存)
    for fn in os.listdir(DATA):
        if fn.endswith(".txt"):
            shutil.copy2(os.path.join(DATA, fn), os.path.join(bdir, fn))
    # 备份 manifest 描述文件
    with open(os.path.join(bdir, "GOVERNANCE.txt"), "w", encoding="utf-8") as f:
        f.write(f"备份时间: {ts}\n治理动作: 剔除{DISCARD} / 降权{DOWNWEIGHT}\n")
        f.write("还原: 把 _discarded/* 移回 data/chinese/, 把 _backup_<ts>/* 拷回 data/chinese/\n")

    # 2) 移动剔除源
    for fn in DISCARD:
        src = os.path.join(DATA, fn)
        if os.path.exists(src):
            shutil.move(src, os.path.join(ddir, fn))
            print(f"  [剔除] {fn} -> _discarded/")
        else:
            print(f"  [跳过] {fn} 不存在")

    # 3) 重跑 prepare.py
    ratio_args = []
    for name, r in DOWNWEIGHT.items():
        ratio_args += ["--source-ratio", f"{name}={r}"]
    cmd = [sys.executable, os.path.join(ROOT, "data", "chinese", "prepare.py"),
           "--char-level", "--insert-eos"] + ratio_args
    print("  ▶ 执行:", " ".join(cmd))
    env = os.environ.copy()
    ret = subprocess.call(cmd, env=env, cwd=ROOT)
    if ret != 0:
        print("  ❌ prepare.py 失败, 开始回滚...")
        rollback(bdir, ddir)
        sys.exit(1)

    # 4) 治理报告
    after = load_manifest_counts()
    print("\n===== 治理前后 (train 侧) =====")
    for k in ("samples", "tokens", "chars"):
        print(f"  {k:<8}: {before.get(k,0):>12,} -> {after.get(k,0):>12,}  "
              f"({(after.get(k,0)/max(before.get(k,0),1)-1)*100:+.1f}%)")
    print("\n✅ 治理完成。备份在:", bdir)

def rollback(bdir, ddir):
    for fn in os.listdir(ddir):
        shutil.move(os.path.join(ddir, fn), os.path.join(DATA, fn))
    print("  已把 _discarded/* 移回 data/chinese/")
    print("  请手动把 _backup_<ts>/* 拷回 data/chinese/ 还原 bin/manifest")

if __name__ == "__main__":
    main()