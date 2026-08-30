"""下载 GLM 开放对话中文子集（HF 流式）→ glm_dialogue.txt。

svjack/GLM-Open-Dialogue-Chinese-Dataset：口语化单轮中文对话（"嘿,吉姆,晚餐后去喝几瓶啤酒怎么样?"）。
只取 cate=='gen'（自然对话生成）；choose/reconstruct 是任务类，跳过。
输出格式与其他语料一致：用户：<上下文>\n模型：<回复>，样本间空行。

用法：uv run python data/chinese/download_glm_chat.py [--n 60000] [--force]
"""
import argparse
import os
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent
OUT = DATA_DIR / "glm_dialogue.txt"

TMPL_PREFIX = "根据上下文，得到后续的对话"
RE_CTX = re.compile(r"上下文：?(.*?)(?:答案：|$)", re.S)


def clean_source(src: str) -> list:
    """从模板里抠出上下文并按 [SEP] 拆段，返回口语多轮段列表（首段=用户）。

    模板形如：'…根据上下文，得到后续的对话：\n上下文：轮1[SEP]轮2…\n答案：\n'
    注意模板里也有"上下文"字样 → 用 rfind 定位最后一个"上下文："。
    gen 样本的上下文是同一对话流不断截断增长的（[SEP] 分隔历史轮）。
    """
    i = src.rfind("上下文：")
    s = src[i + len("上下文："):] if i >= 0 else src.replace(TMPL_PREFIX, "")
    j = s.find("答案：")
    if j >= 0:
        s = s[:j]
    s = re.sub(r"\s+", " ", s).strip()
    segs = [x.strip() for x in re.split(r"\[SEP\]", s) if len(x.strip()) >= 2]
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60000, help="目标提取条数")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if OUT.exists() and not a.force:
        print(f"已存在 {OUT}（{OUT.stat().st_size/1e6:.1f} MB），跳过。--force 重下")
        return

    import datasets
    ds = datasets.load_dataset("svjack/GLM-Open-Dialogue-Chinese-Dataset",
                               split="train", streaming=True)
    tmp = OUT.with_suffix(".txt.tmp")
    written, skipped, bad = 0, 0, 0
    with open(tmp, "w", encoding="utf-8") as f:
        for ex in ds:
            if ex["cate"] != "gen":
                skipped += 1
                continue
            segs = clean_source(ex["source_text"])
            a_ = str(ex["target_text"]).strip()
            if not segs:
                bad += 1
                continue
            if len(segs) % 2 == 0:
                # 末段=模型轮 → 直接当回复（保留交替合法性，target 弃——它是下一条前缀的预告）
                if not (5 <= len(segs[-1]) <= 500):
                    bad += 1
                    continue
                turns = [f"{'用户' if i % 2 == 0 else '模型'}：{seg}" for i, seg in enumerate(segs)]
            else:
                # 末段=用户轮 → target 当模型回复
                if not (5 <= len(a_) <= 500):
                    bad += 1
                    continue
                turns = [f"{'用户' if i % 2 == 0 else '模型'}：{seg}" for i, seg in enumerate(segs)]
                turns.append(f"模型：{a_}")
            f.write("\n".join(turns) + "\n\n")
            written += 1
            if written >= a.n:
                break
    tmp.rename(OUT)
    print(f"完成：{written} 条（跳过 {skipped} 非 gen，{bad} 过滤）→ {OUT}")


if __name__ == "__main__":
    main()
