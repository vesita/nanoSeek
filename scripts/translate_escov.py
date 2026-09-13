#!/usr/bin/env python3
"""Translate the English ESConv-style multi-turn corpus (adafny123/visual_noval_atri
`fine-tune_dst.json`) into Chinese, message by message, preserving structure exactly.

MUST be run with the *independent* translation venv, never the training .venv:

    ~/.venvs/mt/bin/python scripts/translate_escov.py <mode> [options]

Modes
-----
  bench      translate the first N dialogues, report msg/s and full-corpus ETA
  canary     known-answer set + determinism + Chinese cross-talk control
  run        full translation with jsonl checkpoint / resume
  assemble   jsonl -> escov_zh.json + escov_zh.txt (project corpus format)
  validate   structural check of the assembled json + negative controls

Rationale for decode settings (all measured, see analysis/translate_escov.md):
  * MarianMTModel (`Helsinki-NLP/opus-mt-en-zh`) on torch CPU.
  * transformers must be 4.x: 5.x produces重复 loops that never emit EOS.
  * no_repeat_ngram_size=4 is REQUIRED: without it the model loops until
    max_new_tokens on short inputs ("Hello" -> 200x"哈") and is ~14x slower.
  * CPU threads: 4 is fastest; 12 threads is ~4x SLOWER (contention).
  * USE_TF=0 so transformers does not drag TensorFlow (installed only for the
    ModelScope CSANMT candidate) into the Marian path.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "chinese" / "new_dialogue"
WORK = Path("/tmp/mt_work")
JSONL = WORK / "escov_zh.jsonl"
META = WORK / "run_meta.json"

INPUT_CANDIDATES = [
    Path("/home/vesita/datasets/NLP/escov_en_1300_multiturn.json"),
]

MODEL_B = "Helsinki-NLP/opus-mt-en-zh"
MODEL_A = "damo/nlp_csanmt_translation_en2zh"

BATCH = 32
THREADS = 4
MAX_NEW = 512


# --------------------------------------------------------------------------- #
# input
# --------------------------------------------------------------------------- #
def find_input() -> Path:
    """Locate the English source corpus (never hard-code a cache hash)."""
    for p in INPUT_CANDIDATES:
        if p.is_file():
            return p
    # legacy ModelScope cache: metadata stub holds the url; data is the stub minus .json
    import glob

    hub = Path(os.path.expanduser("~/.cache/modelscope/hub/datasets"))
    hits = []
    for pat in (str(hub / "*.json"), str(hub / "downloads" / "*.json")):
        for f in glob.glob(pat):
            try:
                url = str(json.load(open(f)).get("url", ""))
            except Exception:
                continue
            if "visual_noval_atri" in url and "fine-tune_dst" in url:
                hits.append(Path(f[:-5]))  # strip .json -> real payload
    for h in hits:
        if h.is_file():
            return h
    raise SystemExit("input corpus not found; edit INPUT_CANDIDATES")


def load_dialogues() -> list[dict]:
    p = find_input()
    print(f"[input] {p} ({p.stat().st_size} bytes)", flush=True)
    return json.load(open(p))


def flatten(dialogues: list[dict]) -> tuple[list[str], list[tuple[int, int]]]:
    """Global message order -> (texts, [(dialogue_idx, msg_idx), ...])."""
    texts, where = [], []
    for di, d in enumerate(dialogues):
        for mi, m in enumerate(d["messages"]):
            texts.append(m["content"])
            where.append((di, mi))
    return texts, where


# --------------------------------------------------------------------------- #
# text post-processing
# --------------------------------------------------------------------------- #
_CJK = r"\u4e00-\u9fff\u3400-\u4dbf"
_CJK_PUNCT = r"\u3000-\u303f\uff01-\uff65"
_LATIN_TAIL = re.compile(r"^[A-Za-z0-9\s.,!?'\"\\/|*\[\]()+-]*$")


def normalize_zh(t: str) -> str:
    """Whitespace hygiene. Inter-CJK spaces are KEPT: the project corpus
    (multi_turn_dialogue.txt) contains them too, so this stays in-distribution."""
    t = t.replace("\u2581", " ")
    t = re.sub(r"[ \t\u00a0]+", " ", t)
    t = re.sub(rf"\s+(?=[{_CJK_PUNCT}])", "", t)  # "你好 。" -> "你好。"
    t = re.sub(rf"(?<=[{_CJK_PUNCT}])\s+(?=[{_CJK_PUNCT}])", "", t)
    return t.strip()


def collapse_degeneration(s: str) -> tuple[str, str]:
    """Collapse decoder repetition loops, iterated to a fixpoint. Returns
    (text, rule) with rule '' if untouched.

    Conservative by construction: a doubled 2-char unit is only collapsed when the
    whole string is the repetition, or the leftover tail is pure Latin/punctuation
    (the "你好。你好。Hello." echo pattern). Legit emphasis such as
    "不是不是，我是说……" is deliberately left alone.

    The fixpoint matters: '嗨 嗨 嗨 嗨嗨 嗨 嗨 嗨' collapses to the *unit*
    '嗨 嗨 嗨 嗨', which is itself degenerate and must be collapsed again to '嗨'.
    """
    t = s.strip()
    out = t
    rule = ""
    for _ in range(6):
        nxt, r = _collapse_once(out)
        if not r or nxt == out:
            break
        out, rule = nxt, (rule + "+" + r if rule else r)
    if out != t and not rule:
        rule = "changed"
    return out, rule


def _collapse_once(t: str) -> tuple[str, str]:
    t = t.strip()
    if not t:
        return t, ""
    # (1) whole string is a periodic repetition
    for u in range(1, 17):
        if len(t) >= 2 * u and len(t) % u == 0 and t == t[:u] * (len(t) // u):
            reps = len(t) // u
            if reps >= 3 or (reps == 2 and u >= 2):
                unit = t[:u].strip().rstrip("，,、")
                if unit and unit != t:
                    return unit, f"periodic-u{u}-r{reps}"
            break
    # (2) leading immediate repetition, leftover is Latin/punct only
    m = re.match(r"^(.{2,16}?)\1+", t)
    if m:
        unit = m.group(1)
        reps = len(m.group(0)) // len(unit)
        rest = t[m.end():].strip()
        if reps >= 2 and _LATIN_TAIL.match(rest):
            u2 = unit.strip().rstrip("，,、")
            if u2 and u2 != t:
                return u2, f"leading-u{len(unit)}-r{reps}"
    toks = t.split()
    # (3) every whitespace token identical
    if len(toks) >= 2 and len(set(toks)) == 1:
        return toks[0].strip().rstrip("，,、"), f"same-token-x{len(toks)}"
    # (4) one token dominates and nothing else of substance survives
    if len(toks) >= 3:
        tok, n = Counter(toks).most_common(1)[0]
        if n >= 3 and n / len(toks) >= 2 / 3:
            kept = [x for x in toks if x != tok]
            if len(kept) <= 1:
                return tok.strip().rstrip("，,、"), f"dominant-{n}/{len(toks)}"
    return t, ""


# --------------------------------------------------------------------------- #
# backend B: Helsinki-NLP/opus-mt-en-zh
# --------------------------------------------------------------------------- #
class MarianBackend:
    name = MODEL_B

    def __init__(self, threads: int = THREADS, batch: int = BATCH, beams: int = 4):
        import torch
        from transformers import MarianMTModel, MarianTokenizer

        torch.set_num_threads(threads)
        self.torch = torch
        self.batch = batch
        self.beams = beams
        self.tok = MarianTokenizer.from_pretrained(MODEL_B)
        self.model = MarianMTModel.from_pretrained(MODEL_B).eval()
        self.stats = Counter()

    def translate_batch(self, texts: list[str]) -> list[str]:
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        maxnew = min(MAX_NEW, max(64, 3 * int(enc["input_ids"].shape[1])))
        with self.torch.no_grad():
            out = self.model.generate(
                **enc,
                num_beams=self.beams,
                no_repeat_ngram_size=4,
                max_new_tokens=maxnew,
                do_sample=False,
            )
        raw = self.tok.batch_decode(out, skip_special_tokens=True)
        res = []
        for r in raw:
            r = normalize_zh(r)
            r, rule = collapse_degeneration(r)
            if rule:
                self.stats[rule] += 1
            self.stats["total"] += 1
            res.append(r)
        return self._retry_empty(texts, res)

    def _retry_empty(self, texts, res):
        """Bare noun phrases with no terminal punctuation (e.g. 'Maple Syrup Urine
        Disease') make the decoder emit EOS immediately -> empty string.  Deterministic
        repair: append a period (Marian is trained on punctuated sentences) and redo."""
        empt = [k for k, r in enumerate(res) if not r.strip()]
        if not empt:
            return res
        self.stats["empty_initial"] += len(empt)
        retry = []
        for k in empt:
            t = texts[k].rstrip()
            retry.append(t if re.search(r"[.!?]\s*$", t) else t + ".")
        enc = self.tok(retry, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with self.torch.no_grad():
            out = self.model.generate(
                **enc, num_beams=self.beams, no_repeat_ngram_size=4,
                max_new_tokens=64, do_sample=False,
            )
        for k, r in zip(empt, self.tok.batch_decode(out, skip_special_tokens=True)):
            r, rule = collapse_degeneration(normalize_zh(r))
            if rule:
                self.stats[rule] += 1
            res[k] = r
            self.stats["empty_repaired"] += 1
        return res


# --------------------------------------------------------------------------- #
# backend A: ModelScope CSANMT (TensorFlow)
# --------------------------------------------------------------------------- #
class CsanmtBackend:
    name = MODEL_A

    def __init__(self):
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        self.p = pipeline(task=Tasks.translation, model=MODEL_A)
        self.stats = Counter()

    def translate_batch(self, texts: list[str]) -> list[str]:
        res = []
        for t in texts:
            out = self.p(t)
            zh = out["translation"] if isinstance(out, dict) else str(out)
            zh, rule = collapse_degeneration(normalize_zh(zh))
            if rule:
                self.stats[rule] += 1
            self.stats["total"] += 1
            res.append(zh)
        return res


def get_backend(which: str, **kw):
    return CsanmtBackend() if which == "A" else MarianBackend(**kw)


# --------------------------------------------------------------------------- #
# jsonl checkpoint / resume
# --------------------------------------------------------------------------- #
def load_progress(path: Path) -> tuple[dict[int, str], int]:
    done: dict[int, str] = {}
    if path.exists():
        for line in path.open():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a torn final line
            done[int(r["i"])] = r["zh"]
    k = 0
    while k in done:
        k += 1
    return done, k


def truncate_progress(path: Path, done: dict[int, str], keep: int) -> None:
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for i in range(keep):
            f.write(json.dumps({"i": i, "zh": done[i]}, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# structural validation
# --------------------------------------------------------------------------- #
def check_dialogue(src_msgs: list[dict], zh_msgs: list[dict]) -> list[str]:
    """Return a list of structural problems ([] == OK)."""
    problems: list[str] = []
    if len(src_msgs) != len(zh_msgs):
        problems.append(f"message count {len(zh_msgs)} != {len(src_msgs)}")
        return problems  # per-message checks are meaningless after a count mismatch
    for j, (a, b) in enumerate(zip(src_msgs, zh_msgs)):
        if a.get("role") != b.get("role"):
            problems.append(f"role[{j}] {b.get('role')!r} != {a.get('role')!r}")
        if not str(b.get("content", "")).strip():
            problems.append(f"empty content[{j}]")
    return problems


def validate_all(orig: list[dict], trans: list[dict]) -> dict:
    if len(orig) != len(trans):
        return {"fatal": f"dialogue count {len(trans)} != {len(orig)}"}
    viol = {}
    for i, (a, b) in enumerate(zip(orig, trans)):
        p = check_dialogue(a["messages"], b["messages"])
        if p:
            viol[i] = p
    return {
        "dialogues": len(orig),
        "ok": len(orig) - len(viol),
        "violating": len(viol),
        "examples": {str(k): v for k, v in list(viol.items())[:5]},
    }


def negative_controls(trans: list[dict]) -> dict:
    """Deliberately corrupt copies and assert the checker CATCHES them."""
    import copy
    import random

    out = {}
    # c1: swap two adjacent messages with different roles (guaranteed role-seq change)
    bad = copy.deepcopy(trans)
    msgs = bad[0]["messages"]
    for j in range(len(msgs) - 1):
        if msgs[j]["role"] != msgs[j + 1]["role"]:
            msgs[j], msgs[j + 1] = msgs[j + 1], msgs[j]
            break
    out["swap_adjacent"] = check_dialogue(trans[0]["messages"], bad[0]["messages"])
    # c2: shuffle one dialogue's messages
    bad = copy.deepcopy(trans)
    random.Random(1234).shuffle(bad[1]["messages"])
    out["shuffle_dialogue"] = check_dialogue(trans[1]["messages"], bad[1]["messages"])
    # c3: drop one message
    bad = copy.deepcopy(trans)
    bad[2]["messages"].pop()
    out["drop_message"] = check_dialogue(trans[2]["messages"], bad[2]["messages"])
    # c4: blank out a content
    bad = copy.deepcopy(trans)
    bad[3]["messages"][2]["content"] = "   "
    out["blank_content"] = check_dialogue(trans[3]["messages"], bad[3]["messages"])
    out["all_caught"] = all(v for v in out.values() if isinstance(v, list))
    return out


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #
def mode_bench(args):
    dialogues = load_dialogues()
    n = args.dialogues
    texts, where = flatten(dialogues[:n])
    print(f"[bench] {n} dialogues -> {len(texts)} messages", flush=True)
    be = get_backend(args.model, threads=args.threads, batch=args.batch)
    t0 = time.time()
    out_lens: list[int] = []
    for bi, i in enumerate(range(0, len(texts), args.batch)):
        chunk = texts[i:i + args.batch]
        tb = time.time()
        res = be.translate_batch(chunk)
        out_lens += [len(r) for r in res]
        print(f"  batch {bi:3d} i={i:5d} {time.time()-tb:6.2f}s "
              f"maxout={max(len(r) for r in res):4d} elapsed={time.time()-t0:6.1f}s",
              flush=True)
    dt = time.time() - t0
    import statistics
    print(f"[bench] out chars mean={statistics.mean(out_lens):.1f} "
          f"median={statistics.median(out_lens)} max={max(out_lens)} "
          f"hit_cap={sum(1 for x in out_lens if x >= MAX_NEW * 0.9)}", flush=True)
    total_msgs = sum(len(d["messages"]) for d in dialogues)
    rate = len(texts) / dt
    eta = total_msgs / rate
    print(f"[bench] model={be.name} threads={args.threads} batch={args.batch}")
    print(f"[bench] {len(texts)} msgs in {dt:.1f}s = {rate:.2f} msg/s")
    print(f"[bench] FULL corpus {total_msgs} msgs -> ETA {eta/60:.1f} min ({eta/3600:.2f} h)")
    print(f"[bench] collapse rules fired: {dict(be.stats)}")
    META.parent.mkdir(parents=True, exist_ok=True)
    META.write_text(json.dumps({
        "model": be.name, "threads": args.threads, "batch": args.batch,
        "bench_msgs": len(texts), "bench_seconds": dt, "rate_msg_s": rate,
        "full_msgs": total_msgs, "eta_seconds": eta, "collapse": dict(be.stats),
    }, ensure_ascii=False, indent=1))


KNOWN_ANSWERS = [
    ("Hello", {"你好", "您好", "嗨", "哈罗", "你好!", "你好。"}),
    ("How are you?", {"你好吗", "你还好吗", "你好吗?", "你好吗？", "你怎么样"}),
    ("Thank you.", {"谢谢", "谢谢你", "感谢你", "谢谢。"}),
    ("Good morning.", {"早上好", "早安", "早上好。", "早上好!"}),
    ("I love you.", {"我爱你", "我爱你。", "我爱你!"}),
]

CROSSTALK = [
    "我今天心情不太好，有点焦虑。",
    "谢谢你一直陪着我，听我说这些。",
    "你能告诉我该怎么办吗？",
]


def _norm_ans(s: str) -> str:
    s = re.sub(r"\s+", "", s)
    s = s.replace("?", "？").replace("!", "！").replace(",", "，")
    return s.strip("。？！ ").lower()


def mode_canary(args):
    dialogues = load_dialogues()
    be = get_backend(args.model, threads=args.threads, batch=args.batch)
    rep: dict = {"model": be.name}

    # ---- 1. known answers -------------------------------------------------
    print("\n=== 1. known-answer control ===", flush=True)
    src = [a for a, _ in KNOWN_ANSWERS]
    got = []
    for i in range(0, len(src), args.batch):
        got += be.translate_batch(src[i:i + args.batch])
    ka = []
    npass = 0
    for (en, ok), zh in zip(KNOWN_ANSWERS, got):
        hit = _norm_ans(zh) in {_norm_ans(x) for x in ok}
        npass += hit
        ka.append({"en": en, "zh": zh, "accept": sorted(ok), "pass": hit})
        print(f"  [{'PASS' if hit else 'FAIL'}] {en!r} -> {zh!r}   accept={sorted(ok)}", flush=True)
    rep["known_answers"] = {"cases": ka, "pass": npass, "total": len(ka)}
    print(f"  -> {npass}/{len(ka)} pass", flush=True)

    # ---- 2. determinism ---------------------------------------------------
    print("\n=== 2. determinism (same input twice) ===", flush=True)
    probe = [m["content"] for d in dialogues[:3] for m in d["messages"]][:40]
    a1 = []
    for i in range(0, len(probe), args.batch):
        a1 += be.translate_batch(probe[i:i + args.batch])
    a2 = []
    for i in range(0, len(probe), args.batch):
        a2 += be.translate_batch(probe[i:i + args.batch])
    diff = [(p, x, y) for p, x, y in zip(probe, a1, a2) if x != y]
    # batch-composition sensitivity: same texts, one big batch vs small batches
    one = be.translate_batch(probe) if len(probe) <= 64 else None
    comp = sum(1 for x, y in zip(a1, one) if x != y) if one else None
    rep["determinism"] = {
        "n": len(probe), "identical": len(probe) - len(diff),
        "mismatches": diff[:3],
        "batch_composition_mismatches": comp,
    }
    print(f"  repeat-run identical: {len(probe)-len(diff)}/{len(probe)}"
          f"  | batch-composition mismatches: {comp}", flush=True)

    # ---- 3. Chinese cross-talk -------------------------------------------
    print("\n=== 3. Chinese cross-talk control (zh -> en2zh model) ===", flush=True)
    ct = be.translate_batch(CROSSTALK)
    rows = []
    for a, b in zip(CROSSTALK, ct):
        same = _norm_ans(a) == _norm_ans(b)
        keeps_cjk = bool(re.search(rf"[{_CJK}]", b))
        rows.append({"zh_in": a, "zh_out": b, "identical": same, "still_cjk": keeps_cjk})
        print(f"  in : {a}\n  out: {b}\n  identical={same} still_cjk={keeps_cjk}", flush=True)
    rep["crosstalk"] = {"rows": rows, "identical": sum(r["identical"] for r in rows),
                        "harmed": sum(not r["identical"] for r in rows)}

    # ---- 4. structural negative controls on a synthetic copy -------------
    print("\n=== 4. structural negative controls ===", flush=True)
    good = [{"messages": [{"role": m["role"], "content": "中文" + str(i)}
                          for i, m in enumerate(d["messages"], 1)]} for d in dialogues[:6]]
    nc = negative_controls(good)
    for k, v in nc.items():
        if isinstance(v, list):
            print(f"  {k}: {'CAUGHT' if v else 'MISSED'}  {v[:2]}", flush=True)
    print(f"  all_caught={nc['all_caught']}", flush=True)
    rep["negative_controls"] = nc
    (WORK / "canary.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(f"\n[canary] written {WORK/'canary.json'}", flush=True)
    return rep


def mode_run(args):
    dialogues = load_dialogues()
    texts, _ = flatten(dialogues)
    total = len(texts)
    limit = min(args.limit, total) if args.limit else total
    WORK.mkdir(parents=True, exist_ok=True)

    done, k = load_progress(JSONL)
    start = (k // args.batch) * args.batch
    if start:
        truncate_progress(JSONL, done, start)
    print(f"[run] total={total} limit={limit} resume_at={start} "
          f"(contiguous done={k})", flush=True)

    be = get_backend(args.model, threads=args.threads, batch=args.batch)
    t0 = time.time()
    processed = 0
    with JSONL.open("a") as f:
        i = start
        while i < limit:
            chunk = texts[i:min(i + args.batch, limit)]
            out = be.translate_batch(chunk)
            for j, zh in enumerate(out):
                f.write(json.dumps({"i": i + j, "zh": zh}, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
            i += len(chunk)
            processed += len(chunk)
            if processed % (args.batch * 20) == 0 or i >= limit:
                dt = time.time() - t0
                rate = processed / dt if dt else 0
                left = (limit - i) / rate if rate else 0
                print(f"  {i}/{limit}  {rate:.2f} msg/s  elapsed={dt/60:.1f}min  "
                      f"eta={left/60:.1f}min", flush=True)
    dt = time.time() - t0
    print(f"[run] done {processed} msgs in {dt/60:.2f} min "
          f"({processed/dt:.2f} msg/s); collapse={dict(be.stats)}", flush=True)

    # a completed run is not finished until no message is empty (see _retry_empty)
    done2, _ = load_progress(JSONL)
    empties = [i for i in range(limit) if not str(done2.get(i, "")).strip()]
    if empties:
        print(f"[run] repairing {len(empties)} empty outputs: {empties}", flush=True)
        fixed = be.translate_batch([texts[i] for i in empties])
        for i, zh in zip(empties, fixed):
            done2[i] = zh
        truncate_progress(JSONL, done2, limit)
        still = [i for i in empties if not done2[i].strip()]
        print(f"[run] after repair, still empty: {still}", flush=True)

    (WORK / "run_stats.json").write_text(json.dumps({
        "model": be.name, "processed": processed, "seconds": dt,
        "rate_msg_s": processed / dt, "collapse": dict(be.stats),
        "bulk_seconds": dt, "bulk_first": None,
    }, ensure_ascii=False, indent=1))


def mode_repass(args):
    """Re-apply collapse_degeneration (now iterated to a fixpoint) to every stored
    translation. A no-op for converged values; repairs values left degenerate by the
    earlier single-pass collapse. Deterministic text-only transform, no model needed."""
    done, k = load_progress(JSONL)
    if k != len(done):
        raise SystemExit(f"jsonl not contiguous: {k} vs {len(done)}")
    changed = []
    for i in range(k):
        new, rule = collapse_degeneration(done[i])
        if new != done[i]:
            changed.append((i, done[i], new, rule))
            done[i] = new
    truncate_progress(JSONL, done, k)
    print(f"[repass] {len(changed)}/{k} translations changed")
    for i, a, b, r in changed[:25]:
        print(f"   [{i}] {a[:45]!r} -> {b[:45]!r}  ({r})")


def mode_assemble(args):
    dialogues = load_dialogues()
    texts, where = flatten(dialogues)
    done, k = load_progress(JSONL)
    if k < len(texts):
        raise SystemExit(f"jsonl incomplete: {k}/{len(texts)} messages")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out = []
    for d in dialogues:
        out.append({"messages": [{"content": "", "role": m["role"]} for m in d["messages"]]})
    for i, (di, mi) in enumerate(where):
        out[di]["messages"][mi]["content"] = done[i]

    jp = OUT_DIR / "escov_zh.json"
    jp.write_text(json.dumps(out, ensure_ascii=False, indent=1))

    tp = OUT_DIR / "escov_zh.txt"
    with tp.open("w") as f:
        for d in out:
            for m in d["messages"]:
                f.write(("用户：" if m["role"] == "user" else "模型：") + m["content"] + "\n")
            f.write("\n")
    print(f"[assemble] {jp} ({jp.stat().st_size} bytes)")
    print(f"[assemble] {tp} ({tp.stat().st_size} bytes)")
    v = validate_all(dialogues, out)
    print(f"[assemble] structure: ok={v['ok']}/{v['dialogues']} violating={v['violating']}")
    if v.get("examples"):
        print(f"[assemble] examples: {v['examples']}")


def mode_validate(args):
    dialogues = load_dialogues()
    trans = json.load(open(OUT_DIR / "escov_zh.json"))
    v = validate_all(dialogues, trans)
    print(f"[validate] dialogues={v['dialogues']} ok={v['ok']} violating={v['violating']}")
    if v.get("examples"):
        print(f"[validate] examples={v['examples']}")

    # labels / txt cross-check
    txt = (OUT_DIR / "escov_zh.txt").read_text().split("\n\n")
    blocks = [b for b in txt if b.strip()]
    print(f"[validate] txt blocks={len(blocks)} (expect {len(dialogues)})")
    bad_labels = 0
    for d, b in zip(trans, blocks):
        lines = b.split("\n")
        if len(lines) != len(d["messages"]):
            bad_labels += 1
            continue
        for m, ln in zip(d["messages"], lines):
            want = "用户：" if m["role"] == "user" else "模型："
            if not ln.startswith(want):
                bad_labels += 1
                break
    print(f"[validate] txt blocks with wrong line/label structure: {bad_labels}")

    # content hygiene: newlines would break the line-per-message .txt format
    allc = [m["content"] for d in trans for m in d["messages"]]
    n_nl = sum(1 for c in allc if "\n" in c or "\r" in c)
    lens = [len(c) for c in allc]
    print(f"[validate] contents={len(allc)} with-newline={n_nl} "
          f"empty={sum(1 for c in allc if not c.strip())} "
          f"len min/mean/max={min(lens)}/{sum(lens)/len(lens):.1f}/{max(lens)}")
    cjk = sum(1 for c in allc if re.search(rf"[{_CJK}]", c))
    print(f"[validate] contents containing CJK: {cjk}/{len(allc)} = {100*cjk/len(allc):.1f}%")
    if n_nl:
        print("[validate] FATAL: newline inside content would corrupt escov_zh.txt")
        sys.exit(1)

    nc = negative_controls(trans)
    print("[validate] negative controls:")
    for k2, val in nc.items():
        if isinstance(val, list):
            print(f"   {k2}: {'CAUGHT' if val else 'MISSED'}  {val[:2]}")
    print(f"   all_caught={nc['all_caught']}")
    if not nc["all_caught"]:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["bench", "canary", "run", "repass", "assemble", "validate"])
    ap.add_argument("--model", default="B", choices=["A", "B"])
    ap.add_argument("--threads", type=int, default=THREADS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--dialogues", type=int, default=50, help="bench: how many dialogues")
    ap.add_argument("--limit", type=int, default=0, help="run: cap messages")
    args = ap.parse_args()
    {"bench": mode_bench, "canary": mode_canary, "run": mode_run, "repass": mode_repass,
     "assemble": mode_assemble, "validate": mode_validate}[args.mode](args)


if __name__ == "__main__":
    main()
