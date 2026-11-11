"""P1 (bảng nhầm lẫn + LLM chấm câu) và P2 (ghép N-best + LLM chấm lại) trên output `run.py --no-llm` (có ref) -> WER.

LLM không viết câu mới, chỉ chấm log-prob cả câu (lifemate/lmscore.py):
  P1: mỗi từ của câu ASR chính có ứng viên từ (a) cặp lỗi thật ASR -> đáp án của train-split (dump eval_asr.py) và
      (b) đồng âm theo giọng: d/gi/r, s/x, ch/tr, l/n, -n/-ng, -c/-t, hỏi/ngã (chỉ giữ từ có trong từ vựng train-split).
      Câu thay một từ thắng câu gốc >= tau (log-prob) và từ đó có conf âm học < cmax -> đổi (tối đa 3 chỗ, không kề nhau).
  P2: giả thuyết = các bản decode khác nhau của cùng audio (nhiều cấu hình run.py + bản không bias);
      điểm = LLM + alpha * số cấu hình ra cùng câu + gamma * log conf âm học -> chọn điểm cao nhất.
Chấm điểm 1 lần (GPU), lưu scores.jsonl; dò tham số offline (CPU), báo cả số tốt nhất trên held-out (lạc quan) và
cross-validation 2-fold (chọn tham số trên một nửa mẫu, đo nửa kia).

    python scripts/asr_rescore.py --hyps main=nollm_main.jsonl b8=nollm_b8.jsonl ... --dump data/asr_train_plain.jsonl \\
        --out-dir out/rescore
    python scripts/asr_rescore.py --sweep-only --out-dir out/rescore      # dò lại từ scores.jsonl, không cần GPU
"""
import argparse
import itertools
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lifemate.text import is_heldout, norm, wer_counts  # noqa: E402
from lifemate.vi_num import to_spoken  # noqa: E402

NUM_WORDS = set("không một hai ba bốn năm sáu bảy tám chín mười mươi trăm nghìn ngàn triệu tỷ tỉ linh lẻ tư lăm mốt "
                "rưỡi chục".split())
_TONE = {"̀": "1", "́": "2", "̉": "3", "̃": "3", "̣": "5"}  # hỏi = ngã (3)
_ONSET = re.compile(r"^(ngh|ng|nh|ch|gh|gi|kh|ph|th|tr|qu|[bcdđghklmnpqrstvx])?(.*)$")
_ONSET_GROUP = {"d": "z", "gi": "z", "r": "z", "s": "s", "x": "s", "ch": "c", "tr": "c", "l": "l", "n": "l"}


def loose(word):
    """khoá gộp các cặp đồng âm theo giọng: d/gi/r, s/x, ch/tr, l/n, -ng/-n, -c/-t, hỏi/ngã."""
    d = unicodedata.normalize("NFD", word)
    tone = "".join(_TONE.get(c, "") for c in d)
    base = unicodedata.normalize("NFC", "".join(c for c in d if c not in _TONE))
    m = _ONSET.match(base)
    on, rest = m.group(1) or "", m.group(2)
    rest = re.sub(r"ng$", "n", rest)
    rest = re.sub(r"c$", "t", rest)
    return _ONSET_GROUP.get(on, on) + "|" + rest + tone


def align(r, h):
    """căn từ Levenshtein -> [(ref_word | None, hyp_word | None)]."""
    n, m = len(r), len(h)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    out, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (r[i - 1] != h[j - 1]):
            out.append((r[i - 1], h[j - 1]))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            out.append((r[i - 1], None))
            i -= 1
        else:
            out.append((None, h[j - 1]))
            j -= 1
    return out[::-1]


def build_tables(dump, min_pair=2, min_vocab=2):
    """train-split của dump -> (bảng nhầm lẫn {từ ASR: Counter(từ đúng)}, tần suất từ vựng, chỉ mục loose -> từ)."""
    conf, vocab = defaultdict(Counter), Counter()
    for l in open(dump, encoding="utf-8"):
        r = json.loads(l)
        if is_heldout(r["id"]):
            continue
        ref, hyp = norm(r.get("ref_spoken") or r["ref"]).split(), norm(r["hyp"]).split()
        vocab.update(ref)
        for a, b in align(ref, hyp):
            if a and b and a != b:
                conf[b][a] += 1
    conf = {h: Counter({r: c for r, c in cs.items() if c >= min_pair}) for h, cs in conf.items()}
    conf = {h: cs for h, cs in conf.items() if cs}
    idx = defaultdict(list)
    for w, c in vocab.items():
        if c >= min_vocab and w.isalpha():
            idx[loose(w)].append(w)
    return conf, vocab, idx


def candidates(word, conf, vocab, idx, k=6):
    if word in NUM_WORDS or not word.isalpha():
        return []
    out = [w for w, _ in conf.get(word, Counter()).most_common()]
    out += sorted((w for w in idx.get(loose(word), []) if w != word), key=lambda w: -vocab[w])
    return [w for w in dict.fromkeys(out) if w != word][:k]


def load_hyps(specs):
    """name=path ... -> {(id, turn): {"ref", "hyps": {name: (text, conf)}}}; bản không bias của cấu hình đầu = "plain"."""
    turns = {}
    for i, spec in enumerate(specs):
        name, path = spec.split("=", 1)
        for l in open(path, encoding="utf-8"):
            row = json.loads(l)
            for t in row["turns"]:
                if not t.get("ref"):
                    continue
                d = turns.setdefault((row["id"], t["turn"]), {"ref": t["ref"], "hyps": {}})
                d["hyps"][name] = (t["asr"], t.get("conf") or [])
                if i == 0 and "plain" in t:
                    d["hyps"]["plain"] = (t["plain"], t.get("plain_conf") or [])
    first = specs[0].split("=", 1)[0]
    return {k: v for k, v in turns.items() if first in v["hyps"]}, first


def score_all(turns, primary, tables, llm, gpu_mem, k):
    from lifemate.lmscore import SentenceScorer
    conf, vocab, idx = tables
    recs, sents = [], set()
    for (sid, turn), d in sorted(turns.items()):
        text, cf = d["hyps"][primary]
        words = norm(text).split()
        cf = cf if len(cf) == len(words) else [1.0] * len(words)
        pos = []
        for i, w in enumerate(words):
            cs = candidates(w, conf, vocab, idx, k)
            if cs:
                pos.append({"i": i, "w": w, "conf": cf[i], "cands": cs})
                for c in cs:
                    sents.add(" ".join(words[:i] + [c] + words[i + 1:]))
        nb = {}
        for name, (t, c) in d["hyps"].items():
            tn = norm(t)
            e = nb.setdefault(tn, {"text": tn, "configs": [], "ac": None})
            e["configs"].append(name)
            if c and len(c) == len(tn.split()):
                ac = sum(math.log(max(x, 1e-4)) for x in c)
                e["ac"] = ac if e["ac"] is None else max(e["ac"], ac)
            sents.add(tn)
        sents.add(" ".join(words))
        recs.append({"id": sid, "turn": turn, "ref": d["ref"], "primary": " ".join(words), "p1": pos, "nbest": list(nb.values())})
    sents = sorted(sents)
    print(f"{len(recs)} lượt, {len(sents)} câu cần chấm", flush=True)
    sc = SentenceScorer(llm, gpu_mem=gpu_mem)
    lp = dict(zip(sents, sc.score(sents)))
    for r in recs:
        base = lp[r["primary"]][0]
        r["lp"] = base
        words = r["primary"].split()
        for p in r["p1"]:
            p["delta"] = [round(lp[" ".join(words[:p["i"]] + [c] + words[p["i"] + 1:])][0] - base, 3) for c in p["cands"]]
        for e in r["nbest"]:
            e["lp"], e["ntok"] = lp[e["text"]]
    return recs


# ----------------------------------------------------------------------------- dò tham số (CPU)
_wer_cache = {}


def wer_turn(ref, hyp):
    k = (ref, hyp)
    if k not in _wer_cache:
        h = norm(hyp)
        _wer_cache[k] = wer_counts(to_spoken(ref, h) or norm(ref), h)
    return _wer_cache[k]


def wer(recs, pick):
    e = n = 0
    for r in recs:
        a, b = wer_turn(r["ref"], pick(r))
        e, n = e + a, n + b
    return e / max(n, 1)


def p1_pick(tau, cmax, max_subs=3):
    def pick(r):
        words = r["primary"].split()
        best = []
        for p in r["p1"]:
            if p["conf"] >= cmax or not p["delta"]:
                continue
            j = max(range(len(p["delta"])), key=lambda j: p["delta"][j])
            if p["delta"][j] >= tau:
                best.append((p["delta"][j], p["i"], p["cands"][j]))
        used = set()
        for _, i, c in sorted(best, reverse=True):
            if len(used) >= max_subs or {i - 1, i, i + 1} & used:
                continue
            words[i] = c
            used.add(i)
        return " ".join(words)
    return pick


def p2_pick(alpha, gamma, normed, primary):
    def pick(r):
        def s(e):
            lp = e["lp"] / max(e["ntok"], 1) * 10 if normed else e["lp"]
            ac = e["ac"] if e["ac"] is not None else 0.0
            return lp + alpha * len(e["configs"]) + gamma * ac + (1e-6 if primary in e["configs"] else 0)
        return max(r["nbest"], key=s)["text"]
    return pick


def grids(primary):
    p1 = {f"P1 tau={t} cmax={c}": p1_pick(t, c) for t, c in itertools.product([0, 1, 2, 3, 4, 6, 8, 12], [0.3, 0.5, 0.7, 0.9, 1.01])}
    p2 = {f"P2 alpha={a} gamma={g}{' norm' if nm else ''}": p2_pick(a, g, nm, primary)
          for a, g, nm in itertools.product([0, 1, 2, 4, 8, 1000], [0, 0.5, 1, 2, 4], [False, True])}
    return p1, p2


def sweep(recs, primary):
    rows = {"baseline (ASR chính)": wer(recs, lambda r: r["primary"]),
            "oracle N-best (trần P2)": sum(min((wer_turn(r["ref"], e["text"]) for e in r["nbest"]), key=lambda x: x[0])[0]
                                          for r in recs) / max(1, sum(wer_turn(r["ref"], r["primary"])[1] for r in recs))}
    for e_name in sorted({c for r in recs for e in r["nbest"] for c in e["configs"]}):
        rows[f"chỉ cấu hình {e_name}"] = wer(recs, lambda r, n=e_name: next((e["text"] for e in r["nbest"] if n in e["configs"]),
                                                                          r["primary"]))
    p1, p2 = grids(primary)
    fold = lambda r: int(re.search(r"(\d+)$", r["id"]).group(1)) // 10 % 2
    halves = [[r for r in recs if fold(r) == f] for f in (0, 1)]
    out = {"rows": rows, "best": {}, "cv": {}}
    for tag, grid in (("P1", p1), ("P2", p2)):
        full = {k: wer(recs, f) for k, f in grid.items()}
        bk = min(full, key=full.get)
        out["best"][tag] = (bk, full[bk])
        # CV 2-fold: chọn trên nửa này, đo nửa kia; WER gộp theo số từ
        e = n = 0
        chosen = []
        for a, b in ((0, 1), (1, 0)):
            k = min(grid, key=lambda k: wer(halves[a], grid[k]))
            chosen.append(k)
            for r in halves[b]:
                x, y = wer_turn(r["ref"], grid[k](r))
                e, n = e + x, n + y
        out["cv"][tag] = (chosen, e / max(n, 1))
        out[f"grid_{tag}"] = full
    e = n = 0
    for h in halves:
        for r in h:
            x, y = wer_turn(r["ref"], r["primary"])
            e, n = e + x, n + y
    out["cv"]["baseline"] = e / max(n, 1)
    return out


def report(out):
    print("\n== WER held-out")
    for k, v in out["rows"].items():
        print(f"  {k:<40} {v:.4f}")
    for tag in ("P1", "P2"):
        k, v = out["best"][tag]
        print(f"  {tag} tốt nhất trên held-out (lạc quan): {v:.4f}  [{k}]")
        ch, cv = out["cv"][tag]
        print(f"  {tag} cross-validation 2-fold:            {cv:.4f}  (baseline {out['cv']['baseline']:.4f}) [{' | '.join(ch)}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hyps", nargs="+", help="name=path của run.py --no-llm (cái đầu là ASR chính, có --bias)")
    ap.add_argument("--dump", default="data/asr_train_plain.jsonl")
    ap.add_argument("--llm", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--k", type=int, default=6, help="số ứng viên tối đa mỗi từ (P1)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sweep-only", action="store_true")
    a = ap.parse_args()
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    meta = od / "meta.json"
    if not a.sweep_only:
        turns, primary = load_hyps(a.hyps)
        tables = build_tables(a.dump)
        print(f"bảng nhầm lẫn: {len(tables[0])} từ ASR, từ vựng {len(tables[1])}", flush=True)
        recs = score_all(turns, primary, tables, a.llm, a.gpu_mem, a.k)
        with open(od / "scores.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
        json.dump({"primary": primary, "hyps": a.hyps}, open(meta, "w"))
    primary = json.load(open(meta))["primary"]
    recs = [json.loads(l) for l in open(od / "scores.jsonl", encoding="utf-8")]
    out = sweep(recs, primary)
    report(out)
    json.dump(out, open(od / "sweep.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
