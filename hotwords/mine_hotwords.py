"""Khai thác danh sách hotword CỐ ĐỊNH (toàn cục) từ lỗi ASR trên train. Offline, KHÔNG đọc state.json.

    python hotwords/mine_hotwords.py --in data/asr_train_plain.jsonl --train-jsonl DATA/train/train.jsonl
        -> hotwords/hotwords_global.txt  (dòng "CỤM TỪ :score", chữ HOA cho vocab BPE; nạp 1 lần lúc khởi tạo ASR)
        -> hotwords/hotwords_global.json (thống kê từng cụm: n_occ, n_err, err_rate, ASR nghe ra gì, alias để snap_text)

Cách làm:
  1. ref -> dạng đọc (vi_num.to_spoken, chọn cách đọc số theo hyp). Chỉ dùng phần train-split (id % 10 != 9),
     phần held-out (id % 10 == 9) để asr/eval_asr.py đo công bằng.
  2. Ứng viên lấy từ ref viết: mã chữ+số (RON95, VTV3...), viết tắt (SJC, SMS), từ ngoại (check-in, meeting, tivi),
     cụm tên riêng nhiều âm tiết (Sếp Dũng, Đà Nẵng); từ đơn âm tiết tiếng Việt bị loại (dễ kéo sai từ thường).
     Mã có số được sinh thêm theo mẫu (VTV1-9, HTV1-9, SCTV1-20, RON92/95, E5) và đọc theo kiểu Zipformer
     (chữ giữ nguyên, số đọc thành chữ: VTV3 -> "vtv ba").
  3. Căn hyp <-> ref dạng đọc theo từ; đếm số lần cụm bị nghe sai và ASR nghe thành gì.
  4. Giữ cụm có n_err >= --min-err và err_rate >= --min-rate; score = clip(base + k * err_rate, base, max).
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import sys  # noqa: E402

if __package__ in (None, ""):  # chạy trực tiếp: python hotwords/mine_hotwords.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hotwords.bias import _VI_SYL, common_words  # noqa: E402
from textnorm.vi_num import code_variants, to_spoken  # noqa: E402


def is_heldout(sid):
    """train_0019 -> held-out nếu số % 10 == 9 (chia theo sample, mọi lượt của một sample cùng phía)."""
    m = re.search(r"(\d+)$", sid)
    return bool(m) and int(m.group(1)) % 10 == 9


def load_pairs(path, heldout=False):
    """[(id, ref_viết, ref_đọc, hyp)] của train-split (heldout=False) hoặc held-out."""
    out = []
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        if is_heldout(r["id"]) != heldout:
            continue
        hyp = r["hyp"].lower()
        sp = r.get("ref_spoken") or to_spoken(r["ref"], hyp)
        if sp:
            out.append((r["id"], r["ref"], sp, hyp))
    return out


_CODE = re.compile(r"\b(?=[\w-]*\d)(?=[\w-]*[A-Za-z])[A-Za-z][\w-]*\b|\b\d+[A-Za-z]\w*\b")
_ACR = re.compile(r"\b[A-Z]{2,6}\b")
_CAP = re.compile(r"(?<![.!?;:]\s)(?<!^)\b([A-ZĐ][\wÀ-ỹ]*(?:\s+[A-ZĐ][\wÀ-ỹ]*)+)")
_WORD = re.compile(r"[\w'-]+")

# mã hay gặp, sinh thêm theo mẫu để không chỉ nhớ đúng chuỗi đã thấy trong train
PATTERN_CODES = ([f"VTV{i}" for i in range(1, 10)] + [f"HTV{i}" for i in range(1, 10)] +
                 [f"SCTV{i}" for i in range(1, 21)] + ["RON92", "RON95", "RON97", "E5", "VOV1", "VOV2", "VOV3"])


def candidates(ref):
    """{written: kind} các ứng viên hotword trong một câu ref viết."""
    out = {}
    for m in _CODE.finditer(ref):
        out[m.group(0)] = "code"
    for m in _ACR.finditer(ref):
        out.setdefault(m.group(0), "acronym")
    for w in _WORD.findall(ref):
        lw = w.lower()
        if not re.search(r"\d", w) and not _VI_SYL.match(lw) and len(lw) > 1 and w not in out:
            out.setdefault(lw if w[:1].islower() or w.isupper() is False and w[1:].islower() else w, "foreign")
    for m in _CAP.finditer(ref):
        out.setdefault(m.group(1), "name")
    return out


def spoken_forms(written, kind):
    if kind == "code":
        return code_variants(written)
    s = " ".join(_WORD.findall(written.lower().replace("-", " ")))
    return [s] if s else []


def find_all(words, phrase):
    n = len(phrase)
    return [i for i in range(len(words) - n + 1) if words[i:i + n] == phrase]


def aligned(ref_w, hyp_w):
    """map vị trí ref -> (đúng?, chỉ số hyp tương ứng) từ opcodes."""
    ok = [False] * len(ref_w)
    hmap = [None] * len(ref_w)
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, ref_w, hyp_w, autojunk=False).get_opcodes():
        for i in range(i1, i2):
            ok[i] = tag == "equal"
            hmap[i] = (j1 + (i - i1), j1 + (i - i1) + 1) if tag == "equal" else (j1, j2)
    return ok, hmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/asr_train_plain.jsonl",
                    help="jsonl có hyp/ref: output ASR KHÔNG hotword trên train (asr/eval_asr.py --configs beam --dump)")
    ap.add_argument("--train-jsonl", help="DATA/train/train.jsonl (để lấy common_words); bỏ trống = không lọc")
    ap.add_argument("--out", default="hotwords/hotwords_global")
    ap.add_argument("--min-err", type=int, default=2)
    ap.add_argument("--min-rate", type=float, default=0.3)
    ap.add_argument("--base", type=float, default=1.0)
    ap.add_argument("--k", type=float, default=2.0)
    ap.add_argument("--max-score", type=float, default=2.5)
    ap.add_argument("--name-max", type=float, default=1.5, help="trần score cho tên riêng toàn âm tiết tiếng Việt")
    ap.add_argument("--legit-min", type=int, default=3,
                    help="bỏ cụm nếu cách nghe sai phổ biến nhất là cụm có thật xuất hiện >= n lần trong ref")
    ap.add_argument("--zipformer", help="thư mục model (bpe.model) để bỏ hotword có ký tự ngoài vocab")
    a = ap.parse_args()

    pairs = load_pairs(a.inp)
    common = common_words(a.train_jsonl) if a.train_jsonl else frozenset()

    cand = {}  # written -> kind
    for _, ref, _, _ in pairs:
        for w, k in candidates(ref).items():
            cand.setdefault(w, k)
    for c in PATTERN_CODES:
        cand.setdefault(c, "code")

    # dạng đọc -> written (một written có nhiều dạng đọc)
    forms = {w: spoken_forms(w, k) for w, k in cand.items()}
    stats = {w: {"kind": cand[w], "n_occ": 0, "n_err": 0, "heard": Counter(), "form_ok": Counter()} for w in cand}
    index = defaultdict(list)  # first word -> [(written, phrase_words)]
    for w, fs in forms.items():
        for f in fs:
            index[f.split()[0]].append((w, f.split()))

    for _, ref, sp, hyp in pairs:
        rw, hw = sp.split(), hyp.split()
        ok, hmap = aligned(rw, hw)
        seen = set()
        for i, word in enumerate(rw):
            for w, ph in index.get(word, ()):
                n = len(ph)
                if rw[i:i + n] != ph or (w, i) in seen:
                    continue
                seen.add((w, i))
                st = stats[w]
                st["n_occ"] += 1
                good = all(ok[i:i + n])
                j1 = min(h[0] for h in hmap[i:i + n])
                j2 = max(h[1] for h in hmap[i:i + n])
                heard = " ".join(hw[j1:j2])
                if good:
                    st["form_ok"][" ".join(ph)] += 1
                else:
                    st["n_err"] += 1
                    st["heard"][heard] += 1

    ngrams = Counter()  # n-gram (1-4) của ref dạng đọc: phát hiện cách nghe sai trùng cụm có thật ("khách hàng")
    for _, _, sp, _ in pairs:
        w = sp.split()
        for n in range(1, 5):
            ngrams.update(" ".join(w[i:i + n]) for i in range(len(w) - n + 1))

    sp_model = None
    if a.zipformer:
        import sentencepiece as spm
        sp_model = spm.SentencePieceProcessor(model_file=str(Path(a.zipformer) / "bpe.model"))

    keep, report = [], []
    for w, st in stats.items():
        fs = forms[w]
        if not fs:
            continue
        rate = st["n_err"] / st["n_occ"] if st["n_occ"] else 0.0
        words = fs[0].split()
        single_vi = len(words) == 1 and _VI_SYL.match(words[0])
        trivial = all(x in common for x in words) and st["kind"] in ("foreign", "name") and len(words) == 1
        pattern_only = st["n_occ"] == 0 and w in PATTERN_CODES
        if single_vi or trivial:
            continue
        if not pattern_only and (st["n_err"] < a.min_err or rate < a.min_rate):
            continue
        # dạng đọc làm hotword: các biến thể ASR từng nghe đúng; chưa nghe đúng lần nào -> biến thể chuẩn (đầu tiên)
        hot = [f for f in fs if st["form_ok"][f]] or fs[:1]
        top = st["heard"].most_common(1)
        if top and top[0][0] and ngrams[top[0][0]] >= a.legit_min:   # "khách hàn" -> nghe thành "khách hàng" (có thật)
            continue
        cap = a.name_max if st["kind"] == "name" and all(_VI_SYL.match(x) for x in words) else a.max_score
        score = round(min(cap, a.base + a.k * rate), 2) if not pattern_only else a.base + 0.5
        # alias cho snap_text: các cách nghe sai lặp lại >= 2 lần, không phải cụm có thật trong ref
        mis = [h for h, c in st["heard"].most_common() if c >= 2 and h and h not in fs and ngrams[h] == 0][:5]
        if sp_model is not None:
            hot = [f for f in hot if 0 not in sp_model.encode(f.upper()) and 2 not in sp_model.encode(f.upper())]
            if not hot:
                continue
        display = hot[0] if re.search(r"\d", w) else w   # mã có số: giữ dạng đọc ASR hay nghe đúng (ITN lo phần số)
        keep.append((w, hot, score))
        report.append({"written": w, "kind": st["kind"], "display": display, "hotwords": hot, "score": score,
                       "aliases": sorted(set(fs) | set(mis)), "n_occ": st["n_occ"], "n_err": st["n_err"],
                       "err_rate": round(rate, 3), "heard": st["heard"].most_common(8)})

    report.sort(key=lambda r: (-r["n_err"], r["written"]))
    lines = sorted({f"{f.upper()} :{s}" for _, hot, s in keep for f in hot})
    Path(a.out + ".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(a.out + ".json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(pairs)} câu train-split, {len(cand)} ứng viên -> {len(report)} cụm, {len(lines)} dòng hotword "
          f"-> {a.out}.txt / {a.out}.json")
    for r in report[:40]:
        print(f"  {r['written']:<18} {r['n_err']:>3}/{r['n_occ']:<4} s={r['score']:<4} hot={r['hotwords']} "
              f"heard={[h for h, _ in r['heard'][:3]]}")


if __name__ == "__main__":
    main()
