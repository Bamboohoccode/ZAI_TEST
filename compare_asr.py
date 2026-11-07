"""So sánh ChunkFormer-CTC-large-vie vs Zipformer-30M-RNNT (sherpa-onnx) theo loại từ.

Loại từ (gán trên REFERENCE):
  normal      : từ tiếng Việt thường
  vi_proper   : tên riêng tiếng Việt (viết hoa, không đầu câu)
  foreign     : tên riêng/brand nước ngoài (viết hoa, không phải âm tiết Việt: Emily, Zalo, Nakamura...)
  acronym     : viết tắt toàn hoa (SMS, SJC, QR...)
  en_common   : từ tiếng Anh thông dụng viết thường (online, email, laptop, check...)

Chỉ lấy câu KHÔNG chứa chữ số (model đọc số thành chữ nên reference "50" bị phạt oan).
Metric: "word accuracy" = tỉ lệ từ reference được alignment khớp đúng (chuẩn hoá lowercase, bỏ dấu câu).

python compare_asr.py --n 200 --out-dir /tmp/cmp
"""
import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Dual_ChunkFormer"))
sys.path.insert(1, str(ROOT))
import bias  # noqa: E402
from models.text_utils import _align  # noqa: E402

DATA = Path("/home/trinhkhng/Downloads/train+public_test/train+public_test/train")
ZIP = Path("/tmp/claude-1000/-home-trinhkhng-voice-agent-pipeline/1540ed1e-9ffb-4c35-997b-72e097a028cc/scratchpad/zip")

V = "aăâeêioôơuưyáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
ON = "ngh|ng|nh|ch|gh|gi|kh|ph|th|tr|qu|[bcdđghklmnpqrstvx]"
VI_SYL = re.compile(rf"^({ON})?[{V}]+(ch|ng|nh|[cmnpt])?$")
CATS = ["normal", "vi_proper", "foreign", "acronym", "en_common"]


def ref_tokens(text):
    """[(word_lower, category)] cho reference; 'check-in' tách thành 2 từ cùng loại."""
    toks, start, out, raw = re.findall(r"[\w'-]+|[.!?;…]", text), True, [], []
    for w in toks:
        if re.fullmatch(r"[.!?;…]", w):
            start = True
            continue
        raw.append((w, start))
        start = False
    # từ viết thường không phải âm tiết Việt => en_common (dùng để nhận lại dạng viết hoa đầu câu: "Check ...")
    for w, st in raw:
        for p in [x for x in w.lower().split("-") if x]:
            vi = bool(VI_SYL.match(p))
            if w.isupper() and len(w) >= 2 and not vi:
                c = "acronym"
            elif vi:
                c = "vi_proper" if (w[0].isupper() and not st) else "normal"
            elif w[0].isupper() and not st:
                c = "foreign"
            else:
                c = "en_common" if w.islower() or st else "foreign"
            out.append((p, c))
    return out


def norm_hyp(t):
    return re.sub(r"[^\w\s'\-]", " ", t.lower()).replace("-", " ").replace("_", " ").split()


def word_ok(hyp, ref):
    """list[bool] theo ref: từ ref khớp đúng từ hyp trên đường alignment."""
    path, _ = _align(hyp, ref)
    ok = [False] * len(ref)
    for (h0, r0), (h1, r1) in zip(path, path[1:]):
        if h1 == h0 + 1 and r1 == r0 + 1 and hyp[h0] == ref[r0]:
            ok[r0] = True
    return ok


def load_items(n, seed):
    rows = [json.loads(l) for l in open(DATA / "train.jsonl", encoding="utf-8")]
    rows = [r for r in rows if not re.search(r"\d", r["audio_text"])]
    random.Random(seed).shuffle(rows)
    return [{"id": r["id"], "audio": str(DATA / r["audio"]), "text": r["audio_text"], "context": r["context"],
             "state": str(DATA / "train_environment" / r["id"] / "state.json")} for r in rows[:n]]


def lexicon(it):
    return bias.build_lexicon(it["state"], it["context"])


def run_chunkformer(items):
    from chunkformer import ChunkFormerModel
    m = ChunkFormerModel.from_pretrained("khanhld/chunkformer-ctc-large-vie").to("cpu").eval()
    res = {}
    for i, it in enumerate(items):
        t = time.time()
        out = m.endless_decode(audio_path=it["audio"], chunk_size=64, left_context_size=128,
                               right_context_size=128, return_timestamps=False)
        res[it["id"]] = out if isinstance(out, str) else (out[0] if isinstance(out, list) and out and isinstance(out[0], str) else str(out))
        print(f"[chunkformer {i + 1}/{len(items)}] {time.time() - t:.1f}s {res[it['id']]}", flush=True)
    return res


def run_zipformer(items, hotwords=False):
    from asr_zipformer import ZipformerASR
    asr = ZipformerASR(str(ZIP), bias=hotwords)
    res = {}
    for i, it in enumerate(items):
        res[it["id"]] = asr.transcribe([it["audio"]], hotwords=lexicon(it) if hotwords else None)[0]
        print(f"[zipformer{'+hw' if hotwords else ''} {i + 1}/{len(items)}] {res[it['id']]}", flush=True)
    return res


def snap_all(items, hyps, common):
    return {it["id"]: bias.snap_text(hyps[it["id"]], lexicon(it), common)[0] for it in items}


def score(items, hyps):
    tot, ok_n = defaultdict(int), defaultdict(int)
    errs = defaultdict(list)
    for it in items:
        rt = ref_tokens(it["text"])
        ref = [w for w, _ in rt]
        ok = word_ok(norm_hyp(hyps[it["id"]]), ref)
        for (w, c), o in zip(rt, ok):
            tot[c] += 1
            ok_n[c] += o
            if not o and c != "normal":
                errs[c].append((w, it["id"]))
    return tot, ok_n, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--only", choices=["chunkformer", "zipformer"])
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    items = load_items(a.n, a.seed)

    base = {}
    for name, fn in [("chunkformer", run_chunkformer), ("zipformer", run_zipformer),
                     ("zipformer_hw", lambda it: run_zipformer(it, hotwords=True))]:
        p = out / f"{name}.json"
        if p.exists():
            base[name] = json.load(open(p, encoding="utf-8"))
        elif a.only in (None, name):
            base[name] = fn(items)
            json.dump(base[name], open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    common = bias.common_words(DATA / "train.jsonl")
    hyps = dict(base)
    for k in list(base):
        hyps[k + "+snap"] = snap_all(items, base[k], common)

    names = list(hyps)
    print("\n" + "=" * 90)
    print(f"{len(items)} câu không chứa số, word accuracy theo loại từ (ref)")
    print(f"{'loại':<11}{'#từ':>6}" + "".join(f"{n:>19}" for n in names))
    res = {k: score(items, hyps[k]) for k in hyps}
    summary = {}
    for c in CATS + ["ALL"]:
        row = []
        for k in names:
            tot, okn, _ = res[k]
            t = sum(tot.values()) if c == "ALL" else tot[c]
            o = sum(okn.values()) if c == "ALL" else okn[c]
            row.append((t, o))
        summary[c] = row
        print(f"{c:<11}{row[0][0]:>6}" + "".join(f"{o / max(t, 1):>19.1%}" for t, o in row))
    json.dump(summary, open(out / "summary.json", "w"), indent=1)

    for c in ("foreign", "acronym", "en_common", "vi_proper"):
        print(f"\n-- lỗi hay gặp [{c}] (từ ref bị sai) --")
        for k in ("chunkformer+snap", "zipformer_hw+snap"):
            cnt = defaultdict(int)
            for w, _ in res[k][2][c]:
                cnt[w] += 1
            print(f"  {k}: " + ", ".join(f"{w}×{n}" for w, n in sorted(cnt.items(), key=lambda x: -x[1])[:15]))


if __name__ == "__main__":
    main()
