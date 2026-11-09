"""In transcript ASR (Zipformer) trên tập public_test để duyệt bằng mắt (public không có nhãn -> không tính WER).

Mỗi câu in:  greedy (không bias)  |  bias = hotword chia điểm + snap_text + asr_fix (đúng như pipeline.py --asr-model zipformer --bias)
kèm câu trả lời của assistant trong history làm ngữ cảnh. Kết quả cũng ghi ra --out (jsonl).

    python asr/test_asr_public.py --root ~/Downloads/train+public_test/train+public_test
    python asr/test_asr_public.py --root ... --zipformer /path/zipformer-ft     # model fine-tune
    python asr/test_asr_public.py --root ... --n 200 --seed 0 --threads 4
    python asr/test_asr_public.py --root ... --hotwords-file hotwords/hotwords_global.txt --vad --lm lm/rnnlm/rnnlm.onnx   # không đọc state.json
"""
import argparse
import json
import random
import time
from pathlib import Path

import sys

if __package__ in (None, ""):  # chạy trực tiếp: python asr/test_asr_public.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asr.asr_fix import fix_text  # noqa: E402
from asr.asr_zipformer import ZipformerASR  # noqa: E402
from hotwords.bias import build_lexicon, common_words, snap_text  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="thư mục chứa train/ và public_test/")
    ap.add_argument("--split", default="public_test")
    ap.add_argument("--zipformer", default="hynt/Zipformer-30M-RNNT-6000h", help="HF id hoặc thư mục local")
    ap.add_argument("--n", type=int, default=200, help="số câu (lượt user) cần in")
    ap.add_argument("--seed", type=int, default=0, help="chọn ngẫu nhiên sample; -1 = lấy theo thứ tự file")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--hotword-hi", type=float, default=2.5, help="tên riêng / viết tắt")
    ap.add_argument("--hotword-lo", type=float, default=1.5, help="cụm thường")
    ap.add_argument("--glossary-score", type=float, default=1.0, help="từ ngoại viết thường; 0 = không làm hotword")
    ap.add_argument("--no-asr-fix", action="store_true")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--no-greedy", action="store_true", help="bỏ cột greedy (nhanh gấp đôi)")
    ap.add_argument("--out", default="data/asr_public.jsonl")
    ap.add_argument("--hotwords-file", help="hotword cố định (mine_hotwords.py); bật -> KHÔNG đọc state.json, snap theo <file>.json")
    ap.add_argument("--vad", action="store_true")
    ap.add_argument("--lm", help="RNNLM onnx (train_rnnlm.py)")
    ap.add_argument("--lm-scale", type=float, default=0.3)
    ap.add_argument("--beam", type=int, default=4)
    a = ap.parse_args()

    root = Path(a.root).expanduser()
    items = [json.loads(l) for l in open(root / a.split / f"{a.split}.jsonl", encoding="utf-8")]
    if a.seed >= 0:
        random.Random(a.seed).shuffle(items)
    # lấy trọn sample (mọi lượt user) cho tới khi đủ n câu
    picked, total = [], 0
    for it in items:
        if total >= a.n:
            break
        picked.append(it)
        total += 1 + sum(h["role"] == "user" for h in it["history"])

    train = root / "train" / "train.jsonl"
    common = common_words(train) if train.exists() else frozenset()
    biased = ZipformerASR(a.zipformer, int8=a.int8, num_threads=a.threads, bias=True, beam=a.beam,
                          hi=a.hotword_hi, lo=a.hotword_lo, glossary_score=a.glossary_score,
                          hotwords_file=a.hotwords_file, vad=a.vad, lm=a.lm, lm_scale=a.lm_scale)
    g_alias = None
    if a.hotwords_file:
        from hotwords.bias import load_global
        g_alias = load_global(Path(a.hotwords_file).with_suffix(".json"))
    greedy = None if a.no_greedy else ZipformerASR(a.zipformer, int8=a.int8, num_threads=a.threads, bias=False)

    n_done, n_diff, t0 = 0, 0, time.time()
    with open(a.out, "w", encoding="utf-8") as fout:
        for si, it in enumerate(picked, 1):
            # lượt user theo thứ tự hội thoại; ghi nhớ câu trả lời assistant ngay sau mỗi lượt
            turns, replies = [], []
            for h in it["history"]:
                if h["role"] == "user":
                    turns.append(h["audio"]); replies.append(None)
                elif replies:
                    replies[-1] = h.get("text")
            turns.append(it["audio"]); replies.append(None)
            paths = [root / a.split / p for p in turns]

            if g_alias is not None:   # hotword cố định: không đọc state.json
                lex = []
                raw = biased.transcribe(paths)
                snapped = [snap_text(r, [], common, aliases=g_alias) for r in raw]
            else:
                lex = build_lexicon(root / a.split / f"{a.split}_environment" / it["id"] / "state.json", it["context"])
                raw = biased.transcribe(paths, hotwords=lex or None)
                snapped = [snap_text(r, lex, common) for r in raw] if lex else [(r, []) for r in raw]
            fixed = [([], t) if a.no_asr_fix else fix_text(t)[::-1] for t, _ in snapped]   # (changes, text)
            snapped = [(t, ch) for (_, ch), (_, t) in zip(snapped, fixed)]
            g = greedy.transcribe(paths) if greedy else [None] * len(paths)

            print(f"\n{'=' * 100}\n[{si}/{len(picked)}] {it['id']}  ({len(paths)} lượt)  "
                  f"lexicon: {', '.join(lex[:12])}{' …' if len(lex) > 12 else ''}")
            for k, (p, gr, (fin, ch), rep, (fch, _)) in enumerate(zip(turns, g, snapped, replies, fixed)):
                n_done += 1
                tag = "QUERY" if k == len(turns) - 1 else f"u{k + 1}"
                diff = gr is not None and gr != fin
                n_diff += diff
                print(f"  {n_done:>4}. [{tag}] {Path(p).name}")
                if gr is not None:
                    print(f"        greedy: {gr}")
                print(f"        bias  : {fin}{'   <-- khác' if diff else ''}")
                if ch:
                    print(f"        snap  : {ch}")
                if fch:
                    print(f"        fix   : {fch}")
                if rep:
                    print(f"        (bot) : {rep}")
                fout.write(json.dumps({"id": it["id"], "turn": tag, "audio": p, "greedy": gr, "bias_raw": raw[k],
                                       "bias": fin, "snap": ch, "fix": fch}, ensure_ascii=False) + "\n")

    dt = time.time() - t0
    print(f"\n{n_done} câu / {len(picked)} sample, {dt:.0f}s ({dt / max(n_done, 1):.2f}s/câu)"
          + (f", bias khác greedy ở {n_diff} câu" if greedy else "") + f"  -> {a.out}")


if __name__ == "__main__":
    main()
