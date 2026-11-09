"""Đo WER Zipformer trên train (có nhãn) theo nhiều cấu hình: greedy / beam / + hotword toàn cục / + VAD / + LM.
Offline, không đọc state.json. Ref = audio_text đổi sang dạng đọc (vi_num.to_spoken, chấp nhận mọi cách đọc số hợp lệ).

    # 1) decode train không hotword -> dữ liệu cho hotwords/mine_hotwords.py / lm/train_rnnlm.py (thay asr_train.jsonl vốn có --bias)
    python asr/eval_asr.py --root DATA --subset all --configs beam --dump data/asr_train_plain.jsonl

    # 2) so sánh trên held-out (id % 10 == 9, không dùng để khai thác hotword / train LM)
    python asr/eval_asr.py --root DATA --configs greedy,beam,hw,vad+hw,hw+lm,vad+hw+lm+snap+fix \\
        --hotwords hotwords/hotwords_global.txt --lm lm/rnnlm/rnnlm.onnx --lm-scales 0.1,0.2,0.3,0.5

Cấu hình = các phần nối bằng '+': greedy | beam | hw (hotwords_file) | vad | lm | snap (snap_text theo
hotwords_global.json) | fix (asr_fix). "hw"/"lm" ngầm dùng beam search.
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import sys  # noqa: E402

if __package__ in (None, ""):  # chạy trực tiếp: python asr/eval_asr.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asr.asr_fix import fix_text  # noqa: E402
from hotwords.mine_hotwords import is_heldout  # noqa: E402
from textnorm.textutil import norm, wer_counts  # noqa: E402
from textnorm.vi_num import to_spoken  # noqa: E402


def utterances(root, subset):
    """[(id, turn, audio_path, audio_text)] mọi lượt user (history + query) của train."""
    out = []
    for l in open(root / "train" / "train.jsonl", encoding="utf-8"):
        it = json.loads(l)
        if subset == "heldout" and not is_heldout(it["id"]) or subset == "trainsplit" and is_heldout(it["id"]):
            continue
        k = 0
        for h in it["history"]:
            if h["role"] == "user":
                k += 1
                out.append((it["id"], f"u{k}", h["audio"], h["audio_text"]))
        out.append((it["id"], "QUERY", it["audio"], it["audio_text"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--subset", choices=["heldout", "trainsplit", "all"], default="heldout")
    ap.add_argument("--n", type=int, default=0, help="giới hạn số câu (0 = hết)")
    ap.add_argument("--configs", default="greedy,beam,hw")
    ap.add_argument("--zipformer", default="hynt/Zipformer-30M-RNNT-6000h")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--hotwords", default="hotwords/hotwords_global.txt")
    ap.add_argument("--hotwords-score", type=float, default=1.5, help="score mặc định cho dòng không ghi :score")
    ap.add_argument("--snap-json", default=None, help="mặc định: <hotwords>.json")
    ap.add_argument("--lm", default="lm/rnnlm/rnnlm.onnx")
    ap.add_argument("--lm-scales", default="0.3")
    ap.add_argument("--vad-pad", type=float, default=0.15)
    ap.add_argument("--vad-gap", type=float, default=0.5, help="khoảng lặng giữa dài hơn -> rút còn 0.1s (lớn = không rút)")
    ap.add_argument("--vad-threshold", type=float, default=0.5)
    ap.add_argument("--bs", type=int, default=32, help="số file mỗi lần decode_streams")
    ap.add_argument("--dump", help="ghi jsonl (id, turn, audio, hyp, ref) của cấu hình ĐẦU TIÊN (cho mine_hotwords / LM)")
    ap.add_argument("--dump-all", help="thư mục: ghi jsonl cho từng cấu hình")
    a = ap.parse_args()

    from asr.asr_zipformer import ZipformerASR
    from asr.vad import VAD
    root = Path(a.root).expanduser()
    utts = utterances(root, a.subset)
    if a.n:
        utts = utts[:a.n]
    paths = [root / "train" / u[2] for u in utts]

    configs = []
    for c in a.configs.split(","):
        parts = set(c.split("+"))
        if "lm" in parts:
            configs += [(f"{c}@{s}", parts, float(s)) for s in a.lm_scales.split(",")]
        else:
            configs.append((c, parts, None))

    snap_aliases, common = None, frozenset()
    if any("snap" in p for _, p, _ in configs):
        from hotwords.bias import common_words, load_global
        snap_aliases = load_global(a.snap_json or str(Path(a.hotwords).with_suffix(".json")))
        common = common_words(root / "train" / "train.jsonl")

    cache, rows = {}, []
    for name, parts, scale in configs:
        akey = ("greedy" in parts and not parts & {"hw", "lm"}, "hw" in parts, "vad" in parts, scale)
        t0 = time.time()
        if akey not in cache:
            greedy, hw, vad, sc = akey
            asr = ZipformerASR(a.zipformer, int8=a.int8, num_threads=a.threads, bias=False, beam=a.beam,
                               hotwords_file=a.hotwords if hw else None, hotwords_score=a.hotwords_score,
                               lm=a.lm if sc is not None else None, lm_scale=sc or 0.0,
                               vad=VAD(threshold=a.vad_threshold, pad=a.vad_pad, max_gap=a.vad_gap) if vad else None,
                               decoding="greedy_search" if greedy else "modified_beam_search")
            hyps = []
            for i in range(0, len(paths), a.bs):
                hyps += asr.transcribe(paths[i:i + a.bs])
                print(f"\r  {name}: {len(hyps)}/{len(paths)}", end="", flush=True)
            print()
            cache[akey] = hyps
            del asr
        hyps = list(cache[akey])
        if "snap" in parts:
            from hotwords.bias import snap_text
            hyps = [snap_text(h, [], common, aliases=snap_aliases)[0] for h in hyps]
        if "fix" in parts:
            hyps = [fix_text(h)[0] for h in hyps]
        err = tot = err_q = tot_q = 0
        recs = []
        for u, h in zip(utts, hyps):
            hn = norm(h)
            ref = to_spoken(u[3], hn) or norm(u[3])
            e, n = wer_counts(ref, hn)
            err, tot = err + e, tot + n
            if u[1] == "QUERY":
                err_q, tot_q = err_q + e, tot_q + n
            recs.append({"id": u[0], "turn": u[1], "audio": u[2], "hyp": h, "ref": u[3], "ref_spoken": ref})
        rows.append((name, err / max(tot, 1), err_q / max(tot_q, 1), time.time() - t0))
        if a.dump and len(rows) == 1:
            with open(a.dump, "w", encoding="utf-8") as f:
                f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
        if a.dump_all:
            Path(a.dump_all).mkdir(parents=True, exist_ok=True)
            with open(Path(a.dump_all) / f"{name.replace('+', '_').replace('@', '_lm')}.jsonl", "w", encoding="utf-8") as f:
                f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)

    print(f"\n{len(utts)} câu ({a.subset})")
    print(f"{'config':<32}{'WER':>8}{'WER(query)':>12}{'time':>8}")
    for name, w, wq, t in rows:
        print(f"{name:<32}{w:>8.4f}{wq:>12.4f}{t:>7.0f}s")


if __name__ == "__main__":
    main()
