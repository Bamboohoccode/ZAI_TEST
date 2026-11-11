"""P6: dò tên (hotword theo lượt) trên log-prob của đầu CTC của Zipformer (CTC word spotter, kiểu NVIDIA CTC-WS), bắt
được tên mà beam search transducer đã bỏ mất; tên dò được mà câu ASR chưa có -> thay đoạn gần âm nhất trong câu.

Đầu CTC lấy từ jit_script.pt của repo model (torch.jit: model.encoder(features, feature_lengths) + model.ctc_output).
Điểm dò của một tên = Viterbi CTC của chuỗi token tên trên một đoạn bất kỳ, so với best path cùng đoạn (log-prob đã trừ
max theo frame: 0 = tên chính là best path; càng âm càng kém). Chấm 1 lần (GPU), lưu spots.jsonl, dò ngưỡng offline.

    python scripts/ctc_spotter.py --root DATA --hyps nollm_main.jsonl --out-dir out/ctc     # cần --bias trong hyps
    python scripts/ctc_spotter.py --sweep-only --root DATA --out-dir out/ctc
"""
import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lifemate.bias import common_words, key, lexicons_per_turn, ratio, tokens  # noqa: E402
from lifemate.data import load_items, user_turns  # noqa: E402
from lifemate.text import norm, wer_counts  # noqa: E402
from lifemate.vi_num import to_spoken  # noqa: E402

REPO = "hynt/ZipFormer-150M-CR-CTC-RNNT-6000h"


def fbank(path):
    import kaldi_native_fbank as knf
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32")
    x = x.mean(1) if x.ndim > 1 else x
    o = knf.FbankOptions()
    o.frame_opts.dither, o.frame_opts.snip_edges, o.frame_opts.samp_freq = 0, False, sr
    o.mel_opts.num_bins, o.mel_opts.high_freq = 80, -400
    fb = knf.OnlineFbank(o)
    fb.accept_waveform(sr, x.tolist())
    fb.input_finished()
    return np.stack([fb.get_frame(i) for i in range(fb.num_frames_ready)])


class CTC:
    def __init__(self, device="cuda"):
        import sentencepiece as spm
        import torch
        from huggingface_hub import hf_hub_download
        self.torch, self.dev = torch, device
        self.m = torch.jit.load(hf_hub_download(REPO, "jit_script.pt"), map_location=device).eval()
        self.sp = spm.SentencePieceProcessor(model_file=hf_hub_download(REPO, "bpe.model"))
        print("jit:", [n for n, _ in self.m.named_children()], flush=True)

    def logprobs(self, path):
        t = self.torch
        x = t.from_numpy(fbank(path)).unsqueeze(0).to(self.dev)
        with t.no_grad():
            out, lens = self.m.encoder(features=x, feature_lengths=t.tensor([x.shape[1]], device=self.dev))
            lp = self.m.ctc_output(out)[0, :int(lens[0])]
        return lp.float().cpu().numpy()  # (T, V), log_softmax

    def ids(self, text):
        return self.sp.encode(text.upper())


def spot(lp, ids, blank=0):
    """Viterbi CTC của chuỗi ids trên đoạn tốt nhất của lp (log-prob trừ max theo frame) -> (điểm, frame đầu, frame cuối).
    Điểm 0 = trùng best path; chuẩn hoá theo số token (điểm / len(ids))."""
    if not ids:
        return -1e9, 0, 0
    rel = lp - lp.max(1, keepdims=True)
    ext = [blank]
    for i in ids:
        ext += [i, blank]
    S, T = len(ext), len(rel)
    allow2 = np.array([s >= 2 and ext[s] != blank and ext[s] != ext[s - 2] for s in range(S)])
    v = np.full(S, -np.inf)
    st = np.zeros(S, dtype=int)
    best = (-np.inf, 0, 0)
    for t in range(T):
        e = rel[t, ext]
        stay, prev1 = v, np.concatenate([[-np.inf], v[:-1]])
        prev2 = np.where(allow2, np.concatenate([[-np.inf, -np.inf], v[:-2]]), -np.inf)
        cand = np.stack([stay, prev1, prev2])
        arg = cand.argmax(0)
        nv = cand.max(0)
        nst = np.choose(arg, [st, np.concatenate([[0], st[:-1]]), np.concatenate([[0, 0], st[:-2]])])
        # bắt đầu mới ở frame t (blank đầu hoặc token đầu): điểm trước đó = 0 (best path)
        for s in (0, 1):
            if 0 > nv[s]:
                nv[s], nst[s] = 0.0, t
        v, st = nv + e, nst
        for s in (S - 2, S - 1):  # kết thúc ở token cuối hoặc blank sau nó
            if v[s] > best[0]:
                best = (v[s], st[s], t)
    return best[0] / len(ids), best[1], best[2]


def score_all(a):
    root = Path(a.root)
    items = {it["id"]: it for it in load_items(root, "train")}
    common = frozenset(common_words(root / "train" / "train.jsonl"))
    ctc = CTC(a.device)
    recs = []
    rows = [json.loads(l) for l in open(a.hyps, encoding="utf-8")]
    for n, row in enumerate(rows):
        it = items[row["id"]]
        lexs = lexicons_per_turn(it, common=common)
        for k, (u, t) in enumerate(zip(user_turns(it), row["turns"])):
            if not t.get("ref"):
                continue
            lp = ctc.logprobs(root / "train" / u["audio"])
            spots = []
            for e in dict.fromkeys(lexs[k]):
                sc, s0, s1 = spot(lp, ctc.ids(e))
                spots.append({"e": e, "score": round(float(sc), 3), "frames": [int(s0), int(s1)]})
            recs.append({"id": row["id"], "turn": t["turn"], "ref": t["ref"], "asr": t["asr"], "T": len(lp),
                         "spots": sorted(spots, key=lambda x: -x["score"])})
        print(f"[{n + 1}/{len(rows)}] {row['id']}", flush=True)
    return recs


def merge(hyp, spots, thr, sim, max_rep=2):
    """tên có điểm dò >= -thr mà câu chưa có -> thay đoạn (cùng số từ +-1) gần âm nhất nếu ratio >= sim."""
    words = norm(hyp).split()
    done = 0
    for sp in spots:
        if sp["score"] < -thr or done >= max_rep:
            break
        ew = tokens(sp["e"])
        if not ew or " ".join(ew) in " ".join(words):
            continue
        ek, best = key(ew), (0.0, None)
        for n in {max(1, len(ew) - 1), len(ew), len(ew) + 1}:
            for i in range(0, len(words) - n + 1):
                r = ratio(key(words[i:i + n]), ek)
                if r > best[0]:
                    best = (r, (i, n))
        if best[1] and best[0] >= sim:
            i, n = best[1]
            words = words[:i] + ew + words[i + n:]
            done += 1
    return " ".join(words)


def wer(recs, f):
    e = n = 0
    for r in recs:
        h = norm(f(r))
        x, y = wer_counts(to_spoken(r["ref"], h) or norm(r["ref"]), h)
        e, n = e + x, n + y
    return e / max(n, 1)


def sweep(recs):
    grid = {f"thr={t} sim={s}": (lambda r, t=t, s=s: merge(r["asr"], r["spots"], t, s))
            for t, s in itertools.product([0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0], [0.5, 0.6, 0.7, 0.8, 0.9])}
    out = {"baseline": wer(recs, lambda r: r["asr"])}
    full = {k: wer(recs, f) for k, f in grid.items()}
    bk = min(full, key=full.get)
    out["best"] = (bk, full[bk])
    fold = lambda r: int(re.search(r"(\d+)$", r["id"]).group(1)) // 10 % 2
    halves = [[r for r in recs if fold(r) == f] for f in (0, 1)]
    chosen, e, n = [], 0, 0
    for a, b in ((0, 1), (1, 0)):
        k = min(grid, key=lambda k: wer(halves[a], grid[k]))
        chosen.append(k)
        for r in halves[b]:
            h = norm(grid[k](r))
            x, y = wer_counts(to_spoken(r["ref"], h) or norm(r["ref"]), h)
            e, n = e + x, n + y
    out["cv"] = (chosen, e / max(n, 1))
    out["grid"] = full
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--hyps", help="run.py --no-llm --bias (train --heldout), có ref")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sweep-only", action="store_true")
    a = ap.parse_args()
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    if not a.sweep_only:
        recs = score_all(a)
        with open(od / "spots.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
    recs = [json.loads(l) for l in open(od / "spots.jsonl", encoding="utf-8")]
    out = sweep(recs)
    print(f"\n== P6 CTC word spotter, WER held-out ({len(recs)} lượt)")
    print(f"  baseline (ASR chính)              {out['baseline']:.4f}")
    print(f"  tốt nhất trên held-out (lạc quan) {out['best'][1]:.4f}  [{out['best'][0]}]")
    print(f"  cross-validation 2-fold           {out['cv'][1]:.4f}  [{' | '.join(out['cv'][0])}]")
    json.dump(out, open(od / "sweep.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
