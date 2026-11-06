"""Calibrate the CTC confidences.

1. Temperature scaling: T minimizing the CTC NLL of the reference (per token) of each model.
   --manifest: code-switched dev set (audio, text) -> T_vi.   --en-manifest: English-only dev set -> T_en.
2. Threshold: on the Vi words (confidence scaled by T_vi), choose conf_threshold maximizing F-beta of detecting wrong
   words (beta=2: missing an error costs more than a wasted slot).

python train/calibrate_ctc.py --manifest dev.jsonl [--en-manifest dev_en.jsonl] --out configs/calibration.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import load_config  # noqa: E402
from models.dual_chunkformer import ChunkformerCTC, greedy_words, temp_scale  # noqa: E402
from models.text_utils import normalize, word_correct  # noqa: E402

GRID = [round(0.5 + 0.05 * i, 2) for i in range(0, 56)]  # 0.5 .. 3.25


def load(manifest, limit):
    rows = [json.loads(l) for l in open(manifest, encoding="utf-8") if l.strip()]
    return rows[:limit] if limit else rows


def fit_temperature(model, rows):
    """-> (best T, {T: nll per token}) on the rows."""
    data = []
    for r in rows:
        ids = model.encode(" ".join(normalize(r["text"])))
        lp = model.log_probs(r["audio"])
        if ids and len(ids) <= lp.shape[0]:
            data.append((lp, torch.tensor(ids)))
    nll = {}
    for T in GRID:
        tot = n = 0.0
        for lp, ids in data:
            x = temp_scale(lp, T)[:, None, :]
            tot += float(F.ctc_loss(x, ids[None], torch.tensor([x.shape[0]]), torch.tensor([len(ids)]),
                                    blank=0, reduction="sum", zero_infinity=True))
            n += len(ids)
        nll[T] = tot / max(n, 1)
    return min(nll, key=nll.get), nll


def tune_threshold(vi, rows, T_vi, agg, beta=2.0):
    conf, wrong = [], []
    for r in rows:
        words = greedy_words(temp_scale(vi.log_probs(r["audio"]), T_vi), vi.id2tok, vi.frame_ms, agg)
        ok = word_correct([w.text for w in words], normalize(r["text"]))
        conf += [w.conf for w in words]
        wrong += [not o for o in ok]
    conf, wrong = np.array(conf), np.array(wrong)
    best = (0.0, 0.5, 0.0, 0.0)
    for thr in np.unique(np.round(conf, 3)):
        flag = conf < thr
        tp = float((flag & wrong).sum())
        p, rc = tp / max(flag.sum(), 1), tp / max(wrong.sum(), 1)
        f = (1 + beta ** 2) * p * rc / max(beta ** 2 * p + rc, 1e-9)
        if f > best[0]:
            best = (f, float(thr), p, rc)
    return {"conf_threshold": best[1], "f_beta": best[0], "precision": best[2], "recall": best[3],
            "n_words": int(len(conf)), "word_error_rate": float(wrong.mean()) if len(wrong) else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="code-switched dev set: audio + text")
    ap.add_argument("--en-manifest", help="English-only dev set for T_en")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/pipeline_config.yaml"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "configs/calibration.json"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    cfg = load_config(a.config)
    m, c, k = cfg["models"], cfg["chunkformer"], cfg["ctc"]
    kw = dict(chunk_size=c["chunk_size"], left=c["left_context_size"], right=c["right_context_size"])
    res = {}
    vi = ChunkformerCTC(m["chunkformer_vi"], m.get("vi_ckpt"), m["vi_device"], **kw)
    rows = load(a.manifest, a.limit)
    res["T_vi"], nll = fit_temperature(vi, rows)
    print(f"T_vi = {res['T_vi']}  (nll/token {nll[1.0]:.4f} at T=1 -> {nll[res['T_vi']]:.4f})")
    res.update(tune_threshold(vi, rows, res["T_vi"], k["conf_agg"]))
    print(f"conf_threshold = {res['conf_threshold']:.3f}  precision {res['precision']:.2f} recall {res['recall']:.2f}")
    del vi
    if a.en_manifest:
        en = ChunkformerCTC(m["chunkformer_en"], m.get("en_ckpt"), m["en_device"], **kw)
        res["T_en"], nll = fit_temperature(en, load(a.en_manifest, a.limit))
        print(f"T_en = {res['T_en']}  (nll/token {nll[1.0]:.4f} at T=1 -> {nll[res['T_en']]:.4f})")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=2)
    print("->", a.out)


if __name__ == "__main__":
    main()
