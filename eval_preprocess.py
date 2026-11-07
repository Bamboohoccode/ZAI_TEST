"""Which preprocessing actually lowers ASR WER? Runs ChunkFormer on train audio under several preprocess variants.

.venv/bin/python eval_preprocess.py --n 60
.venv/bin/python eval_preprocess.py --n 100 --root ~/Downloads/train+public_test/train+public_test

WER is against train `audio_text` after lowercasing and stripping punctuation. Digits/ITN differences vs the spoken
transcript are the same for every variant, so only the *relative* WER between variants is meaningful.
Also prints the VAD-estimated SNR distribution so you can pick the denoise gate.
"""
import argparse
import json
import random
import re
import tempfile
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

from audio_preprocess import PreprocessConfig, estimate_snr_db, load_wav, preprocess_array, save_wav, speech_segments, to_mono, resample
from pipeline import ASR_MODEL, run_asr

warnings.filterwarnings("ignore")
ROOT = Path.home() / "Downloads/train+public_test/train+public_test"


def norm(t: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", t.lower()).split()


def edit_distance(a: list[str], b: list[str]) -> int:
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[len(b)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--asr", default=ASR_MODEL)
    a = ap.parse_args()

    root = Path(a.root) / "train"
    rows = [json.loads(l) for l in open(root / "train.jsonl")]
    random.Random(a.seed).shuffle(rows)
    rows = rows[: a.n]

    base = PreprocessConfig()
    variants = {
        "raw (no preprocess)": None,
        "basic: energy trim+norm": replace(base, vad="energy", denoise="off"),
        "silero trim+norm": replace(base, denoise="off"),
        "silero + denoise always": replace(base, denoise="always"),
        "silero + denoise auto<15dB": replace(base, denoise="auto", denoise_below_snr_db=15),
        "silero + denoise auto<25dB": replace(base, denoise="auto", denoise_below_snr_db=25),
    }

    from chunkformer import ChunkFormerModel
    asr = ChunkFormerModel.from_pretrained(a.asr)

    tmp = Path(tempfile.mkdtemp(prefix="evalpre_"))
    refs = [norm(r["audio_text"]) for r in rows]
    snrs = []
    paths = {k: [] for k in variants}
    for i, r in enumerate(rows):
        src = root / r["audio"]
        x, sr = load_wav(src)
        x16 = resample(to_mono(x), sr, 16000)
        segs = speech_segments(x16 - x16.mean(), 16000)
        snrs.append(estimate_snr_db(x16 - x16.mean(), segs, 16000))
        for k, cfg in variants.items():
            if cfg is None:
                paths[k].append(src)
                continue
            y, ysr = preprocess_array(x, sr, cfg)
            p = tmp / f"{k.split(':')[0].replace(' ', '_').replace('<', 'lt').replace('+', 'p')}_{i}.wav"
            save_wav(p, y, ysr)
            paths[k].append(p)

    s = np.array([v for v in snrs if v is not None])
    print(f"\nSNR (VAD-estimated) over {len(s)}/{len(snrs)} files with measurable noise: "
          f"min {s.min():.1f} p10 {np.percentile(s, 10):.1f} median {np.median(s):.1f} p90 {np.percentile(s, 90):.1f} dB; "
          f"<15dB: {(s < 15).sum()}, <25dB: {(s < 25).sum()}")

    print(f"\n{'variant':<30} {'WER%':>7} {'errors':>7}")
    for k, ps in paths.items():
        hyps = []
        for j in range(0, len(ps), 8):
            hyps += run_asr(asr, ps[j:j + 8])
        err = sum(edit_distance(norm(h), r) for h, r in zip(hyps, refs))
        words = sum(len(r) for r in refs)
        print(f"{k:<30} {100 * err / words:7.2f} {err:7d}")
    print(f"\n(reference words: {words}; audio in {tmp})")


if __name__ == "__main__":
    main()
