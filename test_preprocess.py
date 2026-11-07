"""Test audio_preprocess: runs it on wav file(s) and WRITES the output wav(s) so you can listen.

python test_preprocess.py                              # Dual_ChunkFormer/a.wav -> preprocess_out/a_pre.wav
python test_preprocess.py x.wav y.wav --out-dir out/   # your files
python test_preprocess.py --synthetic                  # also run on a generated noisy/stereo/44.1k signal
python test_preprocess.py a.wav --no-trim --highpass 0 # toggle steps

Prints before/after stats (sr, channels, duration, RMS dBFS, peak) and asserts the invariants
(16 kHz mono, no clipping, finite). Exits 1 if one fails.
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from audio_preprocess import PreprocessConfig, load_wav, preprocess_array, save_wav

DEFAULT_WAV = Path(__file__).parent / "Dual_ChunkFormer" / "a.wav"


def dbfs(x: np.ndarray) -> float:
    return 20 * np.log10(max(float(np.sqrt(np.mean(x ** 2))), 1e-10)) if len(x) else float("-inf")


def stats(x: np.ndarray, sr: int) -> str:
    ch = 1 if x.ndim == 1 else x.shape[1]
    return (f"sr={sr} ch={ch} dur={len(x) / sr:.2f}s "
            f"rms={dbfs(x):.1f}dBFS peak={float(np.abs(x).max()) if len(x) else 0:.3f}")


def make_synthetic(sr: int = 44100) -> np.ndarray:
    """Stereo, 44.1 kHz, DC offset + 50 Hz hum, 1s silence, 'speech'-like bursts, 1s silence, quiet."""
    t = np.arange(int(sr * 2.0)) / sr
    rng = np.random.default_rng(0)
    speech = (np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 440 * t)) * (np.sin(2 * np.pi * 3 * t) > 0)
    sig = np.concatenate([np.zeros(sr), 0.05 * speech, np.zeros(sr)])
    sig += 0.1 + 0.002 * np.sin(2 * np.pi * 50 * np.arange(len(sig)) / sr) + 0.0002 * rng.standard_normal(len(sig))
    return np.stack([sig, 0.8 * sig], axis=1).astype(np.float32)


def check(name: str, x: np.ndarray, sr: int, cfg: PreprocessConfig, out_dir: Path, raw: np.ndarray, raw_sr: int) -> bool:
    out, out_sr = preprocess_array(raw, raw_sr, cfg)
    path = out_dir / f"{name}_pre.wav"
    save_wav(path, out, out_sr)
    back, back_sr = load_wav(path)  # verify what was actually written

    print(f"\n[{name}]\n  before: {stats(raw, raw_sr)}\n  after : {stats(back, back_sr)}\n  wrote : {path}")
    problems = []
    if back_sr != cfg.target_sr:
        problems.append(f"sr {back_sr} != {cfg.target_sr}")
    if back.ndim != 1:
        problems.append("not mono")
    if not np.isfinite(back).all():
        problems.append("non-finite samples")
    if len(back) and np.abs(back).max() > 1.0:
        problems.append("clipping")
    if len(back) == 0:
        problems.append("empty output")
    if cfg.normalize and len(back) and abs(dbfs(back) - cfg.target_dbfs) > 3 and np.abs(back).max() < cfg.peak_limit - 0.01:
        problems.append(f"rms {dbfs(back):.1f} not near {cfg.target_dbfs}")
    if cfg.trim_silence and len(back) >= len(raw) / raw_sr * cfg.target_sr:
        print("  note  : nothing trimmed (no leading/trailing silence)")
    for p in problems:
        print(f"  FAIL  : {p}")
    return not problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", nargs="*", help=f"input wav(s); default {DEFAULT_WAV}")
    ap.add_argument("--out-dir", default="preprocess_out")
    ap.add_argument("--synthetic", action="store_true", help="also test a generated stereo 44.1 kHz noisy signal")
    ap.add_argument("--no-trim", action="store_true")
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--highpass", type=float, default=PreprocessConfig.highpass_hz, help="Hz; 0 disables")
    a = ap.parse_args()

    cfg = replace(PreprocessConfig(), trim_silence=not a.no_trim, normalize=not a.no_normalize,
                  highpass_hz=a.highpass or None)
    out_dir = Path(a.out_dir)
    paths = a.audio or ([] if a.synthetic else [str(DEFAULT_WAV)])

    ok = True
    for p in paths:
        raw, sr = load_wav(p)
        ok &= check(Path(p).stem, raw, sr, cfg, out_dir, raw, sr)
    if a.synthetic:
        ok &= check("synthetic", None, 44100, cfg, out_dir, make_synthetic(44100), 44100)

    print("\nALL OK" if ok else "\nFAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
