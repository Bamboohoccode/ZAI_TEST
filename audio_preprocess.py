"""Audio pre-processing for the voice pipeline (wav in -> clean 16 kHz mono float32 out).

Steps (each can be switched off via PreprocessConfig):
  1. load       any PCM/float wav -> float32 in [-1, 1]
  2. mono       average channels
  3. dc         remove DC offset (before resampling, to avoid edge ringing)
  4. resample   to target_sr (default 16 kHz, what the ASR expects), then high-pass
                low rumble (Butterworth, zero-phase)
  5. denoise    spectral-gating noise reduction (noisereduce). mode "off" | "always" | "auto";
                "auto" only denoises when the VAD-estimated SNR is below `denoise_below_snr_db`,
                because enhancement on clean audio hurts modern end-to-end ASR (see eval_preprocess.py)
  6. trim       strip leading/trailing silence, Silero VAD (neural) with energy fallback; keeps a margin
  7. normalize  RMS-normalize to target_dbfs, then peak-limit so nothing clips

Usage as a module:
    from audio_preprocess import preprocess_file, preprocess_array
    wav, sr = preprocess_file("a.wav")                    # -> (np.float32 [T], 16000)
    preprocess_file("a.wav", out_path="a_clean.wav")      # also write the result

Usage as CLI:
    python audio_preprocess.py in.wav [more.wav ...] --out-dir cleaned/
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from math import gcd
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfiltfilt


@dataclass
class PreprocessConfig:
    target_sr: int = 16000
    highpass_hz: float | None = 80.0     # None = skip
    denoise: str = "off"                 # "off" | "always" | "auto" (SNR-gated)
    denoise_below_snr_db: float = 15.0   # auto: denoise only if estimated SNR is under this
    denoise_strength: float = 0.8        # noisereduce prop_decrease, 0..1
    vad: str = "silero"                  # "silero" | "energy" (trim method)
    vad_threshold: float = 0.4
    trim_silence: bool = True
    trim_top_db: float = 35.0            # frames quieter than (peak frame - this) count as silence
    trim_pad_ms: int = 150               # margin kept around speech
    frame_ms: int = 20
    normalize: bool = True
    target_dbfs: float = -20.0           # RMS level of the speech
    peak_limit: float = 0.97             # never exceed this absolute peak


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a wav file as float32, shape [T] (mono) or [T, C]."""
    sr, x = wavfile.read(str(path))
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float32) / 2147483648.0
    elif x.dtype == np.uint8:
        x = (x.astype(np.float32) - 128.0) / 128.0
    else:  # float32 / float64
        x = x.astype(np.float32)
    return x, int(sr)


def save_wav(path: str | Path, wav: np.ndarray, sr: int) -> None:
    """Write float [-1, 1] mono audio as 16-bit PCM."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(wav, -1.0, 1.0)
    wavfile.write(str(path), sr, (pcm * 32767.0).astype(np.int16))


def to_mono(x: np.ndarray) -> np.ndarray:
    return x if x.ndim == 1 else x.mean(axis=1)


def resample(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return x
    g = gcd(sr, target_sr)
    return resample_poly(x, target_sr // g, sr // g).astype(np.float32)


def highpass(x: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    sos = butter(4, cutoff_hz, btype="highpass", fs=sr, output="sos")
    # sosfiltfilt needs the signal to be longer than its padding
    if len(x) < 64:
        return x
    return sosfiltfilt(sos, x).astype(np.float32)


def trim_silence(x: np.ndarray, sr: int, top_db: float, pad_ms: int, frame_ms: int) -> np.ndarray:
    """Cut leading/trailing frames whose RMS is > top_db below the loudest frame."""
    frame = max(1, int(sr * frame_ms / 1000))
    n = len(x) // frame
    if n == 0:
        return x
    rms = np.sqrt((x[: n * frame].reshape(n, frame) ** 2).mean(axis=1)) + 1e-10
    db = 20 * np.log10(rms / rms.max())
    voiced = np.flatnonzero(db > -top_db)
    if voiced.size == 0:
        return x
    pad = int(sr * pad_ms / 1000)
    start = max(0, voiced[0] * frame - pad)
    end = min(len(x), (voiced[-1] + 1) * frame + pad)
    return x[start:end]


_SILERO = None


def _silero():
    global _SILERO
    if _SILERO is None:
        from silero_vad import load_silero_vad
        _SILERO = load_silero_vad()
    return _SILERO


def speech_segments(x: np.ndarray, sr: int, threshold: float = 0.4) -> list[tuple[int, int]]:
    """Silero VAD speech spans in samples (sr must be 8k or 16k). Raises ImportError if silero-vad is missing."""
    import torch
    from silero_vad import get_speech_timestamps
    ts = get_speech_timestamps(torch.from_numpy(x), _silero(), sampling_rate=sr, threshold=threshold,
                               min_speech_duration_ms=100, min_silence_duration_ms=100, speech_pad_ms=0)
    return [(t["start"], t["end"]) for t in ts]


def estimate_snr_db(x: np.ndarray, segs: list[tuple[int, int]], sr: int, min_noise_s: float = 0.2) -> float | None:
    """Speech-frame power vs non-speech power. None when there is too little non-speech to measure noise."""
    mask = np.zeros(len(x), dtype=bool)
    for a, b in segs:
        mask[a:b] = True
    if mask.sum() == 0 or (~mask).sum() < min_noise_s * sr:
        return None
    ps, pn = float(np.mean(x[mask] ** 2)), float(np.mean(x[~mask] ** 2))
    return 10 * np.log10(max(ps - pn, 1e-12) / max(pn, 1e-12))


def denoise(x: np.ndarray, sr: int, strength: float, noise: np.ndarray | None = None) -> np.ndarray:
    import noisereduce as nr
    if noise is not None:  # real noise profile from non-speech parts -> stationary gating
        y = nr.reduce_noise(y=x, sr=sr, y_noise=noise, stationary=True, prop_decrease=strength)
    else:
        y = nr.reduce_noise(y=x, sr=sr, stationary=False, prop_decrease=strength)
    return y.astype(np.float32)


def normalize(x: np.ndarray, target_dbfs: float, peak_limit: float) -> np.ndarray:
    rms = float(np.sqrt(np.mean(x ** 2)))
    if rms < 1e-8:  # digital silence: nothing to scale
        return x
    y = x * (10 ** (target_dbfs / 20) / rms)
    peak = float(np.abs(y).max())
    if peak > peak_limit:
        y = y * (peak_limit / peak)
    return y.astype(np.float32)


def preprocess_array(x: np.ndarray, sr: int, cfg: PreprocessConfig | None = None) -> tuple[np.ndarray, int]:
    """Run the whole chain on an in-memory signal. Returns (float32 mono [T], cfg.target_sr)."""
    cfg = cfg or PreprocessConfig()
    x = to_mono(x).astype(np.float32)
    x = x - x.mean() if len(x) else x                     # DC offset first: resample_poly zero-pads, a DC step would ring at the edges
    x = resample(x, sr, cfg.target_sr)
    if cfg.highpass_hz:
        x = highpass(x, cfg.target_sr, cfg.highpass_hz)
    segs = None
    if cfg.vad == "silero" and (cfg.trim_silence or cfg.denoise == "auto") and len(x):
        try:
            segs = speech_segments(x, cfg.target_sr, cfg.vad_threshold)
        except ImportError:
            segs = None  # silero-vad not installed: fall back to energy trim, SNR gate disabled
    if cfg.denoise != "off" and len(x):
        snr = estimate_snr_db(x, segs, cfg.target_sr) if segs else None
        if cfg.denoise == "always" or (snr is not None and snr < cfg.denoise_below_snr_db):
            noise = None
            if segs:
                m = np.ones(len(x), dtype=bool)
                for a, b in segs:
                    m[a:b] = False
                if m.sum() > 0.5 * cfg.target_sr:
                    noise = x[m]
            x = denoise(x, cfg.target_sr, cfg.denoise_strength, noise)
    if cfg.trim_silence and len(x):
        if segs:
            pad = int(cfg.target_sr * cfg.trim_pad_ms / 1000)
            x = x[max(0, segs[0][0] - pad): min(len(x), segs[-1][1] + pad)]
        elif segs is None:
            x = trim_silence(x, cfg.target_sr, cfg.trim_top_db, cfg.trim_pad_ms, cfg.frame_ms)
    if cfg.normalize:
        x = normalize(x, cfg.target_dbfs, cfg.peak_limit)
    return x, cfg.target_sr


def preprocess_file(path: str | Path, out_path: str | Path | None = None,
                    cfg: PreprocessConfig | None = None) -> tuple[np.ndarray, int]:
    """Load `path`, preprocess, optionally write the result to `out_path` (wav, 16-bit)."""
    x, sr = load_wav(path)
    wav, out_sr = preprocess_array(x, sr, cfg)
    if out_path is not None:
        save_wav(out_path, wav, out_sr)
    return wav, out_sr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", nargs="+", help="input wav file(s)")
    ap.add_argument("--out-dir", default="preprocessed", help="where the cleaned wavs go (same file names)")
    ap.add_argument("--sr", type=int, default=PreprocessConfig.target_sr)
    ap.add_argument("--no-trim", action="store_true")
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--denoise", choices=["off", "always", "auto"], default="off")
    ap.add_argument("--vad", choices=["silero", "energy"], default="silero")
    ap.add_argument("--highpass", type=float, default=PreprocessConfig.highpass_hz, help="Hz; 0 disables")
    a = ap.parse_args()

    cfg = replace(PreprocessConfig(), target_sr=a.sr, trim_silence=not a.no_trim,
                  normalize=not a.no_normalize, highpass_hz=a.highpass or None,
                  denoise=a.denoise, vad=a.vad)
    for p in a.audio:
        out = Path(a.out_dir) / (Path(p).stem + ".wav")
        wav, sr = preprocess_file(p, out, cfg)
        print(f"{p} -> {out}  ({len(wav) / sr:.2f}s @ {sr} Hz)")


if __name__ == "__main__":
    main()
