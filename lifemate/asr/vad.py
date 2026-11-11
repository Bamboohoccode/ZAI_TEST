"""VAD (silero, chạy bằng sherpa-onnx, offline): cắt khoảng lặng / nhiễu đầu-cuối và rút ngắn khoảng lặng dài giữa câu
trước khi đưa vào ASR. Model lấy từ pip package `silero_vad` (silero_vad/data/silero_vad.onnx), không tải gì từ mạng.

    from lifemate.asr.vad import VAD
    vad = VAD()
    vad.segments(x, 16000)    # [(start_s, end_s), ...]
    y = vad.trim(x, 16000)    # audio đã cắt; không thấy tiếng nói / còn quá ngắn -> trả nguyên x

    python -m lifemate.asr.vad a.wav [--out a_trim.wav]                 # in các đoạn tiếng nói
    python -m lifemate.asr.vad --dir DATA/train/audio --stats [--n 500]  # thống kê % bị cắt, số file rơi vào fallback
"""
import argparse
import glob
import importlib.resources
import random
from pathlib import Path

import numpy as np

SR = 16000
WINDOW = 512  # silero 16 kHz


def _model_path():
    p = importlib.resources.files("silero_vad") / "data" / "silero_vad.onnx"
    if not Path(str(p)).exists():
        raise FileNotFoundError(f"không thấy {p}; cài `pip install silero-vad` (bản offline có sẵn model)")
    return str(p)


def _resample(x, sr):
    if sr == SR:
        return x
    import librosa
    return librosa.resample(x, orig_sr=sr, target_sr=SR)


class VAD:
    def __init__(self, threshold=0.5, min_silence=0.25, min_speech=0.15, pad=0.15, max_gap=0.5, gap_keep=0.1,
                 min_keep=0.3, model=None):
        import sherpa_onnx
        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = model or _model_path()
        cfg.silero_vad.threshold = threshold
        cfg.silero_vad.min_silence_duration = min_silence
        cfg.silero_vad.min_speech_duration = min_speech
        cfg.silero_vad.window_size = WINDOW
        cfg.silero_vad.max_speech_duration = 30
        cfg.sample_rate = SR
        cfg.num_threads = 1
        self._cfg = cfg
        self._vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)
        self.pad, self.max_gap, self.gap_keep, self.min_keep = pad, max_gap, gap_keep, min_keep

    def segments(self, x, sr=SR):
        """list (start_s, end_s) các đoạn tiếng nói (theo trục thời gian của x)."""
        x = _resample(np.ascontiguousarray(x, dtype=np.float32), sr)
        v = self._vad
        v.reset()
        for i in range(0, len(x), WINDOW):
            v.accept_waveform(x[i:i + WINDOW])
        v.flush()
        segs = []
        while not v.empty():
            s = v.front
            segs.append((s.start / SR, (s.start + len(s.samples)) / SR))
            v.pop()
        return segs

    def trim(self, x, sr=SR):
        """giữ [đầu đoạn nói đầu - pad, cuối đoạn nói cuối + pad]; khoảng lặng giữa > max_gap rút còn gap_keep.
        Fallback: không có đoạn nói hoặc phần giữ lại < min_keep giây -> trả nguyên x (không bao giờ làm mất câu)."""
        segs = self.segments(x, sr)
        if not segs:
            return x
        dur = len(x) / sr
        merged = []  # (start, end) đã pad, gộp các đoạn cách nhau <= max_gap
        for a, b in segs:
            a, b = max(0.0, a - self.pad), min(dur, b + self.pad)
            if merged and a - merged[-1][1] <= self.max_gap:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        if sum(b - a for a, b in merged) < self.min_keep:
            return x
        gap = np.zeros(int(self.gap_keep * sr), dtype=x.dtype)
        parts = []
        for k, (a, b) in enumerate(merged):
            if k:
                parts.append(gap)
            parts.append(x[int(a * sr):int(np.ceil(b * sr))])
        return np.concatenate(parts)


def main():
    import soundfile as sf
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", nargs="?")
    ap.add_argument("--out")
    ap.add_argument("--dir", help="thư mục audio (tìm đệ quy *.wav)")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--n", type=int, default=0, help="chỉ lấy ngẫu nhiên n file (0 = tất cả)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--pad", type=float, default=0.15)
    a = ap.parse_args()
    vad = VAD(threshold=a.threshold, pad=a.pad)

    if a.wav:
        x, sr = sf.read(a.wav, dtype="float32")
        x = x.mean(1) if x.ndim > 1 else x
        for s, e in vad.segments(x, sr):
            print(f"{s:7.2f} - {e:7.2f}")
        y = vad.trim(x, sr)
        print(f"{len(x) / sr:.2f}s -> {len(y) / sr:.2f}s")
        if a.out:
            sf.write(a.out, y, sr)
        return

    files = sorted(glob.glob(str(Path(a.dir).expanduser() / "**" / "*.wav"), recursive=True))
    if a.n:
        random.Random(0).shuffle(files)
        files = files[:a.n]
    tot_in = tot_out = 0.0
    fallback, cut = 0, []
    for f in files:
        x, sr = sf.read(f, dtype="float32")
        x = x.mean(1) if x.ndim > 1 else x
        y = vad.trim(x, sr)
        tot_in, tot_out = tot_in + len(x) / sr, tot_out + len(y) / sr
        fallback += len(y) == len(x)
        cut.append((len(x) - len(y)) / sr)
    cut = np.array(cut)
    print(f"{len(files)} file, {tot_in:.0f}s -> {tot_out:.0f}s (cắt {100 * (1 - tot_out / max(tot_in, 1e-9)):.1f}%), "
          f"giữ nguyên {fallback} file; cắt/file p50 {np.median(cut):.2f}s p95 {np.percentile(cut, 95):.2f}s max {cut.max():.2f}s")


if __name__ == "__main__":
    main()
