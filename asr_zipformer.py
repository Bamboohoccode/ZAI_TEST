"""Zipformer RNNT (sherpa-onnx) ASR: hynt/Zipformer-30M-RNNT-6000h, hỗ trợ hotwords theo từng lần gọi.

    asr = ZipformerASR("hynt/Zipformer-30M-RNNT-6000h")       # HF id hoặc thư mục local
    asr.transcribe(["a.wav"], hotwords=["Zalo", "Cô giáo Mai"])   # -> ["... zalo ..."] (chữ thường, không dấu câu)

Hotwords chỉ có tác dụng khi decoding_method="modified_beam_search" (mặc định khi bias=True).
"""
import os
import tempfile
from pathlib import Path

import numpy as np

DEFAULT_REPO = "hynt/Zipformer-30M-RNNT-6000h"


def _resolve_dir(name):
    if Path(name).is_dir():
        return Path(name)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(name, allow_patterns=["*.onnx", "bpe.model", "tokens.txt", "config.json"]))


def _ensure_tokens(d):
    """repo chỉ có bpe.model; sherpa-onnx cần tokens.txt (id-ordered). Ghi vào thư mục cache nếu thiếu."""
    tok = d / "tokens.txt"
    if not tok.exists():
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=str(d / "bpe.model"))
        tok.write_text("".join(f"{sp.id_to_piece(i)} {i}\n" for i in range(sp.get_piece_size())), encoding="utf-8")
    vocab = d / "bpe.vocab"  # "piece<TAB>score", dùng cho hotwords (modeling_unit=bpe)
    if not vocab.exists():
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=str(d / "bpe.model"))
        vocab.write_text("".join(f"{sp.id_to_piece(i)}\t{sp.get_score(i)}\n" for i in range(sp.get_piece_size())), encoding="utf-8")
    return tok


class ZipformerASR:
    def __init__(self, name=DEFAULT_REPO, int8=False, num_threads=4, bias=True, hotwords_score=2.0, beam=4):
        import sherpa_onnx
        d = _resolve_dir(name)
        suf = ".int8.onnx" if int8 else ".onnx"
        self.bias = bias
        tokens = _ensure_tokens(d)
        kw = {}
        if bias:  # hotwords_file phải có mặt lúc khởi tạo để bật contextual graph; hotword thật truyền theo từng stream
            self._boot = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
            self._boot.write("ZALO :1.0\n")
            self._boot.close()
            kw = dict(hotwords_file=self._boot.name, hotwords_score=hotwords_score, modeling_unit="bpe",
                      bpe_vocab=str(d / "bpe.vocab"))
        self.rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(next(d.glob(f"encoder-*{suf}"))), decoder=str(next(d.glob(f"decoder-*{suf}"))),
            joiner=str(next(d.glob(f"joiner-*{suf}"))), tokens=str(tokens), num_threads=num_threads,
            sample_rate=16000, feature_dim=80, max_active_paths=beam,
            decoding_method="modified_beam_search" if bias else "greedy_search", **kw)

    def __del__(self):
        b = getattr(self, "_boot", None)
        if b and os.path.exists(b.name):
            os.unlink(b.name)

    def transcribe(self, paths, hotwords=None, score=2.0):
        """paths: list wav. hotwords: list[str] dùng chung cho cả lô (hoặc None). Trả list[str] chữ thường."""
        import soundfile as sf
        hw = None
        if hotwords and self.bias:
            from bias import hotwords as fmt
            hw = fmt(hotwords, score)
        streams = []
        for p in paths:
            x, sr = sf.read(str(p), dtype="float32")
            x = x.mean(1) if x.ndim > 1 else x
            s = self.rec.create_stream(hw) if hw else self.rec.create_stream()
            s.accept_waveform(sr, np.ascontiguousarray(x))
            streams.append(s)
        self.rec.decode_streams(streams)
        return [s.result.text.strip().lower() for s in streams]
