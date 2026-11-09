"""Zipformer RNNT (sherpa-onnx) ASR: hynt/ZipFormer-150M-CR-CTC-RNNT-6000h. Chạy offline hoàn toàn (chỉ đọc model từ thư mục
local hoặc HF cache, không tải).

    asr = ZipformerASR("hynt/ZipFormer-150M-CR-CTC-RNNT-6000h")                      # HF id (đã có trong cache) hoặc thư mục
    asr = ZipformerASR(..., hotwords_file="hotwords/hotwords_global.txt")             # hotword cố định, nạp 1 lần lúc khởi tạo
    asr = ZipformerASR(..., lm="lm/rnnlm/rnnlm.onnx", lm_scale=0.3)            # LM fusion (RNNLM, train_rnnlm.py)
    asr = ZipformerASR(..., vad=True)                                         # cắt khoảng lặng bằng silero VAD (vad.py)
    asr.transcribe(["a.wav"])                                                 # -> ["..."] chữ thường, không dấu câu
    asr.transcribe(["a.wav"], hotwords=["Zalo"])                              # (kiểu cũ) hotword theo từng lần gọi

hotwords / LM chỉ có tác dụng với decoding_method="modified_beam_search" (tự bật khi dùng bias / hotwords_file / lm).
"""
import os
import tempfile
from pathlib import Path

import numpy as np

DEFAULT_REPO = "hynt/ZipFormer-150M-CR-CTC-RNNT-6000h"


def _resolve_dir(name):
    """thư mục local, hoặc snapshot đã có trong HF cache (local_files_only: không bao giờ tải từ mạng)."""
    if Path(name).expanduser().is_dir():
        return Path(name).expanduser()
    from huggingface_hub import try_to_load_from_cache
    try:
        p = try_to_load_from_cache(name, "bpe.model")  # snapshot có thể "incomplete" (thiếu README...) vẫn dùng được
        if isinstance(p, str) and any(Path(p).parent.glob("encoder-*.onnx")):
            return Path(p).parent
        raise FileNotFoundError(name)
    except Exception as e:  # noqa: BLE001
        raise FileNotFoundError(f"không thấy model '{name}' trong thư mục local hay HF cache (chế độ offline, không tải). "
                                f"Truyền đường dẫn thư mục chứa encoder/decoder/joiner-*.onnx + bpe.model.") from e


def model_dir(name=DEFAULT_REPO):
    return _resolve_dir(name)


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
    def __init__(self, name=DEFAULT_REPO, int8=False, num_threads=4, bias=True, hotwords_score=2.0, beam=4,
                 hi=2.5, lo=1.5, glossary_score=1.0, hotwords_file=None, lm=None, lm_scale=0.3, vad=None,
                 decoding=None):
        """bias: hotword theo từng lần gọi (transcribe(hotwords=...)). hotwords_file: file "CỤM :score" cố định
        (mine_hotwords.py). lm: RNNLM onnx (train_rnnlm.py). vad: True | vad.VAD | None.
        decoding: ép "greedy_search" / "modified_beam_search" (mặc định: beam nếu có bias/hotwords_file/lm)."""
        import sherpa_onnx
        d = _resolve_dir(name)
        # "encoder-*.onnx" cũng khớp "*.int8.onnx" -> lọc rõ ràng theo int8
        pick = lambda n: str(next(p for p in sorted(d.glob(f"{n}-*.onnx")) if p.name.endswith(".int8.onnx") == int8))
        self.dir = d
        self.bias = bias or bool(hotwords_file)
        self.hi, self.lo, self.glossary_score = hi, lo, glossary_score  # điểm hotword mặc định (xem bias.hotword_lines)
        tokens = _ensure_tokens(d)
        kw = {}
        if hotwords_file:
            if not Path(hotwords_file).exists():
                raise FileNotFoundError(hotwords_file)
            kw = dict(hotwords_file=str(hotwords_file), hotwords_score=hotwords_score, modeling_unit="bpe",
                      bpe_vocab=str(d / "bpe.vocab"))
        elif bias:  # hotwords_file phải có mặt lúc khởi tạo để bật contextual graph; hotword thật truyền theo từng stream
            self._boot = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
            self._boot.write("ZALO :1.0\n")
            self._boot.close()
            kw = dict(hotwords_file=self._boot.name, hotwords_score=hotwords_score, modeling_unit="bpe",
                      bpe_vocab=str(d / "bpe.vocab"))
        if lm:
            if not Path(lm).exists():
                raise FileNotFoundError(lm)
            kw.update(lm=str(lm), lm_scale=lm_scale)
        method = decoding or ("modified_beam_search" if (self.bias or lm) else "greedy_search")
        self.rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=pick("encoder"), decoder=pick("decoder"), joiner=pick("joiner"), tokens=str(tokens), num_threads=num_threads,
            sample_rate=16000, feature_dim=80, max_active_paths=beam, decoding_method=method, **kw)
        if vad is True:
            from asr.vad import VAD
            vad = VAD()
        self.vad = vad or None

    def __del__(self):
        b = getattr(self, "_boot", None)
        if b and os.path.exists(b.name):
            os.unlink(b.name)

    def transcribe(self, paths, hotwords=None, score=None, hi=None, lo=None, glossary_score=None, words=False):
        """paths: list wav. hotwords: list[str] entity dùng chung cho cả lô (hoặc None). Trả list[str] chữ thường.
        words=True -> mỗi file trả [(từ, conf)] (xem word_confidences) thay vì chuỗi.
        Mặc định chia điểm: tên riêng hi, cụm thường lo, từ ngoại viết thường của glossary glossary_score (0 = bỏ).
        score=<số> -> kiểu cũ, mọi hotword cùng một điểm. Không truyền hotwords -> dùng hotwords_file (nếu có)."""
        import soundfile as sf
        hw = None
        if hotwords and self.bias:
            from hotwords.bias import hotword_lines, hotwords as fmt
            hw = fmt(hotwords, score) if score is not None else hotword_lines(
                hotwords, self.hi if hi is None else hi, self.lo if lo is None else lo,
                self.glossary_score if glossary_score is None else glossary_score)
        streams = []
        for p in paths:
            x, sr = sf.read(str(p), dtype="float32")
            x = x.mean(1) if x.ndim > 1 else x
            if self.vad is not None:
                x = self.vad.trim(x, sr)
            s = self.rec.create_stream(hw) if hw else self.rec.create_stream()
            s.accept_waveform(sr, np.ascontiguousarray(x))
            streams.append(s)
        self.rec.decode_streams(streams)
        if words:
            return [word_confidences(s.result) for s in streams]
        return [s.result.text.strip().lower() for s in streams]


def word_confidences(result):
    """[(từ, conf)] từ kết quả sherpa-onnx: gộp token BPE thành từ (token mở đầu bằng khoảng trắng / '▁' = từ mới),
    conf = exp(log-prob nhỏ nhất của các token trong từ), trong khoảng (0, 1]."""
    import math
    lps = list(getattr(result, "ys_log_probs", None) or [])
    toks = list(result.tokens)
    if len(lps) != len(toks):  # bản sherpa-onnx không trả log-prob -> coi như chắc chắn hết
        lps = [0.0] * len(toks)
    out, pending = [], False
    for t, lp in zip(toks, lps):
        new = t.startswith((" ", "▁")) or not out or pending
        piece = t.lstrip(" ▁").lower()
        if not piece:  # token "▁" đứng riêng (BPE tách "▁ổ" -> "▁" + "ổ"): token sau mở từ mới, không dính vào từ trước
            pending = True
            continue
        pending = False
        if new:
            out.append([piece, lp])
        else:
            out[-1][0] += piece
            out[-1][1] = min(out[-1][1], lp)
    return [(w, math.exp(lp)) for w, lp in out]
