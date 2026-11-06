"""Two ChunkFormer CTC models (Vietnamese, English) on the same audio, with frame -> time mapping.

Frame time: the encoder subsamples fbank frames (10 ms) by `subsampling_rate` (8 for ChunkFormer) -> 80 ms/frame.
Both models return log-probs per frame; confidence uses temperature scaling: softmax(logits / T).
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch

BLANK = 0
SPECIAL = {"<blank>", "<unk>", "<sos/eos>", "<s>", "</s>", "<pad>"}
SPACE_PIECES = {"<space>", " ", "|", "▁"}


@dataclass
class Word:
    text: str
    start_ms: float
    end_ms: float
    conf: float


def temp_scale(logp, T):
    """log_softmax(logits / T) from log-probs (they differ from logits by a per-frame constant)."""
    return logp if T == 1.0 else torch.log_softmax(logp / T, dim=-1)


class ChunkformerCTC:
    def __init__(self, name, ckpt=None, device="cuda:0", chunk_size=64, left=128, right=128):
        from chunkformer import ChunkFormerModel
        self.m = ChunkFormerModel.from_pretrained(name)
        if ckpt:  # fine-tuned weights on top of `name`
            sd = torch.load(ckpt, map_location="cpu")
            sd = sd.get("model", sd.get("state_dict", sd))
            self.m.model.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()}, strict=True)
        self.m = self.m.to(device).eval()
        self.device, self.chunk, self.left, self.right = device, chunk_size, left, right
        self.id2tok = dict(self.m.char_dict)  # id -> token
        self.tok2id = {t: i for i, t in self.id2tok.items()}
        self.frame_ms = 10.0 * self.m.model.encoder.embed.subsampling_rate

    @torch.no_grad()
    def log_probs(self, audio_path):
        """(T, V) CTC log-probs on CPU, one row per encoder frame (frame_ms each)."""
        x, x_len = self.m._load_audio_and_extract_features(str(audio_path))
        lens = torch.tensor([x_len], dtype=torch.int, device=self.m.device)
        offset = torch.zeros(1, dtype=torch.int, device=self.m.device)
        enc, enc_lens, *_ = self.m.model.encoder.forward_parallel_chunk(
            xs=[x], xs_origin_lens=lens, chunk_size=self.chunk, left_context_size=self.left,
            right_context_size=self.right, offset=offset)
        lp = self.m.model.ctc.log_softmax(enc)  # (n_chunks, T_chunk, V)
        return lp.reshape(-1, lp.shape[-1])[: int(enc_lens[0])].float().cpu()

    # ---- text <-> ids (for calibration: CTC loss of a reference)
    def encode(self, text):
        """Greedy longest-match tokenization of `text` into vocab ids (SentencePiece-style '▁' or char vocab)."""
        ids, has_sp = [], any(t.startswith("▁") and len(t) > 1 for t in self.tok2id)
        for w in text.split():
            s = ("▁" + w) if has_sp else w
            i = 0
            while i < len(s):
                for j in range(len(s), i, -1):
                    if s[i:j] in self.tok2id:
                        ids.append(self.tok2id[s[i:j]])
                        i = j
                        break
                else:
                    i += 1  # character not in vocab: dropped
            if not has_sp:
                sp = next((self.tok2id[t] for t in SPACE_PIECES if t in self.tok2id), None)
                if sp is not None:
                    ids.append(sp)
        return ids

    def decode(self, ids):
        out = [self.id2tok[i] for i in ids if i != BLANK and self.id2tok.get(i) not in SPECIAL]
        return "".join(out).replace("▁", " ").replace("<space>", " ").strip()


def greedy_words(lp, id2tok, frame_ms, agg="min"):
    """CTC greedy path -> words with start/end (ms) and confidence.
    Token confidence = peak probability over the frames the token occupies; word confidence = min/mean of its tokens."""
    probs, ids = lp.exp(), lp.argmax(-1).tolist()
    words, cur = [], None  # cur: [text, start_frame, end_frame, [token confs]]
    t = 0
    while t < len(ids):
        k = ids[t]
        e = t
        while e + 1 < len(ids) and ids[e + 1] == k:
            e += 1
        if k != BLANK and id2tok.get(k) not in SPECIAL:
            piece = id2tok[k]
            conf = float(probs[t:e + 1, k].max())
            boundary = piece.startswith("▁") or piece in SPACE_PIECES
            text = "" if piece in SPACE_PIECES else piece.removeprefix("▁")
            if boundary or cur is None:
                if cur and cur[0]:
                    words.append(cur)
                cur = [text, t, e + 1, [conf] if text else []]
            elif text:
                cur[0] += text
                cur[2] = e + 1
                cur[3].append(conf)
        t = e + 1
    if cur and cur[0]:
        words.append(cur)
    f = (lambda c: min(c)) if agg == "min" else (lambda c: sum(c) / len(c))
    return [Word(w[0], w[1] * frame_ms, w[2] * frame_ms, f(w[3])) for w in words if w[3]]


@dataclass
class DualOutput:
    vi_lp: torch.Tensor  # temperature-scaled (T_vi)
    en_lp: torch.Tensor  # temperature-scaled (T_en)
    vi_frame_ms: float
    en_frame_ms: float
    words: list  # Vi greedy words with conf


class DualChunkformer:
    """Runs Chunkformer-Vi and Chunkformer-En in parallel (2 threads; use 2 GPUs for real overlap)."""

    def __init__(self, cfg):
        m, c, k = cfg["models"], cfg["chunkformer"], cfg["ctc"]
        kw = dict(chunk_size=c["chunk_size"], left=c["left_context_size"], right=c["right_context_size"])
        self.vi = ChunkformerCTC(m["chunkformer_vi"], m.get("vi_ckpt"), m["vi_device"], **kw)
        self.en = ChunkformerCTC(m["chunkformer_en"], m.get("en_ckpt"), m["en_device"], **kw)
        self.T_vi, self.T_en, self.agg = k["temperature_vi"], k["temperature_en"], k["conf_agg"]
        self.pool = ThreadPoolExecutor(max_workers=2)

    def run(self, audio_path):
        fv, fe = self.pool.submit(self.vi.log_probs, audio_path), self.pool.submit(self.en.log_probs, audio_path)
        vi_lp, en_lp = temp_scale(fv.result(), self.T_vi), temp_scale(fe.result(), self.T_en)
        return DualOutput(vi_lp, en_lp, self.vi.frame_ms, self.en.frame_ms,
                          greedy_words(vi_lp, self.vi.id2tok, self.vi.frame_ms, self.agg))
