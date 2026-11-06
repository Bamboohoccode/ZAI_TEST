"""Uncertain-span detection on the Vi greedy words, and time-aligned Vi/En candidate extraction.

For each span of low-confidence words [i, j): window = [t_start - pad, t_end + pad] (ms). The same window is cut from
the Vi and En CTC posteriors, and a CTC prefix beam search gives each language an N-best list.
"""
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field

import numpy as np

from .dual_chunkformer import BLANK, SPECIAL

NEG = -1e30


def _lse(a, b):
    if a < b:
        a, b = b, a
    return a if b <= NEG else a + math.log1p(math.exp(b - a))


def ctc_prefix_beam_search(logp, beam=8, n_best=4, topk=12):
    """logp: (T, V) numpy log-probs. Returns [(token_ids tuple, log score)] best first (blank-free, collapsed)."""
    beams = {(): (0.0, NEG)}  # prefix -> (log p ending in blank, log p ending in non-blank)
    for row in logp:
        cand = np.argpartition(row, -min(topk, len(row)))[-topk:]
        nxt = defaultdict(lambda: [NEG, NEG])
        for prefix, (pb, pnb) in beams.items():
            tot = _lse(pb, pnb)
            for c in cand:
                p = float(row[c])
                if c == BLANK:
                    nxt[prefix][0] = _lse(nxt[prefix][0], tot + p)
                    continue
                ext = prefix + (int(c),)
                if prefix and prefix[-1] == c:  # repeat: needs a blank in between to be a new token
                    nxt[ext][1] = _lse(nxt[ext][1], pb + p)
                    nxt[prefix][1] = _lse(nxt[prefix][1], pnb + p)
                else:
                    nxt[ext][1] = _lse(nxt[ext][1], tot + p)
        beams = dict(sorted(nxt.items(), key=lambda kv: -_lse(*kv[1]))[:beam])
    out = sorted(((p, _lse(*s)) for p, s in beams.items()), key=lambda x: -x[1])
    return out[:n_best]


@dataclass
class Slot:
    id: int
    word_start: int
    word_end: int
    t_start_ms: float
    t_end_ms: float
    base: str  # Vi greedy text of the span
    vi: list = field(default_factory=list)  # Vi candidates (base first)
    en: list = field(default_factory=list)  # En candidates

    def to_dict(self):
        return asdict(self)


@dataclass
class Aligned:
    base_text: str
    sentence: str  # base text with [SLOT_k] in place of the uncertain spans
    slots: list


class TimeAligner:
    def __init__(self, cfg, vi_model, en_model):
        self.c = cfg["ctc"]
        self.vi, self.en = vi_model, en_model  # ChunkformerCTC (id2tok / decode)

    def spans(self, words):
        """Merge adjacent low-confidence words into (i, j) spans; long spans are split at max_span_words."""
        low = [w.conf < self.c["conf_threshold"] for w in words]
        spans, i = [], 0
        while i < len(words):
            if not low[i]:
                i += 1
                continue
            j = i + 1
            while j < len(words):
                if low[j]:
                    j += 1
                    continue
                k = j  # allow a gap of <= merge_gap_words confident words between two uncertain ones
                while k < len(words) and not low[k]:
                    k += 1
                if k < len(words) and k - j <= self.c["merge_gap_words"]:
                    j = k + 1
                else:
                    break
            for a in range(i, j, self.c["max_span_words"]):
                spans.append((a, min(a + self.c["max_span_words"], j)))
            i = j
        return spans

    def _nbest(self, model, lp, frame_ms, t0, t1):
        a, b = max(int(t0 // frame_ms), 0), min(int(math.ceil(t1 / frame_ms)), lp.shape[0])
        if b <= a:
            return []
        hyps = ctc_prefix_beam_search(lp[a:b].numpy(), self.c["beam_size"], self.c["n_best"], self.c["prune_topk"])
        return [t for t in dict.fromkeys(model.decode(list(ids)) for ids, _ in hyps) if t]

    def build(self, out):
        words = out.words
        spans = self.spans(words)
        base_text = " ".join(w.text for w in words)
        slots, parts, pos = [], [], 0
        for k, (i, j) in enumerate(spans, 1):
            t0, t1 = words[i].start_ms, words[j - 1].end_ms
            w0, w1 = max(t0 - self.c["pad_ms"], 0.0), t1 + self.c["pad_ms"]
            base = " ".join(w.text for w in words[i:j])
            vi = list(dict.fromkeys([base, *self._nbest(self.vi, out.vi_lp, out.vi_frame_ms, t0, t1)]))  # Vi: exact span; En: padded window
            en = [t for t in self._nbest(self.en, out.en_lp, out.en_frame_ms, w0, w1) if t not in vi]
            slots.append(Slot(k, i, j, t0, t1, base, vi, en))
            parts += [w.text for w in words[pos:i]] + [f"[SLOT_{k}]"]
            pos = j
        parts += [w.text for w in words[pos:]]
        return Aligned(base_text, " ".join(parts), slots)
