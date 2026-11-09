"""Đánh dấu từ ASR có confidence thấp + ứng viên gần âm, để LLM sửa lại (pipeline.py --refine).

    cands = Candidates.from_train("data/asr_train_tn.jsonl")      # học từ cặp hyp (ASR) / ref_spoken của train
    marked, n = cands.mark([("sẵn", 0.1), ("lo", 0.9), ...], thr=0.6)
    # -> "[sẵn|xăng|sang] lo chín lăm ...", n = số từ được đánh dấu

Ứng viên của một từ w (không có luật viết tay):
  1. bảng nhầm lẫn: ASR nghe "w" mà ref là "x" (căn hyp/ref bằng difflib, cặp thay thế 1-1), xếp theo số lần;
  2. từ trong từ vựng train có khoá âm (hotwords.bias.phon: bỏ dấu, gộp s/x, ch/tr, d/gi/r...) gần với w, xếp theo tần suất.
Bỏ các câu held-out (id % 10 == 9) để còn đánh giá được.
"""
import difflib
import json
from collections import Counter, defaultdict

from hotwords.bias import phon, ratio


def _heldout(item_id):
    try:
        return int(str(item_id).rsplit("_", 1)[-1]) % 10 == 9
    except ValueError:
        return False


class Candidates:
    def __init__(self, confusion, vocab):
        self.confusion = confusion          # {từ ASR: Counter(từ đúng)}
        self.vocab = vocab                  # Counter(từ đúng)
        self._phon = {}
        for w in vocab:
            self._phon.setdefault(phon(w), []).append(w)

    @classmethod
    def from_train(cls, path, include_heldout=False):
        confusion, vocab = defaultdict(Counter), Counter()
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if not include_heldout and _heldout(r.get("id")):
                continue
            hyp, ref = r.get("hyp", "").lower().split(), r.get("ref_spoken", "").lower().split()
            vocab.update(ref)
            sm = difflib.SequenceMatcher(None, hyp, ref, autojunk=False)
            for op, i1, i2, j1, j2 in sm.get_opcodes():
                if op == "replace" and i2 - i1 == j2 - j1:
                    for h, t in zip(hyp[i1:i2], ref[j1:j2]):
                        confusion[h][t] += 1
        return cls(confusion, vocab)

    def candidates(self, word, k=3, min_ratio=0.75, min_count=2):
        """bảng nhầm lẫn (gặp >= min_count lần) trước, rồi từ gần âm: giống âm hơn trước, cùng mức thì từ hay gặp trước."""
        out = [t for t, n in self.confusion.get(word, Counter()).most_common(k) if t != word and n >= min_count]
        if len(out) < k:
            pw = phon(word)
            near = []
            for key, ws in self._phon.items():
                r = ratio(pw, key)
                if r >= min_ratio:
                    near += [(round(r, 2), self.vocab[v], v) for v in ws if v != word and v not in out]
            out += [v for *_, v in sorted(near, reverse=True)[:k - len(out)]]
        return out

    def mark(self, words, thr=0.6, k=3):
        """words: [(từ, conf)] -> (chuỗi có [từ|ứng viên...] ở chỗ conf < thr, số từ được đánh dấu).
        Từ có conf thấp mà không có ứng viên vẫn được đánh dấu dạng [từ] để LLM biết chỗ nghi ngờ."""
        parts, n = [], 0
        for w, c in words:
            if c < thr:
                parts.append("[" + "|".join([w] + self.candidates(w, k)) + "]")
                n += 1
            else:
                parts.append(w)
        return " ".join(parts), n


REFINE_SYSTEM = """Bạn sửa lỗi nhận dạng giọng nói (ASR) tiếng Việt cho câu người dùng nói với trợ lý ảo \
(đặt lịch, báo thức, chuyển tiền, nhắn tin, gọi xe, đặt vé, nhạc, tivi, nhà thông minh, giá xăng, giá vàng...).
- Chỗ nghi ngờ được đánh dấu [a|b|c]: a là từ ASR nghe được, b, c là các từ có âm gần giống.
- Mỗi chỗ đánh dấu, chọn từ hợp ngữ cảnh nhất: giữ a, chọn b/c, hoặc một từ khác đọc gần giống a nếu rõ ràng đúng hơn.
- Từ ngoài ngoặc giữ nguyên, trừ khi sai hiển nhiên (ví dụ "sang chín mươi lăm" là "xăng chín mươi lăm").
- Giữ nguyên tên riêng, từ địa phương, từ tiếng Anh, không thêm bớt ý.
- Viết chữ thường, không dấu câu, số giữ dạng chữ (hai mươi lăm, không viết 25).
- Chỉ trả về một dòng là câu đã sửa, không giải thích."""


def refine_conv(marked, context=()):
    user = marked if not context else "Các câu trước (chỉ để tham khảo):\n" + "\n".join(context) + "\n\nCâu cần sửa:\n" + marked
    return [{"role": "system", "content": REFINE_SYSTEM}, {"role": "user", "content": user}]


def accept_refine(raw, out):
    """LLM trả về rác / viết lại cả câu -> giữ câu ASR. Chấp nhận khi số từ lệch ít và phần lớn từ còn giữ."""
    out = (out or "").strip().splitlines()[0].strip().lower() if (out or "").strip() else ""
    out = out.strip(" .\"'")
    if not out or "[" in out:
        return raw
    a, b = raw.split(), out.split()
    if abs(len(a) - len(b)) > max(2, len(a) // 5):
        return raw
    same = sum(x.size for x in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks())
    return out if same >= 0.6 * len(a) else raw


def refine(llm, word_lists, cands, thr=0.6, k=3, use_context=True):
    """word_lists: [[(từ, conf)]] các lượt của một sample (history trước, query cuối).
    Trả (texts, marked): chỉ gọi LLM cho lượt có từ được đánh dấu, gọi một lô."""
    raws = [" ".join(w for w, _ in wl) for wl in word_lists]
    marked = [cands.mark(wl, thr, k) for wl in word_lists]
    idx = [i for i, (_, n) in enumerate(marked) if n]
    texts = list(raws)
    if idx and llm is not None:
        convs = [refine_conv(marked[i][0], raws[max(0, i - 2):i] if use_context else ()) for i in idx]
        for i, out in zip(idx, llm.chat(convs)):
            texts[i] = accept_refine(raws[i], out)
    return texts, [m for m, _ in marked]
