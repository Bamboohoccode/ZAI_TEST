"""Tiện ích văn bản dùng chung (ASR, eval, LM): norm về dạng giống output ASR, WER theo từ, chia held-out.

WER so từ sau canon(): bỏ khác biệt KHÔNG phải lỗi nhận dạng
  - vị trí dấu thanh kiểu cũ / mới: huỷ = hủy, khoá = khóa, hoá = hóa, thuý = thúy;
  - biến thể chính tả cùng âm hay gặp trong nhãn: dùm = giùm (VARIANTS).
"""
import re
import unicodedata


def norm(s):
    """Dạng giống output ASR: chữ thường, bỏ dấu câu (giữ '.'/',' giữa hai chữ số: 2.5, 1,5)."""
    s = re.sub(r"(?<!\d)[^\w\s]|[^\w\s](?!\d)", " ", s.lower())
    return " ".join(s.split())


# dấu thanh (dấu kết hợp Unicode): huyền, sắc, hỏi, ngã, nặng
_TONES = {"̀": "1", "́": "2", "̉": "3", "̃": "4", "̣": "5"}
VARIANTS = {"dùm": "giùm"}


def canon(word):
    """khoá so sánh của một từ: bỏ dấu thanh khỏi chữ cái rồi gắn số thanh ở cuối (huỷ / hủy -> 'huy3')."""
    word = VARIANTS.get(word, word)
    d = unicodedata.normalize("NFD", word)
    tone = "".join(_TONES[c] for c in d if c in _TONES)
    base = unicodedata.normalize("NFC", "".join(c for c in d if c not in _TONES))
    return base + tone


def wer_counts(ref, hyp, canonical=True):
    """(số lỗi edit, số từ ref) — Levenshtein trên từ (canonical: so theo canon())."""
    r, h = ref.split(), hyp.split()
    if canonical:
        r, h = [canon(w) for w in r], [canon(w) for w in h]
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)], len(r)


def is_heldout(sid):
    """train_0019 -> held-out nếu số % 10 == 9 (chia theo sample, mọi lượt của một sample cùng phía).
    Held-out không dùng để khai thác hotword / train LM, chỉ để đo."""
    m = re.search(r"(\d+)$", str(sid))
    return bool(m) and int(m.group(1)) % 10 == 9
