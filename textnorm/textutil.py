"""Tiện ích văn bản dùng chung (ASR, mapper, eval): norm về dạng giống output ASR, WER theo từ."""
import re


def norm(s):
    """Dạng giống output ASR: chữ thường, bỏ dấu câu (giữ '.'/',' giữa hai chữ số: 2.5, 1,5)."""
    s = re.sub(r"(?<!\d)[^\w\s]|[^\w\s](?!\d)", " ", s.lower())
    return " ".join(s.split())


assert norm("Giá 2.5 triệu, nhé!") == "giá 2.5 triệu nhé"
assert norm("À mà thôi... cứ để; SJC") == "à mà thôi cứ để sjc"


def wer_counts(ref, hyp):
    """(số lỗi edit, số từ ref) — Levenshtein trên từ."""
    r, h = ref.split(), hyp.split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)], len(r)


assert wer_counts("a b c", "a x c") == (1, 3) and wer_counts("a b", "") == (2, 2)
