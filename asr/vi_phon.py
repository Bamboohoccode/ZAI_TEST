"""Khoảng cách phát âm giữa hai âm tiết tiếng Việt, có tính biến thể vùng miền (dùng cho ứng viên của asr/candidates.py).

Âm tiết được tách thành (phụ âm đầu, âm đệm w, âm chính, âm cuối, thanh) theo chính tả, rồi so từng phần:
cặp hay lẫn theo vùng miền / hay bị ASR nhầm tốn ít, khác hẳn tốn 1. Khoảng cách = tổng chi phí từng phần.

    parse("nghiêng")  -> Syl(onset='ng', medial='', nucleus='iê', coda='ng', tone=0)
    dist("sẵn", "xăng") -> 0.5    (s/x + n/ng: giọng Nam, giọng Bắc s=x)
    dist("mơ", "mưa")   -> 0.4

Biến thể (Wikipedia "Vietnamese phonology" + Hoàng Thị Châu, Phương ngữ học tiếng Việt):
- đầu: Bắc d = gi = r (/z/), ch = tr, s = x, l ~ n (đồng bằng Bắc Bộ); Nam d = gi = v (/j/), r rất biến động;
  quê Nam kh -> ph ("khỏe" -> "phẻ"); c/k/q, g/gh, ng/ngh chỉ là chính tả.
- đệm: Nam rụng w ("qu-", "hu-", "uy" -> "i": "xe buýt" -> "xe bít", "huyền" ~ "hìền").
- cuối: Nam -nh/-ch -> -n/-t, -n ~ -ng, -t ~ -c (sau i, e, ê, a); ông/ôc ~ ong/oc.
- chính: Nam nguyên âm ngắn trung hoá (ă ~ â ~ ơ), iê ~ i, ươ ~ ư, uô ~ u (rút gọn đôi), o ~ ô ~ ơ.
- thanh: Nam hỏi = ngã; ASR hay nhầm thanh (PiDA, arXiv 2606.12911: nhầm thanh ít hơn nhầm nguyên âm / phụ âm).
"""
import unicodedata
from functools import lru_cache
from typing import NamedTuple

# dấu thanh (Unicode tổ hợp): 0 ngang, 1 huyền, 2 sắc, 3 hỏi, 4 ngã, 5 nặng
_TONE = {"̀": 1, "́": 2, "̉": 3, "̃": 4, "̣": 5}
_ONSETS = ["ngh", "ng", "nh", "ch", "tr", "gh", "gi", "kh", "ph", "th", "qu",
           "b", "c", "d", "đ", "g", "h", "k", "l", "m", "n", "p", "r", "s", "t", "v", "x"]
_CODAS = ["ng", "nh", "ch", "c", "m", "n", "p", "t"]
_VOWELS = set("aăâeêioôơuưy")


class Syl(NamedTuple):
    onset: str
    medial: str
    nucleus: str
    coda: str
    tone: int


@lru_cache(maxsize=None)
def parse(word):
    """âm tiết chữ thường -> Syl; None nếu không phải âm tiết tiếng Việt (từ ngoại, số...)."""
    d = unicodedata.normalize("NFD", word.lower())
    tone = next((_TONE[c] for c in d if c in _TONE), 0)
    s = unicodedata.normalize("NFC", "".join(c for c in d if c not in _TONE))
    onset = next((o for o in _ONSETS if s.startswith(o)), "")
    rest = s[len(onset):]
    medial = ""
    if onset == "gi" and (not rest or rest[0] not in _VOWELS):   # gì, gìn: gi là phụ âm + âm chính i
        onset, rest = "gi", "i" + rest
    if onset == "qu":
        onset, medial = "c", "w"
    if onset in ("k", "q"):
        onset = "c"
    elif onset == "ngh":
        onset = "ng"
    elif onset == "gh":
        onset = "g"
    elif onset == "gi":
        onset = "d"   # gi và d cùng âm ở mọi vùng
    coda = next((c for c in _CODAS if rest.endswith(c) and len(rest) > len(c)), "")
    rest = rest[:len(rest) - len(coda)]
    # bán âm cuối: ai, ao, au, ay, oi, ui, iu, êu, iêu, ươi, uôi... (không tách uy, uơ: đó là âm đệm + âm chính)
    if not coda and len(rest) > 1 and rest[-1] in "iyou" and rest[-2:] not in ("uy", "uơ"):
        coda = "i" if rest[-1] in "iy" else "u"
        rest = rest[:-1]
        if rest == "a" and s.endswith(("ay", "au")):   # ay, au: âm chính ngắn ă
            rest = "ă"
    # âm đệm w: oa, oă, oe, uy, uâ, uê, uơ (qu đã xử lý ở trên)
    if not medial and len(rest) > 1 and (rest[0] == "o" and rest[1] in "aăe" or rest[0] == "u" and rest[1] in "yâêơ"):
        medial, rest = "w", rest[1:]
    if not rest or any(c not in _VOWELS for c in rest):
        return None
    nucleus = {"y": "i", "yê": "iê", "ya": "iê", "ia": "iê", "ua": "uô", "ưa": "ươ"}.get(rest, rest)
    return Syl(onset, medial, nucleus, coda, tone)


def _pairs(groups, cost):
    out = {}
    for g in groups:
        for a in g:
            for b in g:
                if a != b:
                    out[(a, b)] = min(out.get((a, b), 9), cost)
    return out


# chi phí khi khác nhau; thiếu trong bảng = 1.0
ONSET_COST = {**_pairs([("d", "r", "v"), ("ch", "tr"), ("s", "x"), ("l", "n")], 0.2),
              **_pairs([("kh", "ph"), ("t", "th"), ("đ", "t"), ("n", "nh"), ("h", "kh"), ("h", ""),
                        ("v", "b"), ("g", "ng"), ("h", "qu")], 0.5)}
MEDIAL_COST = 0.3
NUCLEUS_COST = {**_pairs([("ă", "â", "ơ"), ("iê", "i"), ("ươ", "ư"), ("uô", "u"), ("o", "ô"), ("ơ", "ươ"), ("ê", "e"),
                          ("a", "ă")], 0.25),
                **_pairs([("o", "ơ"), ("ô", "ơ"), ("a", "â"), ("u", "ư"), ("ơ", "ư"), ("i", "ê"), ("a", "ơ"),
                          ("uô", "ô"), ("ươ", "ư"), ("ô", "u"), ("o", "u")], 0.4)}
CODA_COST = {**_pairs([("n", "ng", "nh"), ("t", "c", "ch")], 0.2),
             **_pairs([("m", "n"), ("p", "t"), ("i", ""), ("u", ""), ("m", "ng"), ("p", "c")], 0.5)}
TONE_COST = {(3, 4): 0.1, (4, 3): 0.1, (2, 4): 0.25, (4, 2): 0.25, (0, 1): 0.25, (1, 0): 0.25, (3, 5): 0.25, (5, 3): 0.25,
             (4, 5): 0.25, (5, 4): 0.25}
TONE_OTHER = 0.35


def _c(table, a, b, other=1.0):
    return 0.0 if a == b else table.get((a, b), other)


@lru_cache(maxsize=1 << 20)
def dist(a, b):
    """khoảng cách phát âm giữa hai từ (một âm tiết); từ không parse được -> so chuỗi (0 nếu trùng, 9 nếu khác)."""
    if a == b:
        return 0.0
    pa, pb = parse(a), parse(b)
    if pa is None or pb is None:
        return 9.0
    return (_c(ONSET_COST, pa.onset, pb.onset) + (MEDIAL_COST if pa.medial != pb.medial else 0.0)
            + _c(NUCLEUS_COST, pa.nucleus, pb.nucleus) + _c(CODA_COST, pa.coda, pb.coda)
            + _c(TONE_COST, pa.tone, pb.tone, TONE_OTHER))
