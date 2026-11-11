"""TN offline bằng luật (viết -> dạng đọc, giống output Zipformer): số, mã chữ+số, %, giờ. Không gọi mạng/LLM.

    from lifemate.vi_num import to_spoken, num_variants, code_variants
    to_spoken("Mở VTV3 lúc 7:30 nhé, giảm 50%")          # -> "mở vtv ba lúc bảy giờ ba mươi nhé giảm năm mươi phần trăm"
    to_spoken("Đổ RON95 giúp anh", hyp="đổ ron chín mươi lăm giúp anh")   # chọn cách đọc giống người nói
    num_variants("25")     # ["hai mươi lăm", "hai lăm"]
    code_variants("RON95") # ["ron chín lăm", "ron chín mươi lăm"]

Mọi cách đọc hợp lệ được sinh ra; nếu có `hyp` (output ASR của chính câu đó) thì chọn cách đọc khớp hyp nhất.
"""
import re
from collections import Counter

from lifemate.text import norm

D = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
UNITS = [(10 ** 9, "tỷ"), (10 ** 6, "triệu"), (10 ** 3, "nghìn")]


def digits(s):
    """đọc từng chữ số: '0912' -> 'không chín một hai'."""
    return " ".join(D[int(c)] for c in s if c.isdigit())


def _two(n, short=False):
    """0..99 -> list cách đọc. short: '25' -> 'hai lăm' (bỏ 'mươi')."""
    if n < 10:
        return [D[n]]
    t, u = divmod(n, 10)
    if t == 1:
        return ["mười"] if u == 0 else ["mười " + ("lăm" if u == 5 else D[u])]
    head = D[t] + " mươi"
    if u == 0:
        return [head]
    last = {1: ["mốt"], 4: ["bốn", "tư"], 5: ["lăm"]}.get(u, [D[u]])
    out = [f"{head} {x}" for x in last] + [f"{D[t]} {x}" for x in last]   # hai mươi lăm / hai lăm
    if u == 1:
        out.append(f"{head} một")
    return out


def _three(n, full=False):
    """0..999. full: luôn đọc hàng trăm ('không trăm linh năm') khi đứng sau đơn vị lớn."""
    h, r = divmod(n, 100)
    if h == 0 and not full:
        return _two(r)
    hs = f"{D[h]} trăm"
    if r == 0:
        return [hs]
    if r < 10:
        return [f"{hs} {z} {D[r]}" for z in ("linh", "lẻ")]
    return [f"{hs} {x}" for x in _two(r)]


def int_variants(n, limit=6):
    """số nguyên -> list cách đọc (cách chuẩn trước), tối đa `limit`."""
    if n == 0:
        return ["không"]
    for base, name in UNITS:
        if n >= base:
            q, r = divmod(n, base)
            heads = int_variants(q, limit)
            if r == 0:
                alt = [f"{h} ngàn" for h in heads[:1]] if name == "nghìn" else []
                return ([f"{h} {name}" for h in heads] + alt)[:limit]
            outs = []
            if name == "nghìn":
                tails = _three(r, full=True)
            else:
                tails = int_variants(r, limit)
                if r < 100 * (base // 1000):  # 2.050.000 -> hai triệu không trăm năm mươi nghìn
                    tails = ["không trăm " + t if not t.startswith("không") else t for t in tails] + tails
            for h in heads[:2]:
                for t in tails[:3]:
                    outs.append(f"{h} {name} {t}")
            # "hai triệu tám" (= 2.800.000), "ba nghìn rưỡi" (= 3.500)
            if r * 10 % base == 0:
                d = r * 10 // base
                outs += [f"{h} {name} {D[d]}" for h in heads[:1]]
                if d == 5:
                    outs += [f"{h} {name} rưỡi" for h in heads[:1]]
            outs += [f"{h} {name.replace('nghìn', 'ngàn')} " + o.split(f' {name} ', 1)[1] for h in heads[:1]
                     for o in outs[:1]] if name == "nghìn" else []
            return list(dict.fromkeys(outs))[:limit]
    return _three(n)[:limit]


def num_variants(tok):
    """một token số (đã norm: '25', '2.800.000', '2,5', '7:30', '20/11', '0912345678') -> list cách đọc."""
    if re.fullmatch(r"0\d{6,}", tok) or re.fullmatch(r"\d{7,}", tok) and "." not in tok:
        return [digits(tok)]                                  # số điện thoại / tài khoản
    m = re.fullmatch(r"(\d{1,2})[:h](\d{2})", tok)
    if m:                                                     # 7:30, 7h30
        h, mi = int(m.group(1)), int(m.group(2))
        hs = int_variants(h)[0]
        if mi == 0:
            return [f"{hs} giờ"]
        out = [f"{hs} giờ {x}" for x in int_variants(mi)[:2]] + [f"{hs} giờ {x} phút" for x in int_variants(mi)[:1]]
        return out + ([f"{hs} rưỡi", f"{hs} giờ rưỡi"] if mi == 30 else [])
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})(?:/(\d{4}))?", tok)
    if m:                                                     # 20/11(/2026)
        d, mo = int_variants(int(m.group(1)))[0], int_variants(int(m.group(2)))[0]
        y = f" năm {int_variants(int(m.group(3)))[0]}" if m.group(3) else ""
        return [f"{d} tháng {mo}{y}", f"ngày {d} tháng {mo}{y}"]
    if re.fullmatch(r"\d{1,3}(\.\d{3})+", tok):               # 2.800.000
        return int_variants(int(tok.replace(".", "")))
    m = re.fullmatch(r"(\d+)[.,](\d+)", tok)
    if m:                                                     # 2,5 / 2.5
        a, b = int_variants(int(m.group(1)))[0], m.group(2)
        frac = int_variants(int(b))[0] if len(b) <= 2 and not b.startswith("0") else digits(b)
        out = [f"{a} phẩy {frac}", f"{a} chấm {frac}"]
        return out + ([f"{a} rưỡi"] if b == "5" else [])
    if tok.isdigit():
        n = int(tok)
        out = int_variants(n)
        if len(tok) >= 3 and n < 1000 and tok[0] != "0":
            out.append(digits(tok))                           # mã số: 101 -> một không một
        if len(tok) == 4 and tok[0] != "0":
            if tok[2] != "0":                                 # năm 2025 -> hai không hai lăm (đọc theo cặp)
                out += [f"{digits(tok[:2])} {x}" for x in _two(int(tok[2:]))]
            out.append(digits(tok))                           # mã 4 số: 2015 -> hai không một năm
        if tok[0] == "0" and len(tok) > 1:
            out = [digits(tok)]
        return out
    return [digits(tok)] if any(c.isdigit() for c in tok) else [tok]


def code_variants(code):
    """mã chữ + số (RON95, VTV3, HTV7, SCTV9, E5, RV221, 8A3, Omega-3) -> list cách đọc kiểu Zipformer:
    phần chữ giữ nguyên viết thường, phần số đọc thành chữ (>= 3 chữ số: đọc từng số + cách đọc số nguyên)."""
    parts = re.findall(r"[^\W\d_]+|\d+", code.lower())
    alts = [[]]
    for p in parts:
        if p.isdigit():
            opts = (num_variants(p) if len(p) <= 2 else [digits(p)] + int_variants(int(p))[:1])[:3]
        else:
            opts = [p]
        alts = [a + [o] for a in alts for o in opts][:8]
    return list(dict.fromkeys(" ".join(a) for a in alts))


_SYM = [
    (re.compile(r"(\d)\s*%"), r"\1 phần trăm"),
    (re.compile(r"(\d)\s*°\s*c\b", re.I), r"\1 độ c"),
    (re.compile(r"(\d)\s*°"), r"\1 độ"),
    (re.compile(r"(\d)\s*km\b", re.I), r"\1 ki lô mét"),
    (re.compile(r"(\d)\s*kg\b", re.I), r"\1 ki lô gam"),
    (re.compile(r"(\d)\s*k\b", re.I), r"\1 ca"),
    (re.compile(r"(\d)\s*đ\b"), r"\1 đồng"),
    (re.compile(r"(\d+)h(?!\d)"), r"\1 giờ"),           # 24h -> 24 giờ
    (re.compile(r"\+"), " cộng "),                    # S+ Sport, K+, +84
    (re.compile(r"&"), " và "),
    (re.compile(r"@"), " a còng "),
]
_CODE = re.compile(r"^(?=.*\d)(?=.*[^\W\d_])[\w-]+$")


def _best(opts, hyp_counts):
    """chọn cách đọc có nhiều từ xuất hiện trong hyp nhất (hoà -> cách chuẩn, đứng trước)."""
    if not hyp_counts or len(opts) == 1:
        return opts[0]

    def score(o):
        w = o.split()
        hit = sum(min(c, hyp_counts[x]) for x, c in Counter(w).items())
        return hit - 0.01 * max(0, len(w) - hit)
    return max(opts, key=score)


def to_spoken(text, hyp=None):
    """câu viết -> dạng đọc chữ thường không dấu câu. Trả None nếu còn sót chữ số (không chắc cách đọc)."""
    s = text
    for pat, rep in _SYM:
        s = pat.sub(rep, s)
    hc = Counter(hyp.split()) if hyp else None
    out = []
    for tok in norm(s).split():
        if not any(c.isdigit() for c in tok):
            out.append(tok)
        elif _CODE.match(tok):
            out.append(_best(code_variants(tok), hc))
        else:
            out.append(_best(num_variants(tok), hc))
    res = " ".join(out)
    return None if re.search(r"\d", res) else res


if __name__ == "__main__":
    cases = {
        ("Mở VTV3 giúp anh", None): "mở vtv ba giúp anh",
        ("nạp 50 nghìn", None): "nạp năm mươi nghìn",
        ("RON95 giá bao nhiêu", "ron chín lăm giá bao nhiêu"): "ron chín lăm giá bao nhiêu",
        ("RON95 giá bao nhiêu", "ron chín mươi lăm giá bao nhiêu"): "ron chín mươi lăm giá bao nhiêu",
        ("27 độ", "hai bảy độ"): "hai bảy độ",
        ("27 độ", None): "hai mươi bảy độ",
        ("lúc 7:30", "lúc bảy rưỡi"): "lúc bảy rưỡi",
        ("giảm 50%", None): "giảm năm mươi phần trăm",
        ("gọi 0912345678", None): "gọi không chín một hai ba bốn năm sáu bảy tám",
        ("chuyến RV107", None): "chuyến rv một không bảy",
        ("2.800.000 đồng", "hai triệu tám trăm nghìn đồng"): "hai triệu tám trăm nghìn đồng",
        ("2.800.000 đồng", "hai triệu tám đồng"): "hai triệu tám đồng",
        ("góc phố 24h", "góc phố hai tư giờ"): "góc phố hai tư giờ",
        ("105 người", None): "một trăm linh năm người",
        ("lớp 8A3", None): "lớp tám a ba",
        ("tầng 21", None): "tầng hai mươi mốt",
        ("kênh S+ Sport", "kênh s cộng sport"): "kênh s cộng sport",
        ("năm 2025", "năm hai không hai lăm"): "năm hai không hai lăm",
        ("năm 2025", None): "năm hai nghìn không trăm hai mươi lăm",
        ("gửi a@b.com", None): "gửi a a còng b com",
    }
    bad = [(i, to_spoken(*i), o) for i, o in cases.items() if to_spoken(*i) != o]
    for b in bad:
        print("FAIL", b)
    print(f"{len(cases) - len(bad)}/{len(cases)} ok")
