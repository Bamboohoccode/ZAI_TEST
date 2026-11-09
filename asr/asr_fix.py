"""Luật sửa lỗi ASR có hệ thống (chạy sau snap_text, trước ITN). Input/output: text chữ thường, dạng đọc.

    from asr.asr_fix import fix_text
    fix_text("lịch sang tiếng anh giúp tôi")   # -> ("dịch sang tiếng anh giúp tôi", [("lịch", "dịch")])

Mỗi luật được kiểm tra trên nhãn train: không làm đổi câu đúng nào (xem test cuối file).
"""
import re

_LANG = r"(?:anh|hàn|nhật|trung|pháp|đức|thái|nga|việt|ý|lào|hoa|tây ban nha|bồ đào nha)"
_NUM95 = r"chín\s+(?:mươi\s+)?(?:lăm|năm)"
_NUM92 = r"chín\s+(?:mươi\s+)?hai"

RULES = [
    # "lịch sang tiếng anh", "lịch giùm anh câu này qua tiếng hàn" -> dịch (nhầm l/d đầu câu lệnh dịch)
    ("dịch", re.compile(rf"\blịch\b(?=(?:\s+\S+){{0,12}}?\s+(?:sang|qua|ra|sau)\s+tiếng\s+{_LANG}\b)"), "dịch"),
    # "xăng rồi chín năm", "xăng rông chín lăm" -> xăng ron ... (chỉ sửa chữ "ron" nghe nhầm, giữ nguyên số)
    ("ron", re.compile(rf"\b(xăng(?:\s+\S+){{0,2}}?\s+)(?:rôn|rông|rồi|giôn|zôn|giông)(\s+(?:{_NUM95}|{_NUM92}))\b"), r"\1ron\2"),
    # mã chữ cái đọc rời: "tàu s e ba" -> "tàu se ba", "hầm b ba" -> "hầm b ba" (giữ) ; chỉ ghép phụ âm + 1 chữ cái
    ("letters", re.compile(r"\b([bcdfghjklmnpqrstvwxz])\s+([a-z])(?=\s|$)"), r"\1\2"),
]


def fix_text(text):
    """(new_text, changes) — changes: [(tên luật, đoạn cũ, đoạn mới)]."""
    changes = []
    for name, pat, rep in RULES:
        def sub(m, name=name, rep=rep):
            # "dịch giúp em lịch giao hàng ... sang tiếng anh": đã có "dịch" phía trước -> "lịch" là lịch thật
            if name == "dịch" and re.search(r"\bdịch\b", m.string[:m.start()]):
                return m.group(0)
            new = m.expand(rep)
            if new != m.group(0):
                changes.append((name, m.group(0), new))
            return new
        text = pat.sub(sub, text)
    return text, changes


if __name__ == "__main__":
    cases = {
        "lịch sang tiếng anh giúp tôi câu này": "dịch sang tiếng anh giúp tôi câu này",
        "lịch cao cho tôi xem thực đơn sang tiếng hàn giùm anh": "dịch cao cho tôi xem thực đơn sang tiếng hàn giùm anh",
        "dời lịch sang thứ sáu giúp tôi": "dời lịch sang thứ sáu giúp tôi",
        "xem lịch họp tiếng anh chiều nay": "xem lịch họp tiếng anh chiều nay",
        "mà xăng rồi chín năm hôm nay lên bao nhiêu": "mà xăng ron chín năm hôm nay lên bao nhiêu",
        "giá xăng ron chín lăm": "giá xăng ron chín lăm",
        "xong rồi chín năm nữa": "xong rồi chín năm nữa",
        "hai vé tàu s e ba chuyến bảy giờ": "hai vé tàu se ba chuyến bảy giờ",
        "anh ơi em à": "anh ơi em à",
    }
    bad = [(i, fix_text(i)[0], o) for i, o in cases.items() if fix_text(i)[0] != o]
    for b in bad:
        print("FAIL", b)
    print(f"{len(cases) - len(bad)}/{len(cases)} ok")
