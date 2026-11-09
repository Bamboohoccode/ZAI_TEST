"""Search-key / contextual biasing for ASR output.

Từ khoá (entity) CHỈ lấy từ item của file <split>.jsonl: hồ sơ user (context.user_profile) + tham số / kết quả các tool
call trong history (danh bạ, lịch, nhạc... mà các lượt trước đã đọc) + một glossary nhỏ (Zalo, SMS, SJC, email...).
KHÔNG đọc <split>_environment/<id>/state.json (chỉ ToolEnv được nạp file đó). Dùng ở 2 chỗ:

  1. hotword_lines(): hotword cho Zipformer/sherpa-onnx (biasing lúc decode, modified_beam_search), pipeline.py --bias.
  2. snap_value(): "search key" của tool call (contacts.search query, tv.tune channel...) -> entity gần nhất (snap_search_keys).

So khớp theo phát âm: bỏ dấu thanh, gộp các cặp hay nhầm (d/gi/r/z, ch/tr, s/x, c/k/q, l/n đầu âm...), so chuỗi ghép
không khoảng trắng để bắt lỗi tách/ghép âm tiết. phon() cũng dùng cho ứng viên gần âm của asr/candidates.py.
"""
import json
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

# key (trong tham số / kết quả tool) mà giá trị là thứ người dùng có thể nhắc tên
ENTITY_KEYS = {"name", "title", "artist", "provider", "location", "label", "room", "channel", "merchant", "program",
               "station", "assignee", "restaurant", "dish", "item", "brand", "album", "show", "city", "address"}
PROFILE_KEYS = ("name", "home_address", "work_address", "city")
# từ ngoại / viết tắt hay gặp trong tập train (Zalo, SMS, SJC...), thêm vào mọi mẫu
GLOSSARY = ["Zalo", "SMS", "SJC", "EVN", "QR", "email", "podcast", "voucher", "laptop", "robot", "online", "camera",
            "file", "check-in", "workshop", "playlist", "deadline", "wifi", "app", "link", "taxi", "combo"]
MAX_WORDS = 6


def strip_tones(s):
    s = unicodedata.normalize("NFD", s.lower().replace("đ", "d"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


_ONSET = [("ngh", "g"), ("ng", "g"), ("nh", "n"), ("ch", "c"), ("tr", "c"), ("gi", "z"), ("d", "z"), ("r", "z"),
          ("ph", "f"), ("th", "t"), ("kh", "h"), ("gh", "g"), ("qu", "k"), ("x", "s"), ("k", "k"), ("q", "k")]


def phon(word):
    """khóa phát âm của một từ (đã bỏ dấu): gộp phụ âm đầu hay nhầm."""
    w = strip_tones(word)
    for src, dst in _ONSET:
        if w.startswith(src):
            w = dst + w[len(src):]
            break
    else:
        if w[:1] == "c":
            w = "k" + w[1:]
    w = w.replace("l", "n")  # n/l lẫn nhau (Nam/Lam, Zalo/gia nô)
    # nguyên âm/vần hay nhầm theo vùng
    w = re.sub(r"(nh|ng)$", lambda m: "n" if m.group(1) == "nh" else "g", w)
    return w.replace("ay", "ai").replace("oo", "o")


def key(words):
    return "".join(phon(w) for w in words)


def ratio(a, b):
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def tokens(text):
    return re.findall(r"[\w'-]+", text.lower().replace("-", " "))


# ------------------------------------------------------------------ lexicon

def _walk(o, out, parent_key=""):
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, str) and k in ENTITY_KEYS:
                out.append(v)
            else:
                _walk(v, out, k)
    elif isinstance(o, list):
        for v in o:
            _walk(v, out, parent_key)


def _history_calls(item):
    """[(tool name, arguments, result)] của các tool call trong history của item (kết quả có thể thiếu -> {})."""
    out = []
    for h in (item or {}).get("history") or []:
        calls = h.get("tool_calls") or []
        results = h.get("tool_results") or []
        for i, c in enumerate(calls):
            args = c.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            out.append((c.get("name", ""), args or {}, results[i] if i < len(results) else {}))
    return out


def build_lexicon(item=None, glossary=GLOSSARY):
    """list[str] entity từ item (user_profile + tool call / kết quả trong history) + glossary, bỏ trùng, giữ thứ tự."""
    ents = []
    for _, args, res in _history_calls(item):
        _walk(args, ents)
        _walk(res, ents)
    prof = ((item or {}).get("context") or {}).get("user_profile") or {}
    ents += [prof[k] for k in PROFILE_KEYS if isinstance(prof.get(k), str)]
    ents += glossary
    seen, out = set(), []
    for e in ents:
        e = " ".join(str(e).split())
        n = len(tokens(e))
        if e and 1 <= n <= MAX_WORDS and not re.fullmatch(r"[\d\W_]+", e) and e.lower() not in seen:
            seen.add(e.lower())
            out.append(e)
    return out


def add_to_pool(pools, tool_name, *objs):
    """thêm entity trong tham số / kết quả của một tool call vào pools[app] (app = tiền tố tên tool)."""
    ents = []
    for o in objs:
        _walk(o, ents)
    pool = pools.setdefault(tool_name.split(".")[0], [])
    for e in (" ".join(str(e).split()) for e in ents):
        if e and e not in pool:
            pool.append(e)
    return pools


def app_pools(item):
    """{app: [entity,...]} từ các tool call / kết quả trong history của item: snap search key của tool về tên đã thấy.
    run_agent bổ sung thêm từ kết quả các tool call mới (add_to_pool)."""
    pools = {}
    for name, args, res in _history_calls(item):
        add_to_pool(pools, name, args, res)
    return pools


def common_words(jsonl_path, min_count=3):
    """từ thường (viết thường) hay gặp trong transcript train (mine_hotwords.py lọc hotword theo danh sách này)."""
    c = Counter()
    for l in open(jsonl_path, encoding="utf-8"):
        r = json.loads(l)
        for t in re.findall(r"[\w'-]+", r["audio_text"]):
            if t.islower():
                c[t] += 1
    return {w for w, n in c.items() if n >= min_count}


def hotwords(entities, score=2.0, upper=True):
    """chuỗi nhiều dòng 'ENTITY :score' cho sherpa-onnx (vocab BPE của Zipformer là chữ HOA)."""
    return "\n".join(f"{' '.join(tokens(e)).upper() if upper else e} :{score}" for e in entities if tokens(e))


def is_proper(e):
    """tên riêng / viết tắt: có chữ hoa ngoài chữ cái đầu (Sân bóng Mini, SMS, Phạm Hùng) hoặc 1 từ viết hoa (Lan, Zalo).
    'Tài khoản thanh toán', 'Gia đình' (chỉ hoa chữ đầu câu) là cụm thường."""
    w = e.split()
    return any(c.isupper() for c in e[1:]) or (len(w) == 1 and e[:1].isupper())


def hotword_lines(entities, hi=2.5, lo=1.5, glossary_score=1.0, glossary=GLOSSARY):
    """hotword chia điểm: tên riêng -> hi, cụm thường -> lo, từ ngoại viết thường của glossary (podcast, voucher...)
    -> glossary_score thấp (0 = bỏ hẳn, nhưng trên dev bỏ hẳn làm WER 0.175 -> 0.185). Score đều 2.0 cho mọi thứ từng kéo
    'đặt báo thức' -> 'tập podcast', 'hàng xóm' -> 'hàng sáu'."""
    gl = {g.lower() for g in glossary if g == g.lower()}
    out = []
    for e in entities:
        s = glossary_score if e.lower() in gl else hi if is_proper(e) else lo
        if s > 0 and tokens(e):
            out.append(hotwords([e], s))
    return "\n".join(out)


# ------------------------------------------------------------------ search key

LETTERS = {"s": "ét", "m": "em", "j": "gi", "c": "xê", "e": "e", "v": "vê", "n": "en", "q": "quy", "r": "rờ", "a": "a",
           "b": "bê", "d": "dê", "g": "giê", "h": "hát", "i": "i", "k": "ca", "l": "eo", "o": "o", "p": "pê", "t": "tê",
           "u": "u", "w": "đớp liu", "x": "ích xì", "y": "i", "z": "dét", "f": "ép"}


def _aliases(entities):
    """(surface, words, key, display): entity đầy đủ + từ cuối của tên người nhiều từ (Cô giáo Mai -> Mai)
    + dạng đọc chữ cái của từ viết tắt (SMS -> ét em ét)."""
    out = []
    for e in entities:
        w = tokens(e)
        out.append((e, w, key(w), e))
        if e.isupper() and 2 <= len(e) <= 4 and e.isalpha():
            sp = " ".join(LETTERS[c] for c in e.lower()).split()
            out.append((e, sp, key(sp), e))
        if len(w) > 1 and e.split()[-1][:1].isupper():
            out.append((e.split()[-1], w[-1:], key(w[-1:]), e.split()[-1]))
    return out


_V = "aăâeêioôơuưyáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
_VI_SYL = re.compile(rf"^(ngh|ng|nh|ch|gh|gi|kh|ph|th|tr|qu|[bcdđghklmnpqrstvx])?[{_V}]+(ch|ng|nh|[cmnpt])?$")


def snap_value(value, entities, thr=0.8):
    """search key của tool call -> entity gần nhất theo phát âm; giữ nguyên nếu đã đúng hoặc không đủ gần."""
    if not isinstance(value, str) or not value.strip():
        return value
    w = tokens(value)
    if any(value.lower() == e.lower() for e in entities):
        return value
    k, best = key(w), (0.0, value)
    for surf, ew, ek, disp in _aliases(entities):
        r = ratio(k, ek)
        if r > best[0]:
            best = (r, disp)
    return best[1] if best[0] >= thr else value
