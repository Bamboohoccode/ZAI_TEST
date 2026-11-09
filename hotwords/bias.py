"""Search-key / contextual biasing for ASR output.

Từ khoá (entity) lấy từ chính dữ liệu app của mẫu (<split>_environment/<id>/state.json: danh bạ, lịch, nhạc, nhà hàng,
kênh TV, thiết bị...) + hồ sơ user + một glossary nhỏ (Zalo, SMS, SJC, email...). Dùng ở 3 chỗ:

  1. hotwords(): chuỗi hotword cho Zipformer/sherpa-onnx (biasing lúc decode, modified_beam_search).
  2. snap_text(): sau ASR, thay đoạn từ nghe gần giống một entity bằng đúng entity ("gia lô" -> "Zalo").
  3. snap_value(): "search key" của tool call (contacts.search query, maps.search_place query...) -> entity gần nhất.

So khớp theo phát âm: bỏ dấu thanh, gộp các cặp hay nhầm (d/gi/r/z, ch/tr, s/x, c/k/q, l/n đầu âm...), so chuỗi ghép
không khoảng trắng để bắt lỗi tách/ghép âm tiết. Entity một âm tiết (Lan, Bống...) chỉ được thay cho từ KHÔNG phổ biến
(common_words), tránh "đá bóng" -> "đá Bống".
"""
import json
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

# key trong state.json mà giá trị là thứ người dùng có thể nhắc tên
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


def build_lexicon(state_path=None, context=None, glossary=GLOSSARY):
    """list[str] entity, bỏ trùng, giữ thứ tự (entity dài trước ưu tiên khi snap)."""
    ents = []
    if state_path and Path(state_path).exists():
        _walk(json.load(open(state_path, encoding="utf-8")), ents)
    prof = (context or {}).get("user_profile") or {}
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


def app_pools(state_path):
    """{app: [entity,...]} từ state.json: dùng để snap search key của tool về đúng tên trong app đó (contacts -> danh bạ)."""
    if not state_path or not Path(state_path).exists():
        return {}
    pools = {}
    for app, data in json.load(open(state_path, encoding="utf-8")).items():
        if app != "_meta":
            ents = []
            _walk(data, ents)
            pools[app] = list(dict.fromkeys(" ".join(e.split()) for e in ents if e.strip()))
    return pools


def common_words(jsonl_path, min_count=3):
    """từ thường (viết thường) hay gặp trong transcript train: entity 1 âm tiết không được đè lên chúng."""
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


# ------------------------------------------------------------------ snapping

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


def _has_run(low, ew):
    n = len(ew)
    return any(low[i:i + n] == ew for i in range(len(low) - n + 1))


def _covered(win, ew, thr=0.8):
    """tỉ lệ từ của entity có một từ trong cửa sổ khớp phát âm."""
    return sum(max(ratio(phon(w), phon(x)) for x in win) >= thr for w in ew) / len(ew)


_NUM_WORDS = {"không", "một", "mốt", "hai", "ba", "bốn", "tư", "năm", "lăm", "sáu", "bảy", "tám", "chín", "mười",
              "mươi", "trăm", "nghìn", "ngàn", "linh", "lẻ", "triệu", "tỷ", "tỉ", "rưỡi"}


def _nums(ws):
    """chuỗi từ chỉ số (lăm = năm): 'htv bảy' vs 'htv ba' khác số -> không được snap (bỏ dấu thì gần như trùng)."""
    return ["năm" if w == "lăm" else w for w in ws if w in _NUM_WORDS]


def _subseq(a, b):
    """a là dãy con của b: 'ron chính lâm' ([]) / 'ron chín lâm' ([chín]) vẫn snap được về 'ron chín lăm'."""
    it = iter(b)
    return all(x in it for x in a)


def _dedup(words, rep):
    """'nhắn sms sms' -> 'nhắn sms': hotword boost hay chèn lặp từ ngoại; chỉ gộp từ trong `rep` (không phải số/từ Việt)."""
    out = []
    for w in words:
        if out and w.lower() == out[-1].lower() and w.lower() in rep:
            continue
        out.append(w)
    return out


def snap_text(text, entities, common=frozenset(), thr=0.9, thr_short=0.9, size_penalty=0.04, name_guard=True,
              aliases=None):
    """(new_text, changes). Duyệt entity dài trước; mỗi đoạn hyp chỉ bị thay một lần.
    Bỏ qua entity đã xuất hiện đúng trong câu; entity một âm tiết tiếng Việt (Lan, Bống...) không bao giờ bị snap
    (dễ đè lên từ thường chỉ khác dấu), chỉ snap entity ngoại / viết tắt / nhiều từ."""
    als = aliases if aliases is not None else _aliases(entities)
    words = text.split()
    low = [re.sub(r"[^\w'-]", "", w.lower()) for w in words]
    used, changes, repl = [False] * len(words), [], {}
    for surf, ew, ek, disp in sorted(als, key=lambda a: -len(a[2])):
        n, best = len(ew), None
        if len(ek) < 3 or _has_run(low, tokens(disp)):
            continue
        if n == 1 and _VI_SYL.match(ew[0]):
            continue
        # cụm toàn âm tiết Việt; trừ mã có số hiển thị dạng đọc chữ thường ("ron chín lăm"), còn tên "Anh Mười" vẫn tính
        all_vi = all(_VI_SYL.match(w) for w in ew) and not (_nums(ew) and disp == disp.lower())
        # tên người tiếng Việt (Lê Minh, Phạm Hùng): mọi âm tiết là âm tiết Việt viết hoa
        person = name_guard and n > 1 and all_vi and all(x[:1].isupper() for x in disp.split()[-n:])
        for size in range(max(1, n - 1), n + 2):
            for i in range(len(words) - size + 1):
                if any(used[i:i + size]):
                    continue
                win = low[i:i + size]
                if not _subseq(_nums(win), _nums(ew)):  # "htv bảy" -> "htv ba": không đổi số đã nghe được
                    continue
                exact = win == ew
                # "nhé mình" -> "Lê Minh": cửa sổ toàn từ thường, chỉ khớp nhờ gộp l/n + bỏ dấu -> không phải tên
                if person and all(w in common for w in win) and strip_tones(" ".join(win)) != strip_tones(" ".join(ew)):
                    continue
                # cụm toàn âm tiết Việt (Bến Thành): "bên thanh toán" -> "Bến Thành toán"; chỉ snap khi khớp đúng dấu
                if all_vi and not exact and all(w in common for w in win):
                    continue
                # 1 từ ngoại ghép từ >= 2 âm tiết Việt: "ứng lẹ món" -> "ứng lemon"; trừ cách nghe sai đã mine (alias)
                if n == 1 and size > 1 and not exact and all(_VI_SYL.match(w) for w in win):
                    continue
                r = ratio(key(win), ek)
                if r < 1.0 and size > 1 and any(w in common and len(w) <= 2 for w in win):  # đừng nuốt từ chức năng ngắn
                    continue
                if n > 1 and _covered(win, ew) < 0.75:
                    continue
                adj = r - size_penalty * abs(size - n)
                if adj >= (thr_short if len(ek) <= 5 else thr) and (best is None or adj > best[0]):
                    best = (adj, i, size, r)
        if best:
            _, i, size, r = best
            changes.append((" ".join(words[i:i + size]), disp, round(r, 2)))
            repl[i] = (size, disp)
            for j in range(i, i + size):
                used[j] = True
    out, i = [], 0
    while i < len(words):
        if i in repl:
            size, disp = repl[i]
            out.append(disp)
            i += size
        else:
            out.append(words[i])
            i += 1
    rep = {w for _, ew, _, _ in als for w in ew if not _VI_SYL.match(w) and w not in _NUM_WORDS}
    return " ".join(_dedup(out, rep)), changes


def load_global(path):
    """hotwords_global.json (mine_hotwords.py) -> alias cho snap_text(aliases=...): mỗi dạng đọc / cách nghe sai lặp lại
    của một cụm -> (surface, words, key, display). display = dạng viết (SMS, Zalo, check-in); mã có số giữ dạng đọc
    ("ron chín lăm") để ITN đổi số. Dùng thay cho build_lexicon(state.json) khi không được đọc state."""
    out = []
    for r in json.load(open(path, encoding="utf-8")):
        for al in r["aliases"]:
            w = al.split()
            if w:
                out.append((r["display"], w, key(w), r["display"]))
    return out


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
