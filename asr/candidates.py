"""Đánh dấu từ ASR có confidence thấp + ứng viên gần âm, để LLM sửa lại (pipeline.py, luôn chạy sau ASR).

    cands = Candidates.from_train("data/asr_train_plain.jsonl")   # cặp hyp (ASR) / ref_spoken của train (eval_asr.py --dump)
    marked, n = cands.mark([("sẵn", 0.1), ("lo", 0.9), ...], thr=0.6)
    # -> "[sẵn|xăng|sang] lo chín lăm ...", n = số từ được đánh dấu

Ứng viên của một từ w (không có luật viết tay):
  1. bảng nhầm lẫn: ASR nghe "w" mà ref là "x" (căn hyp/ref bằng difflib, cặp thay thế 1-1), xếp theo số lần;
  2. từ gần âm trong từ vựng train theo asr/vi_phon.dist (tách âm tiết: đầu / đệm / chính / cuối / thanh, cặp lẫn theo
     vùng miền tốn ít: s/x, ch/tr, d/gi/r/v, l/n, -n/-ng, -t/-c, ă/â/ơ, iê/i, hỏi/ngã...), gần hơn trước, cùng mức thì
     từ hay gặp trước. Từ không phải âm tiết tiếng Việt (zalo, sms) dùng khoá hotwords.bias.phon.
Bỏ các câu held-out (id % 10 == 9) để còn đánh giá được.
"""
import difflib
import inspect
import json
import re
from collections import Counter, defaultdict

from asr.vi_phon import dist, parse
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

    @classmethod
    def from_refs(cls, path, include_heldout=False):
        """dự phòng khi chưa có asr_train_plain.jsonl: từ vựng lấy từ audio_text của <split>.jsonl BTC (mọi lượt user),
        không có bảng nhầm lẫn (ứng viên chỉ từ gần âm). Số giữ nguyên dạng nhãn (to_spoken không chạy ở đây)."""
        vocab = Counter()
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            it = json.loads(line)
            if not include_heldout and _heldout(it.get("id")):
                continue
            for u in [h for h in it.get("history", []) if h.get("role") == "user"] + [it]:
                vocab.update(w for w in re.findall(r"[^\W\d_]+", (u.get("audio_text") or "").lower()))
        return cls(defaultdict(Counter), vocab)

    def candidates(self, word, k=3, min_ratio=0.75, min_count=2, max_dist=0.8, method="syl"):
        """bảng nhầm lẫn (gặp >= min_count lần) trước, rồi từ gần âm: giống âm hơn trước, cùng mức thì từ hay gặp trước.
        method="syl": vi_phon.dist <= max_dist; "key" (hoặc từ không parse được): khoá phon, ratio >= min_ratio."""
        out = [t for t, n in self.confusion.get(word, Counter()).most_common(k) if t != word and n >= min_count]
        if len(out) < k and method == "syl" and parse(word) is not None:
            near = [(d, -self.vocab[v], v) for v in self.vocab
                    if v != word and v not in out and (d := dist(word, v)) <= max_dist]
            out += [v for *_, v in sorted(near)[:k - len(out)]]
        elif len(out) < k:
            pw = phon(word)
            near = []
            for key, ws in self._phon.items():
                r = ratio(pw, key)
                if r >= min_ratio:
                    near += [(round(r, 2), self.vocab[v], v) for v in ws if v != word and v not in out]
            out += [v for *_, v in sorted(near, reverse=True)[:k - len(out)]]
        return out

    def mark(self, words, thr=0.6, k=3, keep=frozenset(), alt=None):
        """words: [(từ, conf)] -> (chuỗi có [từ|ứng viên...] ở chỗ conf < thr, số từ được đánh dấu).
        Từ có conf thấp mà không có ứng viên vẫn được đánh dấu dạng [từ] để LLM biết chỗ nghi ngờ.
        keep: từ của hotword (--bias) -> không đánh dấu: hotword kéo tên vào làm conf (log-prob âm học) thấp,
        đánh dấu thì LLM dễ đổi ngược "phạm hùng" -> "phạm hưng".
        alt: {vị trí: [cụm]} chỗ bản decode khác nghe khác (xem alt_spans) -> luôn đánh dấu, cụm đó làm ứng viên đầu
        (kiểu N-best: hai bản decode khác nhau = chỗ ASR không chắc, kể cả khi conf cao)."""
        alt = alt or {}
        parts, n = [], 0
        for i, (w, c) in enumerate(words):
            if i in alt:
                opts = [x for x in dict.fromkeys(alt[i] + (self.candidates(w, k) if c < thr else [])) if x and x != w]
                parts.append("[" + "|".join([w] + opts[:k + 1]) + "]")
                n += 1
            elif c < thr and w not in keep:
                parts.append("[" + "|".join([w] + self.candidates(w, k)) + "]")
                n += 1
            else:
                parts.append(w)
        return " ".join(parts), n


def alt_spans(words, alts):
    """words: [str] câu chính; alts: [[str]] các bản decode khác (không bias, model khác...) của cùng audio.
    -> {i: [cụm]}: chỗ bản khác nghe khác, cụm đặt ở từ đầu của đoạn khác (đoạn dài thì các từ sau đánh dấu không cụm)."""
    out = {}
    for alt in alts or ():
        if not alt or alt == words:
            continue
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, words, alt, autojunk=False).get_opcodes():
            if op == "equal" or i1 == i2:  # bản khác thừa từ: không có vị trí để gắn, LLM vẫn có thể chèn nếu cần
                continue
            if op == "replace" and i2 - i1 == j2 - j1:
                for d in range(i2 - i1):
                    out.setdefault(i1 + d, []).append(alt[j1 + d])
            else:
                ph = " ".join(alt[j1:j2])  # rỗng = bản khác không có đoạn này: vẫn đánh dấu, không thêm ứng viên
                out.setdefault(i1, []).extend([ph] if ph else [])
                for i in range(i1 + 1, i2):
                    out.setdefault(i, [])
    return {i: list(dict.fromkeys(v)) for i, v in out.items()}


REFINE_SYSTEM = """Bạn là bộ sửa lỗi nhận dạng giọng nói (ASR) tiếng Việt. Câu vào là lời người dùng nói với trợ lý ảo \
(đặt lịch, báo thức, nhắc việc, chuyển tiền, ví, hoá đơn, nhắn tin, gọi điện, gọi xe, đặt vé tàu/xe/máy bay, gửi xe, \
nhạc, tivi, nhà thông minh, giá xăng, giá vàng, mua sắm, đơn hàng, trả hàng, công thức nấu ăn, tủ bếp, ghi chú...).

Định dạng câu vào:
- Chỗ ASR không chắc được đánh dấu [a|b|c|d]: a là từ ASR nghe được, b, c, d là các từ có âm gần giống.
- Danh sách ứng viên chỉ là GỢI Ý, có thể thiếu từ đúng. Từ ngoài ngoặc là chỗ ASR khá chắc, nhưng vẫn có thể sai.

Cách sửa:
1. Đọc cả câu và đoán người dùng muốn trợ lý làm gì (đặt gì, mua gì, gửi cho ai, lúc nào).
2. Mỗi chỗ đánh dấu: chọn ứng viên hợp ngữ cảnh nhất; nếu không ứng viên nào hợp, dùng từ khác ĐỌC GẦN GIỐNG a \
(lẫn s/x, ch/tr, d/gi/r, l/n, -n/-ng, -t/-c, hỏi/ngã, sai thanh điệu...); nếu không chắc thì giữ a.
3. Được sửa CẢ CỤM nhiều từ liền nhau (kể cả từ ngoài ngoặc) khi chúng đọc gần giống một cụm quen thuộc của ngữ cảnh: \
tên đồ vật, sản phẩm, đơn vị, địa điểm, thao tác của app (ví dụ "tai nghe bờ lu tút" -> "tai nghe bluetooth", \
"sang chín mươi lăm" -> "xăng chín mươi lăm").
4. Từ tiếng Anh, tên thương hiệu, đơn vị, viết tắt bị ASR phiên âm sai thì viết lại dạng chuẩn, chữ thường \
(bluetooth, laptop, voucher, sms, zalo, sjc, wifi).
5. Tên người, tên riêng: chỉ đổi khi một ứng viên rõ ràng là tên; không tự bịa tên mới.
6. Kiểm tra lại cả câu sau khi sửa: đọc như một yêu cầu thật gửi trợ lý ảo. Nếu còn chỗ vô nghĩa, lạc ngữ cảnh hoặc \
không ai nói như vậy (sai cụm từ, sai đồ vật, sai thao tác, sai đơn vị) thì sửa tiếp chỗ đó, KỂ CẢ khi không được \
đánh dấu, bằng từ/cụm ĐỌC GẦN GIỐNG làm câu hợp lý (ví dụ "nhà thúc" -> "nhà thuốc", "đặt lịch cắt tốc" -> \
"đặt lịch cắt tóc"). Chỉ sửa chỗ sai, phần câu đã hợp lý thì giữ nguyên.

Ràng buộc:
- Không thêm, bớt hay đổi ý; số từ gần như giữ nguyên; giữ từ đệm/từ địa phương (nhé, nha, ạ, ờ, à, nghen, hông, rứa).
- Giữ con số, giờ, ngày, số tiền đúng như người dùng nói; số viết bằng chữ (hai mươi lăm, không viết 25).
- Viết chữ thường, không dấu câu, bỏ hết dấu ngoặc [ ] và dấu |.

Ngôn ngữ (bắt buộc):
- Suy nghĩ và trả lời HOÀN TOÀN bằng tiếng Việt. KHÔNG dịch câu sang tiếng Anh, không suy luận bằng tiếng Anh, \
không viết bước suy luận, không dùng thẻ <think>.
- Chọn từ theo ÂM ĐỌC tiếng Việt và ngữ cảnh tiếng Việt, không theo nghĩa khi dịch: hai từ dịch ra tiếng Anh \
giống nhau không có nghĩa là đọc giống nhau.
- Chỉ trả về MỘT dòng tiếng Việt là câu đã sửa, không giải thích, không thêm bản dịch."""

NO_THINK = {"enable_thinking": False}

# ví dụ tự đặt (KHÔNG lấy từ public test): chọn ứng viên, sửa cả cụm, giữ nguyên khi không chắc
REFINE_SHOTS = [
    ("đổ [sang|xăng|săng] chín mươi lăm hai trăm nghìn rồi [nhắng|nhắn] cho vợ anh [nha|na]",
     "đổ xăng chín mươi lăm hai trăm nghìn rồi nhắn cho vợ anh nha"),
    ("đặt giúp em cái [tay|tai|tây] nghe [bu|bờ|bù] lu tút loại [dưới|giới|rưới] năm trăm nghìn",
     "đặt giúp em cái tai nghe bluetooth loại dưới năm trăm nghìn"),
    ("tám giờ sáng mai nhắc anh đi lấy [hộ|hồ|họ] [chiếu|chíu|triều] ở [phường|phương|thường] nhé",
     "tám giờ sáng mai nhắc anh đi lấy hộ chiếu ở phường nhé"),
    ("chuyển cho [la|lan|nan] ba trăm nghìn [cua|qua|quá] [dí|ví] [điện|đến] tử",
     "chuyển cho lan ba trăm nghìn qua ví điện tử"),
    ("mười giờ sáng mai nhắc anh ghé nhà thúc gần công ty mua [vỉ|vĩ|ví] thuốc [đau|đâu] đầu",
     "mười giờ sáng mai nhắc anh ghé nhà thuốc gần công ty mua vỉ thuốc đau đầu"),
]


NAMES_HINT = ("Tên có trong dữ liệu của người dùng (quán, người, cuộc họp, kênh...): {names}\n"
              "Cụm nào trong câu ĐỌC GẦN GIỐNG một tên ở trên (kể cả tên nước ngoài bị phiên âm, ví dụ \"quán xì ta búc\" -> "
              "\"quán starbucks\") thì thay bằng tên đó, viết chữ thường. Không dùng tên nào không có trong danh sách.\n\n")


def refine_conv(marked, context=(), names=()):
    user = marked if not context else "Các câu trước (chỉ để tham khảo):\n" + "\n".join(context) + "\n\nCâu cần sửa:\n" + marked
    if names:
        user = NAMES_HINT.format(names=", ".join(names)) + ("Câu cần sửa:\n" + marked if not context else user)
    shots = [m for q, a in REFINE_SHOTS for m in ({"role": "user", "content": q}, {"role": "assistant", "content": a})]
    return [{"role": "system", "content": REFINE_SYSTEM}, *shots, {"role": "user", "content": user}]


# đổi số / giờ / tiền / phủ định là đổi nghĩa (refine cũ hay làm vậy): chỉ nhận khi bản decode khác hoặc tên hỗ trợ
PROTECT = frozenset("không một mốt hai ba bốn tư năm lăm sáu bảy tám chín mười mươi linh lẻ rưỡi trăm nghìn ngàn triệu "
                    "tỷ tỉ chưa đừng chẳng chả".split())


def _phon_ok(a, b, max_dist, min_ratio):
    """đoạn a -> b đọc gần giống: cùng số từ thì từng cặp vi_phon.dist <= max_dist (từ ngoại: khoá phon), khác số từ
    thì khoá phon của cả cụm giống >= min_ratio ("bờ lu tút" -> "bluetooth")."""
    if a and b and len(a) == len(b):
        return all(dist(x, y) <= max_dist or (dist(x, y) >= 9 and ratio(phon(x), phon(y)) >= min_ratio)
                   for x, y in zip(a, b))
    if not a or not b:
        return False
    return ratio(phon("".join(a)), phon("".join(b))) >= min_ratio


def _clean_out(out):
    out = re.sub(r"<think>.*?(</think>|$)", "", out or "", flags=re.S).strip()  # Qwen3 thinking -> bỏ phần suy luận
    out = out.splitlines()[0].strip().lower() if out else ""
    return out.strip(" .\"'")


def accept_refine(raw, out, suspects=None, support=(), max_dist=1.0, min_ratio=0.5, free_dist=0.0, stats=None):
    """LLM trả về rác / viết lại cả câu -> giữ câu ASR. Chấp nhận khi số từ lệch ít và phần lớn từ còn giữ.
    suspects (tập vị trí từ bị đánh dấu) khác None -> chế độ chặt: xét TỪNG chỗ LLM sửa, chỉ nhận chỗ
      - chạm vị trí bị đánh dấu, hoặc cụm mới có trong `support` (tên --prefetch, cụm của bản decode khác),
        hoặc (free_dist > 0) đổi 1-1 từ không bị đánh dấu nhưng đọc rất gần (dist <= free_dist);
      - đọc gần giống đoạn cũ (_phon_ok), trừ khi có trong support;
      - không đổi số / phủ định (PROTECT), trừ khi có trong support.
    Chỗ không đạt giữ nguyên chữ ASR, các chỗ khác vẫn nhận. stats: Counter đếm lý do."""
    stats = stats if stats is not None else Counter()
    out = _clean_out(out)
    if not out or "[" in out:
        stats["bỏ cả câu (rỗng/còn ngoặc)"] += 1
        return raw
    a, b = raw.split(), out.split()
    if abs(len(a) - len(b)) > max(2, len(a) // 5):
        stats["bỏ cả câu (lệch số từ)"] += 1
        return raw
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    same = sum(x.size for x in sm.get_matching_blocks())
    # chế độ chặt đã lọc từng chỗ sửa -> nới ngưỡng (cụm "bờ lu tút" -> "bluetooth" làm mất 3 từ khớp)
    if same < (0.6 if suspects is None else 0.4) * len(a):
        stats["bỏ cả câu (viết lại)"] += 1
        return raw
    if suspects is None:
        return out
    ops = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "replace" and i2 - i1 == j2 - j1:  # thay 1-1: xét từng từ (sửa tên đúng, đổi số sai trong cùng đoạn)
            ops += [("replace", i1 + d, i1 + d + 1, j1 + d, j1 + d + 1) for d in range(i2 - i1)]
        else:
            ops.append((op, i1, i2, j1, j2))
    res = []
    for op, i1, i2, j1, j2 in ops:
        if op == "equal":
            res += a[i1:i2]
            continue
        old, new = a[i1:i2], b[j1:j2]
        if len(old) == len(new) == 1 and old == new:
            res += old
            continue
        near = range(max(i1 - 1, 0), i1 + 1) if i1 == i2 else range(i1, i2)
        sup = " ".join(new) in support
        touch = any(i in suspects for i in near)
        free = (free_dist > 0 and len(old) == len(new) == 1 and dist(old[0], new[0]) <= free_dist)
        if not (touch or sup or free):
            why = "ngoài chỗ đánh dấu"
        elif not sup and Counter(w for w in old if w in PROTECT) != Counter(w for w in new if w in PROTECT):
            why = "đổi số/phủ định"
        elif not sup and not _phon_ok(old, new, max_dist, min_ratio):
            why = "khác âm"
        else:
            res += new
            stats["nhận"] += 1
            continue
        res += old
        stats["bỏ: " + why] += 1
    return " ".join(res)


def refine(llm, word_lists, cands, thr=0.6, k=3, use_context=True, keep=frozenset(), names=None, alts=None,
           strict=True, max_dist=1.0, min_ratio=0.5, free_dist=0.0, stats=None):
    """word_lists: [[(từ, conf)]] các lượt của một sample (history trước, query cuối). keep: xem Candidates.mark.
    names: [list tên | None] mỗi lượt (vd. tên lấy từ tool đọc --prefetch): đưa vào prompt, và lượt có tên luôn được
    gửi LLM dù không có từ nào bị đánh dấu (tên nước ngoài hay bị nghe sai với conf cao: "taipei" -> "tay").
    alts: [[[từ]] | None] mỗi lượt các bản decode khác của cùng audio (N-best tự ghép): chỗ khác nhau được đánh dấu.
    strict: chỉ nhận từng chỗ sửa hợp lệ (xem accept_refine); False = kiểu cũ, nhận/bỏ cả câu.
    Trả (texts, marked): chỉ gọi LLM cho lượt có từ được đánh dấu (hoặc có tên), gọi một lô."""
    n = len(word_lists)
    names = names or [None] * n
    alts = alts or [None] * n
    raws = [" ".join(w for w, _ in wl) for wl in word_lists]
    keeps = keep if isinstance(keep, list) else [keep] * n  # list = keep riêng cho từng lượt
    spans = [alt_spans([w for w, _ in wl], al) for wl, al in zip(word_lists, alts)]
    marked = [cands.mark(wl, thr, k, kp, sp) for wl, kp, sp in zip(word_lists, keeps, spans)]
    idx = [i for i, (_, m) in enumerate(marked) if m or names[i]]
    texts = list(raws)
    if idx and llm is not None:
        convs = [refine_conv(marked[i][0], raws[max(0, i - 2):i] if use_context else (), names[i] or ()) for i in idx]
        # tắt thinking cho refine (planner dùng chung LLM có thể đang bật) -> không suy luận / dịch sang tiếng Anh
        takes_kw = "chat_kwargs" in inspect.signature(llm.chat).parameters
        outs = llm.chat(convs, chat_kwargs=NO_THINK) if takes_kw else llm.chat(convs)
        for i, out in zip(idx, outs):
            if not strict:
                texts[i] = accept_refine(raws[i], out, stats=stats)
                continue
            sus = set(spans[i]) | {j for j, (w, c) in enumerate(word_lists[i]) if c < thr and w not in keeps[i]}
            support = {x.lower() for x in names[i] or ()} | {x for v in spans[i].values() for x in v if x}
            texts[i] = accept_refine(raws[i], out, sus, support, max_dist, min_ratio, free_dist, stats)
    return texts, [m for m, _ in marked]
