"""P5: ví dụ few-shot "câu ASR nhiễu -> tool call đúng" lấy từ train-split, chèn vào system prompt của planner.

Nguồn câu nhiễu: bản ASR thật của train (scripts/eval_asr.py --dump data/asr_train_plain.jsonl, lượt QUERY); tool call
lấy từ label. Chỉ dùng mẫu train-split (id % 10 != 9) để không lộ held-out; chọn k ví dụ giống câu hiện tại nhất."""
import json
import re

from lifemate.text import is_heldout, norm


def _words(s):
    return set(norm(s).split())


class FewShot:
    def __init__(self, dump_path, train_jsonl, k=3, max_arg_chars=240):
        hyp = {}
        for l in open(dump_path, encoding="utf-8"):
            r = json.loads(l)
            if r.get("turn") == "QUERY":
                hyp[r["id"]] = r["hyp"]
        self.k, self.max_arg_chars, self.pool = k, max_arg_chars, []
        for l in open(train_jsonl, encoding="utf-8"):
            it = json.loads(l)
            if is_heldout(it["id"]) or it["id"] not in hyp:
                continue
            self.pool.append((it["id"], hyp[it["id"]], _words(hyp[it["id"]]), it["label"]["tool_calls"]))

    def select(self, query, exclude=None):
        q = _words(query)
        scored = [(len(q & w) / max(1, len(q | w)), i) for i, (sid, _, w, _) in enumerate(self.pool) if sid != exclude]
        return [self.pool[i] for _, i in sorted(scored, reverse=True)[:self.k]]

    def render(self, query, exclude=None):
        ex = self.select(query, exclude)
        if not ex:
            return ""
        out = ["", "Ví dụ đã làm đúng (lời người dùng là bản nhận dạng giọng nói CÓ LỖI như thật; ID, tên, giờ chỉ minh hoạ, "
                   "không dùng lại; ví dụ chỉ ghi các tool call đúng, không ghi kết quả):"]
        for _, h, _, calls in ex:
            out.append(f'- Lời: "{h}"')
            if not calls:
                out.append("  -> không gọi tool")
            for c in calls:
                args = json.dumps(c.get("arguments", {}), ensure_ascii=False)
                if len(args) > self.max_arg_chars:
                    args = args[:self.max_arg_chars] + "…"
                out.append(f"  -> {c['name']} {args}")
        return "\n".join(out)


def alt_hints(text, plain, confs, thr=0.5, max_words=6):
    """P4: dòng phụ cho planner ở câu cuối: bản nhận dạng khác (không bias) nếu khác, và các từ conf thấp."""
    lines = []
    if plain and norm(plain) != norm(text):
        lines.append(f'(bản nhận dạng khác của câu này: "{plain}")')
    words = text.split()
    if confs and len(confs) == len(words):
        low = [w for w, c in sorted(zip(words, confs), key=lambda x: x[1]) if c < thr and not re.fullmatch(r"\W+", w)]
        if low:
            lines.append(f"(từ nghe không chắc: {', '.join(dict.fromkeys(low[:max_words]))})")
    return "\n".join(lines)
