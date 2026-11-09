"""TN (written -> spoken) cho trường ref của asr_train.jsonl bằng agy (LLM), để mapper học hyp -> dạng đọc rồi ITN lo phần viết.

    python mapper/tn_agy.py --in data/asr_train.jsonl --out data/asr_train_tn.jsonl          # thêm trường "ref_spoken" (+ "tn_ok")
    python mapper/tn_agy.py ... --workers 4 --batch 40

- Câu không có chữ số: ref_spoken = norm(ref), không gọi LLM.
- Câu có chữ số: agy đổi số sang chữ, ưu tiên CÁCH ĐỌC của người nói (lấy từ hyp) nếu đó là cách đọc hợp lệ của đúng con số.
- Kiểm tra: không còn chữ số, ở dạng norm, và mọi từ không-phải-số của ref còn nguyên theo thứ tự. Sai -> gửi lại theo lô nhỏ;
  vẫn sai -> giữ bản tốt nhất, tn_ok=false (seq2seq_mapper bỏ qua các câu này khi --target spoken).
- Có cache (--cache) nên chạy lại chỉ gọi những câu chưa xong.
"""
import argparse
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import sys

if __package__ in (None, ""):  # chạy trực tiếp: python mapper/tn_agy.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from textnorm.textutil import norm  # noqa: E402

PROMPT = """Bạn là bộ Text Normalization (TN) tiếng Việt: chuyển câu viết (ref) sang DẠNG ĐỌC đúng như người nói đã nói.

Quy tắc cho mỗi item:
1. Output "spoken": chữ thường, không dấu câu, mọi chữ số và ký hiệu (%, +, :, /, đ, ...) đổi thành chữ tiếng Việt.
2. GIỮ NGUYÊN mọi từ không phải số của ref (kể cả từ địa phương như "ni", "chi", viết tắt như "sjc", "sms"), chỉ viết thường.
   TUYỆT ĐỐI không sửa từ theo hyp, không thêm/bớt từ ngoài phần số.
3. "hyp" là kết quả ASR của chính câu đó (có thể sai). Chỉ dùng hyp để CHỌN CÁCH ĐỌC số giống người nói, khi cách đọc trong hyp
   là một cách đọc hợp lệ của ĐÚNG con số trong ref. Ví dụ: 27 -> "hai bảy" hoặc "hai mươi bảy"; 125 -> "một trăm hai lăm" hoặc
   "một trăm hai mươi lăm"; 2.800.000 -> "hai triệu tám" hoặc "hai triệu tám trăm nghìn"; 7:30 -> "bảy giờ ba mươi" hoặc "bảy rưỡi";
   0 -> "không" (người nam có thể nói "không"/"linh"/"lẻ"). Nếu hyp nghe sai số hoặc không rõ, dùng cách đọc tự nhiên phổ biến nhất.
4. Mã chữ + số (RV245, VTV3, VN123): phần chữ giữ nguyên chữ thường (rv, vtv), phần số đọc như người nói (thường từng chữ số: "hai bốn năm").
   Số điện thoại, số tài khoản: đọc từng chữ số. "+" -> "cộng".
5. Đơn vị đi kèm số đọc thành chữ nếu là ký hiệu: 24°C -> "hai mươi bốn độ c", 50% -> "năm mươi phần trăm", 50k -> "năm mươi ca".

Ví dụ:
ref: "Bật điều hoà phòng ngủ 27 độ giúp tôi." hyp: "bật điều hoà phòng ngủ hai bảy độ giúp tôi"
-> "bật điều hoà phòng ngủ hai bảy độ giúp tôi"
ref: "Bật máy lạnh phòng khách 24 độ giúp tôi." hyp: "bằng máy lạnh phòng khách hai bố đội giúp tôi"
-> "bật máy lạnh phòng khách hai bốn độ giúp tôi"
ref: "Đi chuyến RV305 ấy." hyp: "đi chuyến rv ba không năm ấy" -> "đi chuyến rv ba không năm ấy"
ref: "Nhắc tôi 9 giờ sáng mai." hyp: "nhắc tôi chín giờ sáng mai" -> "nhắc tôi chín giờ sáng mai"

Chỉ in đúng một JSON object, không giải thích, không markdown: {"items":[{"i":<số>,"spoken":"<câu>"}, ...]} đủ mọi item theo đúng i.

Items:
"""


def check(ref, spoken):
    """'' nếu hợp lệ, ngược lại lý do."""
    if not spoken:
        return "empty"
    if re.search(r"\d", spoken):
        return "còn chữ số"
    if norm(spoken) != spoken:
        return "chưa norm"
    keep = [w for w in norm(ref).split() if not re.search(r"\d", w)]
    it = iter(spoken.split())
    if not all(any(w == s for s in it) for w in keep):  # subsequence
        return "mất/đổi từ không-phải-số"
    return ""


def call_agy(items, timeout):
    payload = json.dumps([{"i": i, "ref": r["ref"], "hyp": r["hyp"]} for i, r in items], ensure_ascii=False)
    try:
        p = subprocess.run(["agy", "-p", PROMPT + payload, "--print-timeout", f"{timeout}s"], stdin=subprocess.DEVNULL,
                           capture_output=True, text=True, timeout=timeout + 60)
        txt = p.stdout
        m = re.search(r"\{.*\}", txt, re.S)
        return {int(x["i"]): str(x["spoken"]).strip() for x in json.loads(m.group(0))["items"]} if m else {}
    except Exception as e:  # timeout, JSON hỏng... -> cả lô coi như chưa xong, vòng sau gửi lại
        print("  agy lỗi:", type(e).__name__, str(e)[:200])
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/asr_train.jsonl")
    ap.add_argument("--out", default="data/asr_train_tn.jsonl")
    ap.add_argument("--cache", default="data/tn_agy_cache.jsonl", help="kết quả hợp lệ đã có (i -> spoken)")
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=3, help="số vòng gửi lại; lô nhỏ dần /4 mỗi vòng")
    ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8") if l.strip()]
    key = lambda r: f"{r['id']}|{r['turn']}"
    cache = {}
    if Path(a.cache).exists():
        cache = {d["key"]: d["spoken"] for d in map(json.loads, open(a.cache, encoding="utf-8"))}
    todo = [(i, r) for i, r in enumerate(rows) if r.get("ref") and re.search(r"\d", r["ref"]) and key(r) not in cache]
    print(f"{len(rows)} câu | cần TN: {len(todo)} chưa xong (+{len(cache)} trong cache)")

    best, reason, bs = {}, {}, a.batch
    with open(a.cache, "a", encoding="utf-8") as fc:
        for rnd in range(a.rounds):
            if not todo:
                break
            batches = [todo[k:k + bs] for k in range(0, len(todo), bs)]
            print(f"vòng {rnd + 1}: {len(todo)} câu, {len(batches)} lô x {bs}")
            with ThreadPoolExecutor(a.workers) as ex:
                results = list(ex.map(lambda b: (b, call_agy(b, a.timeout)), batches))
            left = []
            for b, res in results:
                for i, r in b:
                    sp = norm(res.get(i, ""))
                    why = check(r["ref"], sp)
                    if not why:
                        cache[key(r)] = sp
                        fc.write(json.dumps({"key": key(r), "spoken": sp}, ensure_ascii=False) + "\n")
                    else:
                        if sp:
                            best[i] = sp
                        reason[i] = why
                        left.append((i, r))
            fc.flush()
            todo, bs = left, max(1, bs // 4)

    n_bad = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            if r.get("ref") and key(r) in cache:
                r["ref_spoken"], r["tn_ok"] = cache[key(r)], True
            elif r.get("ref") and not re.search(r"\d", r["ref"]):
                r["ref_spoken"], r["tn_ok"] = norm(r["ref"]), True
            else:
                r["ref_spoken"], r["tn_ok"] = best.get(i, norm(r.get("ref") or "")), False
                n_bad += 1
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for i, r in todo[:10]:
        print(f"  FAIL ({reason.get(i)}): {r['ref']}\n        -> {best.get(i)}")
    same = sum(norm(r["hyp"]) == r["ref_spoken"] for r in rows if r.get("ref"))
    print(f"xong -> {a.out} | tn_ok=false: {n_bad} | hyp trùng hệt ref_spoken: {same}/{len(rows)}"
          f" (trước TN, so với norm(ref): {sum(norm(r['hyp']) == norm(r['ref']) for r in rows if r.get('ref'))})")


if __name__ == "__main__":
    main()
