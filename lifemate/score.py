"""Chấm submission theo luật LifeMate trên split có nhãn (train).

  - Gold rỗng: nộp rỗng được 1, ngược lại 0.
  - state_hash sai -> 0. Hash gold = chạy lại label.tool_calls trên một ToolEnv mới (state.json chỉ ToolEnv đọc).
  - Hash đúng -> F1 trên tool call: khớp khi đúng tên tool, cùng tập tham số, tham số thường giống hệt, tham số
    x-match = "fuzzy" có token-F1 > 0.6. Không xét thứ tự; mỗi call gold khớp tối đa một lần.
  - In thêm F1 bỏ qua hash (tiêu chí hoà điểm của BTC), theo loại mẫu (empty / ask / act) và theo app.
"""
import json
import unicodedata
from collections import defaultdict


def toks(s):
    return unicodedata.normalize("NFC", str(s)).lower().split()


def f1(a, b):
    a, b = toks(a), toks(b)
    common = sum(min(a.count(w), b.count(w)) for w in set(a))
    return 2 * common / (len(a) + len(b)) if common else 0.0


def match(c, g, catalog):
    if c.get("name") != g["name"]:
        return False
    props = catalog.get(c["name"], {}).get("parameters", {}).get("properties", {})
    ca, ga = c.get("arguments") or {}, g.get("arguments") or {}
    if set(ca) != set(ga):
        return False
    return all(f1(ca[k], ga[k]) > 0.6 if props.get(k, {}).get("x-match") == "fuzzy" else ca[k] == ga[k] for k in ga)


def call_f1(calls, gold, catalog):
    if not gold:
        return float(not calls)
    left, m = list(gold), 0
    for c in calls:
        j = next((j for j, g in enumerate(left) if match(c, g, catalog)), None)
        if j is not None:
            left.pop(j)
            m += 1
    return 2 * m / (len(calls) + len(gold)) if calls or gold else 1.0


def kind(gold):
    return "empty" if not gold else "ask" if any(g["name"] == "sys.ask_missing_info" for g in gold) else "act"


def score(sub_rows, items, catalog, ToolEnv, split, root):
    """-> dict tổng hợp + list chi tiết từng mẫu."""
    gold_items = {it["id"]: it for it in items}
    rows = []
    for s in sub_rows:
        it = gold_items.get(s["id"])
        if it is None:
            continue
        G = it["label"]["tool_calls"]
        env = ToolEnv(split, s["id"], root=str(root))
        for g in G:
            env.call(g["name"], g["arguments"])
        hash_ok = s.get("state_hash") == env.get_state_hash()
        f = call_f1(s.get("tool_calls") or [], G, catalog)
        sc = f if not G else (f if hash_ok else 0.0)
        apps = sorted({g["name"].split(".")[0] for g in G if g["name"] != "sys.ask_missing_info"}) or ["(none)"]
        rows.append({"id": s["id"], "score": sc, "f1": f, "hash_ok": hash_ok, "kind": kind(G), "apps": apps,
                     "n_pred": len(s.get("tool_calls") or []), "n_gold": len(G)})
    if not rows:
        return {"n": 0}, rows
    agg = {"n": len(rows), "score": sum(r["score"] for r in rows) / len(rows),
           "f1_no_hash": sum(r["f1"] for r in rows) / len(rows),
           "hash_fail": sum(not r["hash_ok"] for r in rows if r["n_gold"]) / max(1, sum(bool(r["n_gold"]) for r in rows))}
    by = defaultdict(list)
    for r in rows:
        by["kind:" + r["kind"]].append(r["score"])
        for app in r["apps"]:
            by["app:" + app].append(r["score"])
    agg["groups"] = {k: (sum(v) / len(v), len(v)) for k, v in sorted(by.items())}
    return agg, rows


def report(agg, rows, worst=15):
    if not agg.get("n"):
        print("không có mẫu nào trùng id với nhãn")
        return
    print(f"SCORE {agg['score']:.4f} trên {agg['n']} mẫu | F1 bỏ qua hash {agg['f1_no_hash']:.4f} | "
          f"sai hash {agg['hash_fail']:.1%} (mẫu có gold)")
    for k, (v, n) in agg["groups"].items():
        if k.startswith("kind:"):
            print(f"  {k[5:]:<8} {v:.4f} (n={n})")
    apps = sorted(((v, n, k[4:]) for k, (v, n) in agg["groups"].items() if k.startswith("app:")), key=lambda x: x[0])
    print("  app tệ nhất: " + ", ".join(f"{a} {v:.2f}/{n}" for v, n, a in apps[:10]))
    bad = sorted((r for r in rows if r["score"] < 1), key=lambda r: r["score"])[:worst]
    for r in bad:
        print(f"    {r['id']}: score {r['score']:.2f} f1 {r['f1']:.2f} hash {'ok' if r['hash_ok'] else 'SAI'} "
              f"{r['kind']} pred {r['n_pred']} / gold {r['n_gold']} {','.join(r['apps'])}")


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
