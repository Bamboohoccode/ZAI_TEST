"""Dựng hội thoại cho planner (system + lịch tham chiếu + history), cắt history cho vừa ngữ cảnh, parse output LLM."""
import json
import re
from datetime import datetime, timedelta

from lifemate.agent.prompts import SYSTEM
from lifemate.config import MAX_CTX, MAX_NEW


WEEKDAYS = ["thứ hai", "thứ ba", "thứ tư", "thứ năm", "thứ sáu", "thứ bảy", "chủ nhật"]


def date_table(now):
    t = datetime.fromisoformat(now)
    today = t.date()
    monday = today - timedelta(days=today.weekday())
    tags = {-1: "hôm qua", 0: "hôm nay" + (f", {t:%H:%M}" if "T" in now else ""), 1: "mai", 2: "mốt / ngày mốt / ngày kia"}
    weeks = ["tuần này", "tuần sau", "tuần sau nữa / tuần kia"]
    rows = []
    for i in range(-1, (monday + timedelta(days=21) - today).days):  # through Sunday of the third week
        d = today + timedelta(days=i)
        week = weeks[(d - monday).days // 7] if d >= monday else ""
        rows.append(f"- {d}: {WEEKDAYS[d.weekday()]} {week}" + (f" ({tags[i]})" if i in tags else ""))
    return "\n".join(rows)


AMPM_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b")


CORRECT_RE = re.compile(r"à không|ý (?:tôi|mình) là|nhầm|khoan|đổi (?:lại|thành)")  # self-correction: the last time said wins


def time_hints(text):
    """am/pm times left in the text (ASR / mapper), as 24h: '9am' -> 09:00, '8:30pm' -> 20:30. The day ('mai') is the date table's job.
    shortcut: only am/pm forms are caught; other time spellings are left to the planner."""
    lines = []
    for m in AMPM_RE.finditer(text.lower()):
        h, mi = int(m.group(1)), int(m.group(2) or 0)
        if 1 <= h <= 12 and mi < 60:
            lines.append(f'- "{m.group(0)}" = {h % 12 + (12 if m.group(3) == "pm" else 0):02d}:{mi:02d}')
    return "\n".join(lines[-1:] if CORRECT_RE.search(text.lower()) else lines)


def tool_specs(item, catalog):
    names = [n for app in item["context"]["installed_apps"].values() for n in app["tools"]] + ["sys.ask_missing_info"]
    specs = []
    for n in dict.fromkeys(names):
        if t := catalog.get(n):
            desc = f"[{t['kind']}] {t['description']}" + (f" Trả về: {t['returns']}" if t.get("returns") else "")
            specs.append({"type": "function", "function": {"name": n, "description": desc, "parameters": t["parameters"]}})
    return specs


def clip(obj, n=1500):
    s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + "…"


def assistant_msg(text, calls):
    return {"role": "assistant", "content": text,
            "tool_calls": [{"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]}


def hints(text):
    h = time_hints(text)
    return "\nGiờ trích từ lời người dùng (24h; ngày xem bảng lịch trên, dùng đúng giờ này):\n" + h if h else ""


def build_messages(item, texts, keep):
    """texts: one per user turn (history order, then the latest). keep: how many history user turns to keep."""
    ctx = item["context"]
    system = SYSTEM.format(now=ctx["current_time"], tz=ctx["timezone"], dates=date_table(ctx["current_time"]) + hints(texts[-1]),
                           loc=json.dumps(ctx.get("location"), ensure_ascii=False),
                           profile=json.dumps(ctx.get("user_profile"), ensure_ascii=False))
    hist, ti = [], 0
    for h in item["history"]:
        if h["role"] == "user":
            hist.append({"role": "user", "content": texts[ti]})
            ti += 1
        elif h.get("tool_calls"):
            hist.append(assistant_msg("", h["tool_calls"]))
            hist += [{"role": "tool", "content": clip(r)} for r in h.get("tool_results") or [{}] * len(h["tool_calls"])]
            hist.append({"role": "assistant", "content": h.get("text", "")})
        else:
            hist.append({"role": "assistant", "content": h.get("text", "")})
    starts = [i for i, m in enumerate(hist) if m["role"] == "user"]
    if keep < len(starts):  # drop oldest turns, cutting only at user messages
        hist = hist[starts[len(starts) - keep]:] if keep else []
    return [{"role": "system", "content": system}, *hist, {"role": "user", "content": texts[-1]}]


def fit_messages(item, texts, llm, tools, max_len=MAX_CTX):
    """build_messages, bỏ bớt lượt history cũ nhất tới khi vừa ngữ cảnh (max_len = max_model_len của LLM). -> (msgs, keep)."""
    keep = len(texts) - 1
    msgs = build_messages(item, texts, keep)
    while keep and llm.n_tokens(msgs, tools) > max_len - MAX_NEW * 2:
        keep -= 1
        msgs = build_messages(item, texts, keep)
    return msgs, keep


def parse_calls(text):
    calls = []
    for raw in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S):
        try:
            c = json.loads(raw)
            args = c.get("arguments") or {}
            calls.append({"name": c["name"], "arguments": json.loads(args) if isinstance(args, str) else args})
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
    return calls


def parse_verdict(text):
    """(accept, reason). Anything unparseable counts as ACCEPT."""
    m = re.search(r"\{.*\}", text, re.S)
    try:
        v = json.loads(m.group(0)) if m else {}
    except ValueError:
        v = {}
    if not isinstance(v, dict) or "verdict" not in v:
        return not text.strip().upper().startswith("REJECT"), text.strip()
    return str(v["verdict"]).upper() != "REJECT", str(v.get("reason", ""))


def unseen_ids(call, catalog, msgs):
    """x-ref args whose value never appeared in history or tool results: invented IDs."""
    props = catalog[call["name"]]["parameters"].get("properties", {})
    blob = "\n".join(str(m.get("content", "")) + json.dumps(m.get("tool_calls", ""), ensure_ascii=False) for m in msgs)
    return [k for k, v in call["arguments"].items() if props.get(k, {}).get("x-ref") and isinstance(v, str)
            and not re.search(r"(?<![\w-])" + re.escape(v) + r"(?![\w-])", blob)]
