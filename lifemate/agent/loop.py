"""Vòng lặp agent: planner đề xuất tool call -> verifier ACCEPT/REJECT -> kiểm tra bằng code (schema, ID bịa, tool đọc
lặp) -> chạy trên ToolEnv. Thêm bước prefetch: gọi tool đọc một lần để lấy tên làm hotword decode lại câu cuối."""
import json
import re
import time

from lifemate.agent.messages import assistant_msg, clip, parse_calls, parse_verdict, unseen_ids
from lifemate.agent.prompts import PREFETCH, REPLAN, VERIFY
from lifemate.bias import _walk as walk_entities, add_to_pool, snap_value, text_entities
from lifemate.config import MAX_NEW, MAX_STEPS, VERIFY_ROUNDS


def plan_step(llm, msgs, tools, trace, rounds, deadline):
    """Planner proposes the next calls; verifier ACCEPTs or REJECTs with a reason, at most `rounds` times."""
    query = msgs[-1]["content"] if msgs[-1]["role"] == "user" else next(m["content"] for m in reversed(msgs) if m["role"] == "user")
    feedback = []
    for attempt in range(rounds + 1):
        out = llm.chat([msgs + feedback], tools)[0]
        trace.append({"plan": out})
        calls = parse_calls(out)
        if not calls or attempt == rounds or time.time() > deadline:  # nothing to check, past N rejects, or out of time: run the last plan
            return out, calls
        shown = "\n".join(f"<tool_call>{json.dumps(c, ensure_ascii=False)}</tool_call>" for c in calls)
        verdict = llm.chat([msgs + [{"role": "user", "content": VERIFY.format(query=query, calls=shown, text=re.sub(r"<tool_call>.*?</tool_call>", "", out, flags=re.S).strip())}]], tools)[0]
        ok, reason = parse_verdict(verdict)
        trace.append({"verify": verdict})
        if ok:
            return out, calls
        feedback = [{"role": "assistant", "content": out}, {"role": "user", "content": REPLAN.format(reason=reason)}]


# search keys: the string arguments that must name an entity of the user's own data (contact, channel, room...)
SEARCH_KEYS = {"contacts.search": "query", "tv.guide": "channel", "tv.tune": "channel", "loyalty.list_vouchers": "merchant",
               "loyalty.get_points": "program", "loyalty.redeem": "program", "household_chores.assign": "assignee",
               "smart_home.get_status": "room", "podcast_radio.play": "station"}


def snap_search_keys(call, pools, trace):
    """Search-key biasing: snap the entity argument of a tool call to the closest name already seen for that app (by sound)."""
    k = SEARCH_KEYS.get(call["name"])
    pool = pools.get(call["name"].split(".")[0])
    old = call["arguments"].get(k)
    if k and pool and isinstance(old, str):
        new = snap_value(old, pool)
        if new != old:
            call["arguments"][k] = new
            trace.append({"snap": {"tool": call["name"], "arg": k, "from": old, "to": new}})


def run_agent(env, llm, msgs, tools, catalog, trace, rounds=VERIFY_ROUNDS, deadline=float("inf"), pools=None, pre_done=(),
              max_len=None):
    """pre_done: [(tool, args, result)] đã chạy trên env trước đó (--prefetch), đã có trong msgs: không gọi lại tool đọc đó.
    max_len: max_model_len của LLM; kết quả tool làm hội thoại sắp tràn -> dừng, giữ các call đã chạy (thay vì lỗi vLLM)."""
    done = {json.dumps({"name": n, "arguments": a}, sort_keys=True, ensure_ascii=False) for n, a, _ in pre_done}
    for _ in range(MAX_STEPS):
        if time.time() > deadline:  # keep the calls that already ran
            trace.append({"error": "time limit"})
            return
        if max_len and llm.n_tokens(msgs, tools) > max_len - MAX_NEW * 2:  # chừa chỗ cho plan + prompt verify
            trace.append({"error": "context full"})
            return
        out, calls = plan_step(llm, msgs, tools, trace, rounds, deadline)
        if not calls:
            return
        ai = len(msgs)
        msgs.append(assistant_msg(re.sub(r"<tool_call>.*?</tool_call>", "", out, flags=re.S).strip(), calls))
        stop = False
        for c in calls:
            if pools and pools.get(c["name"].split(".")[0]):
                snap_search_keys(c, pools, trace)
            key = json.dumps(c, sort_keys=True, ensure_ascii=False)
            v = env.validate(c["name"], c["arguments"])  # not logged: invalid calls never reach the submission
            # the planner's own message is excluded: it would make every invented ID look seen
            bad = v.get("ok") and unseen_ids(c, catalog, msgs[:ai] + msgs[ai + 1:])
            if not v.get("ok"):
                r = {"ok": False, "error": v.get("error"), "note": "Chưa thực thi, hãy sửa tham số."}
            elif bad:  # a made-up ID runs as a silent no-op and breaks the state
                r = {"ok": False, "error": f"Giá trị {bad} chưa xuất hiện trong kết quả tool nào, không được tự đặt ID.",
                     "note": "Chưa thực thi. Gọi tool đọc để lấy ID, hoặc sys.ask_missing_info nếu không lấy được."}
            elif key in done and catalog[c["name"]]["kind"] == "read":  # repeated writes can be legit
                r = {"ok": False, "error": "Đã gọi tool đọc này với cùng tham số, dùng lại kết quả trước."}
            else:
                r = env.call(c["name"], c["arguments"])
                done.add(key)
                if pools is not None and isinstance(r, dict) and r.get("ok"):
                    add_to_pool(pools, c["name"], r)
                stop |= c["name"] == "sys.ask_missing_info"
            msgs.append({"role": "tool", "content": clip(r)})
        if stop:
            return


MAX_PREFETCH_CALLS = 4


MAX_PREFETCH_ENTS = 120


def prefetch_entities(llm, msgs, tools, catalog, env, trace, common=frozenset()):
    """Planner xem bản ASR lượt 1 và gọi tool đọc liên quan (tối đa MAX_PREFETCH_CALLS) trên `env` của sample (env thật:
các lần gọi nằm trong tool_calls của bài nộp và state hash) -> [(tool, args, result)] và list tên (ENTITY_KEYS) trong kết
quả để làm hotword cho lần decode lại câu cuối."""
    reads = [t for t in tools if catalog.get(t["function"]["name"], {}).get("kind") == "read"]
    if not reads:
        return [], []
    out = llm.chat([msgs + [{"role": "user", "content": PREFETCH}]], reads)[0]
    calls = [c for c in parse_calls(out) if catalog.get(c["name"], {}).get("kind") == "read"][:MAX_PREFETCH_CALLS]
    if not calls:
        trace.append({"prefetch": {"calls": [], "entities": []}})
        return [], []
    done, ents = [], []
    for c in calls:
        if not env.validate(c["name"], c["arguments"]).get("ok"):
            continue
        r = env.call(c["name"], c["arguments"])
        if isinstance(r, dict) and r.get("ok") is not False:
            done.append((c["name"], c["arguments"], r))
            walk_entities(r, ents)
    ents = [" ".join(str(e).split()) for e in ents if str(e).strip()]
    # chuỗi dài (tiêu đề lịch / cuộc họp: "Gọi video với Hằng ở Taipei") -> thêm tên riêng bên trong (Hằng, Taipei)
    ents += [x for e in ents if len(e.split()) > 3 for x in text_entities(e, common)]
    ents = list(dict.fromkeys(ents))[:MAX_PREFETCH_ENTS]
    trace.append({"prefetch": {"calls": [{"name": n, "arguments": a} for n, a, _ in done], "entities": ents}})
    return done, ents
