"""Chạy một mẫu end to end: ASR các lượt user (+hotword theo lượt) -> [prefetch: tool đọc -> tên làm hotword -> decode lại
câu cuối] -> mapper phương ngữ -> planner <-> verifier -> zaic_tools. Cộng chế độ chỉ ASR để đo WER trên train.

Mọi lựa chọn đến từ `a` (argparse.Namespace của scripts/run.py)."""
import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from lifemate.agent.loop import prefetch_entities, run_agent
from lifemate.agent.messages import assistant_msg, clip, fit_messages, tool_specs
from lifemate.agent.prompts import mapper_conv
from lifemate.asr.decode import decode_turns, run_asr
from lifemate.bias import _dedup, add_to_pool, app_pools, lexicons_per_turn
from lifemate.data import user_turns
from lifemate.text import norm, wer_counts
from lifemate.vi_num import to_spoken


@dataclass
class Result:
    calls: list
    state_hash: str
    raw: list = field(default_factory=list)       # ASR từng lượt
    texts: list = field(default_factory=list)     # sau mapper (vào planner)
    keep: int = 0                                 # số lượt history giữ lại cho vừa ngữ cảnh
    trace: list = field(default_factory=list)
    seconds: float = 0.0

    def submission(self, sid):
        return {"id": sid, "tool_calls": self.calls, "state_hash": self.state_hash}

    def trace_row(self, sid):
        return {"id": sid, "raw_texts": self.raw, "mapped_texts": self.texts, "kept_history": self.keep,
                "steps": self.trace, "seconds": round(self.seconds, 2)}


def asr_turns(a, item, asr, paths, llm, tools, catalog, pools, trace, common, env=None):
    """ASR mọi lượt (bias theo lượt + fallback). a.prefetch và llm không None: planner gọi tool đọc liên quan một lần trên
    `env` -> tên trong kết quả thêm vào hotword của câu cuối -> decode lại câu cuối.
    -> (texts, fell, done): done = [(tool, args, result)] đã chạy trên env, đưa vào hội thoại của planner."""
    lexs = lexicons_per_turn(item, common=common) if a.bias else None
    plain = run_asr(asr, paths, None) if a.bias and a.bias_fallback > 0 else None
    texts, fell = decode_turns(asr, paths, lexs, a.bias_fallback, plain)
    if not (a.prefetch and llm is not None):
        return texts, fell, []
    msgs1, _ = fit_messages(item, texts, llm, tools, a.max_len)
    done, ents = prefetch_entities(llm, msgs1, tools, catalog, env, trace, common)
    if pools is not None:
        for name, args, res in done:  # tên đã đọc được cũng dùng để snap search key của tool call
            add_to_pool(pools, name, args, res)
    new = [e for e in ents if e.lower() not in {x.lower() for x in lexs[-1]}]
    if new:
        lexs[-1] = _dedup(lexs[-1] + new)
        t2, f2 = decode_turns(asr, paths[-1:], lexs[-1:], a.bias_fallback, plain[-1:] if plain else None)
        trace.append({"redecode": {"before": texts[-1], "after": t2[0], "n_new_hotwords": len(new), "fallback": f2[0]}})
        texts[-1], fell[-1] = t2[0], f2[0]
    return texts, fell, done


def run_sample(a, item, ToolEnv, catalog, asr, llm, mapper, common, gold=False, rounds=2):
    """-> Result. Lỗi giữa chừng không làm mất mẫu: các call đã chạy vẫn nằm trong env và được nộp."""
    t0 = time.time()
    env = ToolEnv(a.split, item["id"], root=str(a.root))
    res = Result([], "")
    deadline = t0 + a.time_limit if a.time_limit else float("inf")
    try:
        turns = user_turns(item)
        tools = tool_specs(item, catalog)
        pools = app_pools(item)  # snap search key: tên trong tool call -> tên đã thấy trong history / kết quả tool
        pre_done = []
        if a.text:
            res.raw = [u["audio_text"] for u in turns]
        else:
            paths = [Path(a.root) / a.split / u["audio"] for u in turns]
            res.raw, fell, pre_done = asr_turns(a, item, asr, paths, None if gold else llm, tools, catalog, pools,
                                                res.trace, common, env)
            res.trace.extend({"bias_fallback": k} for k, f in enumerate(fell) if f)
        # mapper phương ngữ (số giữ dạng chữ, planner tự đổi sang tham số); output rỗng -> giữ câu cũ
        res.texts = ([m or r for r, m in zip(res.raw, mapper.chat([mapper_conv(r) for r in res.raw], None,
                                                                   {"enable_thinking": False}))]
                     if mapper else list(res.raw))
        msgs, res.keep = fit_messages(item, res.texts, llm, tools, a.max_len)
        if pre_done:  # tool đọc của bước prefetch đã chạy trên env: planner thấy kết quả, đi tiếp từ đó
            msgs.append(assistant_msg("", [{"name": n, "arguments": x} for n, x, _ in pre_done]))
            msgs += [{"role": "tool", "content": clip(r)} for _, _, r in pre_done]
        if gold:
            llm.queue = list(item["label"]["tool_calls"])
        run_agent(env, llm, msgs, tools, catalog, res.trace, rounds, deadline=deadline, pools=pools, pre_done=pre_done,
                  max_len=a.max_len)
    except Exception as e:  # giữ những gì đã chạy; mẫu lỗi vẫn phải có dòng trong submission
        res.trace.append({"error": f"{type(e).__name__}: {str(e)[:300]}", "traceback": traceback.format_exc()})
    res.calls, res.state_hash = env.get_tool_calls(), env.get_state_hash()
    res.seconds = time.time() - t0
    return res


def run_asr_only(a, items, ToolEnv, catalog, asr, llm, common, out_path):
    """Chỉ phần âm thanh (không mapper / planner): mỗi dòng {"id", "turns": [{"turn", "audio", "asr", ["ref"]}]}.
    Split có audio_text (train) -> in WER dạng đọc (so theo text.canon). --prefetch cần LLM (chạy tool đọc trên env)."""
    tot = [0, 0]
    t_start = time.time()
    with open(out_path, "w", encoding="utf-8") as fout:
        for i, item in enumerate(items):
            t0 = time.time()
            turns = user_turns(item)
            paths = [Path(a.root) / a.split / u["audio"] for u in turns]
            trace = []
            env = ToolEnv(a.split, item["id"], root=str(a.root)) if a.prefetch else None
            tools = tool_specs(item, catalog) if a.prefetch else []
            texts, fell, _ = asr_turns(a, item, asr, paths, llm, tools, catalog, None, trace, common, env)
            rows = []
            for k, (u, hyp) in enumerate(zip(turns, texts)):
                row = {"turn": "QUERY" if k == len(turns) - 1 else f"u{k + 1}", "audio": u["audio"], "asr": hyp}
                if fell[k]:
                    row["bias_fallback"] = True
                if k == len(turns) - 1 and trace:
                    row["prefetch"] = trace
                if u.get("audio_text"):
                    row["ref"] = u["audio_text"]
                    e, n = wer_counts(to_spoken(u["audio_text"], norm(hyp)) or norm(u["audio_text"]), norm(hyp))
                    tot[0], tot[1] = tot[0] + e, tot[1] + n
                rows.append(row)
            fout.write(json.dumps({"id": item["id"], "turns": rows}, ensure_ascii=False) + "\n")
            print(f"[{i + 1}/{len(items)}] {item['id']} {len(rows)} lượt {time.time() - t0:.1f}s | {rows[-1]['asr'][:120]}",
                  flush=True)
    print(f"\n{len(items)} mẫu -> {out_path} | {(time.time() - t_start) / max(len(items), 1):.1f}s/mẫu")
    if tot[1]:
        print(f"WER dạng đọc: {tot[0] / tot[1]:.4f} ({tot[1]} từ)")
