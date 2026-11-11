"""Voice agent: Zipformer ASR (+RNNLM) -> đánh dấu từ conf thấp -> [LLM refine, chỉ khi --refine] -> Qwen3-0.6B dialect mapper
-> Qwen3-4B planner <-> verifier -> zaic_tools. Sơ đồ: docs/asr_workflow.png, chi tiết: docs/PIPELINE.md.

pip install -r requirements.txt     # cần asr_assets.zip (lm/rnnlm/rnnlm.onnx, data/asr_train_plain.jsonl) giải nén ở gốc repo

python pipeline.py --root DATA --split public_test --lm lm/rnnlm/rnnlm.onnx --lm-scale 0.3 [--beam 8] [--vad] [--bias] \
    --conf-thr 0.6 --out submission.jsonl                              # full pipeline
python pipeline.py --root DATA --split train --lm lm/rnnlm/rnnlm.onnx --mark-only --limit 300
    # chỉ đánh dấu (không LLM, không GPU): in WER + bảng chọn --conf-thr (tỉ lệ đánh dấu / bắt được lỗi / precision)
python pipeline.py --root DATA --split train --lm lm/rnnlm/rnnlm.onnx --no-llm --refine --limit 200   # ASR -> refine, in WER
python pipeline.py --root DATA --split train --text --llm gold --mapper none --limit 50   # bỏ ASR, replay nhãn, không GPU
"""
import argparse
import gc
import json
import os
import random
import re
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hotwords.bias import _dedup, _walk as walk_entities, add_to_pool, app_pools, build_lexicon, common_words, lexicons_per_turn, snap_value, text_entities, tokens  # noqa: E402
from asr.candidates import Candidates, refine  # noqa: E402

ZIPFORMER_MODEL = "hynt/ZipFormer-150M-CR-CTC-RNNT-6000h"
MAPPER_MODEL = "Qwen/Qwen3-0.6B"
LLM_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
MAX_CTX = 16384
MAX_NEW = 512
MAX_STEPS = 8
VERIFY_ROUNDS = 2  # N: max REJECT -> re-plan rounds per step; after N the last plan runs unchecked
TIME_LIMIT_S = 30  # only reported at the end, no longer enforced

# ---------------------------------------------------------------- dialect mapper

MAPPER_SYSTEM = """Bạn chuẩn hoá câu nói tiếng Việt: đổi từ địa phương (Bắc, Trung, Nam) sang từ tiếng Việt phổ thông.
- Chỉ đổi từ ngữ địa phương, không thêm, bớt hay đổi ý.
- Giữ nguyên tên riêng, số, ngày giờ, số tiền, từ tiếng Anh (book, AC, wifi...).
- Câu đã chuẩn thì chép lại y nguyên.
- Chỉ trả lời câu đã chuẩn hoá, không giải thích."""

def mapper_conv(text):
    return [{"role": "system", "content": MAPPER_SYSTEM}, {"role": "user", "content": text}]


# ---------------------------------------------------------------- planner / verifier prompts

WEEKDAYS = ["thứ hai", "thứ ba", "thứ tư", "thứ năm", "thứ sáu", "thứ bảy", "chủ nhật"]

SYSTEM = """Bạn là trợ lý giọng nói, thực hiện yêu cầu của người dùng bằng cách gọi tool.
Thời gian hiện tại: {now} ({tz}). Vị trí hiện tại: {loc}.
Hồ sơ người dùng: {profile}
Lịch tham chiếu:
{dates}

Mỗi lượt, trước khi gọi tool, suy nghĩ ngắn gọn theo 3 bước (viết ra 1-3 dòng):
1. Đơn giản hoá: người dùng thực sự muốn gì, nói trong một câu ngắn? Nếu chỉ trò chuyện, cảm ơn, hỏi chung chung, đã tự hủy ("à thôi", "khoan", "thôi khỏi"), không app nào đã cài làm được, hoặc yêu cầu "nếu ... thì ..." mà điều kiện đã biết là sai từ lịch tham chiếu hay ngữ cảnh: KHÔNG gọi tool nào, kể cả tool đọc, chỉ trả lời một câu.
2. Suy nghĩ ngược từ mục tiêu: tool ghi cuối cùng cần gọi là gì -> tham số bắt buộc của nó (xem schema) -> mỗi tham số lấy từ đâu (lời nói, ngữ cảnh, hồ sơ, lịch sử, hay tool đọc nào). Chỉ gọi tool đọc thật sự cần để điền tham số hoặc kiểm tra điều kiện.
3. Sau khi đọc dữ liệu: đối chiếu từng tham số bắt buộc với kết quả. Còn thiếu, hoặc mơ hồ (không tìm thấy, hay có từ 2 kết quả khớp như hai liên hệ cùng tên) thì gọi sys.ask_missing_info(tool_name, params) rồi dừng, không đoán.

Yêu cầu có điều kiện ("nếu ... thì ...", "không ... thì thôi"): lượt đầu viết THỦ TỤC ngắn, mỗi dòng một bước, ví dụ:
THỦ TỤC:
1. đọc thời tiết ngày mai
2. nếu không mưa: dừng, không gọi tool ghi nào
3. nếu mưa: đọc sự kiện chạy bộ -> dời lịch (giờ mới = giờ cũ + 1 tiếng) -> nhắn Sếp Dũng giờ mới (lấy từ kết quả bước dời)
Rồi làm đúng từng bước: mỗi lượt chỉ gọi tool của bước hiện tại (lượt đầu chỉ tool đọc để kiểm tra điều kiện), dừng chờ kết quả, đối chiếu với thủ tục để biết bước kế. Điều kiện sai thì kết thúc, trả lời một câu. Các lượt sau không viết lại thủ tục, chỉ bám theo nó; mỗi bước lấy giá trị từ kết quả bước trước, không đoán.

Quy tắc:
- Chỉ gọi tool trong danh sách. ID (device_id, contact_id, event_id, ...) phải lấy từ kết quả tool hoặc lịch sử, không tự bịa.
- Dữ liệu đã có trong lịch sử thì dùng lại, không gọi lại tool đọc đó. Không làm việc người dùng không yêu cầu.
- Thời gian dùng ISO 8601 có múi giờ, ví dụ 2026-11-11T05:30:00+07:00. Tiền là số nguyên VND.
- Lời người dùng là bản nhận dạng giọng nói: có thể viết thường, không dấu câu, số đọc bằng chữ ("chín giờ ba mươi" = 09:30, "năm trăm nghìn" = 500000), có thể sai chính tả; hiểu theo ý gần nhất.
- Gọi tool bằng <tool_call>{{"name": ..., "arguments": {{...}}}}</tool_call>. Xong việc thì trả lời một câu ngắn, không gọi thêm tool."""

VERIFY = """[KIỂM TRA] Yêu cầu của người dùng: "{query}"
Tool call đề xuất cho bước tiếp theo:
{calls}
Lập luận và thủ tục của Planner ở lượt này (lượt sau không viết lại, thủ tục nằm trong lịch sử):
{text}

Kiểm tra kỹ, tự hỏi:
1. Câu yêu cầu (nghĩ đơn giản) có thật sự cần làm gì không? Trò chuyện, đã hủy, ngoài phạm vi app, điều kiện "nếu" sai -> không được gọi tool.
2. Đúng tool chưa, tham số đúng kiểu, đúng lời nói và lịch tham chiếu chưa? ID có lấy từ kết quả tool không?
3. Mỗi tool đọc có thật sự cần cho tham số của tool mục tiêu không? Có thừa hoặc sót ý nào của người dùng không? Việc đã chạy ở trên thì không làm lại.
4. Có tham số bắt buộc nào thiếu hoặc mơ hồ trong dữ liệu đã đọc không? Nếu có thì phải là sys.ask_missing_info, không đoán.
5. Yêu cầu có điều kiện: tool call có đúng bước kế tiếp của thủ tục không? Tool ghi chỉ được gọi khi điều kiện đã được kiểm tra bằng kết quả tool và đúng. Giá trị chuyển sang bước sau (giờ mới, ID) có lấy đúng từ kết quả bước trước không?
Đây có thể chỉ là một bước: gọi tool đọc để lấy ID cho bước sau là hợp lệ.
Chỉ trả lời JSON: {{"verdict": "ACCEPT"}} hoặc {{"verdict": "REJECT", "reason": "<lý do ngắn, nói cần sửa gì>"}}"""

REPLAN = "[BỊ TỪ CHỐI] {reason}\nSửa lại tool call cho bước này (hoặc không gọi tool nào nếu không cần)."


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
    """am/pm times the ITN step left in the text, as 24h: '9am' -> 09:00, '8:30pm' -> 20:30. The day ('mai') is the date table's job.
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

# ---------------------------------------------------------------- models


class VLLM:
    def __init__(self, name, lora, gpu_mem, seed, max_len, max_new, chat_kwargs=None, eager=False):
        from vllm import LLM, SamplingParams
        kw, self.lora = {}, None
        if lora:
            from vllm.lora.request import LoRARequest
            r = json.load(open(Path(lora) / "adapter_config.json"))["r"]
            kw = {"enable_lora": True, "max_lora_rank": next(x for x in (8, 16, 32, 64, 128, 256, 512) if x >= r)}
            self.lora = LoRARequest("ft", 1, str(lora))
        self.llm = LLM(name, enable_prefix_caching=True, max_model_len=max_len, seed=seed,
                       gpu_memory_utilization=gpu_mem, enforce_eager=eager, **kw)
        self.sp = SamplingParams(temperature=0, max_tokens=max_new)  # greedy: deterministic
        self.kw = chat_kwargs
        self.tok = self.llm.get_tokenizer()

    def n_tokens(self, msgs, tools):
        text = self.tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False,
                                            **(self.kw or {}))
        return len(self.tok.encode(text, add_special_tokens=False))

    def chat(self, convs, tools=None, chat_kwargs=None):
        """chat_kwargs: ghi đè chat_template_kwargs cho lần gọi này (refine ASR tắt thinking dù planner bật)."""
        outs = self.llm.chat(convs, self.sp, tools=tools, chat_template_kwargs=chat_kwargs or self.kw,
                             lora_request=self.lora, use_tqdm=False)
        return [o.outputs[0].text.strip() for o in outs]

    def close(self):
        import torch
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
        del self.llm
        gc.collect()
        cleanup_dist_env_and_memory()
        torch.cuda.empty_cache()


class HFLLM:
    """transformers thay cho vLLM (máy không có GPU / vLLM): cùng giao diện VLLM, greedy, xử lý từng conv một.
    gpu_mem / seed / max_len / eager bỏ qua (giữ chữ ký để dùng thay VLLM)."""

    def __init__(self, name, lora, gpu_mem, seed, max_len, max_new, chat_kwargs=None, eager=False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(name)
        self.m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(self.dev).eval()
        if lora:
            from peft import PeftModel
            self.m = PeftModel.from_pretrained(self.m, lora).eval()
        self.max_new, self.kw, self.prefix = max_new, chat_kwargs, {}

    def n_tokens(self, msgs, tools):
        text = self.tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False, **(self.kw or {}))
        return len(self.tok.encode(text, add_special_tokens=False))

    def _ids(self, conv, tools, kw, gen):
        text = self.tok.apply_chat_template(conv, tools=tools, add_generation_prompt=gen, tokenize=False, **kw)
        return self.tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]

    def chat(self, convs, tools=None, chat_kwargs=None):
        """Cache KV phần đầu chung (mọi message trừ message user cuối): refine dùng chung system + ví dụ mẫu cho mọi câu,
        trên CPU phần đọc prompt tốn nhất nên chỉ tính một lần."""
        import copy
        kw = chat_kwargs or self.kw or {}
        outs = []
        for c in convs:
            ids = self._ids(c, tools, kw, True)
            pre = self._ids(c[:-1], tools, kw, False) if len(c) > 1 else None
            cache = None
            if pre is not None and pre.shape[1] < ids.shape[1] and self.torch.equal(ids[0, :pre.shape[1]], pre[0]):
                key = (tuple(pre[0].tolist()))
                if key not in self.prefix:
                    if len(self.prefix) >= 4:
                        self.prefix.pop(next(iter(self.prefix)))
                    with self.torch.no_grad():
                        self.prefix[key] = self.m(pre.to(self.dev), use_cache=True).past_key_values
                cache = copy.deepcopy(self.prefix[key])
            with self.torch.no_grad():
                o = self.m.generate(ids.to(self.dev), attention_mask=self.torch.ones_like(ids).to(self.dev),
                                    past_key_values=cache, max_new_tokens=self.max_new, do_sample=False,
                                    temperature=None, top_p=None, top_k=None)
            # giữ <tool_call> (token thêm vào vocab), chỉ bỏ token kết thúc
            out = self.tok.decode(o[0, ids.shape[1]:], skip_special_tokens=False)
            outs.append(re.sub(r"<\|(im_end|endoftext)\|>", "", out).strip())
        return outs

    def close(self):
        del self.m
        gc.collect()


def _vllm_worker(conn, device, args, backend="vllm"):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)  # before torch/vllm are imported in this fresh process
    try:
        llm = (HFLLM if backend == "hf" else VLLM)(*args)
        conn.send("ready")
        while True:
            op, payload = conn.recv()
            conn.send(llm.chat(*payload) if op == "chat" else llm.n_tokens(*payload))
    except Exception as e:
        # gửi chuỗi, không gửi object: exception của NVML / CUDA không unpickle được ở process cha (che mất lỗi thật)
        conn.send(RuntimeError(f"[LLM process, GPU {device}] {type(e).__name__}: {e}\n{traceback.format_exc()}"))


class RemoteVLLM:
    """LLM (vLLM hoặc HF) trong process con chỉ thấy một GPU (`device`): LLM chính và mapper nằm được trên hai card khác nhau
    (--llm-device / --mapper-device). Cùng giao diện VLLM: chat(convs, tools, chat_kwargs), n_tokens(msgs, tools)."""

    def __init__(self, device, *args, backend="vllm"):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        ctx.Process(target=_vllm_worker, args=(child, device, args, backend), daemon=True).start()
        self._recv()  # "ready", or the engine's startup error

    def _recv(self):
        r = self.conn.recv()
        if isinstance(r, Exception):
            raise r
        return r

    def chat(self, convs, tools=None, chat_kwargs=None):
        self.conn.send(("chat", (convs, tools, chat_kwargs)))
        return self._recv()

    def n_tokens(self, msgs, tools):
        self.conn.send(("n_tokens", (msgs, tools)))
        return self._recv()


class GoldLLM:
    """Replays label.tool_calls one per step: tests env, executor and output without a GPU."""
    queue = []

    def n_tokens(self, msgs, tools):
        return 0

    def chat(self, convs, tools=None):
        if not self.queue:
            return ["Xong."]
        return [f"<tool_call>{json.dumps(self.queue.pop(0), ensure_ascii=False)}</tool_call>"]


def run_asr(model, paths, hotwords=None):
    """Decode one sample's files (history turns + query) in one batched call (deterministic).
    Zipformer RNNT (+RNNLM); `hotwords` bias the decoding when built with bias=True. -> [[(word, confidence)]] per file."""
    return model.transcribe([str(p) for p in paths], hotwords=hotwords, words=True)


# ---------------------------------------------------------------- agent


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


def run_agent(env, llm, msgs, tools, catalog, trace, rounds=VERIFY_ROUNDS, deadline=float("inf"), pools=None, pre_done=()):
    """pre_done: [(tool, args, result)] đã chạy trên env trước đó (--prefetch), đã có trong msgs: không gọi lại tool đọc đó."""
    done = {json.dumps({"name": n, "arguments": a}, sort_keys=True, ensure_ascii=False) for n, a, _ in pre_done}
    for _ in range(MAX_STEPS):
        if time.time() > deadline:  # keep the calls that already ran
            trace.append({"error": "time limit"})
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


THR_SWEEP = (0.4, 0.5, 0.6, 0.7, 0.8)


def decode_turns(asr, paths, lexs=None, fallback=0.6, plain=None):
    """ASR các lượt; lexs[k] = hotword riêng của lượt k (None = không bias). Các lượt cùng hotword decode chung một lô.
    fallback > 0: lượt có bias ra ít hơn fallback * số từ của bản không bias (hotword khớp nửa chừng làm cụt câu) -> dùng
    bản không bias (`plain`: kết quả không bias đã có sẵn, None = tự decode). -> (wls, keeps, fell)."""
    if not lexs:
        wls = run_asr(asr, paths, None)
        return wls, [frozenset()] * len(paths), [False] * len(paths)
    wls, groups = [None] * len(paths), {}
    for k, lx in enumerate(lexs):
        groups.setdefault(tuple(lx), []).append(k)
    for lx, ks in groups.items():
        for k, wl in zip(ks, run_asr(asr, [paths[k] for k in ks], list(lx) or None)):
            wls[k] = wl
    keeps, fell = [hotword_words(lx) for lx in lexs], [False] * len(paths)
    if fallback > 0:
        for k, pl in enumerate(plain or run_asr(asr, paths, None)):
            if len(wls[k]) < fallback * len(pl):
                wls[k], keeps[k], fell[k] = pl, frozenset(), True
    return wls, keeps, fell


# ---------------------------------------------------------------- prefetch: LLM gọi tool đọc 1 lần -> tên trong kết quả làm hotword

PREFETCH = """[CHUẨN BỊ] Câu cuối của người dùng là bản nhận dạng giọng nói, tên riêng (quán, món, người, kênh, địa điểm,
cuộc họp...) có thể bị nghe sai. Bước đầu của việc thực hiện yêu cầu: gọi các tool ĐỌC mà yêu cầu THẬT SỰ cần (để tìm đúng
đối tượng hoặc lấy ID cho bước sau), ví dụ danh sách quán khi đặt món, danh bạ khi nhắn / gọi ai đó, danh sách cuộc họp khi
vào cuộc họp. Các lần gọi này được tính là một phần của việc thực hiện yêu cầu: không gọi tool không liên quan.
Chỉ gọi tool đọc, không gọi tool ghi, không giải thích. Yêu cầu không cần đọc dữ liệu (trò chuyện, đã huỷ...) thì không gọi tool."""
MAX_PREFETCH_CALLS = 4
MAX_PREFETCH_ENTS = 120


def prefetch_entities(llm, msgs, tools, catalog, env, trace, common=frozenset()):
    """Planner xem bản ASR lượt 1 và gọi tool đọc liên quan (tối đa MAX_PREFETCH_CALLS) trên `env` của sample (env thật:
các lần gọi nằm trong tool_calls của bài nộp và state hash) -> [(tool, args, result)] và list tên (ENTITY_KEYS) trong kết
quả để làm hotword cho lần decode lại và đưa vào prompt refine."""
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


def asr_with_prefetch(a, item, asr, paths, llm, tools, catalog, pools, trace, common, env=None):
    """ASR mọi lượt (bias theo lượt + fallback). --prefetch (llm không None): planner gọi tool đọc liên quan 1 lần trên `env`
    -> tên trong kết quả thêm vào hotword QUERY -> decode lại QUERY; tên cũng đưa vào prompt refine của QUERY.
    -> (wls, keeps, fell, names, done, alts): done = [(tool, args, result)] đã chạy trên env, đưa vào hội thoại của planner;
    alts[k] = các bản decode khác của lượt k (bản không bias, --alt-zipformer) làm N-best cho refine."""
    lexs = lexicons_per_turn(item, common=common) if a.bias else None
    plain = run_asr(asr, paths, None) if a.bias and (a.bias_fallback > 0 or a.refine) else None
    wls, keeps, fell = decode_turns(asr, paths, lexs, a.bias_fallback, plain)
    names = [None] * len(wls)
    alts = [[] for _ in wls]
    for src in (plain, run_asr(a.alt_asr, paths, None) if getattr(a, "alt_asr", None) else None):
        for k, wl in enumerate(src or ()):
            alts[k].append([w for w, _ in wl])
    if not (a.prefetch and llm is not None):
        return wls, keeps, fell, names, [], alts
    pass1 = [" ".join(w for w, _ in wl) for wl in wls]
    msgs1, _ = fit_messages(item, pass1, llm, tools, a.max_len)
    done, ents = prefetch_entities(llm, msgs1, tools, catalog, env, trace, common)
    if pools is not None:
        for name, args, res in done:  # tên đã đọc được cũng dùng để snap search key của tool call
            add_to_pool(pools, name, args, res)
    new = [e for e in ents if e.lower() not in {x.lower() for x in lexs[-1]}]
    if new:
        lexs[-1] = _dedup(lexs[-1] + new)
        w2, k2, f2 = decode_turns(asr, paths[-1:], lexs[-1:], a.bias_fallback, plain[-1:] if plain else None)
        trace.append({"redecode": {"before": pass1[-1], "after": " ".join(w for w, _ in w2[0]),
                                   "n_new_hotwords": len(new), "fallback": f2[0]}})
        wls[-1], keeps[-1], fell[-1] = w2[0], k2[0], f2[0]
    if ents:
        names[-1] = [e for e in ents if len(e.split()) <= 4]  # tên ngắn; tiêu đề dài đã tách ở prefetch
    return wls, keeps, fell, names, done, alts


def fit_messages(item, texts, llm, tools, max_len=MAX_CTX):
    """build_messages, bỏ bớt lượt history cũ nhất tới khi vừa ngữ cảnh (max_len = max_model_len của LLM). -> (msgs, keep)."""
    keep = len(texts) - 1
    msgs = build_messages(item, texts, keep)
    while keep and llm.n_tokens(msgs, tools) > max_len - MAX_NEW * 2:
        keep -= 1
        msgs = build_messages(item, texts, keep)
    return msgs, keep


def refine_kw(a, alts, stats):
    """tham số refine chung cho hai chế độ chạy (xem --refine-loose / --refine-max-dist / --refine-free-dist)."""
    return dict(alts=alts, strict=not a.refine_loose, max_dist=a.refine_max_dist, free_dist=a.refine_free_dist, stats=stats)


def hotword_words(lex):
    """từ (viết thường) của các hotword --bias: không đánh dấu cho LLM sửa (xem Candidates.mark)."""
    return frozenset(t for e in lex for t in tokens(e))


def run_audio_only(a, items, asr, t_start, llm=None, cands=None, catalog=None):
    """--no-llm / --mark-only: VAD/ASR(+LM) -> đánh dấu conf thấp -> [LLM refine] cho mọi lượt user (không ITN: số giữ dạng chữ),
    không mapper/planner/tool. Mỗi dòng --out: {"id", "turns": [{"turn", "audio", "asr", "marked", "refine", ["ref"]}]}.
    Split có audio_text (train): in WER (asr thô / sau refine) và bảng chọn --conf-thr: với mỗi ngưỡng,
    % từ bị đánh dấu, % từ sai bắt được, precision (từ sai căn theo hyp <-> ref dạng đọc)."""
    import difflib
    from textnorm.textutil import norm, wer_counts
    from textnorm.vi_num import to_spoken
    root = Path(a.root)
    if a.prefetch:
        from zaic_tools import ToolEnv  # main() đã thêm thư mục chứa zaic_tools vào sys.path
    # từ thường của tập (để không coi chữ hoa đầu câu là tên riêng khi lấy entity từ text)
    common = frozenset(common_words(root / a.split / f"{a.split}.jsonl")) if a.bias else frozenset()
    tot = {"asr": [0, 0], "refine": [0, 0]}
    rstats = Counter()  # refine: số chỗ sửa nhận / bỏ theo lý do
    sweep = {t: [0, 0] for t in THR_SWEEP}  # thr -> [marked, marked & sai]
    n_words = n_wrong = n_sure = 0
    with open(a.out, "w", encoding="utf-8") as fout:
        for i, item in enumerate(items):
            t0 = time.time()
            turns = [h for h in item["history"] if h["role"] == "user"] + [item]
            paths = [root / a.split / u["audio"] for u in turns]
            trace = []
            env = ToolEnv(a.split, item["id"], root=str(root)) if a.prefetch else None
            wls, keeps, fell, names, _, alts = asr_with_prefetch(a, item, asr, paths, llm, tool_specs(item, catalog) if a.prefetch else [],
                                                           catalog, None, trace, common, env)
            asr_out = [" ".join(w for w, _ in wl) for wl in wls]
            fix, marked = refine(llm if a.refine else None, wls, cands, a.conf_thr, a.cand_k, keep=keeps, names=names,
                                 **refine_kw(a, alts, rstats))
            rows = []
            for k, (u, wl, r0, r1, r2) in enumerate(zip(turns, wls, asr_out, marked, fix)):
                row = {"turn": "QUERY" if k == len(turns) - 1 else f"u{k + 1}", "audio": u["audio"], "asr": r0,
                       "marked": r1, "refine": r2}
                if fell[k]:
                    row["bias_fallback"] = True
                if k == len(turns) - 1 and trace:  # prefetch: tool đọc đã gọi, tên lấy được, ASR trước / sau khi decode lại
                    row["prefetch"] = trace
                if u.get("audio_text"):
                    ref = u["audio_text"]
                    row["ref"] = ref
                    for key, hyp, rf in (("asr", norm(r0), to_spoken(ref, norm(r0)) or norm(ref)),
                                         ("refine", norm(r2), to_spoken(ref, norm(r2)) or norm(ref))):
                        e, n = wer_counts(rf, hyp)
                        tot[key][0] += e
                        tot[key][1] += n
                    hw, rw = [w for w, _ in wl], (to_spoken(ref, norm(r0)) or norm(ref)).split()
                    ok = [False] * len(hw)
                    for blk in difflib.SequenceMatcher(None, hw, rw, autojunk=False).get_matching_blocks():
                        ok[blk.a:blk.a + blk.size] = [True] * blk.size
                    for (w, c), good in zip(wl, ok):
                        n_words, n_wrong, n_sure = n_words + 1, n_wrong + (not good), n_sure + (c >= 0.9999)
                        for t in THR_SWEEP:
                            if c < t and w not in keeps[k]:
                                sweep[t][0] += 1
                                sweep[t][1] += not good
                rows.append(row)
            fout.write(json.dumps({"id": item["id"], "turns": rows}, ensure_ascii=False) + "\n")
            n_mark = sum("[" in m for m in marked)
            print(f"[{i + 1}/{len(items)}] {item['id']} {len(rows)} lượt, đánh dấu {n_mark} "
                  f"{time.time() - t0:.1f}s | {rows[-1]['refine' if a.refine and llm is not None else 'marked'][:120]}", flush=True)
    total = time.time() - t_start
    print(f"\n{len(items)} samples -> {a.out} | avg {total / max(len(items), 1):.1f}s/sample")
    if tot["asr"][1]:
        print(f"WER dạng đọc, ASR thô        : {tot['asr'][0] / tot['asr'][1]:.4f}")
        if llm is not None and a.refine:
            print(f"WER dạng đọc, sau LLM refine : {tot['refine'][0] / tot['refine'][1]:.4f}")
    if rstats:
        print("refine " + ("(kiểu cũ, nhận/bỏ cả câu)" if a.refine_loose else "(chặt, xét từng chỗ sửa)") + ": "
              + ", ".join(f"{k} {v}" for k, v in rstats.most_common()))
    if n_words:
        if n_sure == n_words:
            print("CẢNH BÁO: mọi từ có conf = 1 -> sherpa-onnx không trả ys_log_probs, refine sẽ không đánh dấu gì")
        print(f"\nChọn --conf-thr ({n_words} từ, {n_wrong} từ sai so với ref; từ bị ASR bỏ sót không tính)")
        print(f"{'conf-thr':>8}{'đánh dấu':>10}{'bắt lỗi':>10}{'precision':>11}")
        for t, (m, hit) in sweep.items():
            print(f"{t:>8.2f}{100 * m / n_words:>9.1f}%{100 * hit / max(n_wrong, 1):>9.1f}%{100 * hit / max(m, 1):>10.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder holding the split folders")
    ap.add_argument("--split", default="public_test")
    ap.add_argument("--tools-dir", default=None,
                    help="thư mục CHỨA package zaic_tools (mặc định tự tìm trong root/split, root/train, root)")
    ap.add_argument("--out", default="submission.jsonl")
    ap.add_argument("--zipformer", default=ZIPFORMER_MODEL, help="Zipformer HF id or local (fine-tuned) folder (sherpa-onnx files)")
    ap.add_argument("--zipformer-threads", type=int, default=4)
    ap.add_argument("--bias", action="store_true",
                    help="hotword theo từng mẫu: lexicon từ item (user_profile + tool call/kết quả trong history) -> Zipformer hotwords")
    ap.add_argument("--hotword-score", type=float, default=2.0, help="score khởi tạo contextual graph (hotword thật dùng --hotword-hi/lo)")
    ap.add_argument("--hotword-hi", type=float, default=2.5, help="hotword score cho tên riêng / viết tắt")
    ap.add_argument("--hotword-lo", type=float, default=1.5, help="hotword score cho cụm thường (đèn bếp, tài khoản thanh toán)")
    ap.add_argument("--glossary-score", type=float, default=1.0, help="score cho từ ngoại viết thường (podcast, voucher...); 0 = không làm hotword (dev: 0 kém hơn)")
    ap.add_argument("--hotwords-file", help="hotword CỐ ĐỊNH (mine_hotwords.py -> hotwords_global.txt), không đọc state.json")
    ap.add_argument("--bias-fallback", type=float, default=0.6,
                    help="--bias: lượt có bias ra ít hơn tỉ lệ này số từ của bản không bias (câu bị cụt) -> dùng bản không bias; 0 = tắt")
    ap.add_argument("--prefetch", action="store_true",
                    help="sau ASR lượt 1, LLM gọi tool đọc liên quan 1 lần (trên env thật: tính vào tool_calls / state hash) -> tên trong kết quả "
                         "làm hotword -> decode lại QUERY trước khi refine (kéo theo --bias)")
    ap.add_argument("--heldout", action="store_true", help="chỉ chạy các sample held-out (số id %% 10 == 9, xem mine_hotwords.py)")
    ap.add_argument("--vad", action="store_true", help="cắt khoảng lặng bằng silero VAD trước Zipformer (vad.py)")
    ap.add_argument("--lm", help="RNNLM onnx cho LM fusion (train_rnnlm.py -> rnnlm/rnnlm.onnx)")
    ap.add_argument("--lm-scale", type=float, default=0.3)
    ap.add_argument("--beam", type=int, default=4, help="max_active_paths của modified_beam_search")
    ap.add_argument("--mark-only", action="store_true",
                    help="chỉ đánh dấu [từ|ứng viên...], không gọi LLM (duyệt ứng viên + chọn --conf-thr, không cần GPU); kéo theo --no-llm")
    ap.add_argument("--refine", action="store_true",
                    help="bật LLM refine câu ASR (mặc định TẮT: trên train, refine làm WER dạng đọc tăng 11.6%% -> 15.0%% "
                         "và đổi nghĩa số / thời gian / hành động / phủ định); tắt thì vẫn đánh dấu `marked`, chỉ không gọi LLM sửa")
    ap.add_argument("--refine-loose", action="store_true",
                    help="refine kiểu cũ: nhận/bỏ cả câu LLM trả về. Mặc định (chặt): xét TỪNG chỗ LLM sửa, chỉ nhận chỗ chạm từ bị "
                         "đánh dấu (conf thấp hoặc các bản decode khác nhau), đọc gần giống chữ cũ, không đổi số / phủ định "
                         "(trừ khi bản decode khác hoặc tên --prefetch có đúng cụm đó)")
    ap.add_argument("--refine-max-dist", type=float, default=1.0,
                    help="chặt: đổi 1-1 từng từ phải có vi_phon.dist <= ngần này (sang/xăng 0.45, la/lan 1.0, hai/ba 1.5)")
    ap.add_argument("--refine-free-dist", type=float, default=0.0,
                    help="chặt: cho sửa cả từ KHÔNG bị đánh dấu nếu đọc rất gần (dist <= ngần này, vd. 0.3: thúc/thuốc, "
                         "tốc/tóc); 0 = không cho")
    ap.add_argument("--alt-zipformer",
                    help="model thứ hai decode greedy cùng audio làm N-best cho refine (chỗ hai model nghe khác nhau được "
                         "đánh dấu): HF id / thư mục, vd. hynt/Zipformer-30M-RNNT-6000h; \"same\" = model chính, greedy, không LM. "
                         "Khi có --bias, bản decode không bias đã luôn được dùng làm N-best")
    ap.add_argument("--conf-thr", type=float, default=0.6, help="từ có confidence < ngưỡng thì đánh dấu cho LLM sửa")
    ap.add_argument("--cand-k", type=int, default=3, help="số ứng viên mỗi từ")
    ap.add_argument("--cand-train", default=str(Path(__file__).resolve().parent / "data" / "asr_train_plain.jsonl"),
                    help="jsonl có hyp + ref_spoken (asr/eval_asr.py --dump, bước A) để học bảng nhầm lẫn và từ vựng")
    ap.add_argument("--mapper", default=MAPPER_MODEL,
                    help='HF id or local folder, "none" = off, "same" = dùng chung model với --llm (không nạp model thứ hai)')
    ap.add_argument("--mapper-lora")
    ap.add_argument("--mapper-device", help="GPU index for the mapper, run in its own process (default: share the LLM's GPU)")
    ap.add_argument("--llm-device", help="GPU index cho LLM chính (planner/verifier/refine), chạy trong process riêng "
                                         "(mặc định: GPU mặc định, cùng process). Vd. --llm-device 0 --mapper-device 1")
    ap.add_argument("--llm", default=LLM_MODEL, help='HF id or local folder, "gold" = replay labels (train)')
    ap.add_argument("--llm-lora")
    ap.add_argument("--llm-backend", choices=["vllm", "hf"], default="vllm",
                    help="hf = transformers (chạy được trên CPU khi không có GPU/vLLM; chậm, xử lý từng câu)")
    ap.add_argument("--gpu-mem", type=float, default=0.7, help="main LLM gpu_memory_utilization")
    ap.add_argument("--max-len", type=int, default=MAX_CTX,
                    help="max_model_len của LLM chính; history cũ bị cắt cho vừa. GPU thiếu bộ nhớ KV cache -> giảm (vd. 8192)")
    ap.add_argument("--mapper-gpu-mem", type=float, default=0.1, help="mapper gpu_memory_utilization")
    ap.add_argument("--eager", action="store_true", help="vLLM enforce_eager: skip CUDA graphs (slower, avoids capture errors)")
    ap.add_argument("--verify-rounds", type=int, default=VERIFY_ROUNDS)
    ap.add_argument("--text", action="store_true", help="use audio_text instead of ASR (train only)")
    ap.add_argument("--no-llm", action="store_true",
                    help="chỉ chạy phần xử lý âm thanh (VAD/ASR -> refine), không mapper/planner/tool; "
                         "ghi text từng bước ra --out (mặc định asr_out.jsonl), split có nhãn -> in WER")
    ap.add_argument("--ids", help="chỉ chạy các sample này: \"train_0001,train_0003\" hoặc file txt mỗi dòng 1 id "
                                  "(vd. data/asr_sai_nghia_ids.txt)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.mark_only:  # chỉ đánh dấu: không LLM, không planner
        a.no_llm = True
    if a.prefetch:
        if a.mark_only or a.text or a.llm == "gold":
            sys.exit("--prefetch cần LLM thật + ASR: không dùng với --mark-only / --text / --llm gold")
        a.bias = True  # hotword decode lại cần contextual graph của --bias
    if a.no_llm and a.out == "submission.jsonl":  # đừng ghi đè file nộp bằng output không có tool call
        a.out = "asr_out.jsonl"
    gold = a.llm == "gold"
    rounds = 0 if gold else a.verify_rounds

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # in-process engine: reproducible, freeable
    random.seed(a.seed)
    try:
        import numpy as np, torch
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)
        torch.cuda.manual_seed_all(a.seed)
    except ImportError:
        print("There is some problems when setting seed")
        pass

    LLMClass = HFLLM if a.llm_backend == "hf" else VLLM

    def make_llm(device, *args):  # device: GPU riêng cho LLM này (process con), None = GPU mặc định, cùng process
        if device is None:
            return LLMClass(*args)
        dev = str(device).lower().removeprefix("cuda:").removeprefix("cuda")  # "cuda:1" / "1" -> "1" (CUDA_VISIBLE_DEVICES)
        if not re.fullmatch(r"\d+(,\d+)*", dev):
            sys.exit(f"--llm-device / --mapper-device: cần số GPU (0, 1 hoặc cuda:1), nhận {device!r}")
        return RemoteVLLM(dev, *args, backend=a.llm_backend)
    root = Path(a.root)
    if not a.no_llm or a.prefetch:
        # zaic_tools (BTC) có thể chỉ nằm trong train/: thử --tools-dir, root/split, root/train, root
        dirs = [Path(a.tools_dir)] if a.tools_dir else [root / a.split, root / "train", root]
        found = next((d for d in dirs if (d / "zaic_tools").is_dir() or (d / "zaic_tools.py").is_file()), None)
        if found is None:
            sys.exit(f"không tìm thấy zaic_tools trong {[str(d) for d in dirs]}; chỉ đường bằng --tools-dir")
        sys.path.insert(0, str(found.resolve()))
        from zaic_tools import ToolEnv
    catalog = {t["name"]: t for app in json.load(open(root / a.split / f"{a.split}_tools_catalog.json", encoding="utf-8"))["apps"]
               for t in app["tools"]}
    items = [json.loads(l) for l in open(root / a.split / f"{a.split}.jsonl", encoding="utf-8") if l.strip()]
    if a.heldout:
        from hotwords.mine_hotwords import is_heldout
        items = [it for it in items if is_heldout(it["id"])]
    if a.ids:  # "train_0001,train_0003" hoặc file txt mỗi dòng 1 id (id lượt "train_0008_u2" -> lấy sample train_0008)
        want = Path(a.ids).read_text(encoding="utf-8").split() if Path(a.ids).is_file() else a.ids.split(",")
        want = {re.sub(r"_u\d+$", "", w.strip()) for w in want if w.strip()}
        items = [it for it in items if it["id"] in want]
        print(f"--ids: {len(items)}/{len(want)} sample tìm thấy")
    items = items[:a.limit] if a.limit else items
    t_start = time.time()

    # all models stay resident (GPU split: ASR + mapper --mapper-gpu-mem + llm --gpu-mem); each sample runs end to end
    asr = cands = None
    if not a.text:
        from asr.asr_zipformer import ZipformerASR
        asr = ZipformerASR(a.zipformer, num_threads=a.zipformer_threads, bias=a.bias, hotwords_score=a.hotword_score,
                           hi=a.hotword_hi, lo=a.hotword_lo, glossary_score=a.glossary_score, beam=a.beam,
                           hotwords_file=a.hotwords_file, lm=a.lm, lm_scale=a.lm_scale, vad=a.vad)
        a.alt_asr = None
        if a.alt_zipformer:
            a.alt_asr = ZipformerASR(a.zipformer if a.alt_zipformer == "same" else a.alt_zipformer,
                                     num_threads=a.zipformer_threads, bias=False, vad=a.vad, decoding="greedy_search")
        if Path(a.cand_train).exists():
            cands = Candidates.from_train(a.cand_train)
        else:  # dự phòng: từ vựng từ nhãn train.jsonl, không có bảng nhầm lẫn ASR -> ứng viên chỉ từ gần âm
            refs = root / "train" / "train.jsonl"
            if not refs.exists():
                sys.exit(f"thiếu {a.cand_train} và {refs}: giải nén asr_assets.zip (bước E) hoặc chạy asr/eval_asr.py --dump")
            print(f"CẢNH BÁO: thiếu {a.cand_train} -> ứng viên chỉ lấy từ gần âm trong từ vựng {refs} (không có bảng nhầm "
                  f"lẫn ASR). Đầy đủ: asr/eval_asr.py --root DATA --subset all --configs beam --dump {a.cand_train}")
            cands = Candidates.from_refs(refs)
    if a.no_llm:
        if a.text:
            sys.exit("--no-llm / --mark-only cần ASR, không dùng với --text")
        # LLM ở chế độ audio-only chỉ cần cho --refine hoặc --prefetch
        llm = None if gold or a.mark_only or not (a.refine or a.prefetch) else make_llm(a.llm_device, a.llm, a.llm_lora, a.gpu_mem, a.seed,
                                                         a.max_len if a.prefetch else 4096, 256, {"enable_thinking": False}, a.eager)
        return run_audio_only(a, items, asr, t_start, llm, cands, catalog)
    m_args = (a.mapper, a.mapper_lora, a.mapper_gpu_mem, a.seed, 2048, 256, {"enable_thinking": False}, a.eager)
    llm = GoldLLM() if gold else make_llm(a.llm_device, a.llm, a.llm_lora, a.gpu_mem, a.seed, a.max_len, MAX_NEW, None, a.eager)
    # "same" (hoặc trùng --llm, không LoRA riêng): mapper dùng chung model với LLM chính -> chỉ nạp 1 model lên GPU
    share = a.mapper == "same" or (a.mapper == a.llm and a.mapper_lora == a.llm_lora)
    mapper = None if a.mapper == "none" or gold else llm if share else make_llm(a.mapper_device, *m_args)
    common = frozenset(common_words(root / a.split / f"{a.split}.jsonl")) if a.bias else frozenset()
    slow, shown_tb = [], False
    with open(a.out, "w", encoding="utf-8") as fout, open(a.out + ".trace.jsonl", "w", encoding="utf-8") as ftr:
        for i, item in enumerate(items):
            t0 = time.time()  # the time limit covers ASR and mapper of this sample too
            env = ToolEnv(a.split, item["id"], root=str(root))
            trace, raw, texts, keep, pools, pre_done = [], [], [], 0, None, []
            try:
                # 1. ASR of every user turn: history first, then the query, in one batch
                turns = [h for h in item["history"] if h["role"] == "user"] + [item]
                tools = tool_specs(item, catalog)
                pools = app_pools(item)  # snap_search_keys: tên trong tool call -> tên đã thấy trong history / kết quả tool
                if a.text:
                    raw = [u["audio_text"] for u in turns]
                else:  # ASR(+LM) -> [prefetch: LLM gọi tool đọc -> tên làm hotword -> decode lại QUERY] -> đánh dấu + LLM sửa
                    paths = [root / a.split / u["audio"] for u in turns]
                    wls, keeps, fell, names, pre_done, alts = asr_with_prefetch(a, item, asr, paths, None if gold else llm, tools,
                                                                          catalog, pools, trace, common, env)
                    rst = Counter()
                    raw, marked = refine(llm if a.refine and not gold else None, wls, cands, a.conf_thr, a.cand_k,
                                         keep=keeps, names=names, **refine_kw(a, alts, rst))
                    if a.refine:
                        trace.extend({"refine": {"marked": m, "to": t}} for m, t in zip(marked, raw) if "[" in m)
                        if rst:
                            trace.append({"refine_edits": dict(rst)})
                    trace.extend({"bias_fallback": k} for k, f in enumerate(fell) if f)
                # 2. dialect mapper (không ITN: số giữ dạng chữ, planner tự đổi sang tham số); output rỗng -> giữ câu cũ
                texts = [m or r for r, m in zip(raw, mapper.chat([mapper_conv(r) for r in raw], None, {"enable_thinking": False}))] if mapper else raw
                # 3-5. planner <-> verifier -> executor
                msgs, keep = fit_messages(item, texts, llm, tools, a.max_len)
                if pre_done:  # tool đọc của bước prefetch đã chạy trên env: planner thấy kết quả, đi tiếp từ đó
                    msgs.append(assistant_msg("", [{"name": n, "arguments": a} for n, a, _ in pre_done]))
                    msgs += [{"role": "tool", "content": clip(r)} for _, _, r in pre_done]
                if gold:
                    llm.queue = list(item["label"]["tool_calls"])
                run_agent(env, llm, msgs, tools, catalog, trace, rounds, pools=pools, pre_done=pre_done)
            except Exception as e:  # keep what already ran; a crash must still write the line
                trace.append({"error": f"{type(e).__name__}: {str(e)[:300]}", "traceback": traceback.format_exc()})
                if not shown_tb:  # full traceback of the first failure in the log; the rest are in the trace file
                    shown_tb = True
                    traceback.print_exc()
            calls = env.get_tool_calls()
            dt = time.time() - t0
            if dt > TIME_LIMIT_S:
                slow.append((item["id"], round(dt, 1)))
            fout.write(json.dumps({"id": item["id"], "tool_calls": calls, "state_hash": env.get_state_hash()},
                                  ensure_ascii=False) + "\n")
            ftr.write(json.dumps({"id": item["id"], "raw_texts": raw, "mapped_texts": texts,
                                  "kept_history": keep, "steps": trace, "seconds": round(dt, 2)}, ensure_ascii=False) + "\n")
            err = next((s["error"][:120] for s in trace if "error" in s), "")
            print(f"[{i + 1}/{len(items)}] {item['id']} calls={len(calls)} {dt:.1f}s {err}", flush=True)

    total = time.time() - t_start
    print(f"\n{len(items)} samples -> {a.out} | avg {total / max(len(items), 1):.1f}s/sample"
          f" | agent > {TIME_LIMIT_S}s: {len(slow)}")
    for sid, dt in slow:
        print(f"  {sid}: {dt}s")


if __name__ == "__main__":
    main()
