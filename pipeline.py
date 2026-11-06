"""Voice agent: ChunkFormer ASR -> Qwen3-0.6B dialect mapper -> Qwen3-4B planner <-> verifier -> zaic_tools.

pip install -U vllm chunkformer

python pipeline.py --root DATA --split public_test                                   # full pipeline, 1 GPU
python pipeline.py --root DATA --split train --text --limit 100                      # skip ASR, use audio_text
python pipeline.py --root DATA --split public_test --asr-ckpt ft/asr.pt --llm ./llm_ft --mapper-lora ./mapper_lora
python pipeline.py --root DATA --split train --text --llm gold --mapper none --limit 50   # replay labels, no GPU
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
from datetime import datetime, timedelta
from pathlib import Path

ASR_MODEL = "khanhld/chunkformer-ctc-large-vie"
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

    def chat(self, convs, tools=None):
        outs = self.llm.chat(convs, self.sp, tools=tools, chat_template_kwargs=self.kw,
                             lora_request=self.lora, use_tqdm=False)
        return [o.outputs[0].text.strip() for o in outs]

    def close(self):
        import torch
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
        del self.llm
        gc.collect()
        cleanup_dist_env_and_memory()
        torch.cuda.empty_cache()


def _vllm_worker(conn, device, args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)  # before torch/vllm are imported in this fresh process
    try:
        llm = VLLM(*args)
        conn.send("ready")
        while True:
            conn.send(llm.chat(conn.recv()))
    except Exception as e:
        conn.send(e)


class RemoteVLLM:
    """VLLM in a child process pinned to one GPU, so two engines can sit on different cards."""

    def __init__(self, device, *args):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        ctx.Process(target=_vllm_worker, args=(child, device, args), daemon=True).start()
        self._recv()  # "ready", or the engine's startup error

    def _recv(self):
        r = self.conn.recv()
        if isinstance(r, Exception):
            raise r
        return r

    def chat(self, convs, tools=None):
        self.conn.send(convs)
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


def load_asr(name, ckpt, device):
    import torch
    from chunkformer import ChunkFormerModel
    model = ChunkFormerModel.from_pretrained(name)  # architecture, vocab, CMVN
    if ckpt:  # fine-tuned weights on top of `name`; strict so a key mismatch fails loudly
        sd = torch.load(ckpt, map_location="cpu")
        sd = sd.get("model", sd.get("state_dict", sd))
        model.model.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()}, strict=True)
    return model.to(device).eval()


def load_itn(folder_path, device="cpu", code_dir=None):
    """Load the bert2BERT inverse-text-normalization model trained by ITN/main.py (checkpoint folder with
    config.json + model.safetensors). Returns itn(text) -> text, e.g. 'ngày chín tháng một' -> 'ngày 9/1'.
    code_dir: the ITN folder holding model.py and model_config.py (default: ITN/ next to this script).
    Same tag-then-generate decoding as ITN/infer.py."""
    import torch
    from safetensors.torch import load_file
    from transformers import GenerationConfig
    itn_dir = Path(code_dir) if code_dir else Path(__file__).resolve().parent / "ITN"
    if not ((itn_dir / "model.py").exists() and (itn_dir / "model_config.py").exists()):
        raise FileNotFoundError(f"ITN code (model.py, model_config.py) not found in {itn_dir}; pass --itn-code")
    sys.path.insert(0, str(itn_dir))  # ITN modules import each other as top-level names (model, model_config)
    try:
        import model as itn_model  # makes ./cache in the cwd
        from model_config import InvertTextNormalizationConfig
    finally:
        sys.path.remove(str(itn_dir))
    tokenizer = itn_model.init_tokenizer()
    # Not from_pretrained: transformers 5.x crashes in mark_tied_weights_as_initialized for the custom decoder.
    net = itn_model.InvertTextNormalization(InvertTextNormalizationConfig.from_pretrained(folder_path))
    net.load_state_dict(load_file(os.path.join(folder_path, "model.safetensors")), strict=False)
    net.tie_weights()  # lm_head is tied to the embeddings and absent from the checkpoint
    try:
        net.generation_config = GenerationConfig.from_pretrained(folder_path)
    except OSError:
        net.generation_config.max_length = 200
        net.generation_config.no_repeat_ngram_size = 3
    net = net.to(device).eval()

    def enc(s):
        e = tokenizer([s])
        return {"input_ids": torch.tensor(e["input_ids"], device=device),
                "attention_mask": torch.tensor(e["attention_mask"], device=device)}

    @torch.no_grad()
    def itn(text):
        words = text.split()
        if not words:
            return text
        tags = net.spoken_tagging_classifier(net.dropout(net.encoder(**enc(text))[0]))
        tags = torch.argmax(tags, dim=-1).tolist()[0][1:-1]  # drop <bos>/<eos> labels
        out, span = [], ""

        def flush():
            if span:
                gen = net.generate(**enc(span), num_beams=1, num_return_sequences=1)
                out.append(tokenizer.decode(gen[0].tolist(), skip_special_tokens=True))

        for i, w in enumerate(words):
            tag = tags[i] if i < len(tags) else 0  # 0 = keep as is, 1 = starts a spoken span, else continues it
            if tag in (0, 1):
                flush()
                span = w if tag == 1 else ""
                if tag == 0:
                    out.append(w)
            else:
                span = f"{span} {w}".strip()
        flush()
        return " ".join(out)

    return itn


def run_asr(model, paths):
    """CTC greedy decode of one sample's files (history turns + query) in one batched call (deterministic)."""
    import torch
    with torch.no_grad():
        texts = model.batch_decode(audio_paths=[str(p) for p in paths], chunk_size=64, left_context_size=128,
                                   right_context_size=128, total_batch_duration=1800)
    return [t.strip() for t in texts]

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


def run_agent(env, llm, msgs, tools, catalog, trace, rounds=VERIFY_ROUNDS, deadline=float("inf")):
    done = set()
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
                stop |= c["name"] == "sys.ask_missing_info"
            msgs.append({"role": "tool", "content": clip(r)})
        if stop:
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder holding the split folders")
    ap.add_argument("--split", default="public_test")
    ap.add_argument("--out", default="submission.jsonl")
    ap.add_argument("--asr", default=ASR_MODEL, help="ChunkFormer HF id or local (fine-tuned) folder")
    ap.add_argument("--asr-ckpt", help="fine-tuned ChunkFormer .pt loaded onto --asr")
    ap.add_argument("--asr-device", default="cuda:0")
    ap.add_argument("--itn", help="folder of the ITN checkpoint trained by ITN/main.py; runs on ASR text before the mapper")
    ap.add_argument("--itn-code", help="ITN folder with model.py + model_config.py (default: ITN/ next to pipeline.py)")
    ap.add_argument("--itn-device", default="cuda:0")
    ap.add_argument("--mapper", default=MAPPER_MODEL, help='HF id or local folder, "none" = off')
    ap.add_argument("--mapper-lora")
    ap.add_argument("--mapper-device", help="GPU index for the mapper, run in its own process (default: share the LLM's GPU)")
    ap.add_argument("--llm", default=LLM_MODEL, help='HF id or local folder, "gold" = replay labels (train)')
    ap.add_argument("--llm-lora")
    ap.add_argument("--gpu-mem", type=float, default=0.7, help="main LLM gpu_memory_utilization")
    ap.add_argument("--mapper-gpu-mem", type=float, default=0.1, help="mapper gpu_memory_utilization")
    ap.add_argument("--eager", action="store_true", help="vLLM enforce_eager: skip CUDA graphs (slower, avoids capture errors)")
    ap.add_argument("--verify-rounds", type=int, default=VERIFY_ROUNDS)
    ap.add_argument("--text", action="store_true", help="use audio_text instead of ASR (train only)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
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
        print(f"There is some problems when setting seed")
        pass

    root = Path(a.root)
    sys.path.insert(0, str(root / a.split))
    from zaic_tools import ToolEnv
    catalog = {t["name"]: t for app in json.load(open(root / a.split / f"{a.split}_tools_catalog.json", encoding="utf-8"))["apps"]
               for t in app["tools"]}
    items = [json.loads(l) for l in open(root / a.split / f"{a.split}.jsonl", encoding="utf-8") if l.strip()]
    items = items[:a.limit] if a.limit else items
    t_start = time.time()

    # all models stay resident (GPU split: ASR + mapper --mapper-gpu-mem + llm --gpu-mem); each sample runs end to end
    asr = None if a.text else load_asr(a.asr, a.asr_ckpt, a.asr_device)
    itn = load_itn(a.itn, a.itn_device, a.itn_code) if a.itn else None
    m_args = (a.mapper, a.mapper_lora, a.mapper_gpu_mem, a.seed, 2048, 256, {"enable_thinking": False}, a.eager)
    mapper = None if a.mapper == "none" or gold else RemoteVLLM(a.mapper_device, *m_args) if a.mapper_device else VLLM(*m_args)
    llm = GoldLLM() if gold else VLLM(a.llm, a.llm_lora, a.gpu_mem, a.seed, MAX_CTX, MAX_NEW, None, a.eager)
    slow, shown_tb = [], False
    with open(a.out, "w", encoding="utf-8") as fout, open(a.out + ".trace.jsonl", "w", encoding="utf-8") as ftr:
        for i, item in enumerate(items):
            t0 = time.time()  # the time limit covers ASR and mapper of this sample too
            env = ToolEnv(a.split, item["id"], root=str(root))
            trace, raw, texts, keep = [], [], [], 0
            try:
                # 1. ASR of every user turn: history first, then the query, in one batch
                turns = [h for h in item["history"] if h["role"] == "user"] + [item]
                raw = [u["audio_text"] for u in turns] if a.text else run_asr(asr, [root / a.split / u["audio"] for u in turns])
                # 2. ITN (spoken -> written numbers/dates), then dialect mapper; empty output keeps the previous text
                itn_out = [itn(r) or r for r in raw] if itn else raw
                texts = [m or r for r, m in zip(itn_out, mapper.chat([mapper_conv(r) for r in itn_out]))] if mapper else itn_out
                # 3-5. planner <-> verifier -> executor
                tools = tool_specs(item, catalog)
                keep = len(texts) - 1
                msgs = build_messages(item, texts, keep)
                while keep and llm.n_tokens(msgs, tools) > MAX_CTX - MAX_NEW * 2:
                    keep -= 1
                    msgs = build_messages(item, texts, keep)
                if gold:
                    llm.queue = list(item["label"]["tool_calls"])
                run_agent(env, llm, msgs, tools, catalog, trace, rounds)
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
