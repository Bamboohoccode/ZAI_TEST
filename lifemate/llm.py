"""Backend LLM cùng một giao diện chat(convs, tools, chat_kwargs) / n_tokens(msgs, tools):
VLLM (GPU), HFLLM (transformers, chạy được trên CPU), RemoteVLLM (process con trên một GPU riêng), GoldLLM (phát lại nhãn)."""
import gc
import json
import os
import re
import sys
import traceback
from pathlib import Path


class VLLM:
    def __init__(self, name, lora, gpu_mem, seed, max_len, max_new, chat_kwargs=None, eager=False):
        from vllm import LLM, SamplingParams
        kw, self.lora = {}, None
        if lora:
            from vllm.lora.request import LoRARequest
            r = json.load(open(Path(lora) / "adapter_config.json"))["r"]
            kw = {"enable_lora": True, "max_lora_rank": next(x for x in (8, 16, 32, 64, 128, 256, 512) if x >= r)}
            self.lora = LoRARequest("ft", 1, str(lora))
        import torch
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
            kw["dtype"] = "half"  # T4 / V100 không có bf16 (Qwen3 mặc định bf16) -> vLLM từ chối nếu để auto
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
        """chat_kwargs: ghi đè chat_template_kwargs cho lần gọi này (mapper / prefetch tắt thinking dù planner bật)."""
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
        """Cache KV phần đầu chung (mọi message trừ message user cuối): các lần gọi dùng chung system prompt,
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


def make_llm(device, backend, *args):
    """device: GPU riêng cho LLM này (process con, "1" / "cuda:1"); None = GPU mặc định, cùng process.
    args: (name, lora, gpu_mem, seed, max_len, max_new, chat_kwargs, eager)."""
    if device is None:
        return (HFLLM if backend == "hf" else VLLM)(*args)
    dev = str(device).lower().removeprefix("cuda:").removeprefix("cuda")  # "cuda:1" / "1" -> "1" (CUDA_VISIBLE_DEVICES)
    if not re.fullmatch(r"\d+(,\d+)*", dev):
        sys.exit(f"--llm-device / --mapper-device: cần số GPU (0, 1 hoặc cuda:1), nhận {device!r}")
    return RemoteVLLM(dev, *args, backend=backend)
