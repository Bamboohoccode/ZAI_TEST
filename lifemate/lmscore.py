"""Chấm điểm câu bằng log-prob của LLM (vLLM prompt_logprobs): dùng cho P1 (bảng nhầm lẫn + LLM chấm câu) và
P2 (N-best + LLM chấm lại). LLM không sinh chữ: chỉ đọc câu và trả tổng log-prob của các token của câu.

    sc = SentenceScorer("Qwen/Qwen3-4B-Instruct-2507")
    sc.score(["làm giúp mình năm mươi ngàn tiền điện thoại", "nạp giúp mình năm mươi ngàn tiền điện thoại"])
    -> [(logprob, n_token), ...]
"""

PREFIX = 'Người dùng nói với trợ lý ảo trên điện thoại: "'
SUFFIX = '"'


class SentenceScorer:
    def __init__(self, name, gpu_mem=0.85, max_len=1024, prefix=PREFIX, suffix=SUFFIX):
        import torch
        from vllm import LLM, SamplingParams
        kw = {}
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
            kw["dtype"] = "half"  # T4: không có bf16
        self.llm = LLM(name, max_model_len=max_len, gpu_memory_utilization=gpu_mem, enable_prefix_caching=True, seed=0, **kw)
        self.tok = self.llm.get_tokenizer()
        self.pre = self.tok.encode(prefix, add_special_tokens=False)
        self.suffix, self.max_len = suffix, max_len
        self.sp = SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=0)

    def score(self, sents, batch=512):
        """-> [(tổng log-prob của câu + dấu đóng ngoặc, số token)] theo thứ tự `sents`."""
        out = []
        for i in range(0, len(sents), batch):
            ids = [(self.pre + self.tok.encode(s + self.suffix, add_special_tokens=False))[:self.max_len - 1]
                   for s in sents[i:i + batch]]
            res = self.llm.generate([{"prompt_token_ids": x} for x in ids], self.sp, use_tqdm=False)
            for x, r in zip(ids, res):
                lps = r.prompt_logprobs
                tot = sum(lps[j][x[j]].logprob for j in range(len(self.pre), len(x)) if lps[j] and x[j] in lps[j])
                out.append((tot, len(x) - len(self.pre)))
        return out
