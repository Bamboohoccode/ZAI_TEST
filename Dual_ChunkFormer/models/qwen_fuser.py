"""Qwen wrapper: resolve [SLOT_k] either by generating JSON, or by scoring candidates (constrained choice)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.collator import build_messages, chat_text  # noqa: E402
from models.text_utils import fill_slots, parse_choices  # noqa: E402


class QwenFuser:
    def __init__(self, name, lora=None, device="cuda:0", max_new_tokens=128, length_norm=True, dtype=torch.bfloat16):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype if str(device).startswith("cuda") else torch.float32)
        if lora:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora).merge_and_unload()
        self.model = model.to(device).eval()
        self.device, self.max_new, self.length_norm = device, max_new_tokens, length_norm
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

    @torch.no_grad()
    def generate(self, sentences_slots):
        """[(sentence, slots)] -> raw answers (greedy, deterministic)."""
        texts = [chat_text(self.tok, build_messages(s, sl)) for s, sl in sentences_slots]
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        out = self.model.generate(**enc, max_new_tokens=self.max_new, do_sample=False, pad_token_id=self.tok.pad_token_id)
        return [self.tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for o in out]

    @torch.no_grad()
    def _cand_scores(self, prefix, cands):
        """log P(candidate + closing quote | prefix) for every candidate, one batch."""
        p = self.tok(prefix, add_special_tokens=False)["input_ids"]
        cs = [self.tok(c + '"', add_special_tokens=False)["input_ids"] for c in cands]
        n = max(len(c) for c in cs)
        pad = self.tok.pad_token_id
        ids = torch.tensor([p + c + [pad] * (n - len(c)) for c in cs], device=self.device)
        mask = torch.tensor([[1] * (len(p) + len(c)) + [0] * (n - len(c)) for c in cs], device=self.device)
        logp = torch.log_softmax(self.model(input_ids=ids, attention_mask=mask).logits.float(), dim=-1)
        scores = []
        for b, c in enumerate(cs):
            tgt = torch.tensor(c, device=self.device)
            lp = logp[b, len(p) - 1:len(p) - 1 + len(c)].gather(1, tgt[:, None]).sum()
            scores.append(float(lp) / (len(c) if self.length_norm else 1))
        return scores

    def score(self, sentence, slots):
        """Constrained decoding: per slot, pick the candidate (vi + en) with the best log-prob, left to right."""
        prefix = chat_text(self.tok, build_messages(sentence, slots)) + "{"
        choices = {}
        for i, s in enumerate(slots):
            cands = list(dict.fromkeys(s["vi"] + s["en"]))
            prefix += ("" if i == 0 else ", ") + f'"SLOT_{s["id"]}": "'
            sc = self._cand_scores(prefix, cands)
            best = cands[sc.index(max(sc))]
            choices[s["id"]] = best
            prefix += best + '"'
        return choices

    def resolve(self, sentence, slots, mode="generate"):
        """-> (final transcript, {slot_id: text}). No slots: the sentence is returned unchanged."""
        if not slots:
            return sentence, {}
        choices = self.score(sentence, slots) if mode == "score" else parse_choices(self.generate([(sentence, slots)])[0])
        return fill_slots(sentence, slots, choices), choices
