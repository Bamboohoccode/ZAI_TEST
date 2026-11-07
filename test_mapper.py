"""Test the dialect mapper (Qwen3-0.6B, optionally + the LoRA from train_mapper.py).

python test_mapper.py                                  # built-in cases, base model
python test_mapper.py --lora mapper_lora               # built-in cases, fine-tuned adapter
python test_mapper.py --lora mapper_lora --data        # + ViDia2Std test split: exact match, WER, over-correction
python test_mapper.py --lora mapper_lora --file my.txt # map every line of a text file, print before -> after
python test_mapper.py --text "tui đang ở mô rứa"       # one sentence
python test_mapper.py --lora mapper_lora --shots       # prompt WITH the 6 few-shots (what an adapter trained with shots expects)

Uses pipeline.mapper_conv (system prompt + the sentence), thinking off, greedy: the same input the pipeline builds.
Works on CPU (slow) or GPU. Exits with status 1 if a built-in case fails.
"""
import argparse
import re
import sys

import torch

from pipeline import MAPPER_MODEL, mapper_conv

# Few-shot examples of the earlier prompt version; some adapters were trained with them in the prompt (--shots).
SHOTS = [
    ("mai mấy giờ tàu ra huế rứa", "mai mấy giờ có tàu ra huế vậy"),
    ("đặt cho tui cái lịch họp chiều ni", "đặt cho tôi cái lịch họp chiều nay"),
    ("chừ mở cái quạt trong buồng ngủ đi", "bây giờ mở cái quạt trong phòng ngủ đi"),
    ("ổng nhắn gì cho tui hông", "ông ấy nhắn gì cho tôi không"),
    ("bật đèn phòng khách", "bật đèn phòng khách"),
    ("book xe về nhà lúc 5 giờ chiều", "book xe về nhà lúc 5 giờ chiều"),
]


def conv_with_shots(text):
    """pipeline.mapper_conv's system prompt + the 6 few-shots as chat turns."""
    c = mapper_conv(text)
    shots = [m for q, a in SHOTS for m in ({"role": "user", "content": q}, {"role": "assistant", "content": a})]
    return [c[0], *shots, c[-1]]


# (ASR-style input, expected output, tokens that must survive).
# `keep` tokens (numbers, English words, names) must appear unchanged in the output.
CASES = [
    ("tui đang ở mô rứa", "tôi đang ở đâu vậy", []),
    ("răng hôm ni trời nóng rứa", "sao hôm nay trời nóng vậy", []),
    ("má tui đi chợ chưa", "mẹ tôi đi chợ chưa", []),
    ("bữa nay có mưa hông", "hôm nay có mưa không", []),
    ("chuyển cho lan hai trăm nghìn vô tài khoản", "chuyển cho lan hai trăm nghìn vào tài khoản", ["lan", "hai trăm nghìn"]),
    # already standard: must be copied as is (over-correction check)
    ("đặt vé máy bay đi đà nẵng lúc 9 giờ sáng", "đặt vé máy bay đi đà nẵng lúc 9 giờ sáng", ["9", "đà nẵng"]),
    ("mở wifi và tắt ac phòng ngủ", "mở wifi và tắt ac phòng ngủ", ["wifi", "ac"]),
    ("nhắn tin cho sếp dũng là họp lúc 14 giờ 30", "nhắn tin cho sếp dũng là họp lúc 14 giờ 30", ["dũng", "14", "30"]),
]


def norm(s):
    """ASR-like text: lowercase, no punctuation (a '.' or ',' between digits stays: 2.5). Same as train_mapper.norm."""
    s = re.sub(r"(?<!\d)[^\w\s]|[^\w\s](?!\d)", " ", s.lower())
    return " ".join(s.split())


def wer(hyp, ref):
    h, r = hyp.split(), ref.split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(len(r), 1)


class Mapper:
    def __init__(self, model, lora, device, conv=mapper_conv):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model, padding_side="left")
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
        m = AutoModelForCausalLM.from_pretrained(model, torch_dtype=dtype)
        if lora:
            from peft import PeftModel
            m = PeftModel.from_pretrained(m, lora).merge_and_unload()
        self.model, self.device, self.conv = m.to(device).eval(), device, conv

    @torch.no_grad()
    def __call__(self, texts, bs=16):
        """Greedy mapping; an empty answer keeps the input (as pipeline.py does)."""
        out = []
        for i in range(0, len(texts), bs):
            chunk = texts[i:i + bs]
            prompts = [self.tok.apply_chat_template(self.conv(t), add_generation_prompt=True, tokenize=False,
                                                    enable_thinking=False) for t in chunk]
            enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            gen = self.model.generate(**enc, max_new_tokens=128, do_sample=False, pad_token_id=self.tok.pad_token_id)
            out += [self.tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() or t
                    for t, g in zip(chunk, gen)]
        return out


def run_cases(mapper):
    preds = mapper([c[0] for c in CASES])
    ok = 0
    print(f"{'':2}{'input':<46}{'output':<46}expected")
    for (src, exp, keep), pred in zip(CASES, preds):
        hit = norm(pred) == norm(exp)
        lost = [k for k in keep if k not in norm(pred)]
        ok += hit and not lost
        print(f"{'OK' if hit and not lost else 'XX':<2}{src:<46}{pred:<46}{'' if hit else exp}"
              + (f"   [lost: {lost}]" if lost else ""))
    print(f"\nbuilt-in cases: {ok}/{len(CASES)} pass")
    return ok == len(CASES)


def run_dataset(mapper, name, limit):
    from datasets import load_dataset
    try:
        test = load_dataset(name)["test"]
    except Exception:  # repo without a loading script: its csv files (same as train_mapper.py)
        test = load_dataset("csv", data_files=f"hf://datasets/{name}/test.csv")["train"]
    rows = [(norm(r["dialect"]), norm(r["standard"])) for r in test if r["dialect"] and r["standard"]]
    rows = [r for r in rows if r[0] and r[1]][:limit or None]
    preds = [norm(p) for p in mapper([r[0] for r in rows])]
    em = sum(p == g for p, (_, g) in zip(preds, rows))
    w = sum(wer(p, g) for p, (_, g) in zip(preds, rows)) / len(rows)
    base = sum(wer(s, g) for s, g in rows) / len(rows)  # copying the input unchanged
    same = [(s, p) for p, (s, g) in zip(preds, rows) if s == g]  # rows already standard
    over = sum(p != s for s, p in same)
    print(f"\n{name} test, {len(rows)} rows:")
    print(f"  exact match            {em / len(rows):.3f}")
    print(f"  WER vs gold            {w:.3f}   (copy-the-input baseline {base:.3f})")
    if same:
        print(f"  over-correction        {over}/{len(same)} = {over / len(same):.3f}  (already-standard rows that were changed)")
    for (s, g), p in [x for x in zip(rows, preds) if x[1] != x[0][1]][:5]:
        print(f"  miss  src : {s}\n        gold: {g}\n        pred: {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MAPPER_MODEL)
    ap.add_argument("--lora", help="adapter folder from train_mapper.py (mapper_lora)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--shots", action="store_true", help="put the 6 few-shots in the prompt (adapter trained with them)")
    ap.add_argument("--text", nargs="*", help="map these sentences and print them")
    ap.add_argument("--file", help="map every non-empty line of this file")
    ap.add_argument("--data", nargs="?", const="Biu3010/ViDia2Std", help="also score a dataset's test split")
    ap.add_argument("--limit", type=int, default=200, help="rows of the dataset test split (0 = all)")
    a = ap.parse_args()

    print(f"model {a.model} | lora {a.lora or 'none (base model)'} | {a.device} | prompt: "
          + ("system + 6 few-shots" if a.shots else "system only (pipeline.mapper_conv)"))
    mapper = Mapper(a.model, a.lora, a.device, conv_with_shots if a.shots else mapper_conv)
    passed = True
    if a.text or a.file:
        lines = (a.text or []) + ([l.strip() for l in open(a.file, encoding="utf-8") if l.strip()] if a.file else [])
        for s, p in zip(lines, mapper(lines)):
            print(f"{s}\n  -> {p}")
    else:
        passed = run_cases(mapper)
    if a.data:
        run_dataset(mapper, a.data, a.limit)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
