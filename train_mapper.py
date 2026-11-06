"""LoRA fine-tune Qwen3-0.6B as the dialect mapper on Biu3010/ViDia2Std (dialect -> standard Vietnamese).

pip install -U transformers peft datasets accelerate

python train_mapper.py                          # ~3 epochs, saves ./mapper_lora
python train_mapper.py --epochs 1 --limit 2000  # quick smoke test
then: python pipeline.py ... --mapper-lora ./mapper_lora

Training uses pipeline.mapper_conv (same system prompt + chat template, thinking off) so train and inference
see identical prompts. Text is lowercased and stripped of punctuation because the mapper's input is raw ASR output.
"""
import argparse
import re

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from pipeline import MAPPER_MODEL, mapper_conv


def norm(s):
    """ASR-like text: lowercase, no punctuation (a '.' or ',' between digits stays: 2.5)."""
    s = re.sub(r"(?<!\d)[^\w\s]|[^\w\s](?!\d)", " ", s.lower())
    return " ".join(s.split())


assert norm("Giá 2.5 triệu, nhé!") == "giá 2.5 triệu nhé"


def encode(tok, row, max_len):
    prompt = tok.apply_chat_template(mapper_conv(row["src"]), add_generation_prompt=True, tokenize=False,
                                     enable_thinking=False)
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    t = tok(row["tgt"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    ids = (p + t)[:max_len]
    return {"input_ids": ids, "labels": ([-100] * len(p) + t)[:max_len]}  # loss on the answer only


def collate(tok):
    def fn(batch):
        n = max(len(b["input_ids"]) for b in batch)
        pad = lambda x, v: x + [v] * (n - len(x))
        return {"input_ids": torch.tensor([pad(b["input_ids"], tok.pad_token_id) for b in batch]),
                "attention_mask": torch.tensor([pad([1] * len(b["input_ids"]), 0) for b in batch]),
                "labels": torch.tensor([pad(b["labels"], -100) for b in batch])}
    return fn


@torch.no_grad()
def evaluate(model, tok, rows, bs=16):
    """Greedy exact match on normalized text; prints a few misses."""
    model.eval()
    tok.padding_side = "left"
    hit, misses = 0, []
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        prompts = [tok.apply_chat_template(mapper_conv(r["src"]), add_generation_prompt=True, tokenize=False,
                                           enable_thinking=False) for r in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        out = model.generate(**enc, max_new_tokens=128, do_sample=False, pad_token_id=tok.pad_token_id)
        for r, o in zip(chunk, out):
            pred = norm(tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True))
            if pred == r["tgt"]:
                hit += 1
            elif len(misses) < 5:
                misses.append((r["src"], r["tgt"], pred))
    print(f"exact match {hit}/{len(rows)} = {hit / len(rows):.3f}")
    for m in misses:
        print("  src :", m[0], "\n  gold:", m[1], "\n  pred:", m[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MAPPER_MODEL)
    ap.add_argument("--data", default="Biu3010/ViDia2Std")
    ap.add_argument("--out", default="mapper_lora")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=768, help="prompt without shots is ~100 tokens")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N train rows (smoke test)")
    ap.add_argument("--eval", type=int, default=200, help="test rows for the final exact-match check, 0 = skip")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ds = load_dataset(a.data)
    prep = lambda split, n=0: [{"src": norm(r["dialect"]), "tgt": norm(r["standard"])}
                               for r in (ds[split].select(range(n)) if n else ds[split])
                               if r["dialect"] and r["standard"] and norm(r["dialect"]) and norm(r["standard"])]
    train, dev, test = prep("train", a.limit), prep("validation", 500), prep("test", a.eval)

    tok = AutoTokenizer.from_pretrained(a.model)
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()  # T4 has no bf16: fp32 weights + fp16 autocast
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16 if bf16 else torch.float32)
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                             "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    enc = lambda rows: [encode(tok, r, a.max_len) for r in rows]
    tr, dv = enc(train), enc(dev)
    # labels must start exactly where the prompt ends, or the answer is trained on the wrong tokens
    assert tr[0]["labels"][next(i for i, x in enumerate(tr[0]["labels"]) if x != -100) - 1] == -100
    assert tok.decode([x for x in tr[0]["labels"] if x != -100]) == train[0]["tgt"] + "<|im_end|>"

    args = TrainingArguments(output_dir=a.out + "_ckpt", num_train_epochs=a.epochs, learning_rate=a.lr,
                             per_device_train_batch_size=a.bs, per_device_eval_batch_size=a.bs, warmup_ratio=0.03,
                             lr_scheduler_type="cosine", logging_steps=50, eval_strategy="epoch", save_strategy="no",
                             bf16=bf16, fp16=torch.cuda.is_available() and not bf16, seed=a.seed, report_to="none",
                             remove_unused_columns=False, group_by_length=False)
    Trainer(model=model, args=args, train_dataset=tr, eval_dataset=dv, data_collator=collate(tok)).train()

    model.save_pretrained(a.out)  # adapter_config.json + adapter weights, loaded by pipeline.py --mapper-lora
    print("saved ->", a.out)
    if a.eval:
        evaluate(model, tok, test)


if __name__ == "__main__":
    main()
