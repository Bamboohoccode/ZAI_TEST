"""Stage 1: LoRA SFT of Qwen to resolve slots as JSON ({"SLOT_1": "...", ...}).

python train/train_sft_lora.py --train data/train_slots.jsonl [--valid data/dev_slots.jsonl] --out ckpt/sft_lora
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.collator import SFTCollator  # noqa: E402
from models import load_config  # noqa: E402


def read(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main():
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/pipeline_config.yaml"))
    ap.add_argument("--base", help="override models.qwen")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = load_config(a.config)
    s = cfg["sft"]
    name = a.base or cfg["models"]["qwen"]

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(r=s["lora_r"], lora_alpha=s["lora_alpha"], lora_dropout=s["lora_dropout"],
                                             target_modules=s["target_modules"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    class DS(torch.utils.data.Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            return self.rows[i]

    train, valid = DS(read(a.train)), DS(read(a.valid)) if a.valid else None
    args = TrainingArguments(
        output_dir=a.out, num_train_epochs=s["epochs"], per_device_train_batch_size=s["batch_size"],
        per_device_eval_batch_size=s["batch_size"], gradient_accumulation_steps=s["grad_accum"], learning_rate=s["lr"],
        lr_scheduler_type="cosine", warmup_ratio=0.03, bf16=True, logging_steps=20, save_strategy="epoch",
        save_total_limit=1, eval_strategy="epoch" if valid else "no", remove_unused_columns=False, seed=a.seed,
        report_to="none")
    Trainer(model=model, args=args, train_dataset=train, eval_dataset=valid,
            data_collator=SFTCollator(tok, s["max_len"])).train()
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print("LoRA adapter ->", a.out)


if __name__ == "__main__":
    main()
