"""Stage 2: GRPO on top of the SFT LoRA, with three rewards:
  format               answer is a JSON that covers every slot
  wer                  1 - WER(filled transcript, reference)
  anti_overcorrection  penalty for every slot whose Vi base was already right but the model changed it

python train/train_grpo.py --train data/train_slots.jsonl --init-lora ckpt/sft_lora --out ckpt/grpo_lora
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.collator import build_messages  # noqa: E402
from models import load_config  # noqa: E402
from models.text_utils import fill_slots, parse_choices, wer  # noqa: E402


def _text(c):
    return c[0]["content"] if isinstance(c, list) else c


def reward_format(completions, slots_json, **kw):
    out = []
    for c, sj in zip(completions, slots_json):
        got, need = parse_choices(_text(c)), {s["id"] for s in json.loads(sj)}
        out.append(1.0 if need and need <= set(got) else 0.0)
    return out


def reward_wer(completions, sentence, slots_json, reference, **kw):
    out = []
    for c, st, sj, ref in zip(completions, sentence, slots_json, reference):
        final = fill_slots(st, json.loads(sj), parse_choices(_text(c)))
        out.append(1.0 - min(1.0, wer(final, ref)))
    return out


def reward_anti_overcorrection(completions, slots_json, target_json, **kw):
    """-(fraction of slots changed although the Vi base already equals the gold) in [-1, 0]."""
    out = []
    for c, sj, tj in zip(completions, slots_json, target_json):
        slots, gold, got = json.loads(sj), json.loads(tj), parse_choices(_text(c))
        bad = sum(1 for s in slots if gold.get(str(s["id"])) == s["base"] and got.get(s["id"], s["base"]) != s["base"])
        out.append(-bad / max(len(slots), 1))
    return out


def main():
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init-lora", help="SFT adapter to continue from (recommended)")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/pipeline_config.yaml"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = load_config(a.config)
    g, name = cfg["grpo"], cfg["models"]["qwen"]

    rows = [json.loads(l) for l in open(a.train, encoding="utf-8") if l.strip()]
    ds = Dataset.from_list([{
        "prompt": build_messages(r["sentence"], r["slots"]), "sentence": r["sentence"],
        "slots_json": json.dumps(r["slots"], ensure_ascii=False), "target_json": json.dumps(r["target"], ensure_ascii=False),
        "reference": r["reference"]} for r in rows])

    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16)
    if a.init_lora:
        model = PeftModel.from_pretrained(model, a.init_lora, is_trainable=True)
    w = g["reward_weights"]
    args = GRPOConfig(
        output_dir=a.out, learning_rate=g["lr"], beta=g["beta"], num_generations=g["num_generations"],
        max_completion_length=g["max_completion_length"], per_device_train_batch_size=g["batch_size"],
        gradient_accumulation_steps=g["grad_accum"], num_train_epochs=g["epochs"], bf16=True, logging_steps=10,
        save_strategy="epoch", save_total_limit=1, seed=a.seed, report_to="none",
        reward_weights=[w["format"], w["wer"], w["anti_overcorrection"]])
    trainer = GRPOTrainer(model=model, processing_class=tok, args=args, train_dataset=ds,
                          reward_funcs=[reward_format, reward_wer, reward_anti_overcorrection],
                          **({} if a.init_lora else {"peft_config": __import__("peft").LoraConfig(
                              r=cfg["sft"]["lora_r"], lora_alpha=cfg["sft"]["lora_alpha"],
                              target_modules=cfg["sft"]["target_modules"], task_type="CAUSAL_LM")}))
    trainer.train()
    trainer.save_model(a.out)
    print("LoRA adapter ->", a.out)


if __name__ == "__main__":
    main()
