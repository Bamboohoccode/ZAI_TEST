"""End-to-end: audio -> Dual-Chunkformer -> uncertain spans + aligned Vi/En candidates -> Qwen -> transcript.

python infer_pipeline.py --audio a.wav b.wav
python infer_pipeline.py --manifest dev.jsonl --out pred.jsonl --lora ckpt/sft_lora [--mode score] [--no-llm]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import load_config  # noqa: E402
from models.dual_chunkformer import DualChunkformer  # noqa: E402
from models.text_utils import wer  # noqa: E402
from models.time_aligner import TimeAligner  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent / "configs/pipeline_config.yaml"))
    ap.add_argument("--audio", nargs="*", default=[])
    ap.add_argument("--manifest", help="jsonl with audio (+ optional text -> WER is reported)")
    ap.add_argument("--out", default="pred.jsonl")
    ap.add_argument("--lora", help="override models.qwen_lora")
    ap.add_argument("--mode", choices=["generate", "score"], help="override qwen.mode")
    ap.add_argument("--no-llm", action="store_true", help="stop after the Vi base sentence + slots (no Qwen)")
    a = ap.parse_args()

    cfg = load_config(a.config)
    items = [{"audio": p} for p in a.audio]
    if a.manifest:
        items += [json.loads(l) for l in open(a.manifest, encoding="utf-8") if l.strip()]
    if not items:
        ap.error("give --audio or --manifest")

    dual = DualChunkformer(cfg)
    aligner = TimeAligner(cfg, dual.vi, dual.en)
    fuser = None
    if not a.no_llm:
        from models.qwen_fuser import QwenFuser
        q = cfg["qwen"]
        fuser = QwenFuser(cfg["models"]["qwen"], a.lora or cfg["models"].get("qwen_lora"), cfg["models"]["qwen_device"],
                          q["max_new_tokens"], q["score_length_norm"])
    mode = a.mode or cfg["qwen"]["mode"]

    w_base = w_final = n_ref = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for i, it in enumerate(items):
            al = aligner.build(dual.run(it["audio"]))
            final, choices = (al.base_text, {}) if fuser is None else fuser.resolve(al.sentence, [s.to_dict() for s in al.slots], mode)
            rec = {"audio": it["audio"], "base": al.base_text, "sentence": al.sentence,
                   "slots": [s.to_dict() for s in al.slots], "choices": choices, "final": final}
            if it.get("text"):
                rec["reference"] = it["text"]
                rec["wer_base"], rec["wer_final"] = wer(al.base_text, it["text"]), wer(final, it["text"])
                w_base += rec["wer_base"]
                w_final += rec["wer_final"]
                n_ref += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[{i + 1}/{len(items)}] slots={len(al.slots)} | {al.sentence}\n    -> {final}", flush=True)
    if n_ref:
        print(f"\nmean WER (per utterance): base {w_base / n_ref:.4f} -> final {w_final / n_ref:.4f}  ({n_ref} utts)")
    print("->", a.out)


if __name__ == "__main__":
    main()
