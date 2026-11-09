"""Seq2seq (ViT5 / BARTpho) map transcript ASR -> câu reference (audio_text): sửa lỗi nhận dạng + ITN + dấu câu.

Bước 1 — tạo cặp (hyp, ref) bằng cách chạy Zipformer trên mọi lượt user của train (cache ra jsonl):
    python mapper/seq2seq_mapper.py hyps --root ~/Downloads/train+public_test/train+public_test --out data/asr_train.jsonl --bias
    # --bias = hotword + snap_text + asr_fix giống pipeline.py --bias; bỏ đi để lấy greedy thô.
    # LƯU Ý: nếu Zipformer đã fine-tune trên chính train thì hyp sẽ "sạch" bất thường -> mapper học được ít.
    #        Dùng model gốc (mặc định) hoặc model fine-tune theo k-fold để hyp giống lỗi thật trên test.

Bước 2 — fine-tune (nên chạy GPU: Colab/Kaggle T4 ~10 phút cho vit5-base, 5k câu):
    python mapper/seq2seq_mapper.py train --data data/asr_train.jsonl --out mapper_s2s
    python mapper/seq2seq_mapper.py train --data data/asr_train.jsonl --model vinai/bartpho-syllable --out mapper_bartpho
    python mapper/seq2seq_mapper.py train --data data/asr_train.jsonl --target norm   # chỉ sửa lỗi, giữ dạng chữ thường/không dấu câu
    python mapper/tn_agy.py && python mapper/seq2seq_mapper.py train --data data/asr_train_tn.jsonl --target spoken   # hyp -> dạng đọc, ITN sau

Bước 3 — predict:
    python mapper/seq2seq_mapper.py predict --model mapper_s2s --text "đặt báo thức năm giờ rưỡi sáng mai nhé"
    python mapper/seq2seq_mapper.py predict --model mapper_s2s --in ~/Downloads/asr_public.jsonl --field bias --out mapped.jsonl

Dùng trong code:
    from mapper.seq2seq_mapper import Seq2SeqMapper
    m = Seq2SeqMapper("mapper_s2s"); m(["nạp năm mươi nghìn cho số không chín..."])
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

if __package__ in (None, ""):  # chạy trực tiếp: python mapper/seq2seq_mapper.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from textnorm.textutil import norm, wer_counts  # noqa: E402,F401

DEFAULT_MODEL = "VietAI/vit5-base"
PREFIX = ""  # ViT5/BARTpho không cần task prefix; để chuỗi rỗng cho đơn giản


# ---------------------------------------------------------------- text utils


def score(pairs):
    """pairs: [(ref, pred)] dạng gốc. Trả dict WER (sau norm), CER (sau norm), exact match (dạng gốc)."""
    we = wn = ce = cn = em = 0
    for ref, pred in pairs:
        e, n = wer_counts(norm(ref), norm(pred)); we += e; wn += n
        e, n = wer_counts(" ".join(norm(ref).replace(" ", "_")), " ".join(norm(pred).replace(" ", "_"))); ce += e; cn += n
        em += ref.strip() == pred.strip()
    return {"wer": we / max(wn, 1), "cer": ce / max(cn, 1), "exact": em / max(len(pairs), 1), "n": len(pairs)}


# ---------------------------------------------------------------- 1. hyps

def cmd_hyps(a):
    from asr.asr_zipformer import ZipformerASR
    root = Path(a.root).expanduser()
    split_dir = root / a.split
    items = [json.loads(l) for l in open(split_dir / f"{a.split}.jsonl", encoding="utf-8") if l.strip()]
    if a.limit:
        items = items[:a.limit]

    lexicon_tools = None
    if a.bias:
        from asr.asr_fix import fix_text
        from hotwords.bias import build_lexicon, common_words, snap_text
        common = common_words(root / "train" / "train.jsonl")
        lexicon_tools = (build_lexicon, snap_text, fix_text, common)
    asr = ZipformerASR(a.zipformer, int8=a.int8, num_threads=a.threads, bias=a.bias)

    done = set()
    out = Path(a.out)
    if out.exists() and not a.overwrite:  # resume: bỏ qua sample đã chạy
        done = {json.loads(l)["id"] for l in open(out, encoding="utf-8") if l.strip()}
        print(f"resume: {len(done)} sample đã có trong {out}")

    n = 0
    with open(out, "a" if done else "w", encoding="utf-8") as f:
        for si, it in enumerate(items, 1):
            if it["id"] in done:
                continue
            turns = [h for h in it["history"] if h["role"] == "user"] + [it]
            paths = [split_dir / u["audio"] for u in turns]
            if lexicon_tools:
                build_lexicon, snap_text, fix_text, common = lexicon_tools
                lex = build_lexicon(split_dir / f"{a.split}_environment" / it["id"] / "state.json", it["context"])
                raw = asr.transcribe(paths, hotwords=lex or None)
                hyps = [snap_text(r, lex, common)[0] if lex else r for r in raw]
                hyps = [fix_text(h)[0] for h in hyps]
            else:
                hyps = asr.transcribe(paths)
            for k, (u, h) in enumerate(zip(turns, hyps)):
                f.write(json.dumps({"id": it["id"], "turn": "QUERY" if k == len(turns) - 1 else f"u{k + 1}",
                                    "audio": u["audio"], "hyp": h, "ref": u.get("audio_text")},
                                   ensure_ascii=False) + "\n")
                n += 1
            f.flush()
            if si % 50 == 0:
                print(f"[{si}/{len(items)}] {n} câu")
    print(f"xong: {n} câu mới -> {out}")


# ---------------------------------------------------------------- 2. train

def load_pairs(path, target):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("ref") and r.get("hyp") is not None]
    if target == "spoken":  # cần file đã chạy tn_agy.py; bỏ câu TN không qua kiểm tra
        assert all("ref_spoken" in r for r in rows), "thiếu ref_spoken: chạy tn_agy.py trước"
        rows = [r for r in rows if r.get("tn_ok", True)]
    for r in rows:
        r["src"] = norm(r["hyp"])
        r["tgt"] = {"norm": lambda: norm(r["ref"]), "spoken": lambda: r["ref_spoken"],
                    "raw": lambda: " ".join(r["ref"].split())}[target]()
    return rows


def split_by_dialog(rows, dev_frac, seed):
    """Chia theo id hội thoại để các lượt cùng một sample không rơi vào cả train lẫn dev."""
    ids = sorted({r["id"] for r in rows})
    random.Random(seed).shuffle(ids)
    dev_ids = set(ids[:max(1, int(len(ids) * dev_frac))])
    return [r for r in rows if r["id"] not in dev_ids], [r for r in rows if r["id"] in dev_ids]


def load_tokenizer(name):
    """AutoTokenizer, fallback cho transformers 5: T5Tokenizer mới ép model Unigram, nhưng tokenizer.json
    của ViT5 là BPE (vocab dict) -> TypeError 'dict' object is not an instance of 'Sequence'.
    Khi đó nạp thẳng tokenizer.json (giữ nguyên BPE + post-processor thêm </s>)."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    try:
        return AutoTokenizer.from_pretrained(name)
    except TypeError:
        return PreTrainedTokenizerFast.from_pretrained(name)


def cmd_train(a):
    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer,
                              Seq2SeqTrainingArguments)

    rows = load_pairs(a.data, a.target)
    train, dev = split_by_dialog(rows, a.dev_frac, a.seed)
    if a.limit:
        train = train[:a.limit]
    # augmentation: câu "đúng" -> ref. Dạy "câu đúng thì giữ nguyên" (chống sửa thừa); với raw/norm còn dạy ITN/dấu câu.
    # spoken: input sạch chính là ref_spoken (dạng đọc) -> cặp đồng nhất.
    clean = (lambda r: r["tgt"]) if a.target == "spoken" else (lambda r: norm(r["ref"]))
    aug = [{"src": clean(r), "tgt": r["tgt"]} for r in train] if a.aug_clean else []
    train_pairs = [{"src": r["src"], "tgt": r["tgt"]} for r in train] + aug
    print(f"train {len(train)} câu (+{len(aug)} aug) | dev {len(dev)} câu | target={a.target}")

    base = score([(r["tgt"], r["src"]) for r in dev])
    print(f"baseline (giữ nguyên hyp) trên dev: WER {base['wer']:.4f}  CER {base['cer']:.4f}  exact {base['exact']:.3f}")

    tok = load_tokenizer(a.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(a.model)

    def enc(batch):
        x = tok([PREFIX + s for s in batch["src"]], max_length=a.max_len, truncation=True)
        x["labels"] = tok(text_target=batch["tgt"], max_length=a.max_len, truncation=True)["input_ids"]
        return x

    tr = Dataset.from_list(train_pairs).map(enc, batched=True, remove_columns=["src", "tgt"])
    dv = Dataset.from_list([{"src": r["src"], "tgt": r["tgt"]} for r in dev]).map(enc, batched=True,
                                                                                   remove_columns=["src", "tgt"])
    # T5 + fp16 dễ ra NaN -> chỉ dùng bf16 nếu GPU hỗ trợ, ngược lại fp32 (T4: fp32). BART thì fp16 được.
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    is_t5 = "t5" in model.config.model_type
    fp16 = torch.cuda.is_available() and not bf16 and not is_t5

    def metrics(p):
        preds, labels = p
        preds = np.where(preds < 0, tok.pad_token_id, preds)
        labels = np.where(labels < 0, tok.pad_token_id, labels)
        P = tok.batch_decode(preds, skip_special_tokens=True)
        L = tok.batch_decode(labels, skip_special_tokens=True)
        return score(list(zip(L, P)))

    import inspect
    warm = ({"warmup_ratio": 0.05} if "warmup_ratio" in inspect.signature(Seq2SeqTrainingArguments).parameters
            else {"warmup_steps": 0.05})  # transformers 5: warmup_steps float = tỉ lệ
    args = Seq2SeqTrainingArguments(
        output_dir=a.out + "_ckpt", num_train_epochs=a.epochs, learning_rate=a.lr, weight_decay=0.01,
        per_device_train_batch_size=a.bs, per_device_eval_batch_size=a.bs * 2, **warm,
        lr_scheduler_type="linear", logging_steps=50, eval_strategy="epoch", save_strategy="epoch",
        save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="wer", greater_is_better=False,
        predict_with_generate=True, generation_max_length=a.max_len, generation_num_beams=1,
        bf16=bf16, fp16=fp16, seed=a.seed, report_to="none")
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=tr, eval_dataset=dv, processing_class=tok,
                             data_collator=DataCollatorForSeq2Seq(tok, model=model), compute_metrics=metrics)
    trainer.train()
    trainer.save_model(a.out)
    tok.save_pretrained(a.out)
    json.dump({"target": a.target, "max_len": a.max_len, "base": a.model}, open(Path(a.out) / "mapper_config.json", "w"))
    print("saved ->", a.out)

    # đánh giá cuối bằng beam search + guard, in vài lỗi để soi
    m = Seq2SeqMapper(a.out, model=trainer.model, tok=tok)
    preds = m([r["src"] for r in dev], beams=a.beams)
    s = score([(r["tgt"], p) for r, p in zip(dev, preds)])
    print(f"mapper (beam={a.beams}) trên dev: WER {s['wer']:.4f}  CER {s['cer']:.4f}  exact {s['exact']:.3f}"
          f"   | baseline WER {base['wer']:.4f}")
    worse = [(r, p) for r, p in zip(dev, preds)
             if wer_counts(norm(r["tgt"]), norm(p))[0] > wer_counts(norm(r["tgt"]), r["src"])[0]]
    print(f"{len(worse)}/{len(dev)} câu mapper làm TỆ hơn hyp. Ví dụ:")
    for r, p in worse[:8]:
        print(f"  hyp : {r['src']}\n  ref : {r['tgt']}\n  pred: {p}\n")


# ---------------------------------------------------------------- 3. inference

class Seq2SeqMapper:
    def __init__(self, path, device=None, model=None, tok=None, max_ratio=1.6, min_ratio=0.6):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        cfg_p = Path(path) / "mapper_config.json"
        cfg = json.load(open(cfg_p)) if cfg_p.exists() else {}
        self.max_len = cfg.get("max_len", 128)
        self.tok = tok or load_tokenizer(path)
        self.model = model or AutoModelForSeq2SeqLM.from_pretrained(path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.max_ratio, self.min_ratio = max_ratio, min_ratio

    def __call__(self, texts, beams=4, bs=32):
        """texts: list[str] ASR. Trả list[str]. Guard: output dài/ngắn bất thường (ảo giác, cụt câu) -> trả lại input."""
        import torch
        out = []
        for i in range(0, len(texts), bs):
            chunk = [norm(t) for t in texts[i:i + bs]]
            enc = self.tok([PREFIX + t for t in chunk], return_tensors="pt", padding=True, truncation=True,
                           max_length=self.max_len).to(self.device)
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=self.max_len, num_beams=beams, do_sample=False)
            for src, pred in zip(chunk, self.tok.batch_decode(gen, skip_special_tokens=True)):
                ns, np_ = len(src.split()), len(norm(pred).split())
                ok = ns == 0 or self.min_ratio <= np_ / ns <= self.max_ratio
                out.append(pred.strip() if ok else src)
        return out


def cmd_predict(a):
    m = Seq2SeqMapper(a.model)
    if a.text:
        for t in a.text:
            print(f"{t}\n -> {m([t], beams=a.beams)[0]}")
        return
    rows = [json.loads(l) for l in open(Path(a.inp).expanduser(), encoding="utf-8") if l.strip()]
    preds = m([r[a.field] or "" for r in rows], beams=a.beams)
    with open(a.out, "w", encoding="utf-8") as f:
        for r, p in zip(rows, preds):
            r["mapped"] = p
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if all(r.get("ref") for r in rows):
        s0 = score([(r["ref"], r[a.field]) for r in rows])
        s1 = score([(r["ref"], p) for r, p in zip(rows, preds)])
        print(f"WER {s0['wer']:.4f} -> {s1['wer']:.4f} | CER {s0['cer']:.4f} -> {s1['cer']:.4f}")
    print(f"{len(rows)} câu -> {a.out}")


# ---------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    h = sp.add_parser("hyps", help="chạy Zipformer trên train -> jsonl (hyp, ref)")
    h.add_argument("--root", required=True)
    h.add_argument("--split", default="train")
    h.add_argument("--zipformer", default="hynt/Zipformer-30M-RNNT-6000h")
    h.add_argument("--bias", action="store_true", help="hotword + snap_text + asr_fix như pipeline")
    h.add_argument("--int8", action="store_true")
    h.add_argument("--threads", type=int, default=8)
    h.add_argument("--limit", type=int, default=0)
    h.add_argument("--overwrite", action="store_true", help="không resume, ghi đè file cũ")
    h.add_argument("--out", default="data/asr_train.jsonl")

    t = sp.add_parser("train")
    t.add_argument("--data", default="data/asr_train.jsonl", nargs="?")
    t.add_argument("--model", default=DEFAULT_MODEL, help="VietAI/vit5-base | vinai/bartpho-syllable | ...")
    t.add_argument("--out", default="mapper_s2s")
    t.add_argument("--target", choices=["raw", "norm", "spoken"], default="raw",
                   help="raw = đúng audio_text (số, hoa, dấu câu); norm = chữ thường không dấu câu; "
                        "spoken = ref_spoken từ tn_agy.py (dạng đọc, ITN làm sau) — khuyến nghị")
    t.add_argument("--epochs", type=float, default=10)
    t.add_argument("--lr", type=float, default=3e-4, help="ViT5 ~3e-4; BARTpho nên ~5e-5")
    t.add_argument("--bs", type=int, default=16)
    t.add_argument("--max-len", type=int, default=128)
    t.add_argument("--dev-frac", type=float, default=0.1)
    t.add_argument("--no-aug-clean", dest="aug_clean", action="store_false")
    t.add_argument("--beams", type=int, default=4)
    t.add_argument("--limit", type=int, default=0)
    t.add_argument("--seed", type=int, default=0)

    p = sp.add_parser("predict")
    p.add_argument("--model", default="mapper_s2s")
    p.add_argument("--text", nargs="*")
    p.add_argument("--in", dest="inp")
    p.add_argument("--field", default="hyp", help="trường chứa ASR trong jsonl (asr_public.jsonl: bias / greedy)")
    p.add_argument("--out", default="mapped.jsonl")
    p.add_argument("--beams", type=int, default=4)

    a = ap.parse_args()
    {"hyps": cmd_hyps, "train": cmd_train, "predict": cmd_predict}[a.cmd](a)


if __name__ == "__main__":
    main()
