"""Fine-tune a ChunkFormer CTC model (Vietnamese or English) with CTC loss.

--lang vi : code-switched Vietnamese data (manifest text = full transcript, English words included)
--lang en : English audio (ideally crops of English words/phrases spoken by Vietnamese speakers + their text)

manifest.jsonl: {"audio": "path.wav", "text": "reference transcript", "duration": optional seconds}
The result is out/pytorch_model.bin (state_dict of model.model); set it as models.vi_ckpt / models.en_ckpt in the config.

python train/finetune_chunkformer.py --lang vi --train vi_train.jsonl --valid vi_dev.jsonl --out ckpt/chunkformer_vi
python train/finetune_chunkformer.py --lang en --train en_train.jsonl --valid en_dev.jsonl --out ckpt/chunkformer_en
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import load_config  # noqa: E402
from models.dual_chunkformer import ChunkformerCTC  # noqa: E402
from models.text_utils import normalize, wer  # noqa: E402


def read(path, limit=0):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return rows[:limit] if limit else rows


def spec_augment(x, n_freq=2, f=15, n_time=2, t=40):
    """x: (T, F) fbank. Zero a few frequency bands and time spans (in place, on a copy)."""
    x = x.clone()
    T, Fd = x.shape
    for _ in range(n_freq):
        w = random.randint(0, f)
        s = random.randint(0, max(Fd - w, 0))
        x[:, s:s + w] = 0
    for _ in range(n_time):
        w = random.randint(0, min(t, T // 5))
        s = random.randint(0, max(T - w, 0))
        x[s:s + w] = 0
    return x


def make_batches(rows, batch_size, max_seconds, shuffle):
    """Batches of similar length when durations are known, else plain shuffled batches of batch_size."""
    order = list(range(len(rows)))
    if all("duration" in r for r in rows):
        order.sort(key=lambda i: rows[i]["duration"])
        batches, cur, tot = [], [], 0.0
        for i in order:
            d = rows[i]["duration"]
            if cur and (len(cur) >= batch_size or (len(cur) + 1) * d > max_seconds):
                batches.append(cur)
                cur = []
            cur.append(i)
        if cur:
            batches.append(cur)
    else:
        if shuffle:
            random.shuffle(order)
        batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if shuffle:
        random.shuffle(batches)
    return batches


def encode_batch(cm, rows, idxs, augment, device):
    feats, lens, ys = [], [], []
    for i in idxs:
        ids = cm.encode(" ".join(normalize(rows[i]["text"])))
        x, x_len = cm.m._load_audio_and_extract_features(rows[i]["audio"])
        if not ids or len(ids) > x_len // 8:  # CTC needs at least as many frames as tokens
            continue
        feats.append(spec_augment(x) if augment else x)
        lens.append(x_len)
        ys.append(torch.tensor(ids))
    if not feats:
        return None
    return (pad_sequence(feats, batch_first=True).to(device), torch.tensor(lens, dtype=torch.int, device=device),
            pad_sequence(ys, batch_first=True).to(device), torch.tensor([len(y) for y in ys], device=device))


def ctc_forward(asr, feats, lens, ys, ys_lens):
    enc, mask = asr.encoder(feats, lens)
    enc_lens = mask.squeeze(1).sum(1)
    loss, _ = asr.ctc(enc, enc_lens, ys, ys_lens)  # mean over the batch
    return loss


@torch.no_grad()
def evaluate(cm, rows, device, chunk, left, right):
    """Mean word error rate with CTC greedy decoding (chunked inference, same as at test time)."""
    cm.m.eval()
    tot = n = 0
    for r in rows:
        lp = cm.log_probs(r["audio"])
        hyp = cm.decode(torch.unique_consecutive(lp.argmax(-1)).tolist())
        tot += wer(hyp, r["text"])
        n += 1
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["vi", "en"], required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/pipeline_config.yaml"))
    ap.add_argument("--init-ckpt", help="continue from a fine-tuned pytorch_model.bin")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5, help="small: ChunkFormer is already trained")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-seconds", type=float, default=240.0, help="audio seconds per batch (needs 'duration' in the manifest)")
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--freeze-encoder", action="store_true", help="train only the CTC head")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    cfg = load_config(a.config)
    m, c = cfg["models"], cfg["chunkformer"]
    device = m[f"{a.lang}_device"]
    cm = ChunkformerCTC(m[f"chunkformer_{a.lang}"], a.init_ckpt or m.get(f"{a.lang}_ckpt"), device,
                        c["chunk_size"], c["left_context_size"], c["right_context_size"])
    asr = cm.m.model
    if a.freeze_encoder:
        for p in asr.encoder.parameters():
            p.requires_grad = False
    params = [p for p in asr.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)

    train, valid = read(a.train, a.limit), read(a.valid, a.limit) if a.valid else []
    steps_total = max(1, a.epochs * (len(make_batches(train, a.batch_size, a.max_seconds, False)) // a.grad_accum))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / a.warmup, max(0.0, (steps_total - s) / max(steps_total - a.warmup, 1))))
    use_amp = str(device).startswith("cuda")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    if valid:
        best = evaluate(cm, valid, device, c["chunk_size"], c["left_context_size"], c["right_context_size"])
        print(f"[{a.lang}] WER before fine-tuning: {best:.4f}", flush=True)
    step, t0 = 0, time.time()
    for ep in range(a.epochs):
        asr.train()
        run, k = 0.0, 0
        for bi, idxs in enumerate(make_batches(train, a.batch_size, a.max_seconds, True)):
            batch = encode_batch(cm, train, idxs, not a.no_augment, device)
            if batch is None:
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = ctc_forward(asr, *batch) / a.grad_accum
            loss.backward()
            run += loss.item() * a.grad_accum
            k += 1
            if (bi + 1) % a.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    print(f"[{a.lang}] ep {ep + 1} step {step}/{steps_total} loss {run / k:.3f} "
                          f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
                    run, k = 0.0, 0
        score = evaluate(cm, valid, device, c["chunk_size"], c["left_context_size"], c["right_context_size"]) if valid else -ep
        print(f"[{a.lang}] epoch {ep + 1}: valid WER {score:.4f}" if valid else f"[{a.lang}] epoch {ep + 1} done", flush=True)
        if score < best or not valid:  # keep the best epoch (the last one when there is no dev set)
            best = score
            torch.save(asr.state_dict(), out / "pytorch_model.bin")
    if (out / "pytorch_model.bin").exists():
        print(f"[{a.lang}] saved {out / 'pytorch_model.bin'}  -> set models.{a.lang}_ckpt to it")
    else:
        print(f"[{a.lang}] no epoch beat the baseline WER, nothing saved (lower --lr or check the data)")


if __name__ == "__main__":
    main()
