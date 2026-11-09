"""Train RNNLM (LSTM trên token BPE của Zipformer) cho LM fusion trong sherpa-onnx, rồi export ONNX. Offline.

    python lm/train_rnnlm.py --data data/asr_train_plain.jsonl --out lm/rnnlm   # -> lm/rnnlm/rnnlm.onnx, model.pt
    python asr/eval_asr.py --root DATA --configs hw,hw+lm --lm lm/rnnlm/rnnlm.onnx --lm-scales 0.1,0.2,0.3

Corpus: câu user của train-split (id % 10 != 9) ở dạng đọc (chữ thường không dấu câu, số đọc thành chữ: trường
`ref_spoken` của asr/eval_asr.py --dump, hoặc vi_num.to_spoken(audio_text)), viết HOA rồi mã hoá bằng bpe.model của Zipformer.
Câu có số được thêm các cách đọc khác (hai mươi lăm / hai lăm). Held-out (id % 10 == 9) để early-stop theo perplexity.

ONNX (khớp sherpa-onnx OfflineRnnLM::Rescore(x, x_lens)):
  input  x      int64 (N, L): token id của giả thuyết, KHÔNG có sos/eos, pad 0
  input  x_lens int64 (N,)
  output nll    float (N,):   -sum log p(token_t | sos, token_<t)  (+ eos nếu export --with-eos)
sherpa-onnx gọi lại mỗi frame của modified_beam_search: lm_log_prob = -nll * lm_scale.
"""
import argparse
import json
import math
import os
import random
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import sys  # noqa: E402

if __package__ in (None, ""):  # chạy trực tiếp: python lm/train_rnnlm.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hotwords.mine_hotwords import is_heldout  # noqa: E402
from textnorm.vi_num import num_variants, to_spoken  # noqa: E402

SOS = EOS = 1   # tokens.txt của Zipformer: <blk>=0, <sos/eos>=1, <unk>=2


class RnnLm(nn.Module):
    def __init__(self, vocab, emb=256, hidden=256, layers=2, dropout=0.3, tie=False):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.rnn = nn.LSTM(emb, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden, emb) if tie and hidden != emb else nn.Identity()
        self.out = nn.Linear(emb if tie else hidden, vocab)
        if tie:
            self.out.weight = self.emb.weight

    def logits(self, inp):
        h, _ = self.rnn(self.drop(self.emb(inp)))
        return self.out(self.proj(self.drop(h)))

    def token_nll(self, x, x_lens, with_eos):
        """x (N, L) không sos/eos -> nll (N, L[+1]) theo vị trí, 0 ở chỗ pad."""
        n, l = x.shape
        sos = torch.full((n, 1), SOS, dtype=x.dtype, device=x.device)
        inp = torch.cat([sos, x], 1)                                   # (N, L+1)
        pos = torch.arange(l + 1, device=x.device).unsqueeze(0)
        if with_eos:
            tgt = torch.cat([x, torch.zeros_like(sos)], 1)
            tgt = torch.where(pos == x_lens.unsqueeze(1), torch.full_like(tgt, EOS), tgt)
            mask = pos <= x_lens.unsqueeze(1)
        else:
            inp, tgt = inp[:, :l], x
            mask = pos[:, :l] < x_lens.unsqueeze(1)
        lp = torch.log_softmax(self.logits(inp), -1)
        nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        return nll * mask


class OnnxWrapper(nn.Module):
    def __init__(self, lm, with_eos):
        super().__init__()
        self.lm, self.with_eos = lm, with_eos

    def forward(self, x, x_lens):
        return self.lm.token_nll(x, x_lens, self.with_eos).sum(1)


# ---------------------------------------------------------------- data

def spoken_alts(text, max_alts=2):
    """các cách đọc khác của câu có số (mỗi số đổi sang cách đọc thứ 2) để LM không chỉ thuộc một kiểu."""
    import re
    out = []
    toks = text.split()
    nums = [i for i, t in enumerate(toks) if re.search(r"\d", t)]
    if not nums:
        return out
    for k in range(1, max_alts + 1):
        alt = [num_variants(t)[min(k, len(num_variants(t)) - 1)] if i in nums else t for i, t in enumerate(toks)]
        out.append(" ".join(alt))
    return out


def load_corpus(data, train_jsonl):
    """(train_sents, heldout_sents) dạng đọc chữ thường."""
    tr, ho = [], []
    if data and Path(data).exists():
        for l in open(data, encoding="utf-8"):
            r = json.loads(l)
            s = r.get("ref_spoken") or to_spoken(r["ref"], r.get("hyp"))
            if s:
                (ho if is_heldout(r["id"]) else tr).append(s)
            if not is_heldout(r["id"]):
                from textnorm.textutil import norm
                for alt in spoken_alts(norm(r["ref"])):
                    s2 = to_spoken(alt)
                    if s2 and s2 != s:
                        tr.append(s2)
    elif train_jsonl:
        for l in open(train_jsonl, encoding="utf-8"):
            it = json.loads(l)
            texts = [h["audio_text"] for h in it["history"] if h["role"] == "user"] + [it["audio_text"]]
            for t in texts:
                s = to_spoken(t)
                if s:
                    (ho if is_heldout(it["id"]) else tr).append(s)
    else:
        raise SystemExit("cần --data (jsonl có ref/ref_spoken) hoặc --train-jsonl")
    return tr, ho


def batches(seqs, bs, shuffle):
    idx = list(range(len(seqs)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), bs):
        chunk = [seqs[j] for j in idx[i:i + bs]]
        L = max(len(s) for s in chunk)
        x = torch.zeros(len(chunk), L, dtype=torch.long)
        for k, s in enumerate(chunk):
            x[k, :len(s)] = torch.tensor(s)
        yield x, torch.tensor([len(s) for s in chunk])


def evaluate(model, seqs, with_eos, device):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for x, xl in batches(seqs, 128, False):
            x, xl = x.to(device), xl.to(device)
            tot += model.token_nll(x, xl, with_eos).sum().item()
            n += (xl + int(with_eos)).sum().item()
    model.train()
    return math.exp(tot / max(n, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/asr_train_plain.jsonl", help="jsonl có id/ref[/ref_spoken/hyp] (eval_asr.py --dump)")
    ap.add_argument("--train-jsonl", help="dùng audio_text của DATA/train/train.jsonl nếu không có --data")
    ap.add_argument("--zipformer", default="hynt/Zipformer-30M-RNNT-6000h")
    ap.add_argument("--out", default="lm/rnnlm")
    ap.add_argument("--emb", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--tie", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--with-eos", action="store_true", help="tính cả xác suất kết câu (mặc định: không, vì sherpa chấm "
                                                            "lại giả thuyết dở dang mỗi frame)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    import sentencepiece as spm
    from asr.asr_zipformer import model_dir
    sp = spm.SentencePieceProcessor(model_file=str(model_dir(a.zipformer) / "bpe.model"))
    vocab = sp.get_piece_size()

    tr_txt, ho_txt = load_corpus(a.data, a.train_jsonl)
    enc = lambda ss: [ids for ids in (sp.encode(s.upper()) for s in ss) if ids]
    tr, ho = enc(tr_txt), enc(ho_txt)
    unk = sum(t == 2 for s in tr for t in s) / max(1, sum(map(len, tr)))
    print(f"train {len(tr)} câu / {sum(map(len, tr))} token, held-out {len(ho)} câu, vocab {vocab}, unk {unk:.2%}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RnnLm(vocab, a.emb, a.hidden, a.layers, a.dropout, a.tie).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    best, bad = float("inf"), 0
    print(f"epoch 0  held-out ppl {evaluate(model, ho, a.with_eos, device):.1f}")
    for ep in range(1, a.epochs + 1):
        tot, n = 0.0, 0
        for x, xl in batches(tr, a.bs, True):
            x, xl = x.to(device), xl.to(device)
            nll = model.token_nll(x, xl, a.with_eos)
            loss = nll.sum() / (xl + int(a.with_eos)).sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot, n = tot + loss.item() * len(xl), n + len(xl)
        ppl = evaluate(model, ho, a.with_eos, device)
        sched.step(ppl)
        print(f"epoch {ep:>2}  train ppl {math.exp(tot / n):7.1f}  held-out ppl {ppl:7.1f}")
        if ppl < best - 0.05:
            best, bad = ppl, 0
            torch.save({"state": model.state_dict(), "args": vars(a), "vocab": vocab}, out / "model.pt")
        else:
            bad += 1
            if bad >= a.patience:
                break

    ck = torch.load(out / "model.pt", map_location="cpu")
    model = RnnLm(vocab, a.emb, a.hidden, a.layers, 0.0, a.tie)
    model.load_state_dict(ck["state"])
    model.eval()
    wrap = OnnxWrapper(model, a.with_eos)
    x = torch.randint(3, vocab, (2, 7))
    xl = torch.tensor([7, 4])
    onnx_path = out / "rnnlm.onnx"
    torch.onnx.export(wrap, (x, xl), str(onnx_path), input_names=["x", "x_lens"], output_names=["nll"],
                      dynamic_axes={"x": {0: "N", 1: "L"}, "x_lens": {0: "N"}, "nll": {0: "N"}}, opset_version=17,
                      dynamo=False)
    import onnx
    m = onnx.load(str(onnx_path))
    meta = {"model_type": "rnnlm", "version": "1", "vocab_size": str(vocab), "embedding_dim": str(a.emb),
            "hidden_dim": str(a.hidden), "num_layers": str(a.layers), "sos_id": str(SOS), "eos_id": str(EOS),
            "with_eos": str(int(a.with_eos)), "comment": "train_rnnlm.py (voice-agent-pipeline)"}
    for k, v in meta.items():
        p = m.metadata_props.add()
        p.key, p.value = k, v
    onnx.save(m, str(onnx_path))

    try:
        import numpy as np
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        got = sess.run(None, {"x": x.numpy(), "x_lens": xl.numpy()})[0]
        assert np.allclose(got, wrap(x, xl).detach().numpy(), atol=1e-3)
        chk = "onnxruntime khớp torch"
    except ImportError:
        chk = "không có onnxruntime để so; kiểm tra bằng ZipformerASR(lm=...)"
    print(f"best held-out ppl {best:.1f} -> {onnx_path} ({chk})")


if __name__ == "__main__":
    main()
