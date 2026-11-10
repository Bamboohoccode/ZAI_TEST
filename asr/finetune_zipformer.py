#!/usr/bin/env python3
"""Fine-tune Zipformer-150M (hynt/ZipFormer-150M-CR-CTC-RNNT-6000h) trên các câu pipeline.py nhận dạng SAI,
trộn 95% dữ liệu cũ (các tập public mà model gốc đã train) để không quên kiến thức cũ, 5% là dữ liệu mới.

Đầu vào: output của `pipeline.py --no-llm --prefetch` trên split train (mỗi dòng {"id", "turns": [{"turn", "audio",
"asr", "marked", "refine", "ref", ["prefetch"]}]}). Câu được chọn để fine-tune = lượt user có `ref` mà bản `--hyp-field`
(mặc định "asr": ASR sau prefetch/decode lại, trước LLM refine) còn sai so với ref dạng đọc.

Chạy trên Kaggle (GPU T4/P100, bật Internet; HF_TOKEN trong Kaggle Secrets nếu muốn dùng tập gated bud500/VietSpeech):

    # 0) output pipeline (đã chạy xong):
    python pipeline.py --root DATA --split train --no-llm --prefetch --lm lm/rnnlm/rnnlm.onnx --out asr_out.jsonl
    # 1) cài k2/icefall/lhotse  2) lọc câu sai + tải replay  3) train + export ONNX   (hoặc "all" = cả 3)
    python asr/finetune_zipformer.py setup   --work /kaggle/working/zf150
    python asr/finetune_zipformer.py prepare --work /kaggle/working/zf150 --root DATA --asr-out asr_out.jsonl \
        [--ids-file data/asr_sai_nghia_ids.txt] [--correct-ratio 1.0] [--max-per-template 3]
    python asr/finetune_zipformer.py train   --work /kaggle/working/zf150 [--resume]
    # dùng model mới:
    python pipeline.py ... --zipformer /kaggle/working/zf150/zipformer-ft

Dữ liệu:
  - mới (5%)   : câu sai của train-split (id % 10 != 9). Nhãn = ref -> dạng đọc (textnorm.vi_num.to_spoken, chọn cách
                 đọc số theo hyp), chữ HOA như vocab BPE. Mỗi epoch dùng 1 bản speed perturb (luân phiên --speed-new).
  - cũ (95%)   : stream từ HF (--replay), lưu flac 16 kHz. Mỗi epoch bốc ngẫu nhiên đúng 19 x (giờ mới của epoch)
                 từ pool -> tỉ lệ 95/5 tính theo THỜI LƯỢNG audio, in ra ở đầu mỗi epoch.
  - --epoch-hours (mặc định 20): ~4 h câu sai -> 95/5 đầy đủ là ~80 h/epoch, quá nặng cho T4. Mỗi epoch chỉ lấy
                 epoch_hours x 5% giờ câu mới (cửa sổ xoay vòng, qua vài epoch đủ mọi câu) + 19 lần chừng đó dữ liệu cũ.
                 --epoch-hours 0 = mọi câu mới mỗi epoch.
  - dev_new    : bộ đánh giá CỐ ĐỊNH 200 câu, chọn từ sample held-out (id % 10 == 9; thiếu thì bù từ sample khác):
                 100 câu "fix" = đang sai (có trong --ids-file) + 100 câu "keep" = model gốc nhận đúng. Ghi ra
                 WORK/data/eval_ids.txt; lần sau truyền --eval-ids <file đó> để đánh giá đúng các câu này. Chỉ các
                 sample chứa câu đánh giá bị loại khỏi train (cùng người nói), sample held-out còn lại vẫn được train.
  - dev_old    : FLEURS vi test (khác người nói với train) — đo mức "quên".
  Chọn epoch theo TB WER(dev_new, dev_old); epoch nào làm dev_old tệ hơn model gốc quá --max-old-degrade thì loại.
  Mỗi epoch in: dev_fix sửa được x/100 câu, dev_keep bị hỏng y/100 câu. Epoch làm hỏng nhiều hơn model gốc quá
  --keep-max-broken câu keep thì loại; hỏng quá ngưỡng --keep-patience epoch liên tiếp thì dừng.

License model gốc cc-by-nc-nd-4.0: không thương mại, KHÔNG chia sẻ weights đã fine-tune.
"""
import argparse
import difflib
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from textnorm.textutil import norm, wer_counts  # noqa: E402
from textnorm.vi_num import to_spoken  # noqa: E402

HF_REPO = "hynt/ZipFormer-150M-CR-CTC-RNNT-6000h"
ICEFALL_COMMIT = "9c966762e058bd7ba79d9da71e42fc06b64b703e"  # bản notebooks/finetune_zipformer.ipynb đã đối chiếu API
SR = 16000
# NAME=NGUỒN|SPLIT|CỘT_TEXT|TRỌNG_SỐ. NGUỒN: repo HF (có thể "repo:config") hoặc glob parquet "hf://datasets/...".
# Đều nằm trong dữ liệu train của model gốc (README hynt): VLSP2020, VIET_BUD500, VietSpeech, FPT, FLEURS.
DEFAULT_REPLAY = [
    "vlsp2020=doof-ferb/vlsp2020_vinai_100h|train|transcription|0.25",
    "bud500=linhtran92/viet_bud500|train|transcription|0.25",          # gated: chấp nhận điều khoản + HF_TOKEN
    "vietspeech=NhutP/VietSpeech|train|transcription|0.20",            # gated
    "fpt=doof-ferb/fpt_fosd|train|transcription|0.15",
    "fleurs=hf://datasets/google/fleurs/parquet-data/vi_vn/train-*.parquet|train|transcription|0.15",
]
DEFAULT_OLD_DEV = "fleurs_test=hf://datasets/google/fleurs/parquet-data/vi_vn/test-*.parquet|train|transcription|1"
VI_TEXT = re.compile(r"[a-zàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ' ]+")


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def is_heldout(sid):
    """train_0019 -> held-out nếu số % 10 == 9 (cùng quy tắc hotwords/mine_hotwords.py, asr/candidates.py)."""
    m = re.search(r"(\d+)$", str(sid))
    return bool(m) and int(m.group(1)) % 10 == 9


# ---------------------------------------------------------------- so khớp chữ: dấu thanh đặt khác chỗ (huỷ/hủy) không tính lỗi

_TONE = {"̀": "1", "́": "2", "̃": "3", "̉": "4", "̣": "5"}


def tone_key(w):
    """'huỷ' và 'hủy' -> cùng khoá (bỏ dấu thanh khỏi vị trí, giữ loại dấu)."""
    d = unicodedata.normalize("NFD", w)
    return unicodedata.normalize("NFC", "".join(c for c in d if c not in _TONE)) + "".join(_TONE.get(c, "") for c in d)


def keyed(s):
    return " ".join(tone_key(w) for w in s.split())


def utt_errors(ref_sp, hyp):
    """(số lỗi, số từ ref) theo từ, không phân biệt vị trí dấu thanh."""
    return wer_counts(keyed(ref_sp), keyed(hyp))


_OLD_STYLE = re.compile("(?<![qQ])([oOuU])([aAeEyY])([̣̀́̃̉])$")


def new_style(s):
    """dấu thanh kiểu mới như vocab BPE của model (▁HỦY, ▁HÒA, ▁KHỎE): 'huỷ'->'hủy', 'hoà'->'hòa', 'khoẻ'->'khỏe'.
    Chỉ đổi vần oa/oe/uy không có phụ âm cuối; 'quý', 'hoàng', 'thuế' giữ nguyên."""
    out = []
    for w in s.split():
        d = unicodedata.normalize("NFD", w)
        out.append(unicodedata.normalize("NFC", _OLD_STYLE.sub(r"\1\3\2", d)))
    return " ".join(out)


assert new_style("huỷ hoà khoẻ quý hoàng thuế HUỶ") == "hủy hòa khỏe quý hoàng thuế HỦY"


def harmonize(ref_sp, hyp):
    """từ ref trùng (theo tone_key) với hyp -> lấy chính tả của hyp (kiểu model đã quen: 'hủy' thay 'huỷ')."""
    r, h = ref_sp.split(), hyp.split()
    out = list(r)
    sm = difflib.SequenceMatcher(None, [tone_key(w) for w in r], [tone_key(w) for w in h], autojunk=False)
    for b in sm.get_matching_blocks():
        out[b.a:b.a + b.size] = h[b.b:b.b + b.size]
    return " ".join(out)


def label_of(ref, hyp=None):
    """ref viết -> nhãn train (chữ HOA, dạng đọc). None nếu còn chữ số không chắc cách đọc."""
    h = norm(hyp) if hyp else None
    sp = to_spoken(ref, h) if h else to_spoken(ref)
    if not sp:
        return None
    if h:
        sp = harmonize(sp, h)
    return new_style(" ".join(sp.upper().split()))


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def duration(p):
    import soundfile as sf
    return sf.info(str(p)).duration


def hf_token_from_kaggle():
    if os.environ.get("HF_TOKEN"):
        return
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
        log("HF_TOKEN lấy từ Kaggle Secrets")
    except Exception:  # noqa: BLE001 — không có secret: chỉ dùng được tập không gated
        pass


# ================================================================ stage: setup

def stage_setup(a):
    import urllib.request
    import torch

    def pip(*x):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *x])

    log("torch", torch.__version__, "| CUDA", torch.version.cuda,
        "| GPU", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    pip("lhotse", "lilcom", "sentencepiece", "sherpa-onnx", "kaldialign", "tensorboard", "onnx", "onnxruntime",
        "onnxmltools", "soundfile", "scipy", "huggingface_hub", "datasets", "dill", "typeguard",
        "pypinyin", "packaging")  # icefall/utils.py import pypinyin, packaging ngay đầu file
    try:
        import k2  # noqa: F401
    except ImportError:
        tv, cu = torch.__version__.split("+")[0], torch.version.cuda
        py = f"cp{sys.version_info.major}{sys.version_info.minor}"
        idx = "https://k2-fsa.github.io/k2/cuda.html" if cu else "https://k2-fsa.github.io/k2/cpu.html"
        tag = f"+cuda{cu}.torch{tv}-" if cu else f"+cpu.torch{tv}-"
        html = urllib.request.urlopen(idx).read().decode()
        urls = [u for u in set(re.findall(r'href="([^"]+\.whl)"', html))
                if tag in u and f"-{py}-{py}-" in u and "linux" in u and "x86_64" in u]
        if not urls:
            sys.exit(f"Không có wheel k2 cho torch {tv} / CUDA {cu} / {py}. Cài torch khớp một wheel trong {idx} "
                     "(vd pip install torch==<ver> --index-url https://download.pytorch.org/whl/cu<xx>) rồi chạy lại.")
        url = max(urls, key=lambda u: int(re.search(r"dev(\d+)", u).group(1)) if re.search(r"dev(\d+)", u) else 0)
        log("k2 wheel:", url.rsplit("/", 1)[-1])
        pip(url)
    icefall = Path(a.work) / "icefall"
    if not icefall.exists():
        subprocess.check_call(["git", "clone", "-q", "https://github.com/k2-fsa/icefall", str(icefall)])
        subprocess.check_call(["git", "-C", str(icefall), "checkout", "-q", ICEFALL_COMMIT])
    log("setup xong:", icefall)


def icefall_paths(work):
    icefall = Path(work) / "icefall"
    zdir = icefall / "egs/librispeech/ASR/zipformer"
    if not zdir.exists():
        sys.exit(f"thiếu {zdir}: chạy stage setup trước")
    for p in (str(zdir), str(icefall)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ["PYTHONPATH"] = f"{icefall}:{zdir}:{os.environ.get('PYTHONPATH', '')}"
    return icefall, zdir


# ================================================================ stage: prepare

def template_cap(rows, cap, thr=0.6):
    """khử câu cùng mẫu (vd. hàng chục câu "vàng sjc bán ra trên ... thì nhắc tôi ..." chỉ khác tên / số):
    gom cụm tham lam theo Jaccard tập từ >= thr, mỗi cụm giữ tối đa `cap` câu (theo thứ tự rows: ưu tiên câu đứng trước).
    cap <= 0 = không khử. Trả (rows giữ lại, số cụm)."""
    if cap <= 0:
        return rows, len(rows)
    reps, index, count, out = [], {}, [], []
    for r in rows:
        ws = set(r["text"].lower().split())
        cand = {c for w in ws for c in index.get(w, ())}
        hit = next((c for c in sorted(cand) if len(ws & reps[c]) / max(len(ws | reps[c]), 1) >= thr), None)
        if hit is None:
            hit = len(reps)
            reps.append(ws)
            count.append(0)
            for w in ws:
                index.setdefault(w, []).append(hit)
        if count[hit] < cap:
            count[hit] += 1
            out.append(r)
    return out, len(reps)


def greedy_asr_out(a):
    """không có --asr-out (chỉ --ids-file): dựng file cùng định dạng output pipeline từ train.jsonl, hyp = greedy model
    gốc (dùng để chọn cách đọc số cho nhãn và tìm câu đúng làm điểm neo). Cache ở WORK/data/asr_greedy.jsonl."""
    out = Path(a.work) / "data" / "asr_greedy.jsonl"
    if out.exists():
        log(f"dùng lại {out}")
        return out
    from asr.asr_zipformer import ZipformerASR
    root = Path(a.root) / a.split
    items = read_jsonl(root / f"{a.split}.jsonl")
    asr = ZipformerASR(a.zipformer, int8=True, bias=False, decoding="greedy_search", num_threads=os.cpu_count() or 4)
    rows, t0 = [], time.time()
    for i, it in enumerate(items):
        us = [h for h in it.get("history", []) if h.get("role") == "user"] + [it]
        us = [u for u in us if u.get("audio") and (root / u["audio"]).exists()]
        hyps = asr.transcribe([root / u["audio"] for u in us]) if us else []
        rows.append({"id": it["id"], "turns": [
            {"turn": "QUERY" if u is it else f"u{k + 1}", "audio": u["audio"], "asr": h, "refine": h,
             "ref": u.get("audio_text")} for k, (u, h) in enumerate(zip(us, hyps))]})
        if (i + 1) % 100 == 0:
            log(f"  greedy {i + 1}/{len(items)} sample, {time.time() - t0:.0f}s")
    write_jsonl(out, rows)
    log(f"greedy model gốc trên {len(rows)} sample -> {out}")
    return out


def load_ids(a):
    """--ids-file -> set id lượt (tên file audio không đuôi: train_0008, train_0008_u2); cột thêm / .wav / '#' bỏ qua."""
    if not a.ids_file:
        return None
    ids = {Path(l.split()[0]).stem for l in open(a.ids_file, encoding="utf-8") if l.strip() and not l.startswith("#")}
    log(f"--ids-file: {len(ids)} id lượt sai")
    return ids


def build_eval(a, ids):
    """bộ đánh giá CỐ ĐỊNH lấy từ sample held-out (id % 10 == 9): --eval-fix-n câu đang SAI (có trong --ids-file; không có
    --ids-file thì greedy model gốc sai) + --eval-keep-n câu model gốc nhận ĐÚNG (canh việc làm hỏng câu vốn đúng).
    Ghi ra --eval-ids ("fix train_0019" / "keep train_0029_u1"); file đã có -> dùng lại y nguyên, không chọn lại.
    -> (rows dev_new, tập sample bị loại khỏi train = các sample chứa câu đánh giá)."""
    root = Path(a.root) / a.split
    eval_file = Path(a.eval_ids) if a.eval_ids else Path(a.work) / "data" / "eval_ids.txt"
    utts = {}
    for r in read_jsonl(root / f"{a.split}.jsonl"):
        for u in [h for h in r.get("history", []) if h.get("role") == "user"] + [r]:
            if u.get("audio") and u.get("audio_text") and (root / u["audio"]).exists():
                stem = Path(u["audio"]).stem
                utts.setdefault(stem, {"id": "dev_" + stem, "stem": stem, "audio": str((root / u["audio"]).resolve()),
                                       "ref": u["audio_text"], "sample": r["id"]})
    fixed = None
    if eval_file.exists():
        fixed = {}
        for l in open(eval_file, encoding="utf-8"):
            if l.strip() and not l.startswith("#"):
                g, s = l.split()[:2]
                fixed[Path(s).stem] = g
        miss = [s for s in fixed if s not in utts]
        log(f"bộ đánh giá cố định: dùng lại {eval_file} ({sum(g == 'fix' for g in fixed.values())} fix, "
            f"{sum(g == 'keep' for g in fixed.values())} keep)" + (f" | KHÔNG thấy {len(miss)} id: {miss[:5]}" if miss else ""))
        cand = [utts[s] for s in fixed if s in utts]
    else:  # held-out trước; thiếu câu thì lấy bù từ sample khác (sample đó cũng bị loại khỏi train)
        rng = random.Random(a.seed)
        held = sorted((u for u in utts.values() if is_heldout(u["sample"])), key=lambda u: u["stem"])
        rest = sorted((u for u in utts.values() if not is_heldout(u["sample"])), key=lambda u: u["stem"])
        rng.shuffle(held)
        rng.shuffle(rest)
        cand = held + rest
    is_wrong = (lambda u: u["stem"] in ids) if ids is not None else (lambda u: u["base_err"] > 0)
    from asr.asr_zipformer import ZipformerASR
    asr = ZipformerASR(a.zipformer, int8=True, bias=False, decoding="greedy_search", num_threads=os.cpu_count() or 4)
    rows, n_fix, n_keep = [], 0, 0
    for k in range(0, len(cand), 32):  # greedy model gốc: chọn cách đọc số cho nhãn + biết câu nào đang đúng
        if fixed is None and k >= len(held) and n_fix >= a.eval_fix_n and n_keep >= a.eval_keep_n:
            break  # đã đủ, không decode thêm sample ngoài held-out
        chunk = cand[k:k + 32]
        for u, h in zip(chunk, asr.transcribe([u["audio"] for u in chunk])):
            lab = label_of(u["ref"], h)
            if lab:
                rows.append({**u, "text": lab, "base_hyp": h, "base_err": utt_errors(lab.lower(), norm(h))[0]})
                n_fix += is_wrong(rows[-1])
                n_keep += not is_wrong(rows[-1]) and rows[-1]["base_err"] == 0
    if fixed is None:
        fix = [u for u in rows if is_wrong(u)][:a.eval_fix_n]
        keep = [u for u in rows if not is_wrong(u) and u["base_err"] == 0][:a.eval_keep_n]
        n_out = sum(not is_heldout(u["sample"]) for u in fix + keep)
        if n_out:
            log(f"held-out không đủ: lấy bù {n_out} câu từ sample khác (các sample này cũng không được train)")
        for name, got, want in (("fix", fix, a.eval_fix_n), ("keep", keep, a.eval_keep_n)):
            if len(got) < want:
                log(f"CẢNH BÁO: chỉ tìm được {len(got)} câu {name} (< {want})")
        fixed = {u["stem"]: "fix" for u in fix} | {u["stem"]: "keep" for u in keep}
        eval_file.parent.mkdir(parents=True, exist_ok=True)
        with open(eval_file, "w", encoding="utf-8") as f:
            f.write(f"# bộ đánh giá cố định (seed {a.seed}): fix = câu đang sai, keep = model gốc nhận đúng. "
                    f"Truyền lại bằng --eval-ids để lần sau đánh giá đúng các câu này\n")
            for g in ("fix", "keep"):
                f.writelines(f"{g} {s}\n" for s, gg in sorted(fixed.items()) if gg == g)
        log(f"bộ đánh giá cố định: {len(fix)} fix + {len(keep)} keep -> {eval_file}")
    out = []
    for u in rows:
        if u["stem"] in fixed:
            g = fixed[u["stem"]]
            out.append({**u, "group": g, "hard": g == "fix", "dur": round(duration(u["audio"]), 3)})
    out.sort(key=lambda u: (u["group"], u["stem"]))
    n_keep_bad = sum(u["group"] == "keep" and u["base_err"] > 0 for u in out)
    if n_keep_bad:
        log(f"  lưu ý: lần decode này {n_keep_bad} câu keep lệch so với lúc chọn (sherpa decode theo lô lệch nhẹ, hoặc "
            f"đổi model gốc). Không sao: lúc train số câu keep bị hỏng được so với chính model gốc")
    excl = {u["sample"] for u in out}
    log(f"dev_new = {len(out)} câu đánh giá | loại {len(excl)} sample chứa chúng khỏi train (cùng người nói); "
        f"các sample held-out còn lại được train")
    return out, excl


def select_new(a, sp=None, ids=None, exclude=None):
    """asr_out.jsonl -> dữ liệu mới của train-split.
    Dữ liệu mới = câu SAI (bản --hyp-field sai >= --min-errors từ, hoặc nằm trong --ids-file nếu có)
    + câu ĐÚNG cùng miền làm điểm neo (--correct-ratio x số câu sai) để model không học "câu miền này hay sai -> đổi từ".
    Cả hai đều khử trùng mẫu câu (--max-per-template). exclude: sample không được train (chứa câu đánh giá);
    None = mọi sample held-out (id % 10 == 9)."""
    root = Path(a.root) / a.split
    if ids is None and a.ids_file:
        ids = load_ids(a)
    wrong, right, hard_dev = [], [], set()
    n_turn = n_ref = n_wrong = n_skip_lab = n_skip_wer = n_minor = 0
    fields = ["asr", "refine"] if a.hyp_field == "either" else [a.hyp_field]
    src = a.asr_out or greedy_asr_out(a)
    found = set()
    for item in read_jsonl(src):
        held = item["id"] in exclude if exclude is not None else is_heldout(item["id"])
        for t in item["turns"]:
            n_turn += 1
            ref = t.get("ref")
            if not ref:
                continue
            n_ref += 1
            hyps = [norm(t.get(f) or "") for f in fields]
            lab = label_of(ref, hyps[0])
            if not lab:
                n_skip_lab += 1
                continue
            errs = [utt_errors(lab.lower(), h) for h in hyps]
            e, n = max(errs)
            stem = Path(t["audio"]).stem
            listed = ids is not None and stem in ids
            found |= {stem} if listed else set()
            if held:  # held-out không bao giờ vào train; dev_hard = lượt held-out pipeline nhận sai
                if listed or (ids is None and e >= a.min_errors):
                    hard_dev.add(t["audio"])
                continue
            if listed:  # có trong --ids-file: luôn là câu sai, kể cả khi greedy tình cờ đúng
                kind = "wrong"
            elif e == 0:
                kind = "right"
            elif (stem in ids) if ids is not None else e >= a.min_errors:
                kind = "wrong"
            else:  # sai nhẹ (không có trong --ids-file): không học, không làm điểm neo
                n_minor += 1
                continue
            if kind == "wrong":
                n_wrong += 1
                if ids is None and e / max(n, 1) > a.max_wer:  # sai gần hết câu: thường nhãn/audio lệch (id tự chọn: tin người chọn)
                    n_skip_wer += 1
                    continue
            wav = root / t["audio"]
            if not wav.exists():
                continue
            if sp is not None and 2 in sp.encode(lab):  # từ ngoài vocab BPE (<unk>): không học được
                continue
            (wrong if kind == "wrong" else right).append(
                {"id": "new_" + stem, "audio": str(wav.resolve()), "text": lab, "hyp": hyps[0],
                 "wer": round(e / max(n, 1), 3), "sample": item["id"], "turn": t["turn"], "kind": kind})

    if ids is not None:
        miss = sorted(ids - found)
        log(f"--ids-file: tìm thấy {len(found)}/{len(ids)} id" + (f", không thấy {len(miss)} (vd {miss[:5]})" if miss else ""))

    def uniq(rows):  # cùng audio có thể lặp (history của nhiều sample)
        seen, out = set(), []
        for r in rows:
            if r["audio"] not in seen:
                seen.add(r["audio"])
                out.append(r)
        return out
    wrong, right = uniq(wrong), uniq(right)
    rng = random.Random(a.seed)
    rng.shuffle(wrong)
    rng.shuffle(right)
    n_w0, n_r0 = len(wrong), len(right)
    # khử mẫu câu trên cả hai tập cùng lúc (câu sai đứng trước -> được giữ trước), rồi lấy câu đúng theo tỉ lệ
    kept, n_tpl = template_cap(wrong + right, a.max_per_template)
    wrong = [r for r in kept if r["kind"] == "wrong"]
    right = [r for r in kept if r["kind"] == "right"][:int(round(a.correct_ratio * len(wrong)))]
    rows = wrong + right
    for r in rows:
        r["dur"] = round(duration(r["audio"]), 3)
    rows = [r for r in rows if 0.3 <= r["dur"] <= a.max_dur]
    log(f"asr_out: {n_turn} lượt, {n_ref} có ref | sai ở train-split: {n_wrong} "
        f"({'theo --ids-file' if ids is not None else f'>= {a.min_errors} lỗi, {a.hyp_field}'}), bỏ vì WER > {a.max_wer}: "
        f"{n_skip_wer}, sai nhẹ bỏ qua: {n_minor}, bỏ vì không đọc được số: {n_skip_lab} | held-out sai: {len(hard_dev)}")
    log(f"khử mẫu câu (<= {a.max_per_template}/mẫu, {n_tpl} mẫu): sai {n_w0} -> {len(wrong)}, đúng {n_r0} -> "
        f"{sum(r['kind'] == 'right' for r in rows)} (tỉ lệ đúng/sai {a.correct_ratio}) | sau lọc thời lượng: {len(rows)} câu")
    return rows, hard_dev


def parse_spec(s):
    name, rest = s.split("=", 1)
    src, split, col, w = rest.split("|")
    return {"name": name, "src": src, "split": split, "col": col, "w": float(w)}


def open_stream(spec, seed):
    from datasets import Audio, load_dataset
    if spec["src"].startswith("hf://"):
        ds = load_dataset("parquet", data_files={"train": spec["src"]}, split="train", streaming=True)
    else:
        repo, _, cfg = spec["src"].partition(":")
        ds = load_dataset(repo, cfg or None, split=spec["split"], streaming=True)
    try:
        ds = ds.cast_column("audio", Audio(decode=False))  # tự giải mã (soundfile) -> không phụ thuộc torchcodec
    except Exception:  # noqa: BLE001
        pass
    return ds.shuffle(seed=seed, buffer_size=256)


def decode_audio(a):
    import numpy as np
    import soundfile as sf
    if isinstance(a, dict) and "array" in a:
        x, sr = np.asarray(a["array"], dtype="float32"), a["sampling_rate"]
    else:
        b, p = (a.get("bytes"), a.get("path")) if isinstance(a, dict) else (a, None)
        try:
            x, sr = sf.read(io.BytesIO(b) if b else p, dtype="float32")
        except Exception:  # noqa: BLE001 — mp3/định dạng lạ
            import librosa
            x, sr = librosa.load(io.BytesIO(b) if b else p, sr=None, mono=True)
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        from scipy.signal import resample_poly
        g = math.gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g).astype("float32")
    return x


def download_source(spec, hours, out_dir, sp, a, prefix):
    """stream tới khi đủ `hours` giờ câu hợp lệ -> flac 16 kHz + rows."""
    import soundfile as sf
    d = Path(out_dir) / spec["name"]
    d.mkdir(parents=True, exist_ok=True)
    rows, tot, scanned = [], 0.0, 0
    t0 = time.time()
    for ex in open_stream(spec, a.seed):
        scanned += 1
        if scanned > a.max_scan:
            break
        text = ex.get(spec["col"]) or ""
        if re.search(r"[<\[(]", text):  # nhãn có chú thích (<unk>, [noise], (cười)...)
            continue
        lab = to_spoken(text)
        if not lab or not VI_TEXT.fullmatch(lab) or len(lab.split()) < 2:
            continue
        lab = new_style(" ".join(lab.upper().split()))
        if 2 in sp.encode(lab):  # <unk>
            continue
        try:
            x = decode_audio(ex["audio"])
        except Exception:  # noqa: BLE001
            continue
        dur = len(x) / SR
        if not (a.min_dur <= dur <= a.max_dur):
            continue
        i = f"{prefix}{spec['name']}_{len(rows):07d}"
        p = d / f"{i}.flac"
        sf.write(str(p), x, SR, subtype="PCM_16")
        rows.append({"id": i, "audio": str(p.resolve()), "text": lab, "dur": round(dur, 3), "src": spec["name"]})
        tot += dur
        if len(rows) % 500 == 0:
            log(f"  {spec['name']}: {len(rows)} câu, {tot / 3600:.2f}/{hours:.2f} h, {time.time() - t0:.0f}s")
        if tot >= hours * 3600:
            break
    log(f"  {spec['name']}: {len(rows)} câu, {tot / 3600:.2f} h (quét {scanned} dòng)")
    return rows


def probe(spec, a):
    try:
        next(iter(open_stream(spec, a.seed)))
        return True
    except Exception as e:  # noqa: BLE001
        msg = str(e).splitlines()[0][:200] if str(e) else type(e).__name__
        log(f"  ! bỏ nguồn {spec['name']} ({spec['src']}): {msg}"
            + ("  -> tập gated: vào trang dataset bấm chấp nhận điều khoản, đặt HF_TOKEN" if "gated" in msg.lower()
               or "401" in msg or "403" in msg else ""))
        return False


def build_replay(a, new_hours, sp):
    work = Path(a.work)
    out = work / "data" / "replay.jsonl"
    per_epoch_new = min(new_hours, a.epoch_hours * a.new_frac) if a.epoch_hours else new_hours
    target = per_epoch_new * (1 - a.new_frac) / a.new_frac * a.replay_pool_factor
    if out.exists():
        rows = read_jsonl(out)
        have = sum(r["dur"] for r in rows) / 3600
        if have >= 0.97 * target:
            log(f"replay: dùng lại {out} ({len(rows)} câu, {have:.2f} h >= cần {target:.2f} h)")
            return rows
        log(f"replay: {out} chỉ có {have:.2f} h < cần {target:.2f} h -> tải lại")
    specs = [parse_spec(s) for s in (a.replay or DEFAULT_REPLAY)]
    log(f"replay: cần {target:.2f} h (= {per_epoch_new:.2f} h mới/epoch x {(1 - a.new_frac) / a.new_frac:.0f} x pool "
        f"{a.replay_pool_factor}) | kiểm tra truy cập nguồn...")
    specs = [s for s in specs if probe(s, a)]
    if not specs:
        sys.exit("không truy cập được nguồn replay nào (bật Internet trên Kaggle? HF_TOKEN?)")
    wsum = sum(s["w"] for s in specs)
    rows = []
    for s in specs:
        rows += download_source(s, target * s["w"] / wsum, (Path(a.cache) if a.cache else work) / "replay", sp, a, "old_")
    short = target * 3600 - sum(r["dur"] for r in rows)
    if short > 60:  # nguồn hết dữ liệu / quét quá max_scan: lấy bù từ nguồn khác (seed khác, lọc trùng text)
        log(f"replay thiếu {short / 3600:.2f} h -> lấy bù")
        seen = {r["text"] for r in rows}
        a2 = argparse.Namespace(**{**vars(a), "seed": a.seed + 1})
        for s in specs:
            extra = [r for r in download_source({**s, "name": s["name"] + "_b"}, short / 3600 / len(specs),
                                                (Path(a.cache) if a.cache else work) / "replay", sp, a2, "old_") if r["text"] not in seen]
            rows += extra
    write_jsonl(out, rows)
    by = {}
    for r in rows:
        by[r["src"]] = by.get(r["src"], 0) + r["dur"] / 3600
    log("replay:", len(rows), "câu |", ", ".join(f"{k} {v:.2f} h" for k, v in by.items()))
    return rows


def stage_prepare(a):
    hf_token_from_kaggle()
    import sentencepiece as spm
    from asr.asr_zipformer import model_dir
    work = Path(a.work)
    (work / "data").mkdir(parents=True, exist_ok=True)
    md = model_dir(a.zipformer)
    sp = spm.SentencePieceProcessor(model_file=str(md / "bpe.model"))

    ids = load_ids(a)
    dev_rows, exclude = build_eval(a, ids)  # trước select_new: biết sample nào phải loại khỏi train
    write_jsonl(work / "data" / "dev_new.jsonl", dev_rows)
    new, _ = select_new(a, sp, ids, exclude)
    if not new:
        sys.exit("không có câu sai nào để fine-tune (kiểm tra --asr-out có trường ref / id trong --ids-file khớp tên "
                 "file audio, --root/--split đúng)")
    new_h = sum(r["dur"] for r in new) / 3600
    write_jsonl(work / "data" / "new.jsonl", new)
    log(f"new: {sum(r['kind'] == 'wrong' for r in new)} câu sai + {sum(r['kind'] == 'right' for r in new)} câu đúng (neo), "
        f"{new_h:.2f} h -> {work / 'data' / 'new.jsonl'}")
    for r in new[:5]:
        log(f"   ví dụ  HYP: {r['hyp']}\n               LAB: {r['text'].lower()}")

    old_dev_out = work / "data" / "dev_old.jsonl"
    if not old_dev_out.exists():
        s = parse_spec(a.old_dev)
        rows = download_source(s, 1e9, (Path(a.cache) if a.cache else work) / "replay", sp,
                               argparse.Namespace(**{**vars(a), "max_scan": a.old_dev_n * 3}), "devold_")[:a.old_dev_n] \
            if probe(s, a) else []
        write_jsonl(old_dev_out, rows)
    build_replay(a, new_h, sp)
    if not read_jsonl(old_dev_out):  # không có FLEURS test: tách một phần pool replay làm dev_old
        rows = read_jsonl(work / "data" / "replay.jsonl")
        random.Random(a.seed).shuffle(rows)
        write_jsonl(old_dev_out, rows[:a.old_dev_n])
        write_jsonl(work / "data" / "replay.jsonl", rows[a.old_dev_n:])
        log(f"dev_old: tách {a.old_dev_n} câu từ replay")
    json.dump({"new_frac": a.new_frac, "new_hours": new_h, "hyp_field": a.hyp_field, "zipformer": str(md)},
              open(work / "data" / "meta.json", "w"), indent=1)
    log("prepare xong")


# ================================================================ stage: train

LARGE_UNMASKED = {(192, 256, 512, 768, 512, 256): (192, 192, 256, 320, 256, 192),
                  (192, 256, 384, 512, 384, 256): (192, 192, 256, 256, 256, 192)}


def infer_arch(sd):
    """kiến trúc Zipformer2 suy ra từ shape tensor của jit_script.pt (đã đối chiếu với bản 150M: 2,2,4,5,4,2 /
    192,256,512,768,512,256). Giả định head dim mặc định icefall: query 32, pos 4, value 12."""
    qhd, phd, vhd = 32, 4, 12
    pre = "encoder.encoder.encoders."
    n_stack = 1 + max(int(m[1]) for k in sd if (m := re.match(r"encoder\.encoder\.encoders\.(\d+)\.", k)))
    L, D, FF, H, K, DS = [], [], [], [], [], []
    causal = False
    for i in range(n_stack):
        base = f"{pre}{i}.encoder.layers." if any(k.startswith(f"{pre}{i}.encoder.layers.") for k in sd) \
            else f"{pre}{i}.layers."
        L.append(1 + max(int(k[len(base):].split(".")[0]) for k in sd if k.startswith(base)))
        ff, d = sd[base + "0.feed_forward2.in_proj.weight"].shape
        FF.append(ff)
        D.append(d)
        nh = sd[base + "0.self_attn1.in_proj.weight"].shape[0] // vhd
        assert sd[base + "0.self_attn_weights.in_proj.weight"].shape[0] == (2 * qhd + phd) * nh, "head dim khác mặc định"
        H.append(nh)
        dw = base + "0.conv_module1.depthwise_conv.weight"
        if dw not in sd:
            causal = True
            dw = base + "0.conv_module1.depthwise_conv.chunkwise_conv.weight"
        K.append(sd[dw].shape[-1])
        dsb = f"{pre}{i}.downsample.bias"
        DS.append(sd[dsb].shape[0] if dsb in sd else 1)
    pos_dim = sd[f"{pre}0.layers.0.self_attn_weights.linear_pos.weight" if f"{pre}0.layers.0.self_attn_weights.linear_pos.weight"
                 in sd else f"{pre}0.encoder.layers.0.self_attn_weights.linear_pos.weight"].shape[1]
    um = LARGE_UNMASKED.get(tuple(D), tuple(min(d, 192) for d in D))
    j = ",".join
    vocab, dec_dim = sd["decoder.embedding.weight"].shape
    arch = dict(num_encoder_layers=j(map(str, L)), downsampling_factor=j(map(str, DS)),
                feedforward_dim=j(map(str, FF)), num_heads=j(map(str, H)), encoder_dim=j(map(str, D)),
                encoder_unmasked_dim=j(map(str, um)), cnn_module_kernel=j(map(str, K)), query_head_dim=str(qhd),
                value_head_dim=str(vhd), pos_head_dim=str(phd), pos_dim=pos_dim, decoder_dim=dec_dim,
                joiner_dim=sd["joiner.encoder_proj.weight"].shape[0],
                context_size=sd["decoder.conv.weight"].shape[-1] if "decoder.conv.weight" in sd else 1,
                causal=causal, chunk_size="16,32,64,-1", left_context_frames="64,128,256,-1")
    return arch, vocab, "ctc_output.1.weight" in sd


def stage_train(a):
    import numpy as np
    import sentencepiece as spm
    import soundfile as sf
    import torch
    from huggingface_hub import snapshot_download
    from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter, Recording, RecordingSet, SupervisionSegment, \
        SupervisionSet
    from lhotse.dataset import DynamicBucketingSampler, K2SpeechRecognitionDataset, SimpleCutSampler, SpecAugment
    from torch.utils.data import DataLoader

    icefall, zdir = icefall_paths(a.work)
    import k2
    from beam_search import greedy_search_batch
    from icefall.utils import create_grad_scaler, torch_autocast
    from optim import Eden, ScaledAdam
    from train import get_model, get_params, set_batch_count

    work = Path(a.work)
    data = work / "data"
    exp = work / "exp"
    exp.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp16 = dev.type == "cuda" and a.fp16
    if dev.type != "cuda":
        log("CẢNH BÁO: không có GPU — train 150M trên CPU rất chậm")

    # ---------------- model gốc từ TorchScript -> icefall AsrModel
    md = Path(a.zipformer) if Path(a.zipformer).is_dir() else \
        Path(snapshot_download(a.zipformer, allow_patterns=["*.onnx", "bpe.model", "jit_script.pt"]))
    if not (md / "jit_script.pt").exists():
        sys.exit(f"{md} thiếu jit_script.pt (cần để dựng lại model PyTorch)")
    sp = spm.SentencePieceProcessor(model_file=str(md / "bpe.model"))
    lang = work / "lang"
    lang.mkdir(exist_ok=True)
    shutil.copy(md / "bpe.model", lang / "bpe.model")
    (lang / "tokens.txt").write_text("".join(f"{sp.id_to_piece(i)} {i}\n" for i in range(sp.get_piece_size())),
                                     encoding="utf-8")
    (lang / "bpe.vocab").write_text("".join(f"{sp.id_to_piece(i)}\t{sp.get_score(i)}\n"
                                            for i in range(sp.get_piece_size())), encoding="utf-8")
    jit = torch.jit.load(str(md / "jit_script.pt"), map_location="cpu").eval()
    jsd = jit.state_dict()
    arch, vocab, has_ctc = infer_arch(jsd)
    log("kiến trúc:", arch, "| vocab", vocab, "| ctc head", has_ctc)
    params = get_params()
    params.update(arch)
    params.update(dict(vocab_size=vocab, blank_id=0, use_transducer=True, use_ctc=has_ctc,
                       use_attention_decoder=False, use_cr_ctc=False))
    model = get_model(params)
    sd = {}
    for k, v in jsd.items():
        if k.startswith("encoder.encoder_embed."):  # bản trùng của encoder_embed.* (EncoderModel lúc export)
            continue
        sd["encoder." + k[len("encoder.encoder."):] if k.startswith("encoder.encoder.") else k] = v
    if not any(k.startswith("encoder_embed.") for k in sd):
        sd.update({k[len("encoder."):]: v for k, v in jsd.items() if k.startswith("encoder.encoder_embed.")})
    r = model.load_state_dict(sd, strict=False)
    pnames = {n for n, _ in model.named_parameters()}
    miss = [k for k in r.missing_keys if k in pnames]
    assert not miss and not r.unexpected_keys, f"kiến trúc không khớp: missing {miss[:5]} unexpected {r.unexpected_keys[:5]}"
    log(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    fbank = Fbank(FbankConfig(num_mel_bins=80))
    dev_new_rows = read_jsonl(data / "dev_new.jsonl")
    x, srate = sf.read(dev_new_rows[0]["audio"], dtype="float32")
    feats = torch.from_numpy(fbank.extract(x, srate))[None]
    lens = torch.tensor([feats.shape[1]])
    set_batch_count(model, 100000.0)
    model.eval()
    with torch.no_grad():
        o1, _ = jit.encoder(feats, lens)
        o2, _ = model.forward_encoder(feats, lens)
    diff = (o1 - o2).abs().max().item()
    log(f"encoder icefall vs jit: max|diff| = {diff:.2e}")
    assert o1.shape == o2.shape and diff < 1e-3, "output lệch bản jit -> kiến trúc suy ra sai, dừng"
    del jit, jsd, sd
    # trọng số gốc (CPU) cho bước nội suy WiSE-FT cuối: theta = alpha * theta_ft + (1 - alpha) * theta_goc
    base_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # ---------------- cuts + fbank (cache trong work/feats)
    def make_cuts(rows):
        recs, sups = [], []
        for r in rows:
            rec = Recording.from_file(r["audio"], recording_id=r["id"])
            if rec.sampling_rate != SR:
                rec = rec.resample(SR)
            recs.append(rec)
            sups.append(SupervisionSegment(id=r["id"], recording_id=r["id"], start=0.0, duration=rec.duration,
                                           channel=0, text=r["text"], language="vi"))
        return CutSet.from_manifests(recordings=RecordingSet.from_recordings(recs),
                                     supervisions=SupervisionSet.from_segments(sups)).to_eager()

    cache = Path(a.cache) if a.cache else work

    def with_feats(cuts, name):
        out = cache / "feats" / f"cuts_{name}.jsonl.gz"
        if out.exists():
            c = CutSet.from_file(out).to_eager()
            if len(c) == len(cuts):
                return c
        out.parent.mkdir(parents=True, exist_ok=True)
        log(f"fbank {name}: {len(cuts)} cuts ...")
        c = cuts.compute_and_store_features(extractor=fbank, storage_path=str(cache / "feats" / name),
                                            num_jobs=max(1, (os.cpu_count() or 2) // 2),
                                            storage_type=LilcomChunkyWriter).to_eager()
        c.to_file(out)
        return c

    new_rows = read_jsonl(data / "new.jsonl")
    old_rows = read_jsonl(data / "replay.jsonl")
    dev_old_rows = read_jsonl(data / "dev_old.jsonl")
    speeds = [float(s) for s in a.speed_new.split(",")] if a.speed_new else [1.0]
    base_new = make_cuts(new_rows)
    variants = [c if f == 1.0 else c.perturb_speed(f) for f in speeds for c in base_new]
    new_cuts = with_feats(CutSet.from_cuts(variants), "new_" + "_".join(map(str, speeds)))
    old_cuts = with_feats(make_cuts(old_rows), "replay")
    dev_new_cuts = with_feats(make_cuts(dev_new_rows), "dev_new")
    dev_old_cuts = with_feats(make_cuts(dev_old_rows), "dev_old") if dev_old_rows else None
    hard_ids = {r["id"] for r in dev_new_rows if r.get("hard")}
    groups = {}  # câu gốc -> các bản speed (lhotse perturb_speed thêm hậu tố "_sp0.9" vào id)
    for c in new_cuts:
        groups.setdefault(re.sub(r"_sp[\d.]+$", "", c.id), []).append(c)
    old_pool = list(old_cuts)
    new_h = sum(c.duration for c in base_new) / 3600
    old_h = sum(c.duration for c in old_pool) / 3600
    want_old_h = (min(new_h, a.epoch_hours * a.new_frac) if a.epoch_hours else new_h) * (1 - a.new_frac) / a.new_frac
    log(f"new {len(groups)} câu {new_h:.2f} h (x{len(speeds)} speed) | replay pool {len(old_pool)} câu {old_h:.2f} h, "
        f"mỗi epoch {want_old_h:.2f} h | dev_new {len(dev_new_cuts)} (hard {len(hard_ids)}) | "
        f"dev_old {len(dev_old_cuts) if dev_old_cuts else 0}")
    if old_h < 0.97 * want_old_h:
        log(f"CẢNH BÁO: pool replay ({old_h:.2f} h) < cần ({want_old_h:.2f} h) -> dữ liệu mới sẽ chiếm "
            f"{new_h / (new_h + old_h):.1%} thay vì {a.new_frac:.0%}")

    order = sorted(groups)
    random.Random(a.seed).shuffle(order)

    def epoch_cuts(epoch):
        rng = random.Random(a.seed * 1000 + epoch)
        if a.epoch_hours:  # chỉ lấy một cửa sổ câu mới mỗi epoch, xoay vòng -> qua vài epoch đủ mọi câu
            want_new, picked, tot_new = a.epoch_hours * a.new_frac * 3600, [], 0.0
            start = ((epoch - 1) * int(len(order) * min(1.0, want_new / max(new_h * 3600, 1)))) % len(order)
            for k in range(len(order)):
                if tot_new >= want_new:
                    break
                vs = groups[order[(start + k) % len(order)]]
                picked.append(vs[epoch % len(vs)])
                tot_new += picked[-1].duration
            new = picked
        else:
            new = [vs[epoch % len(vs)] for vs in (groups[o] for o in order)]
        want_old = sum(c.duration for c in new) * (1 - a.new_frac) / a.new_frac
        pool = old_pool[:]
        rng.shuffle(pool)
        old, tot = [], 0.0
        for c in pool:
            if tot >= want_old:
                break
            old.append(c)
            tot += c.duration
        nh = sum(c.duration for c in new) / 3600
        return CutSet.from_cuts(new + old).filter(lambda c: a.min_dur <= c.duration <= a.max_dur).to_eager(), nh, tot / 3600

    # ---------------- decode + WER (không phân biệt vị trí dấu thanh)
    @torch.no_grad()
    def decode(cuts):
        model.eval()
        dl = DataLoader(K2SpeechRecognitionDataset(return_cuts=True),
                        sampler=SimpleCutSampler(cuts, max_duration=a.max_duration * 2), batch_size=None,
                        num_workers=a.num_workers)
        out = {}
        for b in dl:
            xx, xl = b["inputs"].to(dev), b["supervisions"]["num_frames"].to(dev)
            with torch_autocast(enabled=fp16):
                enc, encl = model.forward_encoder(xx, xl)
            hyps = greedy_search_batch(model=model, encoder_out=enc.float(), encoder_out_lens=encl)
            for c, h in zip(b["supervisions"]["cut"], hyps):
                out[c.supervisions[0].id] = (c.supervisions[0].text.lower(), sp.decode(h).lower())
        return out

    def wer_of(pairs):
        e = n = 0
        for ref, hyp in pairs:
            de, dn = utt_errors(ref, hyp)
            e, n = e + de, n + dn
        return e / max(n, 1)

    def evaluate():
        hn = decode(dev_new_cuts)
        r = {"new": wer_of(hn.values()), "hard": wer_of(v for k, v in hn.items() if k in hard_ids) if hard_ids else float("nan")}
        r["old"] = wer_of(decode(dev_old_cuts).values()) if dev_old_cuts else float("nan")
        r["mean"] = (r["new"] + r["old"]) / 2 if dev_old_cuts else r["new"]
        # dev_keep: câu held-out model GỐC đã nhận đúng -> đếm câu bị làm hỏng; dev_fix: câu model gốc sai -> đếm câu sửa được
        ks, fs = state.get("keep_ids") or [], state.get("fix_ids") or []
        if ks:
            r["keep_broken"] = sum(utt_errors(*hn[k])[0] > 0 for k in ks if k in hn)
            r["keep"] = wer_of(hn[k] for k in ks if k in hn)
        if fs:
            r["fix_fixed"] = sum(utt_errors(*hn[k])[0] == 0 for k in fs if k in hn)
            r["fix"] = wer_of(hn[k] for k in fs if k in hn)
        return r, hn

    def fmt(r):
        s = f"dev_new {r['new']:.4f} | dev_hard {r['hard']:.4f} | dev_old {r['old']:.4f} | mean {r['mean']:.4f}"
        if "keep_broken" in r:
            s += f" | dev_keep hỏng {r['keep_broken']}/{len(state['keep_ids'])} (WER {r['keep']:.4f})"
        if "fix_fixed" in r:
            s += f" | dev_fix sửa được {r['fix_fixed']}/{len(state['fix_ids'])} (WER {r['fix']:.4f})"
        return s
    model.to(dev)

    # ---------------- optimizer / resume
    opt = ScaledAdam(get_parameter_groups(model, a.lr), lr=a.lr, clipping_scale=2.0)
    sch = Eden(opt, a.lr_batches, a.lr_epochs, warmup_batches=a.warmup_batches)
    scaler = create_grad_scaler(enabled=fp16, init_scale=1.0)
    state = {"epoch": 0, "step": 0, "history": [], "base": None}
    last = exp / "last.pt"
    if a.resume and last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sch.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        state = ck["state"]
        log(f"resume từ epoch {state['epoch']}, step {state['step']}")
    if state["base"] is None and any("group" in r for r in dev_new_rows):
        # bộ đánh giá cố định từ prepare (eval_ids.txt): keep / fix đã chọn sẵn, không chọn lại
        state["keep_ids"] = sorted(r["id"] for r in dev_new_rows if r.get("group") == "keep")
        state["fix_ids"] = sorted(r["id"] for r in dev_new_rows if r.get("group") == "fix")
        state["base"], _ = evaluate()
        log("MODEL GỐC   ", fmt(state["base"]))
    elif state["base"] is None:
        base_r, base_h = evaluate()
        # chọn dev_keep / dev_fix theo đúng cách giải mã dùng ở mỗi epoch (greedy, model gốc), cố định cho cả lần train
        rng = random.Random(a.seed)
        ok = sorted(k for k, v in base_h.items() if utt_errors(*v)[0] == 0)
        wrong = sorted(k for k, v in base_h.items() if utt_errors(*v)[0] > 0)
        rng.shuffle(ok)
        rng.shuffle(wrong)
        wrong.sort(key=lambda k: k not in hard_ids)  # ưu tiên câu pipeline cũng nhận sai
        state["keep_ids"], state["fix_ids"] = ok[:a.dev_keep_n], wrong[:a.dev_fix_n]
        if len(state["keep_ids"]) < a.dev_keep_n:
            log(f"CẢNH BÁO: chỉ có {len(ok)} câu dev_new model gốc nhận đúng (< --dev-keep-n {a.dev_keep_n})")
        base_r, _ = evaluate()  # tính lại để base có keep/fix (keep_broken = 0 theo định nghĩa)
        state["base"] = base_r
        log("MODEL GỐC   ", fmt(state["base"]))
    elif "keep_ids" not in state:
        log("CẢNH BÁO: last.pt cũ chưa có dev_keep/dev_fix -> tắt điều kiện dừng theo dev_keep")
    base = state["base"]

    def keep_ok(h):
        # so với chính model gốc (greedy PyTorch có thể lệch vài câu so với sherpa int8 lúc chọn keep)
        return h.get("keep_broken", 0) <= base.get("keep_broken", 0) + a.keep_max_broken

    def eligible(h):
        return keep_ok(h) and not (h["old"] == h["old"] and base["old"] == base["old"]
                                   and h["old"] > base["old"] + a.max_old_degrade)

    def ranked():
        return sorted([h for h in state["history"] if eligible(h)], key=lambda h: h[a.select])

    train_tf = [SpecAugment(time_warp_factor=80, num_frame_masks=10, features_mask_size=27, num_feature_masks=2,
                            frames_mask_size=100)]
    if a.feat_aug_p > 0:
        train_tf.insert(0, FeatNoiseGain(a.feat_aug_p))
    bad = 0
    for epoch in range(state["epoch"] + 1, a.epochs + 1):
        cuts, nh, oh = epoch_cuts(epoch)
        log(f"epoch {epoch}: {len(cuts)} cuts, mới {nh:.2f} h + cũ {oh:.2f} h -> mới chiếm {nh / (nh + oh):.1%}")
        sampler = DynamicBucketingSampler(cuts, max_duration=a.max_duration, shuffle=True, num_buckets=20,
                                          drop_last=True, seed=a.seed + epoch)
        dl = DataLoader(K2SpeechRecognitionDataset(return_cuts=True, input_transforms=train_tf), sampler=sampler,
                        batch_size=None, num_workers=a.num_workers)
        sch.step_epoch(epoch - 1)
        model.train()
        t0, tot_loss, tot_frames, skipped, applied, inf_skip = time.time(), 0.0, 0, 0, 0, 0
        for bi, b in enumerate(dl):
            state["step"] += 1
            step = state["step"]
            set_batch_count(model, 100000.0 + step * a.max_duration / 600)  # ScheduledFloat ở giá trị cuối (như finetune.py)
            xx, xl = b["inputs"].to(dev), b["supervisions"]["num_frames"].to(dev)
            y = k2.RaggedTensor(sp.encode(b["supervisions"]["text"], out_type=int))
            with torch_autocast(enabled=fp16):
                simple, pruned, ctc, _, _ = model(x=xx, x_lens=xl, y=y, prune_range=5, am_scale=0.0, lm_scale=0.25)
                loss = 0.5 * simple + pruned + (a.ctc_scale * ctc if has_ctc and torch.is_tensor(ctc) else 0.0)
            if not torch.isfinite(loss):
                skipped += 1
                opt.zero_grad()
                continue
            scaler.scale(loss).backward()
            sch.step_batch(step)
            s0 = scaler.get_scale() if fp16 else 1.0
            scaler.step(opt)  # fp16: grad inf/NaN -> GradScaler LẶNG LẼ bỏ bước này, chỉ hạ scale
            scaler.update()
            opt.zero_grad()
            if fp16 and scaler.get_scale() < s0:
                inf_skip += 1
            else:
                applied += 1
            if fp16 and bi + 1 == a.fp16_check and applied == 0:
                log(f"CẢNH BÁO: {a.fp16_check} batch đầu đều bị GradScaler bỏ qua (grad fp16 inf/NaN, scale "
                    f"{scaler.get_scale():.2e}) -> model không học. Chuyển sang fp32 cho phần còn lại.")
                fp16 = False
                scaler = create_grad_scaler(enabled=False, init_scale=1.0)
            if fp16 and step % 100 == 0:  # như icefall: đừng để grad scale tụt quá thấp
                s = scaler._scale.item()
                if s < 8.0 or (s < 32.0 and step % 400 == 0):
                    scaler.update(s * 2.0)
            fr = (xl // 4).sum().item()
            tot_loss += loss.item()
            tot_frames += fr
            if bi % a.log_every == 0:
                log(f"  ep {epoch} batch {bi} | loss/frame {loss.item() / max(fr, 1):.4f} | "
                    f"lr {sch.get_last_lr()[0]:.2e} | cập nhật {applied}, bị bỏ (grad inf) {inf_skip}"
                    + (f", scale {scaler.get_scale():.1e}" if fp16 else "") + f" | {time.time() - t0:.0f}s")
        with torch.no_grad():  # weights đã đi xa model gốc bao nhiêu (0 = không học gì)
            num = den = 0.0
            for k, v in model.state_dict().items():
                if v.is_floating_point() and k in base_sd and k in pnames:
                    b0 = base_sd[k].to(v.device, torch.float32)
                    num += (v.float() - b0).pow(2).sum().item()
                    den += b0.pow(2).sum().item()
        drift = (num / max(den, 1e-12)) ** 0.5
        log(f"epoch {epoch}: {applied} bước cập nhật thật, {inf_skip} bước bị GradScaler bỏ, {skipped} batch loss NaN | "
            f"weights lệch model gốc {drift:.2e} (tương đối)")
        if applied == 0 or drift < 1e-6:
            log("CẢNH BÁO: model KHÔNG thay đổi trong epoch này -> kết quả eval sẽ y hệt model gốc. "
                "Thử --no-fp16 hoặc tăng --lr")
        rd, hn = evaluate()
        rd.update(epoch=epoch, loss=tot_loss / max(tot_frames, 1), drift=drift, applied=applied)
        broken = [k for k in state.get("keep_ids") or [] if k in hn and utt_errors(*hn[k])[0] > 0]
        for k in broken[:a.show_broken]:  # xem model làm hỏng câu đúng kiểu gì (nhãn lệch? đổi phương ngữ? bỏ từ?)
            log(f"  keep hỏng {k}\n      REF: {hn[k][0]}\n      HYP: {hn[k][1]}")
        state["history"].append(rd)
        state["epoch"] = epoch
        torch.save({"model": model.state_dict()}, exp / f"epoch-{epoch}.pt")
        log(f"epoch {epoch} xong | loss/frame {rd['loss']:.4f} | {fmt(rd)} | {time.time() - t0:.0f}s"
            + (f" | bỏ {skipped} batch NaN" if skipped else "")
            + ("" if keep_ok(rd) else f" | dev_keep hỏng > {a.keep_max_broken} câu")
            + ("" if eligible(rd) or not keep_ok(rd) else " | dev_old tệ quá ngưỡng"))
        keep = {h["epoch"] for h in ranked()[:a.avg_top_k]} | {epoch} | \
            {min(state["history"], key=lambda h: (h.get("keep_broken", 0), h[a.select]))["epoch"]}  # cho WiSE-FT dự phòng
        for h in state["history"]:  # chỉ giữ top-K checkpoint (đỡ tốn ổ /kaggle/working)
            p = exp / f"epoch-{h['epoch']}.pt"
            if h["epoch"] not in keep and p.exists():
                p.unlink()
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sch.state_dict(),
                    "scaler": scaler.state_dict(), "state": state}, last)
        # dev_keep hỏng quá ngưỡng k epoch liên tiếp -> đang học vẹt câu sai, phá câu vốn đúng: dừng
        broke = 0
        for h in reversed(state["history"]):
            if keep_ok(h):
                break
            broke += 1
        if broke >= a.keep_patience:
            log(f"dừng: dev_keep hỏng > {a.keep_max_broken} câu {broke} epoch liên tiếp")
            break
        best = ranked()[:1]
        if best and best[0]["epoch"] == epoch:
            bad = 0
        else:
            bad += 1
            if bad >= a.early_stop:
                log("early stop")
                break

    # ---------------- chọn model cuối: epoch tốt nhất / trung bình top-K, so với model gốc
    top = [h["epoch"] for h in ranked()[:a.avg_top_k] if (exp / f"epoch-{h['epoch']}.pt").exists()]
    only_wise = False
    if not top:  # không epoch nào đạt: vẫn thử WiSE-FT (nội suy về model gốc) với epoch ít làm hỏng dev_keep nhất
        have = [h for h in state["history"] if (exp / f"epoch-{h['epoch']}.pt").exists()]
        if not have or not a.wise_alphas:
            log("Không epoch nào đạt điều kiện (dev_old tệ quá ngưỡng hoặc làm hỏng dev_keep) -> giữ model gốc, không export")
            return
        top = [min(have, key=lambda h: (h.get("keep_broken", 0), h[a.select]))["epoch"]]
        only_wise = True
        log(f"Không epoch nào đạt điều kiện -> thử WiSE-FT trên epoch {top[0]} (α {a.wise_alphas})")
    load = lambda e: torch.load(exp / f"epoch-{e}.pt", map_location="cpu", weights_only=False)["model"]
    cands = {f"epoch {top[0]}": load(top[0])}
    if len(top) > 1:
        sds = [load(e) for e in top]
        cands[f"avg {sorted(top)}"] = {k: (sum(s[k].double() for s in sds) / len(sds)).to(sds[0][k].dtype)
                                       if sds[0][k].is_floating_point() else sds[0][k] for k in sds[0]}
    # WiSE-FT: nội suy từng ứng viên với model gốc; alpha nhỏ = gần model gốc (ít quên, ít lợi)
    alphas = sorted({float(x) for x in a.wise_alphas.split(",") if x.strip()} - {1.0}) if a.wise_alphas else []

    def interp(s, al):
        return {k: (al * v.double() + (1 - al) * base_sd[k].double()).to(v.dtype)
                if v.is_floating_point() and k in base_sd else v for k, v in s.items()}
    for name in list(cands):
        for al in alphas:
            cands[f"{name} α={al:g}"] = interp(cands[name], al)
    results = {}
    for name, s in cands.items():
        model.load_state_dict(s)
        results[name] = evaluate()[0]
        log(f"{name:24s} {fmt(results[name])}" + ("" if eligible(results[name]) else " | không đạt dev_keep/dev_old"))
    if only_wise:
        results.pop(f"epoch {top[0]}", None)  # bản gốc của epoch đã biết là không đạt
    final = min(results, key=lambda n: (not eligible(results[n]), results[n][a.select]))  # bản avg cũng phải qua dev_keep
    log(f"MODEL GỐC   {fmt(base)}\nCHỌN {final}: {fmt(results[final])}")
    if not eligible(results[final]):
        log("Không bản nào (kể cả WiSE-FT) qua dev_keep/dev_old -> giữ model gốc, không export")
        return
    if results[final][a.select] >= base[a.select]:
        log(f"CẢNH BÁO: model fine-tune không tốt hơn model gốc theo '{a.select}' — vẫn export để so, cân nhắc không dùng")
    torch.save({"model": cands[final]}, exp / "epoch-99.pt")  # export-onnx.py --epoch 99 --avg 1

    # ---------------- export ONNX (fp32 + int8) cho sherpa-onnx -> --zipformer <MODEL_OUT>
    flags = []
    for k, v in arch.items():
        flags += [f"--{k.replace('_', '-')}", str(v)]
    cmd = [sys.executable, str(zdir / "export-onnx.py"), "--tokens", str(lang / "tokens.txt"), "--exp-dir", str(exp),
           "--epoch", "99", "--avg", "1", "--use-averaged-model", "0", "--enable-int8-quantization", "1",
           "--use-transducer", "1", "--use-ctc", str(int(has_ctc)), *flags]
    log(" ".join(cmd))
    subprocess.run(cmd, cwd=zdir, check=True, env={**os.environ, "PYTHONPATH": f"{icefall}:{zdir}"})
    out = Path(a.model_out) if a.model_out else work / "zipformer-ft"
    out.mkdir(parents=True, exist_ok=True)
    for f in exp.glob("*-epoch-99-avg-1*.onnx"):
        shutil.copy(f, out / f.name)
    for f in ("tokens.txt", "bpe.model", "bpe.vocab"):
        shutil.copy(lang / f, out / f)
    log("MODEL_OUT =", out, sorted(p.name for p in out.iterdir()))

    # ---------------- kiểm tra bản ONNX bằng sherpa-onnx (greedy) trên dev_new, so với model gốc
    import sherpa_onnx

    def sherpa_wer(d, rows, int8):
        g = lambda n: str(next(p for p in sorted(Path(d).glob(f"{n}-*.onnx")) if p.name.endswith(".int8.onnx") == int8))
        rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=g("encoder"), decoder=g("decoder"), joiner=g("joiner"), tokens=str(lang / "tokens.txt"),
            num_threads=4, sample_rate=SR, feature_dim=80, decoding_method="greedy_search")
        pairs = []
        for k in range(0, len(rows), 16):
            ss = []
            for r in rows[k:k + 16]:
                x, srate = sf.read(r["audio"], dtype="float32")
                s = rec.create_stream()
                s.accept_waveform(srate, np.ascontiguousarray(x.mean(1) if x.ndim > 1 else x))
                ss.append(s)
            rec.decode_streams(ss)
            pairs += [(r["text"].lower(), s.result.text.strip().lower()) for r, s in zip(rows[k:k + 16], ss)]
        return wer_of(pairs)

    for name, d in (("gốc", md), ("fine-tuned", out)):
        for q in (False, True):
            log(f"sherpa {name:10s} {'int8' if q else 'fp32'}: dev_new {sherpa_wer(d, dev_new_rows, q):.4f}"
                + (f" | dev_old {sherpa_wer(d, dev_old_rows, q):.4f}" if dev_old_rows else ""))
    log(f"xong. Dùng: python pipeline.py ... --zipformer {out}")


def get_parameter_groups(model, lr):
    from icefall.utils import get_parameter_groups_with_lrs
    return get_parameter_groups_with_lrs(model, lr=lr, include_names=True)


def FeatNoiseGain(p):  # noqa: N802 — tạo lớp lúc cần (torch chỉ import trong stage train)
    import torch

    class _F(torch.nn.Module):
        """augment trên log-mel đã tính sẵn: gain ±6 dB và nhiễu SNR 10–30 dB (như notebook fine-tune 30M)."""

        def __init__(self, p, gain_db=6.0, snr_db=(10.0, 30.0)):
            super().__init__()
            self.p, self.gain_db, self.snr_db = p, gain_db, snr_db

        def forward(self, x, *args, **kwargs):
            x = x.clone()
            for i in range(x.shape[0]):
                if random.random() >= self.p:
                    continue
                f = x[i]
                valid = f.mean(-1) > -20.0  # bỏ frame padding
                p_sig = f[valid].exp().mean() if valid.any() else f.exp().mean()
                noise = p_sig * 10 ** (-random.uniform(*self.snr_db) / 10) * 2 * torch.rand_like(f)
                g = random.uniform(-self.gain_db, self.gain_db) * math.log(10) / 10
                x[i] = torch.where(valid[:, None], torch.log(f.exp() + noise) + g, f)
            return x

    return _F(p)


# ================================================================ main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["setup", "prepare", "train", "all"])
    ap.add_argument("--work", default="/kaggle/working/zf150", help="thư mục làm việc: icefall, data, feats, exp, model")
    ap.add_argument("--zipformer", default=HF_REPO, help="HF id hoặc thư mục model gốc (cần jit_script.pt để train)")
    ap.add_argument("--model-out", help="thư mục ONNX cuối (mặc định WORK/zipformer-ft)")
    ap.add_argument("--seed", type=int, default=0)
    g = ap.add_argument_group("prepare: dữ liệu mới (câu pipeline nhận sai)")
    g.add_argument("--root", help="thư mục chứa các split của BTC (như pipeline.py --root)")
    g.add_argument("--split", default="train")
    g.add_argument("--asr-out", help="output của pipeline.py --no-llm [--prefetch]. Bỏ trống khi có --ids-file: lấy câu từ "
                                     "train.jsonl, hyp = greedy model gốc (chỉ để chọn cách đọc số + câu đúng làm neo)")
    g.add_argument("--hyp-field", choices=["asr", "refine", "either"], default="asr",
                   help="so ref với: asr = ASR sau prefetch/decode lại (lỗi của chính model âm học, mặc định); "
                        "refine = sau LLM sửa (chỉ câu cả pipeline vẫn sai); either = sai ở một trong hai")
    g.add_argument("--min-errors", type=int, default=1, help="số từ sai tối thiểu để coi là câu sai")
    g.add_argument("--ids-file", help="txt mỗi dòng 1 id lượt (tên file audio không đuôi, vd. train_0008_u2): chỉ các lượt "
                                      "này là câu sai để học (vd. data/asr_sai_nghia_ids.txt); thay cho --min-errors")
    g.add_argument("--correct-ratio", type=float, default=1.0,
                   help="thêm câu ASR đã ĐÚNG của train-split = tỉ lệ này x số câu sai (điểm neo chống học vẹt); 0 = tắt")
    g.add_argument("--max-per-template", type=int, default=3,
                   help="tối đa bao nhiêu câu cùng mẫu (Jaccard tập từ >= 0.6) trong dữ liệu mới; 0 = không khử")
    g.add_argument("--max-wer", type=float, default=0.6, help="bỏ câu WER cao hơn (thường do nhãn/audio lệch)")
    g.add_argument("--eval-fix-n", type=int, default=100, help="số câu ĐANG SAI (held-out) trong bộ đánh giá cố định")
    g.add_argument("--eval-keep-n", type=int, default=100,
                   help="số câu held-out model gốc nhận ĐÚNG trong bộ đánh giá cố định (canh việc làm hỏng câu đúng)")
    g.add_argument("--eval-ids", help="file bộ đánh giá cố định (dòng 'fix|keep <id>'). Đã có -> dùng lại; chưa có -> chọn "
                                      "rồi ghi vào đây (mặc định WORK/data/eval_ids.txt)")
    g = ap.add_argument_group("prepare: dữ liệu cũ (replay chống quên)")
    g.add_argument("--new-frac", type=float, default=0.05, help="tỉ lệ dữ liệu mới theo thời lượng (0.05 = 5%%)")
    g.add_argument("--replay", action="append",
                   help="nguồn replay NAME=REPO[:CONFIG]|SPLIT|CỘT_TEXT|TRỌNG_SỐ hoặc NAME=hf://...parquet|train|CỘT|W; "
                        "lặp lại được, mặc định: " + "; ".join(DEFAULT_REPLAY))
    g.add_argument("--replay-pool-factor", type=float, default=1.0,
                   help="pool replay = factor x 19 x giờ mới/epoch; mỗi epoch bốc lại 19 x giờ mới từ pool (>1: đa dạng "
                        "hơn, tốn ổ hơn)")
    g.add_argument("--epoch-hours", type=float, default=20,
                   help="giới hạn tổng giờ audio mỗi epoch (vẫn giữ --new-frac): mỗi epoch lấy một phần câu mới, xoay "
                        "vòng qua các epoch. 0 = mọi câu mới mỗi epoch (4 h mới -> ~80 h/epoch). Kaggle T4: 15–25")
    g.add_argument("--cache", help="thư mục chứa audio replay + fbank (lớn; mặc định WORK). Kaggle: /kaggle/tmp nếu "
                               "/kaggle/working (20 GB) không đủ — dữ liệu ở đó mất khi hết phiên")
    g.add_argument("--old-dev", default=DEFAULT_OLD_DEV, help="tập đo quên (cùng cú pháp --replay)")
    g.add_argument("--old-dev-n", type=int, default=300)
    g.add_argument("--max-scan", type=int, default=400000, help="số dòng tối đa quét mỗi nguồn replay")
    g.add_argument("--min-dur", type=float, default=1.0)
    g.add_argument("--max-dur", type=float, default=20.0)
    g = ap.add_argument_group("train")
    g.add_argument("--epochs", type=int, default=8)
    g.add_argument("--lr", type=float, default=0.001, help="base lr ScaledAdam (model 150M + ít dữ liệu: thấp)")
    g.add_argument("--lr-epochs", type=float, default=3.5)
    g.add_argument("--lr-batches", type=float, default=5000.0)
    g.add_argument("--warmup-batches", type=float, default=200.0)
    g.add_argument("--max-duration", type=float, default=200.0, help="giây audio / batch (T4 16GB + 150M: 150–250)")
    g.add_argument("--num-workers", type=int, default=2)
    g.add_argument("--ctc-scale", type=float, default=0.1, help="giữ đầu CTC của model gốc cùng học (0 = bỏ)")
    g.add_argument("--speed-new", default="1.0,0.9,1.1", help="speed perturb dữ liệu mới, mỗi epoch 1 mức (luân phiên)")
    g.add_argument("--feat-aug-p", type=float, default=0.3, help="xác suất nhiễu + gain trên fbank mỗi câu")
    g.add_argument("--no-fp16", dest="fp16", action="store_false")
    g.add_argument("--fp16-check", type=int, default=50,
                   help="nếu ngần này batch đầu mỗi epoch đều bị GradScaler bỏ (grad inf) -> tự chuyển fp32")
    g.add_argument("--select", choices=["mean", "new", "hard", "old"], default="mean", help="chọn epoch theo WER này")
    g.add_argument("--max-old-degrade", type=float, default=0.01,
                   help="loại epoch có WER dev_old > model gốc + ngưỡng (tuyệt đối, 0.01 = 1 điểm WER)")
    g.add_argument("--early-stop", type=int, default=2)
    g.add_argument("--dev-keep-n", type=int, default=100,
                   help="số câu dev_new (held-out) model gốc nhận ĐÚNG, dùng canh việc phá câu vốn đúng")
    g.add_argument("--dev-fix-n", type=int, default=10, help="số câu dev_new model gốc nhận SAI, xem có sửa được không")
    g.add_argument("--keep-max-broken", type=int, default=2,
                   help="epoch làm hỏng > ngần này câu dev_keep thì không được chọn (100 câu: 2 = 2%%)")
    g.add_argument("--keep-patience", type=int, default=2,
                   help="dừng khi dev_keep hỏng quá ngưỡng k epoch liên tiếp")
    g.add_argument("--avg-top-k", type=int, default=2)
    g.add_argument("--show-broken", type=int, default=8, help="mỗi epoch in tối đa ngần này câu dev_keep bị làm hỏng")
    g.add_argument("--wise-alphas", default="0.3,0.5,0.7",
                   help="WiSE-FT: thử nội suy model chọn được với model gốc ở các alpha này (1.0 = bản fine-tune luôn được thử); "
                        "chọn bản tốt nhất theo --select trong số bản qua dev_keep/dev_old; rỗng = tắt")
    g.add_argument("--log-every", type=int, default=50)
    g.add_argument("--resume", action="store_true", help="train tiếp từ WORK/exp/last.pt (phiên Kaggle bị ngắt)")
    a = ap.parse_args()
    if not 0 < a.new_frac < 1:
        sys.exit("--new-frac phải trong (0, 1)")
    if a.stage in ("setup", "all"):
        stage_setup(a)
    if a.stage in ("prepare", "all"):
        if not a.root:
            sys.exit("prepare cần --root")
        if not a.asr_out and not a.ids_file:
            sys.exit("prepare cần --asr-out (output pipeline) hoặc --ids-file (txt id các câu sai)")
        stage_prepare(a)
    if a.stage in ("train", "all"):
        stage_train(a)


if __name__ == "__main__":
    main()
