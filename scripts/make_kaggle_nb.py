"""Sinh notebooks/kaggle.ipynb: chạy tự động toàn bộ thí nghiệm (Kaggle batch, không cần chỉnh tay)."""
import json
import sys

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--stage", default="all", choices=["all", "asr", "nollm"])
A = ap.parse_args()
out, smoke = A.out, A.smoke

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)})


md("""
# LifeMate trên Kaggle: ASR (không agent gọi tool)

Settings: **GPU T4 x2**, **Internet on**, Input: dataset BTC (`train/`, `public_test/`).
Chạy được bằng *Save & Run All* (không cần chỉnh tay): mọi số đo trên **held-out** (id % 10 == 9).

1. Dump ASR train, train RNNLM, so WER mọi cấu hình decode (greedy / beam / hotword / VAD / RNNLM nhiều scale / beam 8).
2. `run.py --no-llm` với cấu hình thắng: hotword cố định / + bias / + prefetch, có và không có mapper -> WER.
Tổng kết ở `results.json`, log từng bước ở `logs/`.

`SMOKE = True`: chạy thử vài mẫu mỗi bước để bắt lỗi môi trường (vài chục phút).
""")

code(f"""
SMOKE = {smoke}
# all = bước 1 rồi bước 2 | asr = chỉ bước 1 | nollm = chỉ bước 2 với ASR_DEFAULT (không cần RNNLM)
STAGE = {A.stage!r}
ASR_DEFAULT = "--beam 4 --hotwords-file assets/hotwords_global.txt --hotword-score 1.5"
BRANCH = 'refactor/rebuild'   # đổi thành 'main' sau khi merge
BUDGET_H = 11.0               # Kaggle chặn 12 h: quá ngân sách thì bỏ các biến thể còn lại
!git clone -q -b $BRANCH https://github.com/Bamboohoccode/ZAI_TEST.git /tmp/ZAI_TEST || (cd /tmp/ZAI_TEST && git pull -q)
%cd /tmp/ZAI_TEST
!git log --oneline -1
!pip install -q sherpa-onnx==2.0.1 sentencepiece soundfile silero-vad librosa onnx onnxruntime vllm 2>&1 | grep -v -i "warning\\|dependency conflicts\\|incompatible" | tail -20
!nvidia-smi --query-gpu=name,memory.total --format=csv
""")

code("""
import glob, json, os, re, subprocess, sys, time
T0 = time.time()
W = '/kaggle/working'
os.makedirs(f'{W}/logs', exist_ok=True)
DATA = os.path.dirname(os.path.dirname(sorted(glob.glob('/kaggle/input/**/train/train.jsonl', recursive=True))[0]))
os.environ['DATA'] = DATA
print(DATA, os.listdir(DATA))
RES = {'smoke': SMOKE, 'stage': STAGE, 'data': DATA, 'asr': [], 'nollm': {}, 'errors': []}

def save():
    RES['hours'] = round((time.time() - T0) / 3600, 2)
    json.dump(RES, open(f'{W}/results.json', 'w'), ensure_ascii=False, indent=1)

def sh(cmd, tag):
    \"\"\"chạy lệnh, in + ghi log logs/<tag>.log; lỗi không dừng notebook (ghi vào RES['errors']).\"\"\"
    print(f'\\n$ {cmd}', flush=True)
    t = time.time()
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=os.getcwd())
    lines = []
    with open(f'{W}/logs/{tag}.log', 'w') as f:
        for l in p.stdout:
            lines.append(l); f.write(l); f.flush()
            print(l, end='', flush=True)
    p.wait()
    out = ''.join(lines)
    if p.returncode:
        RES['errors'].append({'tag': tag, 'code': p.returncode, 'tail': out[-3000:]})
    print(f'[{tag}] exit {p.returncode}, {time.time() - t:.0f}s, tổng {(time.time() - T0) / 3600:.2f}h', flush=True)
    save()
    return p.returncode, out

def hours():
    return (time.time() - T0) / 3600

# tải sẵn model (eval_asr / train_rnnlm đặt HF_HUB_OFFLINE=1 nên phải có trong cache trước)
from huggingface_hub import snapshot_download
sys.path.insert(0, os.getcwd())
from lifemate import config
snapshot_download(config.ZIPFORMER_MODEL, allow_patterns=['*.onnx', 'bpe.model'])
for m in (config.LLM_MODEL, config.MAPPER_MODEL):
    snapshot_download(m, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', 'merges.txt'])
print('models ok', f'{hours():.2f}h')
""")

code("""
sh('python tests/test_unit.py', 'unit')
""")

md("""
## 1–3. ASR (STAGE all / asr): dump train -> RNNLM -> WER held-out của mọi cấu hình decode
""")
code("""
J = f'{W}/asr.json'
if STAGE in ('all', 'asr'):
    N = ' --n 200' if SMOKE else ''
    sh(f'python scripts/eval_asr.py --root $DATA --subset all --configs beam --threads 4 --dump data/asr_train_plain.jsonl{N}', 'asr_dump')
    sh('python scripts/train_rnnlm.py --data data/asr_train_plain.jsonl --out models/rnnlm' + (' --epochs 1' if SMOKE else ''), 'rnnlm')
    HAS_LM = os.path.exists('models/rnnlm/rnnlm.onnx')
    N = ' --n 60' if SMOKE else ''
    if os.path.exists(J): os.remove(J)
    SCALES = '0.1,0.3' if SMOKE else '0.1,0.2,0.3,0.4,0.5'
    base = f'python scripts/eval_asr.py --root $DATA --threads 4 --json {J}{N}'
    sh(f'{base} --configs greedy,beam,hw,vad+hw', 'asr_cfg')
    if HAS_LM:
        sh(f'{base} --configs lm,hw+lm,vad+hw+lm --lm models/rnnlm/rnnlm.onnx --lm-scales {SCALES}', 'asr_cfg_lm')
    rows = json.load(open(J)) if os.path.exists(J) else []
    top = [r for r in sorted(rows, key=lambda r: r['wer']) if r['config'] != 'greedy'][:3]   # beam 8 cho 3 cấu hình tốt nhất
    by_parts = {}
    for r in top:
        c, _, s = r['config'].partition('@')
        by_parts.setdefault(c, []).append(s)
    for c, ss in by_parts.items():
        lm = f' --lm models/rnnlm/rnnlm.onnx --lm-scales {",".join(ss)}' if ss[0] else ''
        sh(f'{base} --beam 8 --configs {c}{lm}', f'asr_beam8_{c.replace("+", "_")}')
    rows = json.load(open(J)) if os.path.exists(J) else []
    RES['asr'] = sorted(rows, key=lambda r: r['wer'])
    if HAS_LM:
        !mkdir -p {W}/rnnlm && cp -r models/rnnlm/. {W}/rnnlm/
HAS_LM = os.path.exists('models/rnnlm/rnnlm.onnx')
RES['has_lm'] = HAS_LM; save()
for r in RES['asr']:
    print(f"{r['config']:<22} beam {r['beam']}  WER {r['wer']:.4f}  WER(query) {r['wer_query']:.4f}  {r['seconds']:.0f}s")

def asr_flags(r):
    c, _, s = r['config'].partition('@')
    parts = set(c.split('+'))
    f = [f"--beam {r['beam']}"]
    if 'hw' in parts: f.append('--hotwords-file assets/hotwords_global.txt --hotword-score 1.5')
    if 'vad' in parts: f.append('--vad')
    if 'lm' in parts and HAS_LM: f.append(f'--lm models/rnnlm/rnnlm.onnx --lm-scale {s}')
    return ' '.join(f)

cands = [r for r in RES['asr'] if r['config'] != 'greedy']   # run.py luôn dùng beam search
ASR = asr_flags(cands[0]) if cands else ASR_DEFAULT
RES['best_asr'] = {'config': cands[0] if cands else None, 'flags': ASR}; save()
print('cấu hình ASR dùng cho bước 4:', ASR)
LLM = '--llm-device 1 --gpu-mem 0.85 --max-len 16384 --mapper-gpu-mem 0.4'   # LLM (prefetch) GPU 1, mapper GPU 0
LIM = ' --limit 6' if SMOKE else ''
""")

md("""
## 4. Xử lý lời nói không có agent gọi tool (`--no-llm`, STAGE all / nollm)
Hotword cố định / + bias theo mẫu (hồ sơ, history) / + prefetch (LLM gọi tool đọc lấy tên làm hotword, decode lại câu cuối),
mỗi mức có và không có mapper phương ngữ. In WER của bản ASR và bản sau mapper.
""")
code("""
def wers(o):
    a = re.search(r'WER dạng đọc: ([0-9.]+)', o)
    m = re.search(r'WER sau mapper: ([0-9.]+)', o)
    t = re.search(r'\\| ([0-9.]+)s/mẫu', o)
    return {'wer_asr': float(a.group(1)) if a else None, 'wer_mapped': float(m.group(1)) if m else None,
            'sec_per_sample': float(t.group(1)) if t else None}

NOLLM = {
    'static':                '--mapper none',
    'static+mapper':         '',
    'bias':                  '--bias --mapper none',
    'bias+mapper':           '--bias',
    'bias+prefetch':         f'--prefetch {LLM} --mapper none',
    'bias+prefetch+mapper':  f'--prefetch {LLM}',
}
if STAGE in ('all', 'nollm'):
    for name, extra in NOLLM.items():
        if hours() > BUDGET_H:
            RES['nollm'][name] = 'skipped (time)'; continue
        _, o = sh(f'python scripts/run.py --root $DATA --split train --heldout --no-llm {ASR} {extra}{LIM} '
                  f'--mapper-device 0 --mapper-gpu-mem 0.4 --out {W}/nollm_{name.replace("+", "_")}.jsonl', f'nollm_{name}')
        RES['nollm'][name] = {'flags': f'{ASR} {extra}'.strip(), **wers(o)}; save()
for k, v in RES['nollm'].items():
    print(f'{k:<22}', v if isinstance(v, str) else f"WER ASR {v['wer_asr']}  WER mapper {v['wer_mapped']}  {v['sec_per_sample']}s/mẫu")
save()
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}
json.dump(nb, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", out, "smoke" if smoke else "full")
