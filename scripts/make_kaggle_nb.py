"""Sinh notebooks/kaggle.ipynb: chạy tự động toàn bộ thí nghiệm (Kaggle batch, không cần chỉnh tay)."""
import json
import sys

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--stage", default="all", choices=["all", "asr", "text", "pipe", "submit"])
ap.add_argument("--variants", default="", help="pipe: tên biến thể, phẩy")
ap.add_argument("--submit", default="", help="submit: tên biến thể thắng")
A = ap.parse_args()
out, smoke = A.out, A.smoke

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)})


md("""
# LifeMate trên Kaggle: so sánh mọi phương pháp, tự chọn cấu hình tốt nhất, tạo bài nộp

Settings: **GPU T4 x2**, **Internet on**, Input: dataset BTC (`train/`, `public_test/`).
Chạy được bằng *Save & Run All* (không cần chỉnh tay): mọi số đo trên **held-out** (id % 10 == 9).

1. ASR: so WER các cấu hình decode (greedy / beam / hotword / VAD / RNNLM fusion ở nhiều scale / beam 8) -> chọn WER thấp nhất.
2. Pipeline: chạy từng biến thể (bật / tắt mapper, prefetch, bias, verify, mapper dùng chung LLM, text oracle) -> chọn SCORE cao nhất.
3. Bài nộp public test bằng cấu hình thắng. Tổng kết ở `results.json`, log từng bước ở `logs/`.

`SMOKE = True`: chạy thử vài mẫu mỗi bước để bắt lỗi môi trường (vài chục phút).
""")

code(f"""
SMOKE = {smoke}
# all = chạy hết tuần tự | asr = ASR + RNNLM + WER | text = gold + biến thể LLM với --text (không cần ASR)
# pipe = các biến thể trong VARIANTS_SEL (cần output của kernel asr làm Input) | submit = bài nộp bằng SUBMIT
STAGE = {A.stage!r}
VARIANTS_SEL = {[v for v in A.variants.split(",") if v]!r}
SUBMIT = {A.submit!r}
BRANCH = 'refactor/rebuild'   # đổi thành 'main' sau khi merge
BUDGET_H = 11.0               # Kaggle chặn 12 h: quá ngân sách thì bỏ các biến thể còn lại, vẫn tạo bài nộp
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
RES = {'smoke': SMOKE, 'stage': STAGE, 'data': DATA, 'asr': [], 'asr_bias': {}, 'pipeline': {}, 'errors': []}

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

md("## 0. Kiểm tra code và scorer: `--text --llm gold` phải cho SCORE ≈ 1.0")
code("""
if STAGE in ('all', 'text'):
  sh('python tests/test_unit.py', 'unit')
  _, o = sh(f'python scripts/run.py --root $DATA --split train --heldout --text --llm gold --mapper none --out {W}/sub_gold.jsonl'
            + (' --limit 20' if SMOKE else ''), 'gold')
  m = re.search(r'SCORE ([0-9.]+)', o)
  RES['gold_score'] = float(m.group(1)) if m else None
  print('gold SCORE', RES['gold_score']); save()
""")

md("""
## 1–3. ASR (STAGE all / asr): dump train -> RNNLM -> WER held-out của mọi cấu hình decode
""")
code("""
DO_ASR = STAGE in ('all', 'asr')
J = f'{W}/asr.json'
if DO_ASR:
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
elif STAGE in ('pipe', 'submit'):
    # lấy kết quả ASR + RNNLM từ output của kernel asr (gắn làm Input)
    prev = [p for p in glob.glob('/kaggle/input/**/results.json', recursive=True)
            if json.load(open(p)).get('stage') in ('asr', 'all') and json.load(open(p)).get('asr')]
    assert prev, 'không thấy results.json của kernel asr trong /kaggle/input'
    RES['asr'] = json.load(open(prev[0]))['asr']
    lm = glob.glob(os.path.join(os.path.dirname(prev[0]), '**/rnnlm.onnx'), recursive=True)
    if lm:
        !mkdir -p models/rnnlm && cp -r {os.path.dirname(lm[0])}/. models/rnnlm/
    print('ASR từ', prev[0], '| rnnlm:', lm)
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
best_asr = cands[0] if cands else {'config': 'hw', 'beam': 4}
ASR = asr_flags(best_asr)
RES['best_asr'] = {'config': best_asr, 'flags': ASR}; save()
print('ASR thắng:', best_asr, '->', ASR)
LLM = '--llm-device 1 --gpu-mem 0.85 --max-len 16384 --mapper-gpu-mem 0.4'   # LLM chính GPU 1, mapper GPU 0
LIM = ' --limit 6' if SMOKE else ''
""")

md("""
## 4. Hotword theo từng mẫu (bias) và prefetch: WER held-out, ASR-only (STAGE all / asr)
""")
code("""
def wer_of(o):
    m = re.search(r'WER dạng đọc: ([0-9.]+)', o)
    return float(m.group(1)) if m else None

if DO_ASR:
    for name, extra in [('static', ''), ('bias', ' --bias'), ('bias+prefetch', f' --prefetch {LLM}')]:
        _, o = sh(f'python scripts/run.py --root $DATA --split train --heldout --asr-only {ASR}{extra}{LIM} '
                  f'--out {W}/asr_only_{name.replace("+", "_")}.jsonl', f'asronly_{name}')
        RES['asr_bias'][name] = wer_of(o); save()
    print(RES['asr_bias'])
""")

md("""
## 5. Biến thể pipeline trên held-out (tự chấm)
`full` = bias + prefetch + mapper + verify 2; mỗi biến thể đổi một thành phần. `text_*` = nhãn audio_text thay ASR
(cô lập phía LLM; trần trên nếu ASR hoàn hảo, không chọn để nộp).
""")
code("""
VARIANTS = {
    'full':             f'--bias --prefetch {ASR} {LLM}',
    'no_mapper':        f'--bias --prefetch {ASR} {LLM} --mapper none',
    'no_prefetch':      f'--bias {ASR} {LLM}',
    'no_verify':        f'--bias --prefetch {ASR} {LLM} --verify-rounds 0',
    'mapper_same':      f'--bias --prefetch {ASR} {LLM} --mapper same',
    'no_bias':          f'{ASR} {LLM}',
    'verify1':          f'--bias --prefetch {ASR} {LLM} --verify-rounds 1',
    'text_oracle':      f'--text {LLM}',
    'text_no_mapper':   f'--text {LLM} --mapper none',
    'text_no_verify':   f'--text {LLM} --verify-rounds 0',
    'text_verify1':     f'--text {LLM} --verify-rounds 1',
    'text_mapper_same': f'--text {LLM} --mapper same',
}
if STAGE == 'all':
    RUN = list(VARIANTS)
elif STAGE == 'text':
    RUN = VARIANTS_SEL or [k for k in VARIANTS if k.startswith('text_')]
elif STAGE == 'pipe':
    RUN = VARIANTS_SEL
else:
    RUN = []

def parse_score(o):
    m = re.search(r'SCORE ([0-9.]+) trên (\\d+) mẫu \\| F1 bỏ qua hash ([0-9.]+) \\| sai hash ([0-9.]+)%', o)
    t = re.search(r'TB ([0-9.]+)s/mẫu \\| > \\d+s: (\\d+)', o)
    if not m:
        return None
    return {'score': float(m.group(1)), 'n': int(m.group(2)), 'f1_no_hash': float(m.group(3)),
            'hash_fail_pct': float(m.group(4)), 'sec_per_sample': float(t.group(1)) if t else None,
            'n_over_30s': int(t.group(2)) if t else None}

per_run_h = None
for name in RUN:
    flags = VARIANTS[name]
    need = (per_run_h or 0.8) + (1.0 if STAGE == 'all' else 0.2)   # 'all' chừa ~1 h cho bài nộp
    if hours() + need > BUDGET_H:
        print(f'bỏ {name}: hết ngân sách thời gian ({hours():.2f}h)'); RES['pipeline'][name] = 'skipped (time)'; continue
    t = time.time()
    _, o = sh(f'python scripts/run.py --root $DATA --split train --heldout {flags}{LIM} --out {W}/sub_{name}.jsonl', f'pipe_{name}')
    per_run_h = max(per_run_h or 0, (time.time() - t) / 3600)
    RES['pipeline'][name] = {'flags': flags, **(parse_score(o) or {'score': None})}; save()

for k, v in RES['pipeline'].items():
    print(f'{k:<17}', v if isinstance(v, str) else f"SCORE {v.get('score')}  F1 {v.get('f1_no_hash')}  hash sai {v.get('hash_fail_pct')}%  {v.get('sec_per_sample')}s/mẫu")
""")

md("## 6. Bài nộp public test (STAGE all: biến thể thắng; STAGE submit: SUBMIT), chặn 30 s/mẫu")
code("""
if STAGE in ('all', 'submit'):
    ok = {k: v for k, v in RES['pipeline'].items() if isinstance(v, dict) and v.get('score') is not None and not k.startswith('text_')}
    win = SUBMIT or (max(ok, key=lambda k: ok[k]['score']) if ok else 'full')
    FLAGS = VARIANTS[win]
    RES['winner'] = {'variant': win, 'flags': FLAGS, 'score': ok.get(win, {}).get('score')}; save()
    print('biến thể nộp:', win, FLAGS)
    sh(f'python scripts/run.py --root $DATA --split public_test {FLAGS} --time-limit 30{LIM} --out {W}/submission.jsonl', 'public_test')
    RES['submission_lines'] = sum(1 for _ in open(f'{W}/submission.jsonl')) if os.path.exists(f'{W}/submission.jsonl') else 0
save()
print(json.dumps(RES, ensure_ascii=False, indent=1)[:8000])
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}
json.dump(nb, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", out, "smoke" if smoke else "full")
