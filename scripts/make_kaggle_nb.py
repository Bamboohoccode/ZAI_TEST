"""Sinh notebook Kaggle chạy tự động (Save & Run All, không chỉnh tay), mỗi STAGE một phần thí nghiệm:

    asr    dump ASR train -> RNNLM -> WER held-out của mọi cấu hình decode (eval_asr.py)
    nollm  run.py --no-llm (không agent gọi tool): hotword cố định / bias / prefetch, hoặc lưới --grid
    p12    dump ASR train + nhiều bản decode held-out -> P1 (bảng nhầm lẫn + LLM chấm), P2 (N-best + LLM chấm lại),
           P6 (CTC word spotter)
    agent  SCORE held-out của các biến thể agent: baseline, mốc 0 (--text), P3 --snap-all, P4 --alt-hyp, P5 --fewshot

    python scripts/make_kaggle_nb.py notebooks/kaggle.ipynb                         # stage asr (mặc định)
    python scripts/make_kaggle_nb.py out/kaggle.ipynb --stage agent --variants base,p3,p4
    python scripts/make_kaggle_nb.py out/kaggle.ipynb --stage nollm --grid grid.json --smoke
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--smoke", action="store_true", help="vài mẫu mỗi bước, để bắt lỗi môi trường")
ap.add_argument("--stage", default="asr", choices=["asr", "nollm", "p12", "agent"])
ap.add_argument("--grid", default="", help="nollm: file JSON {tên: cờ run.py}")
ap.add_argument("--variants", default="base,text_oracle,p3,p4,p5", help="agent: các biến thể, phẩy")
ap.add_argument("--branch", default="refactor/rebuild")
A = ap.parse_args()

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)})


md(f"""
# LifeMate trên Kaggle — stage `{A.stage}`

Settings: **GPU T4 x2**, **Internet on**, Input: dataset BTC (`train/`, `public_test/`); stage `p12` / `agent` có thể gắn thêm
output của kernel `asr` (RNNLM) / `p12` (dump ASR train cho few-shot). Mọi số đo trên **held-out** (id % 10 == 9).
Sinh bởi `scripts/make_kaggle_nb.py`; tổng kết ở `results.json`, log ở `logs/`.
""")

code(f"""
SMOKE = {A.smoke}
STAGE = {A.stage!r}
GRID = {json.load(open(A.grid)) if A.grid else {}!r}
VARIANTS = {[v for v in A.variants.split(",") if v]!r}
BRANCH = {A.branch!r}
BUDGET_H = 11.0   # Kaggle chặn 12 h: quá ngân sách thì bỏ các biến thể còn lại
!git clone -q -b $BRANCH https://github.com/Bamboohoccode/ZAI_TEST.git /tmp/ZAI_TEST || (cd /tmp/ZAI_TEST && git pull -q)
%cd /tmp/ZAI_TEST
!git log --oneline -1
!pip install -q sherpa-onnx==2.0.1 sentencepiece soundfile silero-vad librosa onnx onnxruntime kaldi-native-fbank vllm 2>&1 | grep -v -i "warning\\|dependency conflicts\\|incompatible" | tail -20
!nvidia-smi --query-gpu=name,memory.total --format=csv
""")

code("""
import glob, json, os, re, shutil, subprocess, sys, time
T0 = time.time()
W = '/kaggle/working'
os.makedirs(f'{W}/logs', exist_ok=True)
DATA = os.path.dirname(os.path.dirname(sorted(glob.glob('/kaggle/input/**/train/train.jsonl', recursive=True))[0]))
os.environ['DATA'] = DATA
print(DATA, os.listdir(DATA))
RES = {'smoke': SMOKE, 'stage': STAGE, 'data': DATA, 'errors': []}

def save():
    RES['hours'] = round((time.time() - T0) / 3600, 2)
    json.dump(RES, open(f'{W}/results.json', 'w'), ensure_ascii=False, indent=1)

def sh(cmd, tag):
    \"\"\"chạy lệnh, in + ghi logs/<tag>.log; lỗi không dừng notebook (ghi vào RES['errors']).\"\"\"
    print(f'\\n$ {cmd}', flush=True)
    t = time.time()
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    with open(f'{W}/logs/{tag}.log', 'w') as f:
        for l in p.stdout:
            lines.append(l); f.write(l); f.flush()
            print(l, end='', flush=True)
    p.wait()
    out = ''.join(lines)
    if p.returncode:
        RES['errors'].append({'tag': tag, 'code': p.returncode, 'tail': out[-3000:]})
    print(f'[{tag}] exit {p.returncode}, {time.time() - t:.0f}s, tổng {hours():.2f}h', flush=True)
    save()
    return p.returncode, out

def hours():
    return (time.time() - T0) / 3600

def find_input(name):
    \"\"\"file `name` trong output của kernel khác gắn làm Input (vd. rnnlm.onnx, asr_train_plain.jsonl).\"\"\"
    hits = sorted(glob.glob(f'/kaggle/input/**/{name}', recursive=True))
    return hits[0] if hits else None

# tải sẵn model (eval_asr / train_rnnlm đặt HF_HUB_OFFLINE=1)
from huggingface_hub import snapshot_download
sys.path.insert(0, os.getcwd())
from lifemate import config
snapshot_download(config.ZIPFORMER_MODEL, allow_patterns=['*.onnx', 'bpe.model'])
snapshot_download(config.LLM_MODEL, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', 'merges.txt'])
sh('python tests/test_unit.py', 'unit')

HW = '--hotwords-file assets/hotwords_global.txt --hotword-score 1.5'
LLM = '--llm-device 1 --gpu-mem 0.85 --max-len 16384'   # LLM chính trên GPU 1
LIM = ' --limit 6' if SMOKE else ''
lm = find_input('rnnlm.onnx')
if lm:
    os.makedirs('models/rnnlm', exist_ok=True)
    shutil.copytree(os.path.dirname(lm), 'models/rnnlm', dirs_exist_ok=True)
HAS_LM = os.path.exists('models/rnnlm/rnnlm.onnx')
print('RNNLM:', HAS_LM, lm, '| tổng', f'{hours():.2f}h')
""")

md("## Dump ASR train (stage asr / p12): đầu vào RNNLM, bảng nhầm lẫn P1, few-shot P5")
code("""
DUMP = 'data/asr_train_plain.jsonl'
if STAGE in ('asr', 'p12'):
    N = ' --n 200' if SMOKE else ''
    sh(f'python scripts/eval_asr.py --root $DATA --subset all --configs beam --threads 4 --dump {DUMP}{N}', 'asr_dump')
    if os.path.exists(DUMP):
        shutil.copy(DUMP, f'{W}/asr_train_plain.jsonl')
""")

md("## Stage asr: RNNLM + WER mọi cấu hình decode")
code("""
if STAGE == 'asr':
    J = f'{W}/asr.json'
    sh('python scripts/train_rnnlm.py --data data/asr_train_plain.jsonl --out models/rnnlm' + (' --epochs 1' if SMOKE else ''), 'rnnlm')
    HAS_LM = os.path.exists('models/rnnlm/rnnlm.onnx')
    N = ' --n 60' if SMOKE else ''
    SCALES = '0.1,0.3' if SMOKE else '0.1,0.2,0.3,0.4,0.5'
    base = f'python scripts/eval_asr.py --root $DATA --threads 4 --json {J}{N}'
    sh(f'{base} --configs greedy,beam,hw,vad+hw', 'asr_cfg')
    if HAS_LM:
        sh(f'{base} --configs lm,hw+lm,vad+hw+lm --lm models/rnnlm/rnnlm.onnx --lm-scales {SCALES}', 'asr_cfg_lm')
        shutil.copytree('models/rnnlm', f'{W}/rnnlm', dirs_exist_ok=True)
    rows = json.load(open(J)) if os.path.exists(J) else []
    top = [r for r in sorted(rows, key=lambda r: r['wer']) if r['config'] != 'greedy'][:3]
    by = {}
    for r in top:
        c, _, s = r['config'].partition('@')
        by.setdefault(c, []).append(s)
    for c, ss in by.items():
        x = f' --lm models/rnnlm/rnnlm.onnx --lm-scales {",".join(ss)}' if ss[0] else ''
        sh(f'{base} --beam 8 --configs {c}{x}', f'asr_beam8_{c.replace("+", "_")}')
    RES['asr'] = sorted(json.load(open(J)) if os.path.exists(J) else [], key=lambda r: r['wer']); save()
    for r in RES['asr']:
        print(f"{r['config']:<22} beam {r['beam']}  WER {r['wer']:.4f}  WER(query) {r['wer_query']:.4f}  {r['seconds']:.0f}s")
""")

md("## Stage nollm / p12: `run.py --no-llm` trên held-out")
code("""
def wer_of(o):
    a = re.search(r'WER dạng đọc: ([0-9.]+)', o)
    t = re.search(r'\\| ([0-9.]+)s/mẫu', o)
    return {'wer': float(a.group(1)) if a else None, 'sec_per_sample': float(t.group(1)) if t else None}

def run_nollm(variants, key):
    RES[key] = {}
    for name, flags in variants.items():
        if hours() > BUDGET_H:
            RES[key][name] = 'skipped (time)'; continue
        _, o = sh(f'python scripts/run.py --root $DATA --split train --heldout --no-llm {flags}{LIM} --out {W}/nollm_{name}.jsonl',
                  f'nollm_{name}')
        RES[key][name] = {'flags': flags, **wer_of(o)}; save()
    for k, v in RES[key].items():
        print(f'{k:<22}', v)

if STAGE == 'nollm':
    run_nollm(GRID or {'static': f'--beam 4 {HW}', 'bias': f'--beam 4 {HW} --bias',
                       'bias_prefetch': f'--beam 4 {HW} --prefetch {LLM}'}, 'nollm')

if STAGE == 'p12':
    # các bản decode khác nhau của cùng audio = N-best tự ghép (sherpa-onnx không trả N-best); 'main' là ASR chính
    NB = {'main': f'--beam 4 {HW} --bias', 'b8': f'--beam 8 {HW} --bias', 'nohw': '--beam 4 --bias',
          'vad': f'--beam 4 {HW} --bias --vad'}
    if HAS_LM:
        NB['lm'] = f'--beam 4 {HW} --bias --lm models/rnnlm/rnnlm.onnx --lm-scale 0.2'
    run_nollm(NB, 'nbest')
    hyps = ' '.join(f'{n}={W}/nollm_{n}.jsonl' for n in NB if os.path.exists(f'{W}/nollm_{n}.jsonl'))
    _, o = sh(f'python scripts/asr_rescore.py --hyps {hyps} --dump {DUMP} --out-dir {W}/rescore', 'rescore')
    if os.path.exists(f'{W}/rescore/sweep.json'):
        sw = json.load(open(f'{W}/rescore/sweep.json'))
        RES['rescore'] = {'rows': sw['rows'], 'best': sw['best'], 'cv': sw['cv']}; save()
    # P6: CTC word spotter trên đầu CTC (jit_script.pt), hotword theo lượt, gộp vào câu ASR chính
    sh(f'python scripts/ctc_spotter.py --root $DATA --hyps {W}/nollm_main.jsonl --out-dir {W}/ctc', 'ctc_spotter')
    if os.path.exists(f'{W}/ctc/sweep.json'):
        sw = json.load(open(f'{W}/ctc/sweep.json'))
        RES['ctc_spotter'] = {'baseline': sw['baseline'], 'best': sw['best'], 'cv': sw['cv']}; save()
""")

md("## Stage agent: SCORE held-out (mốc 0 = --text; P3 / P4 / P5 bật từng cái trên baseline)")
code("""
def parse_score(o):
    m = re.search(r'SCORE ([0-9.]+) trên (\\d+) mẫu \\| F1 bỏ qua hash ([0-9.]+) \\| sai hash ([0-9.]+)%', o)
    t = re.search(r'TB ([0-9.]+)s/mẫu \\| > \\d+s: (\\d+)', o)
    if not m:
        return {'score': None}
    return {'score': float(m.group(1)), 'n': int(m.group(2)), 'f1_no_hash': float(m.group(3)),
            'hash_fail_pct': float(m.group(4)), 'sec_per_sample': float(t.group(1)) if t else None,
            'n_over_30s': int(t.group(2)) if t else None}

if STAGE == 'agent':
    BASE = f'--beam 4 {HW} --bias --prefetch {LLM}'
    FS = find_input('asr_train_plain.jsonl')
    AG = {'base': BASE, 'text_oracle': f'--text {LLM}', 'p3': f'{BASE} --snap-all', 'p4': f'{BASE} --alt-hyp',
          'p5': f'{BASE} --fewshot 3 --fewshot-file {FS}', 'p345': f'{BASE} --snap-all --alt-hyp --fewshot 3 --fewshot-file {FS}'}
    RES['agent'] = {}
    per = None
    for name in VARIANTS:
        if name in ('p5', 'p345') and not FS:
            RES['agent'][name] = 'skipped (không có asr_train_plain.jsonl trong Input)'; continue
        if hours() + (per or 1.3) > BUDGET_H:
            RES['agent'][name] = 'skipped (time)'; continue
        t = time.time()
        _, o = sh(f'python scripts/run.py --root $DATA --split train --heldout {AG[name]}{LIM} --out {W}/sub_{name}.jsonl', f'agent_{name}')
        per = max(per or 0, (time.time() - t) / 3600)
        RES['agent'][name] = {'flags': AG[name], **parse_score(o)}; save()
    for k, v in RES['agent'].items():
        print(f'{k:<12}', v)
""")

code("""
save()
print(json.dumps({k: v for k, v in RES.items() if k != 'errors'}, ensure_ascii=False, indent=1)[:8000])
for e in RES['errors']:
    print('LỖI', e['tag'], e['code'], e['tail'][-800:])
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}
json.dump(nb, open(A.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", A.out, A.stage, "smoke" if A.smoke else "")
