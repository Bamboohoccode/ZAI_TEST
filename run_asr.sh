#!/usr/bin/env bash
# Chạy phần ASR của pipeline (docs/asr_workflow.png) và in output từng bước cho mỗi lượt user:
#   audio -> [VAD] -> Zipformer(+RNNLM, [hotword]) -> đánh dấu conf thấp -> LLM refine -> ITN
# Không chạy mapper / planner / tool. Split có nhãn (train) -> in thêm WER + bảng chọn --conf-thr.
#
#   bash run_asr.sh                                   # mặc định: train, 20 mẫu
#   DATA=/kaggle/input/... SPLIT=public_test LIMIT=50 bash run_asr.sh
#   BIAS=1 VAD=1 CONF_THR=0.5 bash run_asr.sh
#   ITN=path/to/itn_ckpt bash run_asr.sh              # thêm bước ITN
#
# Không có vLLM/GPU -> tự chuyển sang --mark-only (chỉ đánh dấu, không LLM refine).
set -euo pipefail
cd "$(dirname "$0")"

DATA=${DATA:-$HOME/Downloads/train+public_test/train+public_test}
SPLIT=${SPLIT:-train}
LIMIT=${LIMIT:-20}
LM=${LM:-lm/rnnlm/rnnlm.onnx}
LM_SCALE=${LM_SCALE:-0.3}
BEAM=${BEAM:-4}
CONF_THR=${CONF_THR:-0.6}
CAND=${CAND:-data/asr_train_plain.jsonl}
ITN=${ITN:-}
ZIPFORMER=${ZIPFORMER:-hynt/Zipformer-30M-RNNT-6000h}
OUT=${OUT:-asr_out.jsonl}

args=(--root "$DATA" --split "$SPLIT" --limit "$LIMIT" --zipformer "$ZIPFORMER" --beam "$BEAM"
      --conf-thr "$CONF_THR" --cand-train "$CAND" --out "$OUT")
[[ -f "$LM" ]] && args+=(--lm "$LM" --lm-scale "$LM_SCALE") || echo "! không có $LM -> chạy không RNNLM"
[[ -n "${BIAS:-}" ]] && args+=(--bias)
[[ -n "${VAD:-}" ]] && args+=(--vad)
[[ -n "$ITN" ]] && args+=(--itn "$ITN")
[[ -f "$CAND" ]] || { echo "! thiếu $CAND: giải nén asr_assets.zip, hoặc chạy bước A:"
                      echo "  python asr/eval_asr.py --root $DATA --subset all --configs beam --dump $CAND"; exit 1; }
if python -c "import vllm" 2>/dev/null; then
    args+=(--no-llm)            # có LLM refine (Qwen qua vLLM), không planner
else
    echo "! không có vLLM -> --mark-only (chỉ đánh dấu, không LLM refine)"
    args+=(--mark-only)
fi

python pipeline.py "${args[@]}"

python - "$OUT" <<'EOF'
import json, sys
for line in open(sys.argv[1], encoding="utf-8"):
    r = json.loads(line)
    print(f"\n=== {r['id']}")
    for t in r["turns"]:
        print(f"  [{t['turn']}] {t['audio']}")
        if t.get("ref"):
            print(f"    ref    : {t['ref']}")
        print(f"    asr    : {t['asr']}")
        if "[" in t["marked"]:
            print(f"    marked : {t['marked']}")
        if t["refine"] != t["asr"]:
            print(f"    refine : {t['refine']}")
        if t["itn"] != t["refine"]:
            print(f"    itn    : {t['itn']}")
EOF
