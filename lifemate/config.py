"""Model mặc định và giới hạn của pipeline (luật LifeMate: tổng < 6B tham số, LLM < 5B, ngữ cảnh 16.384, 30 s/mẫu)."""
ZIPFORMER_MODEL = "hynt/ZipFormer-150M-CR-CTC-RNNT-6000h"
MAPPER_MODEL = "Qwen/Qwen3-0.6B"
LLM_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

MAX_CTX = 16384      # max_model_len của LLM chính
MAX_NEW = 512        # token sinh tối đa mỗi lượt planner
MAX_STEPS = 8        # số lượt planner -> tool tối đa mỗi mẫu
VERIFY_ROUNDS = 2    # N: số lần REJECT -> lập lại plan mỗi bước; quá N thì chạy plan cuối, không kiểm nữa
TIME_LIMIT_S = 30    # giới hạn thời gian mỗi mẫu của BTC (scripts/run.py --time-limit để chặn thật)

HOTWORDS_FILE = "assets/hotwords_global.txt"
RNNLM_FILE = "models/rnnlm/rnnlm.onnx"
ASR_DUMP = "data/asr_train_plain.jsonl"
