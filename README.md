# ZAIC 2026 · LifeMate

Agent nhận giọng nói tiếng Việt và gọi đúng chuỗi tool của các app mô phỏng (Zalo AI Challenge 2026, bảng LifeMate).
Luồng xử lý và cách đo: [docs/PIPELINE.md](docs/PIPELINE.md). Chạy trên Kaggle: [notebooks/kaggle.ipynb](notebooks/kaggle.ipynb).

## Luật cần nhớ

- **Chấm điểm:** `state_hash` sai thì mẫu đó 0 điểm. Hash đúng thì điểm là F1 trên tool call (đúng tên tool, tham số thường
  giống hệt, tham số `x-match: fuzzy` có token-F1 > 0.6, không xét thứ tự). Gold rỗng: nộp rỗng được 1. Mọi call đã chạy
  trên ToolEnv, kể cả call đọc, đều nằm trong bài nộp, nên **một call thừa có thể làm hỏng hash của cả mẫu**.
- **Giới hạn:** tổng < 6B tham số (LLM < 5B, ngữ cảnh 16.384), 30 s/mẫu trên 1 GPU A30, chạy offline, seed cố định.
- **Cấm:** đọc `state.json` hay `<split>_environment/` (chỉ ToolEnv được đọc), phát tán dữ liệu BTC, gán nhãn tay dữ liệu test.
- **Mốc:** nộp code 16/11 23:59 · đóng băng model trước 20/11 11:00 (mở private test) · nộp private 23/11 23:59.

## Cấu trúc

```
lifemate/                 # package dùng chung
├── config.py             #   model mặc định, giới hạn (ngữ cảnh, số bước, verify, 30 s)
├── data.py               #   đọc split / catalog, tìm zaic_tools, chọn mẫu (--heldout / --ids / --limit)
├── text.py               #   norm, WER (canon: huỷ=hủy, giùm=dùm), is_heldout (id % 10 == 9)
├── vi_num.py             #   đọc số thành chữ cho ref (chọn cách đọc giống ASR)
├── bias.py               #   hotword theo lượt (hồ sơ + history), snap search key về tên đã thấy
├── asr/                  #   zipformer.py (sherpa-onnx + RNNLM fusion), vad.py, decode.py (bias + fallback)
├── agent/                #   prompts.py, messages.py (lịch tham chiếu, cắt history), loop.py (planner ↔ verifier, prefetch)
├── llm.py                #   VLLM / HFLLM / RemoteVLLM (--llm-device) / GoldLLM
├── pipeline.py           #   run_sample: một mẫu end to end; run_asr_only: chỉ ASR + WER
└── score.py              #   scorer theo luật BTC
scripts/
├── run.py                # CLI chính -> submission.jsonl (+ .trace.jsonl), split có nhãn thì tự chấm
├── score.py              # chấm lại / so sánh nhiều submission
├── eval_asr.py           # WER theo cấu hình decode (greedy / beam / hotword / VAD / LM) + dump ASR train
├── train_rnnlm.py        # RNNLM cho LM fusion -> models/rnnlm/rnnlm.onnx
└── mine_hotwords.py      # hotword cố định từ lỗi ASR trên train -> assets/hotwords_global.txt
assets/                   # hotwords_global.txt / .json
tests/test_unit.py        # chạy không cần GPU / dữ liệu: python tests/test_unit.py
data/                     # KHÔNG commit dữ liệu BTC; chỉ có danh sách id câu ASR sai (asr_sai_nghia_*.txt)
```

Mọi lệnh chạy từ gốc repo, ví dụ `python scripts/run.py --root DATA --split train --heldout --bias --prefetch`.

## Quy tắc

- Không commit dữ liệu BTC hay trọng số (`data/*`, `models/` đã nằm trong `.gitignore`).
- Mọi thay đổi pipeline phải đi kèm điểm của `scripts/score.py` trên held-out, so với bản trước.
- Nhánh: `<loại>/<module>-<tên>`, ví dụ `feat/asr-khanh`.
