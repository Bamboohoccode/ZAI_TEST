# ZAIC_2026

Repo dự án **LifeMate** cho Zalo AI Challenge 2026: agent nhận giọng nói tiếng Việt và gọi đúng chuỗi tool của các app mô phỏng.

## Tài liệu

- [docs/Overview_PJ.md](docs/Overview_PJ.md): tổng quan dự án, pipeline, phân công, cấu trúc repo, rủi ro.
- [docs/PIPELINE.md](docs/PIPELINE.md): luồng ASR → mapper → planner ↔ verifier → tool.

## Cấu trúc

```
├── pipeline.py              # entry: ASR -> ITN -> mapper -> planner <-> verifier -> zaic_tools
├── asr/                     # Zipformer (sherpa-onnx), VAD, luật sửa lỗi, eval/test ASR
│   ├── asr_zipformer.py     #   ZipformerASR(hotwords_file=, lm=, vad=) — offline
│   ├── vad.py               #   silero VAD (trim khoảng lặng)
│   ├── asr_fix.py           #   luật sửa lỗi hệ thống (lịch->dịch, ron...)
│   ├── eval_asr.py          #   WER trên held-out theo cấu hình (greedy/beam/hw/vad/lm/snap/fix)
│   └── test_asr_public.py   #   in transcript public_test để duyệt
├── hotwords/                # biasing: hotword cố định + snap
│   ├── bias.py              #   hotword lines, snap_text / snap_value, load_global
│   ├── mine_hotwords.py     #   khai thác hotword từ lỗi ASR trên train
│   └── hotwords_global.txt/.json
├── lm/train_rnnlm.py        # RNNLM cho LM fusion -> lm/rnnlm/rnnlm.onnx
├── textnorm/                # dùng chung: norm, WER, đọc số tiếng Việt (vi_num)
├── mapper/                  # seq2seq mapper (ViT5/BARTpho), tn_agy.py (TN bằng LLM, chỉ dùng lúc chuẩn bị dữ liệu)
├── ITN/                     # inverse text normalization (bert2BERT)
├── notebooks/               # fine-tune Zipformer / mapper, Kaggle hotword + RNNLM
├── docs/
└── data/                    # không commit: dữ liệu BTC và jsonl sinh ra (asr_train*.jsonl...)
```

Chạy mọi lệnh từ gốc repo, ví dụ `python asr/eval_asr.py ...`, `python hotwords/mine_hotwords.py`,
`python lm/train_rnnlm.py` (hoặc `python -m asr.eval_asr ...`).

## Quy tắc nhanh

- Không commit `data/` hay trọng số model (luật cuộc thi cấm phát tán dữ liệu).
- Nhánh: `<loại>/<module>-<tên>`, ví dụ `feat/asr-khanh`. Chi tiết trong [docs/Overview_PJ.md](docs/Overview_PJ.md#31-quy-ước-nhánh).
- Chạy offline hoàn toàn khi suy luận.
# ZAI_TEST
