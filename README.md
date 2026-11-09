# ZAIC_2026

Repo dự án **LifeMate** cho Zalo AI Challenge 2026: agent nhận giọng nói tiếng Việt và gọi đúng chuỗi tool của các app mô phỏng.

## Tài liệu

- [docs/Overview_PJ.md](docs/Overview_PJ.md): tổng quan dự án, pipeline, phân công, cấu trúc repo, rủi ro.
- [docs/PIPELINE.md](docs/PIPELINE.md): luồng ASR → mapper → planner ↔ verifier → tool; mục 4 là ASR (confidence + LLM refine), sơ đồ `docs/asr_workflow.png`.

## Cấu trúc

```
├── pipeline.py              # entry: Zipformer(+LM) -> refine -> ITN -> mapper -> planner <-> verifier -> zaic_tools
├── asr/                     # Zipformer (sherpa-onnx), VAD, refine, eval ASR
│   ├── asr_zipformer.py     #   ZipformerASR(hotwords_file=, lm=, vad=) — offline; transcribe(words=True) -> (từ, conf)
│   ├── candidates.py        #   đánh dấu từ conf thấp + ứng viên gần âm, prompt LLM sửa
│   ├── vad.py               #   silero VAD (trim khoảng lặng)
│   └── eval_asr.py          #   dump train (bước A) + WER held-out theo cấu hình (greedy/beam/hw/vad/lm)
├── hotwords/                # biasing: hotword + snap search key của tool
│   ├── bias.py              #   hotword lines, snap_value, phon
│   ├── mine_hotwords.py     #   khai thác hotword từ lỗi ASR trên train
│   └── hotwords_global.txt/.json
├── lm/train_rnnlm.py        # RNNLM cho LM fusion -> lm/rnnlm/rnnlm.onnx
├── textnorm/                # dùng chung: norm, WER, đọc số tiếng Việt (vi_num)
├── ITN/                     # inverse text normalization (bert2BERT)
├── notebooks/               # fine-tune Zipformer, Kaggle hotword + RNNLM (bước A–E)
├── docs/
└── data/                    # không commit: dữ liệu BTC và asr_train_plain.jsonl (từ asr_assets.zip)
```

Chạy mọi lệnh từ gốc repo, ví dụ `python asr/eval_asr.py ...`, `python hotwords/mine_hotwords.py`,
`python lm/train_rnnlm.py` (hoặc `python -m asr.eval_asr ...`).

## Quy tắc nhanh

- Không commit `data/` hay trọng số model (luật cuộc thi cấm phát tán dữ liệu).
- Nhánh: `<loại>/<module>-<tên>`, ví dụ `feat/asr-khanh`. Chi tiết trong [docs/Overview_PJ.md](docs/Overview_PJ.md#31-quy-ước-nhánh).
- Chạy offline hoàn toàn khi suy luận.
# ZAI_TEST
