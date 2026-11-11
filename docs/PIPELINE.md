# Pipeline LifeMate

Một mẫu = hội thoại nhiều lượt; lượt user chỉ có audio. Đầu ra: `tool_calls` đã chạy trên ToolEnv và `state_hash`.

```
audio các lượt user (history + câu cuối)
  │
  ▼ 1. ASR — Zipformer 150M (sherpa-onnx, CPU), modified beam search
  │      + hotword theo lượt (--bias): tên trong hồ sơ user và tool call / kết quả của các lượt TRƯỚC
  │      + hotword cố định (--hotwords-file assets/hotwords_global.txt)
  │      + RNNLM fusion (--lm models/rnnlm/rnnlm.onnx --lm-scale)
  │      bias làm câu cụt (< 60% số từ bản không bias) -> dùng bản không bias
  │
  ▼ 2. Prefetch (--prefetch): planner gọi tool ĐỌC liên quan một lần (danh bạ, quán, cuộc họp...)
  │      -> tên trong kết quả làm hotword -> decode lại câu cuối. Các call này nằm trong bài nộp.
  │
  ▼ 3. Mapper phương ngữ — Qwen3-0.6B (--mapper; "none" để tắt): từ địa phương -> phổ thông, giữ tên / số
  │
  ▼ 4. Planner — Qwen3-4B-Instruct, greedy: system có lịch tham chiếu (hôm nay / mai / tuần sau...), giờ am/pm
  │      quy đổi, hồ sơ user; history cũ bị cắt cho vừa 16.384 token
  │
  ▼ 5. Verifier — cùng LLM, ACCEPT / REJECT + lý do, tối đa --verify-rounds lần lập lại plan
  │
  ▼ 6. Kiểm tra bằng code trước khi chạy: schema (env.validate), ID không có trong history / kết quả tool (bịa),
  │      tool đọc gọi lặp; snap search key (contacts.search query...) về tên đã thấy theo phát âm
  │
  ▼ 7. ToolEnv (zaic_tools) — dừng sau sys.ask_missing_info; tối đa 8 lượt; --time-limit để chặn 30 s
```

## Đo

Mọi số đo trên **held-out** của train: id % 10 == 9 (`--heldout`). Phần còn lại dùng để khai thác hotword và train RNNLM.

| Tầng | Lệnh | Chỉ số |
|---|---|---|
| Cả pipeline | `scripts/run.py --split train --heldout ...` (tự chấm), `scripts/score.py` | SCORE, F1 bỏ qua hash, % sai hash, theo loại mẫu / app |
| ASR | `scripts/eval_asr.py --configs beam,hw,hw+lm --lm-scales ...` | WER dạng đọc (so theo `text.canon`) |
| ASR trong pipeline | `scripts/run.py --asr-only [--bias --prefetch]` | WER sau hotword / prefetch |
| Kiểm tra env + scorer | `scripts/run.py --text --llm gold --mapper none` | SCORE phải ≈ 1.0 |

WER không tính các khác biệt không phải lỗi nhận dạng: vị trí dấu thanh (huỷ = hủy, khoá = khóa) và biến thể giùm / dùm.

## Lỗi ASR hiện tại (2745 lượt sai trong `data/asr_sai_nghia_chitiet.txt`)

| Loại | % chỗ lỗi | Hướng sửa |
|---|---|---|
| Đồng âm theo giọng (giời/dời, xếp/sếp) | 8.8 | LM: RNNLM fusion, hotword |
| Gần âm (nhấn/nhắn, hàng/hàn) | 24.3 | LM fusion, beam rộng hơn |
| Khác âm hẳn | 38.0 | model âm học (fine-tune encoder chỉ +1/100 câu, đã bỏ) |
| Thêm / bớt từ, cụm khác độ dài | 23.7 | — |
| Từ ngoại | 5.2 | hotword cố định |

Đã thử và bỏ: LLM sửa câu ASR (WER 11.6% -> 15.0%), fine-tune Zipformer (dev_fix 7 -> 8/100), ITN bert2BERT
(planner tự đổi số đọc bằng chữ sang tham số).
