# Voice Agent Pipeline

Pipeline xử lý yêu cầu bằng giọng nói: **ASR → Chuẩn hoá phương ngữ → Lập plan → Verify → Gọi tool**.

![Pipeline](pipeline.png)

```mermaid
flowchart TD
    A[Audio input] --> B[ASR<br/>speech → text]
    B -->|transcript| C[Small LLM<br/>Dialect Mapper]
    C -->|query chuẩn| D[Main LLM<br/>Planner]
    D -->|plan + query| E[LLM Verifier]
    E -- ACCEPT --> F[Tool Executor]
    E -- "REJECT + lý do (≤ N lần)" --> D
    E -. "quá N lần: chạy plan cuối" .-> F
    F --> G[Result]
```

---

## 1. Tổng quan luồng

| # | Bước | Input | Output | Model |
|---|------|-------|--------|-------|
| 1 | ASR | Audio | Transcript thô | Whisper / model ASR tiếng Việt |
| 2 | Dialect Mapper | Transcript thô | Query tiếng Việt chuẩn | Small LLM |
| 3 | Planner | Query chuẩn | Plan (các bước, tool, args) | Main LLM |
| 4 | Verifier | Plan + query chuẩn | `ACCEPT` / `REJECT` + lý do | LLM (cùng hoặc khác Main LLM) |
| 5 | Tool Executor | Plan đã ACCEPT | Kết quả tool | Code (không phải LLM) |

---

## 2. Chi tiết từng bước

### 2.1 ASR (Automatic Speech Recognition)

ASR chuyển giọng nói thành văn bản, giữ nguyên cách nói của người dùng kể cả từ địa phương.

- Output là transcript thô, có thể lẫn từ vùng miền và lỗi nhận dạng.
- Không chuẩn hoá ở bước này, để Dialect Mapper xử lý tập trung.

### 2.2 Small LLM – Dialect Mapper

Small LLM chuyển từ ngữ vùng miền sang tiếng Việt chuẩn mà không đổi ý nghĩa câu.

- Câu đã chuẩn thì **trả nguyên văn**, nên không cần thêm bước classifier "có phương ngữ hay không".
- Dùng model nhỏ vì đây là tác vụ rewrite đơn giản, cần nhanh và rẻ.
- Chỉ được đổi từ ngữ, không được thêm hay bớt ý (tên riêng, số, ngày giờ phải giữ nguyên).

Ví dụ:

| Transcript thô | Query chuẩn |
|---|---|
| "Mai mấy giờ tàu ra Huế rứa?" | "Ngày mai mấy giờ có tàu ra Huế vậy?" |
| "Đặt cho tui cái lịch họp chiều ni" | "Đặt cho tôi lịch họp chiều nay" |
| "Bật đèn phòng khách" | "Bật đèn phòng khách" *(giữ nguyên)* |

### 2.3 Main LLM – Planner

Main LLM đọc query chuẩn cùng danh sách tool có sẵn rồi lập plan gồm các bước, tool cần gọi và tham số.

Output nên là JSON để Verifier và Executor parse được:

```json
{
  "intent": "tra cứu lịch tàu",
  "steps": [
    {"tool": "search_train", "args": {"to": "Huế", "date": "2026-11-01"}}
  ]
}
```

Khi nhận REJECT, Planner dùng **lý do reject** để sửa plan, không lập lại từ đầu.

### 2.4 LLM Verifier

Verifier nhận **plan + query chuẩn** và kiểm tra plan trước khi cho phép gọi tool.

Các điểm cần check:
1. **Đúng tool:** tool có tồn tại và có phù hợp với ý định của query không.
2. **Đúng args:** đủ tham số bắt buộc, đúng kiểu, giá trị khớp với query (ngày, tên, số).
3. **Khớp ý định:** plan làm đúng việc user yêu cầu, không thừa và không thiếu.
4. **An toàn:** hành động không đảo ngược được (xoá, thanh toán, gửi tin) thì cần user xác nhận.

Output:

```json
{"verdict": "REJECT", "reason": "thiếu args 'date', query nói 'ngày mai'"}
```

### 2.5 Tool Executor

Dựa trên ToolEnv của Ban Tổ Chức
---

## 3. Vòng lặp REJECT

Sau khi Verifier REJECT, plan quay về Planner kèm lý do, **tối đa N lần** (gợi ý N = 2–3).

- Không giới hạn số vòng thì Planner và Verifier có thể lặp mãi, tốn token và tăng độ trễ.
- Quá N lần thì **plan cuối cùng được chạy luôn, không kiểm tra nữa** (code: `plan_step` trong `pipeline.py`).

```
Planner ──plan──► Verifier ──REJECT + lý do──► Planner   (retry ≤ N)
                     │
                     ├── ACCEPT ──► Tool Executor
                     └── quá N lần ──► Tool Executor (plan cuối, không kiểm tra)
```

---

---

## 4. ASR: ChunkFormer hoặc Zipformer, kèm search-key biasing

`pipeline.py --asr-model {chunkformer,zipformer}` (mặc định `chunkformer`). Zipformer là
`hynt/Zipformer-30M-RNNT-6000h` chạy bằng sherpa-onnx (`asr/asr_zipformer.py`): `pip install sherpa-onnx sentencepiece soundfile`.

`--bias` bật 3 lớp, đều lấy từ khoá từ `<split>_environment/<id>/state.json` (danh bạ, lịch, nhạc, kênh TV, thiết bị...),
hồ sơ user và một glossary nhỏ (Zalo, SMS, SJC...), xem `hotwords/bias.py`:

| Lớp | Chỗ chạy | Việc làm |
|---|---|---|
| Hotwords | lúc decode, chỉ Zipformer | `modified_beam_search` + hotword theo từng mẫu |
| Snap text | sau ASR, trước ITN | thay đoạn nghe gần giống entity theo phát âm ("gia lô" → "Zalo", "ét em ét" → "SMS") |
| Snap search key | trước khi gọi tool | `contacts.search`, `tv.*`... đổi tham số về tên có thật trong app (`SEARCH_KEYS` trong `pipeline.py`) |

Snap không đụng entity tiếng Việt một âm tiết (Lan, Bống...) vì dễ đè lên từ thường chỉ khác dấu ("đá bóng" → "đá Bống").
Mọi thay đổi được ghi trong `*.trace.jsonl` (`snap_text`, `snap`).

Đo bằng `compare_asr.py` (đã xoá; nay dùng `asr/eval_asr.py`) (300 câu train không chứa số, word accuracy):

| | ChunkFormer | Zipformer | Zipformer + hotwords |
|---|---|---|---|
| gốc | 81.4% | 83.2% | 84.2% |
| + snap text | 81.7% | 83.5% | 84.4% |

---

## 5. Refine: đánh dấu từ confidence thấp rồi cho LLM sửa (`--refine`)

Cách thay cho `snap_text` + `asr_fix`, không cần luật hay alias viết tay. Chỉ chạy với Zipformer.

```
Audio ─► Zipformer (+RNNLM) ─► [(từ, conf)] ─► đánh dấu conf < ngưỡng + ứng viên ─► LLM sửa ─► ITN ─► mapper ─► planner
```

1. **Confidence theo từ** (`asr/asr_zipformer.py`, `transcribe(..., words=True)`): sherpa-onnx trả `ys_log_probs` cho từng
   token BPE. Token gộp thành từ, conf của từ = `exp(log-prob nhỏ nhất)` trong các token của nó.
2. **Ứng viên** (`asr/candidates.py`, `Candidates.from_train`): học từ `data/asr_train_tn.jsonl` (bỏ held-out `id % 10 == 9`):
   - bảng nhầm lẫn: căn `hyp` (ASR) với `ref_spoken`, đếm cặp thay thế 1-1 (giữ cặp gặp ≥ 2 lần);
   - từ gần âm trong từ vựng train theo `hotwords.bias.phon` (bỏ dấu, gộp s/x, ch/tr, d/gi/r...), giống âm hơn và hay gặp hơn xếp trước.
3. **Đánh dấu**: từ có conf < `--conf-thr` được viết thành `[từ ASR|ứng viên 1|ứng viên 2|...]`, ví dụ
   `nếu mai [mơ|mai|mở|mô] thì [giờ|dời|dò|nhờ] chạy bộ ...`. Lượt không có từ nào bị đánh dấu thì không gọi LLM.
4. **LLM sửa** (`REFINE_SYSTEM` trong `asr/candidates.py`, dùng `--llm`): mỗi chỗ đánh dấu giữ từ cũ, chọn ứng viên, hoặc chọn từ gần âm
   khác hợp ngữ cảnh; từ ngoài ngoặc giữ nguyên; trả chữ thường, số dạng chữ. Kèm tối đa 2 lượt trước làm ngữ cảnh.
   `accept_refine` giữ câu ASR nếu LLM trả rỗng, còn ngoặc, lệch số từ nhiều hoặc viết lại phần lớn câu.
5. **ITN** chạy trên câu đã sửa, sau đó như cũ (mapper → planner).

Tham số: `--conf-thr` (mặc định 0.6), `--cand-k` (số ứng viên, mặc định 3), `--cand-train` (jsonl có `hyp` + `ref_spoken`).
Trace ghi `{"refine": {"marked", "to"}}`; `--no-llm` ghi các trường `asr / marked / refine / itn` cho từng lượt.

```bash
# chỉ phần âm thanh, có nhãn -> in WER (so với luồng cũ snap + fix khi bỏ --refine)
python pipeline.py --root DATA --split train --asr-model zipformer --lm lm/rnnlm/rnnlm.onnx --refine --no-llm --limit 200
# cả pipeline
python pipeline.py --root DATA --split public_test --asr-model zipformer --lm lm/rnnlm/rnnlm.onnx --refine --itn ITN_CKPT
```

Chọn ngưỡng (150 câu held-out train, Zipformer beam, chưa có hotword/LM):

| `--conf-thr` | từ bị đánh dấu | từ sai bắt được | precision |
|---|---|---|---|
| 0.5 | 10.9% | 29.7% | 43.4% |
| 0.6 | 17.4% | 41.2% | 37.6% |
| 0.7 | 24.2% | 52.0% | 34.0% |

Ngưỡng cao bắt được nhiều lỗi hơn nhưng cũng cho LLM nhiều cơ hội sửa nhầm từ đúng; nên đo lại sau khi bật LM.

Giới hạn:
- Từ bị ASR bỏ sót không có confidence nên không đánh dấu được.
- Ứng viên chỉ lấy từ từ vựng train: tên riêng mới (danh bạ, quán, kênh của từng mẫu) để planner tra bằng tool đọc
  (`contacts.search`, `tv.guide`, `calendar.list_events`...), `snap_search_keys` vẫn chạy trước khi gọi tool.
- Một số ứng viên còn thiếu ("mơ" không ra "mưa", "sẵn" không ra "xăng"); prompt cho phép LLM tự chọn từ gần âm ngoài danh sách.
- Không dùng với `--text`; `--llm gold` chỉ đánh dấu, không gọi LLM.
