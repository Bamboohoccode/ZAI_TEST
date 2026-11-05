# ZAIC_2026

Repo dự án **LifeMate** cho Zalo AI Challenge 2026: agent nhận giọng nói tiếng Việt và gọi đúng chuỗi tool của các app mô phỏng.

## Tài liệu

- [Overview_PJ.md](Overview_PJ.md): tổng quan dự án, pipeline, phân công, cấu trúc repo, rủi ro.

## Cấu trúc hiện tại

```
ZAIC_2026/
├── README.md
├── Overview_PJ.md
├── .gitignore
└── data/        # không commit, chứa dữ liệu của BTC
```

Cấu trúc mục tiêu (code, eval, docker...) xem [Overview_PJ.md](Overview_PJ.md#31-đề-xuất-kế-hoạch-chưa-tạo).

## Quy tắc nhanh

- Không commit `data/` hay trọng số model (luật cuộc thi cấm phát tán dữ liệu).
- Nhánh: `<loại>/<module>-<tên>`, ví dụ `feat/asr-khanh`. Chi tiết trong [Overview_PJ.md](Overview_PJ.md#31-quy-ước-nhánh).
- Chạy offline hoàn toàn khi suy luận.
# ZAI_TEST
