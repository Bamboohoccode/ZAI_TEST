"""Vẽ docs/asr_workflow.png (quy trình ASR: Kaggle chuẩn bị -> máy local chạy pipeline).

    python docs/make_asr_workflow.py            # ghi đè docs/asr_workflow.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

W, H = 1556, 1900
STYLE = {  # fill, edge
    "kaggle": ("#dbe8fd", "#1d4ed8"),
    "file": ("#fdf7c3", "#a16207"),
    "new": ("#ffedd5", "#c2410c"),
    "old": ("#dcfce7", "#15803d"),
    "opt": ("#f3f4f6", "#4b5563"),
}
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")


def box(x, y, w, h, title, sub="", style="old", dashed=False, mono=True):
    fc, ec = STYLE[style]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=10", fc=fc, ec=ec, lw=2,
                                ls=(0, (4, 3)) if dashed else "-"))
    cy = y + h / 2 if not sub else y + h * 0.36
    ax.text(x + w / 2, cy, title, ha="center", va="center", fontsize=11.5, weight="bold", color="#111827")
    if sub:
        ax.text(x + w / 2, y + h * 0.70, sub, ha="center", va="center", fontsize=8.6, color="#374151",
                family="monospace" if mono else None, linespacing=1.5)
    return (x, y, w, h)


def arrow(p, q, color="#374151"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, lw=2, color=color))


def down(a, b, dx_a=0.5, dx_b=0.5):
    arrow((a[0] + a[2] * dx_a, a[1] + a[3]), (b[0] + b[2] * dx_b, b[1]))


def panel(x, y, w, h, color, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=22", fc="none", ec=color, lw=2.5,
                                ls=(0, (6, 4))))
    ax.text(x + 24, y + 32, title, fontsize=15, weight="bold", color=color, va="center")


ax.text(W / 2, 45, "Quy trình ASR: Kaggle (chuẩn bị 1 lần) → máy local (chạy pipeline offline)", ha="center",
        va="center", fontsize=17, weight="bold", color="#111827")

# ------------------------------------------------------------------ ① Kaggle
panel(35, 105, 695, 1225, "#1d4ed8", "① Kaggle — notebooks/kaggle_hotwords_lm.ipynb")
A = box(65, 200, 635, 135, "A. Decode train KHÔNG hotword (beam)",
        "asr/eval_asr.py --subset all --configs beam\n--dump data/asr_train_plain.jsonl  (hyp + ref_spoken)", "kaggle")
B = box(52, 418, 320, 145, "B. (tuỳ chọn) Khai thác hotword",
        "hotwords/mine_hotwords.py\nlist global dễ chèn tên sai\n→ ưu tiên --bias theo state.json", "kaggle",
        mono=False)
C = box(393, 418, 320, 145, "C. Train RNNLM (GPU)", "lm/train_rnnlm.py\nearly-stop theo held-out", "kaggle")
down(A, B, 0.28)
down(A, C, 0.72)
Bf = box(52, 638, 320, 105, "hotwords_global.txt / .json", style="file")
Cf = box(393, 638, 320, 105, "lm/rnnlm/rnnlm.onnx", style="file")
down(B, Bf)
down(C, Cf)
D = box(65, 830, 635, 155, "D. Eval trên held-out (id%10==9)",
        "asr/eval_asr.py --configs beam,hw,hw+lm,...\n→ chọn: lm_scale, beam 4/8, VAD bật/tắt\n"
        "pipeline.py --split train --mark-only → bảng chọn --conf-thr", "kaggle")
arrow((Bf[0] + Bf[2] / 2, Bf[1] + Bf[3]), (D[0] + 146, D[1]))
arrow((Cf[0] + Cf[2] / 2, Cf[1] + Cf[3]), (D[0] + 487, D[1]))
E = box(65, 1055, 635, 125, "E. Tải asr_assets.zip",
        "lm/rnnlm/rnnlm.onnx  data/asr_train_plain.jsonl (cho --cand-train)\nhotwords/…  eval_beam*.txt  eval_conf_thr.txt", "file")
down(D, E)
ax.text(382, 1262, "Trọng số Zipformer không đổi ở đây — chỉ thêm RNNLM (+ hotword tuỳ chọn)", ha="center",
        fontsize=10.5, style="italic", color="#1d4ed8")

# ------------------------------------------------------------------ ② local
X, BW = 806, 686
panel(776, 105, 746, 1700, "#15803d", "② Máy local — offline")
U = box(X, 197, BW, 98, "Giải nén asr_assets.zip ở gốc repo", style="file")
ax.plot([700, 753, 753], [1118, 1118, 246], color="#a16207", lw=2, solid_joinstyle="miter")
arrow((753, 246), (X, 246), color="#a16207")
ax.text(744, 690, "copy về máy local", rotation=90, ha="center", va="center", fontsize=10.5, style="italic",
        color="#a16207")
O = box(X, 313, BW, 115, "(tuỳ chọn) duyệt ứng viên, không cần GPU",
        "python pipeline.py --root DATA --split train --lm lm/rnnlm/rnnlm.onnx \\\n"
        "  --mark-only --out asr_marked.jsonl", "opt", dashed=True)
down(U, O)
ax.text(X + BW / 2, 470, "python pipeline.py — các bước bên trong", ha="center", fontsize=13, weight="bold",
        color="#15803d")
s1 = box(X, 510, BW, 75, "Audio: các lượt user (history) + query", style="old")
s2 = box(X, 615, BW, 95, "VAD — cắt khoảng lặng", "--vad   (chỉ bật nếu bước D cho thấy giảm WER)", "new")
s3 = box(X, 740, BW, 120, "Zipformer beam search + RNNLM → (từ, confidence)",
         "--lm lm/rnnlm/rnnlm.onnx --lm-scale X --beam N  [--bias]\nconf = exp(min log-prob token) từ ys_log_probs",
         "new")
s4 = box(X, 890, BW, 130, "Đánh dấu từ conf < --conf-thr + ứng viên gần âm",
         "asr/candidates.py: bảng nhầm lẫn (hyp↔ref train) + từ gần âm (phon)\n"
         "nếu mai [mơ|mai|mở|mô] thì [giờ|dời|dò|nhờ] chạy bộ ...", "new")
s5 = box(X, 1050, BW, 110, "LLM refine (Qwen) — chọn từ hợp ngữ cảnh",
         "chỉ gọi cho lượt có đánh dấu; accept_refine giữ câu ASR nếu LLM viết lại", "new")
s6 = box(X, 1190, BW, 75, "ITN (số, ngày giờ) → mapper", style="old")
s7 = box(X, 1295, BW, 95, "Planner ⇄ Verifier (Qwen) → Tool executor",
         "tên riêng: tra bằng tool đọc + snap_search_keys trước khi gọi tool", "old", mono=False)
s8 = box(X, 1420, BW, 75, "submission.jsonl", style="file")
for a_, b_ in zip([s1, s2, s3, s4, s5, s6, s7], [s2, s3, s4, s5, s6, s7, s8]):
    down(a_, b_)

ax.add_patch(FancyBboxPatch((X, 1540), BW, 210, boxstyle="round,pad=0,rounding_size=10", fc="#111827", ec="#111827"))
ax.text(X + 16, 1645,
        "python pipeline.py --root DATA --split public_test \\\n"
        "    --lm lm/rnnlm/rnnlm.onnx --lm-scale <tốt nhất từ D> [--beam 8] [--vad] \\\n"
        "    --conf-thr <từ D> --itn <ITN ckpt> --out submission.jsonl\n\n"
        "# --cand-train mặc định data/asr_train_plain.jsonl (bước A)",
        fontsize=9, family="monospace", color="#e5e7eb", va="center", linespacing=1.6)

# ------------------------------------------------------------------ chú thích
ax.text(54, 1392, "Chú thích", fontsize=13, weight="bold", color="#111827", va="center")
for i, (st, label) in enumerate([("kaggle", "bước chạy trên Kaggle"), ("file", "file / kết quả"),
                                 ("new", "thành phần ASR mới (VAD, LM, confidence, ứng viên, LLM refine)"),
                                 ("old", "bước pipeline có sẵn"), ("opt", "tuỳ chọn")]):
    fc, ec = STYLE[st]
    y = 1428 + i * 55
    ax.add_patch(FancyBboxPatch((52, y), 54, 38, boxstyle="round,pad=0,rounding_size=4", fc=fc, ec=ec, lw=2))
    ax.text(126, y + 19, label, fontsize=11, va="center", color="#111827")
ax.text(54, 1745,
        "Lưu ý: hotword / bảng nhầm lẫn nên lấy từ chính model ASR sẽ dùng.\n"
        "Dùng Zipformer fine-tune → chạy lại A→D với --zipformer <thư mục fine-tune>\n"
        "(fine-tune trên toàn bộ train làm hyp train quá sạch: dùng k-fold).\n"
        "data/ không commit: mang asr_train_plain.jsonl theo asr_assets.zip.",
        fontsize=10.5, color="#9a3412", va="center", linespacing=1.6)

out = Path(__file__).resolve().parent / "asr_workflow.png"
fig.savefig(out, dpi=100, facecolor="white")
print(out)
