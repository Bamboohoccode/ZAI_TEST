"""Chạy pipeline LifeMate -> submission.jsonl (+ .trace.jsonl). Split có nhãn (train) -> tự chấm điểm.

    # nộp bài
    python scripts/run.py --root DATA --split public_test --bias --prefetch --out submission.jsonl
    # đo trên held-out của train (id % 10 == 9), tự chấm
    python scripts/run.py --root DATA --split train --heldout --bias --prefetch --out sub_heldout.jsonl
    # chỉ ASR -> WER (không mapper / planner)
    python scripts/run.py --root DATA --split train --heldout --asr-only --lm models/rnnlm/rnnlm.onnx --beam 8
    # kiểm tra env / scorer không cần GPU: bỏ ASR, phát lại nhãn (điểm phải ~1.0)
    python scripts/run.py --root DATA --split train --text --llm gold --mapper none --limit 50
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lifemate import config  # noqa: E402
from lifemate.bias import common_words  # noqa: E402
from lifemate.data import import_tool_env, load_catalog, load_items, select  # noqa: E402
from lifemate.llm import GoldLLM, make_llm  # noqa: E402
from lifemate.pipeline import run_asr_only, run_sample  # noqa: E402
from lifemate.score import report, score  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("dữ liệu")
    g.add_argument("--root", required=True, help="thư mục chứa các split (train/, public_test/)")
    g.add_argument("--split", default="public_test")
    g.add_argument("--tools-dir", help="thư mục CHỨA package zaic_tools (mặc định tự tìm trong root/split, root/train, root)")
    g.add_argument("--out", default="submission.jsonl")
    g.add_argument("--heldout", action="store_true", help="chỉ các mẫu held-out (id %% 10 == 9)")
    g.add_argument("--ids", help="chỉ các mẫu này: \"train_0001,train_0003\" hoặc file txt mỗi dòng 1 id")
    g.add_argument("--limit", type=int, default=0)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--no-score", action="store_true", help="không tự chấm khi split có nhãn")

    g = ap.add_argument_group("ASR (Zipformer, sherpa-onnx, CPU)")
    g.add_argument("--zipformer", default=config.ZIPFORMER_MODEL, help="HF id hoặc thư mục (encoder/decoder/joiner-*.onnx + bpe.model)")
    g.add_argument("--zipformer-threads", type=int, default=4)
    g.add_argument("--beam", type=int, default=4, help="max_active_paths của modified_beam_search")
    g.add_argument("--lm", help=f"RNNLM onnx cho LM fusion (scripts/train_rnnlm.py -> {config.RNNLM_FILE})")
    g.add_argument("--lm-scale", type=float, default=0.3)
    g.add_argument("--vad", action="store_true", help="cắt khoảng lặng bằng silero VAD trước Zipformer")
    g.add_argument("--hotwords-file", help=f"hotword CỐ ĐỊNH (scripts/mine_hotwords.py -> {config.HOTWORDS_FILE})")
    g.add_argument("--bias", action="store_true",
                   help="hotword theo từng mẫu: tên từ hồ sơ user + tool call / kết quả trong history")
    g.add_argument("--hotword-score", type=float, default=2.0, help="score khởi tạo contextual graph")
    g.add_argument("--hotword-hi", type=float, default=2.5, help="score hotword cho tên riêng / viết tắt")
    g.add_argument("--hotword-lo", type=float, default=1.5, help="score hotword cho cụm thường")
    g.add_argument("--glossary-score", type=float, default=1.0, help="score cho từ ngoại viết thường (podcast, voucher...)")
    g.add_argument("--bias-fallback", type=float, default=0.6,
                   help="lượt có bias ra ít hơn tỉ lệ này số từ của bản không bias (câu cụt) -> dùng bản không bias; 0 = tắt")
    g.add_argument("--prefetch", action="store_true",
                   help="sau ASR, LLM gọi tool đọc liên quan một lần (trên env thật, tính vào tool_calls) -> tên trong kết quả "
                        "làm hotword -> decode lại câu cuối (kéo theo --bias)")
    g.add_argument("--asr-only", action="store_true", help="chỉ ASR (+prefetch), không mapper / planner: in WER")
    g.add_argument("--text", action="store_true", help="dùng audio_text thay cho ASR (chỉ train)")

    g = ap.add_argument_group("LLM (vLLM)")
    g.add_argument("--llm", default=config.LLM_MODEL, help='HF id hoặc thư mục; "gold" = phát lại nhãn (train)')
    g.add_argument("--llm-lora")
    g.add_argument("--llm-backend", choices=["vllm", "hf"], default="vllm", help="hf = transformers (chạy trên CPU, chậm)")
    g.add_argument("--llm-device", help="GPU riêng cho LLM chính (process con), vd. 1")
    g.add_argument("--gpu-mem", type=float, default=0.7, help="gpu_memory_utilization của LLM chính")
    g.add_argument("--max-len", type=int, default=config.MAX_CTX, help="max_model_len; GPU thiếu bộ nhớ KV -> giảm (8192)")
    g.add_argument("--eager", action="store_true", help="vLLM enforce_eager (bỏ CUDA graph)")
    g.add_argument("--mapper", default=config.MAPPER_MODEL, help='"none" = tắt, "same" = dùng chung model với --llm')
    g.add_argument("--mapper-lora")
    g.add_argument("--mapper-device", help="GPU riêng cho mapper")
    g.add_argument("--mapper-gpu-mem", type=float, default=0.1)
    g.add_argument("--verify-rounds", type=int, default=config.VERIFY_ROUNDS)
    g.add_argument("--time-limit", type=float, default=0,
                   help=f"giây mỗi mẫu: quá hạn thì dừng gọi tool, nộp những gì đã chạy (BTC: {config.TIME_LIMIT_S}); 0 = không chặn")
    a = ap.parse_args(argv)
    if a.prefetch:
        if a.text or a.llm == "gold":
            ap.error("--prefetch cần LLM thật + ASR: không dùng với --text / --llm gold")
        a.bias = True  # hotword decode lại cần contextual graph của --bias
    if a.asr_only and a.text:
        ap.error("--asr-only cần ASR, không dùng với --text")
    if a.asr_only and a.out == "submission.jsonl":  # đừng ghi đè file nộp bằng output không có tool call
        a.out = "asr_out.jsonl"
    a.root = Path(a.root).expanduser()
    return a


def seed_all(seed):
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # engine trong process: tái lập được, giải phóng được
    random.seed(seed)
    try:
        import numpy as np
        import torch
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def main(argv=None):
    a = parse_args(argv)
    seed_all(a.seed)
    gold = a.llm == "gold"
    ToolEnv = import_tool_env(a.root, a.split, a.tools_dir)
    catalog = load_catalog(a.root, a.split)
    all_items = load_items(a.root, a.split)
    items = select(all_items, a.heldout, a.ids, a.limit)
    # từ thường của split (không coi chữ hoa đầu câu là tên riêng khi lấy entity từ text)
    common = frozenset(common_words(a.root / a.split / f"{a.split}.jsonl")) if a.bias else frozenset()

    asr = None
    if not a.text:
        from lifemate.asr.zipformer import ZipformerASR
        asr = ZipformerASR(a.zipformer, num_threads=a.zipformer_threads, bias=a.bias, hotwords_score=a.hotword_score,
                           hi=a.hotword_hi, lo=a.hotword_lo, glossary_score=a.glossary_score, beam=a.beam,
                           hotwords_file=a.hotwords_file, lm=a.lm, lm_scale=a.lm_scale, vad=a.vad)

    if a.asr_only:
        llm = make_llm(a.llm_device, a.llm_backend, a.llm, a.llm_lora, a.gpu_mem, a.seed, a.max_len, 256,
                       {"enable_thinking": False}, a.eager) if a.prefetch else None
        return run_asr_only(a, items, ToolEnv, catalog, asr, llm, common, a.out)

    llm = GoldLLM() if gold else make_llm(a.llm_device, a.llm_backend, a.llm, a.llm_lora, a.gpu_mem, a.seed, a.max_len,
                                          config.MAX_NEW, None, a.eager)
    share = a.mapper == "same" or (a.mapper == a.llm and a.mapper_lora == a.llm_lora)  # chỉ nạp 1 model lên GPU
    mapper = None if a.mapper == "none" or gold else llm if share else make_llm(
        a.mapper_device, a.llm_backend, a.mapper, a.mapper_lora, a.mapper_gpu_mem, a.seed, 2048, 256,
        {"enable_thinking": False}, a.eager)
    rounds = 0 if gold else a.verify_rounds

    t_start, slow, shown_tb, sub = time.time(), [], False, []
    with open(a.out, "w", encoding="utf-8") as fout, open(a.out + ".trace.jsonl", "w", encoding="utf-8") as ftr:
        for i, item in enumerate(items):
            res = run_sample(a, item, ToolEnv, catalog, asr, llm, mapper, common, gold, rounds)
            err = next((s for s in res.trace if "error" in s), None)
            if err and "traceback" in err and not shown_tb:  # traceback đầy đủ của lỗi đầu tiên; các lỗi sau xem trace
                shown_tb = True
                print(err["traceback"], file=sys.stderr)
            if res.seconds > config.TIME_LIMIT_S:
                slow.append((item["id"], round(res.seconds, 1)))
            row = res.submission(item["id"])
            sub.append(row)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            ftr.write(json.dumps(res.trace_row(item["id"]), ensure_ascii=False) + "\n")
            print(f"[{i + 1}/{len(items)}] {item['id']} calls={len(res.calls)} {res.seconds:.1f}s "
                  f"{(err or {}).get('error', '')[:120]}", flush=True)

    total = time.time() - t_start
    print(f"\n{len(items)} mẫu -> {a.out} | TB {total / max(len(items), 1):.1f}s/mẫu | > {config.TIME_LIMIT_S}s: {len(slow)}")
    for sid, dt in slow:
        print(f"  {sid}: {dt}s")
    if not a.no_score and items and "label" in items[0]:
        report(*score(sub, all_items, catalog, ToolEnv, a.split, a.root))


if __name__ == "__main__":
    main()
