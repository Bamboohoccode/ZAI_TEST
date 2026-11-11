"""Decode các lượt user của một mẫu: hotword riêng từng lượt (--bias), fallback về bản không bias khi câu bị cụt."""


def run_asr(model, paths, hotwords=None):
    """Decode các file của một mẫu (history + query) trong một lô (deterministic). -> [chuỗi chữ thường] theo file.
    Ghép từ của word_confidences (gộp token BPE, xử lý token '▁' đứng riêng) thay vì result.text."""
    return [" ".join(w for w, _ in wl) for wl in model.transcribe([str(p) for p in paths], hotwords=hotwords, words=True)]


def decode_turns(asr, paths, lexs=None, fallback=0.6, plain=None):
    """ASR các lượt; lexs[k] = hotword riêng của lượt k (None = không bias). Các lượt cùng hotword decode chung một lô.
    fallback > 0: lượt có bias ra ít hơn fallback * số từ của bản không bias (hotword khớp nửa chừng làm cụt câu) -> dùng
    bản không bias (`plain`: kết quả không bias đã có sẵn, None = tự decode). -> (texts, fell)."""
    if not lexs:
        return run_asr(asr, paths, None), [False] * len(paths)
    texts, groups = [None] * len(paths), {}
    for k, lx in enumerate(lexs):
        groups.setdefault(tuple(lx), []).append(k)
    for lx, ks in groups.items():
        for k, t in zip(ks, run_asr(asr, [paths[k] for k in ks], list(lx) or None)):
            texts[k] = t
    fell = [False] * len(paths)
    if fallback > 0:
        for k, pl in enumerate(plain or run_asr(asr, paths, None)):
            if len(texts[k].split()) < fallback * len(pl.split()):
                texts[k], fell[k] = pl, True
    return texts, fell
