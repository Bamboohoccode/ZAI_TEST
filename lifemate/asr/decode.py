"""Decode các lượt user của một mẫu: hotword riêng từng lượt (--bias), fallback về bản không bias khi câu bị cụt."""


def run_asr_words(model, paths, hotwords=None):
    """Decode các file của một mẫu (history + query) trong một lô (deterministic). -> [[(từ, conf)]] theo file."""
    return model.transcribe([str(p) for p in paths], hotwords=hotwords, words=True)


def run_asr(model, paths, hotwords=None, confs=None):
    """-> [chuỗi chữ thường] theo file. Ghép từ của word_confidences (gộp token BPE, xử lý token '▁' đứng riêng) thay vì
    result.text. confs: list -> nối thêm [conf từng từ] của mỗi file (cùng thứ tự)."""
    wls = run_asr_words(model, paths, hotwords)
    if confs is not None:
        confs.extend([round(c, 4) for _, c in wl] for wl in wls)
    return [" ".join(w for w, _ in wl) for wl in wls]


def decode_turns(asr, paths, lexs=None, fallback=0.6, plain=None, confs=None):
    """ASR các lượt; lexs[k] = hotword riêng của lượt k (None = không bias). Các lượt cùng hotword decode chung một lô.
    fallback > 0: lượt có bias ra ít hơn fallback * số từ của bản không bias (hotword khớp nửa chừng làm cụt câu) -> dùng
    bản không bias (`plain`: kết quả không bias đã có sẵn, None = tự decode; plain_confs tương ứng nếu muốn conf).
    confs: list rỗng -> điền conf từng từ của bản được chọn cho mỗi lượt. -> (texts, fell)."""
    if not lexs:
        return run_asr(asr, paths, None, confs), [False] * len(paths)
    texts, cf, groups = [None] * len(paths), [None] * len(paths), {}
    for k, lx in enumerate(lexs):
        groups.setdefault(tuple(lx), []).append(k)
    for lx, ks in groups.items():
        c = []
        for k, t, ck in zip(ks, run_asr(asr, [paths[k] for k in ks], list(lx) or None, c), c):
            texts[k], cf[k] = t, ck
    fell = [False] * len(paths)
    if fallback > 0:
        pc = []
        if plain is None:
            plain = run_asr(asr, paths, None, pc)
        for k, pl in enumerate(plain):
            if len(texts[k].split()) < fallback * len(pl.split()):
                texts[k], fell[k] = pl, True
                cf[k] = pc[k] if pc else [1.0] * len(pl.split())
    if confs is not None:
        confs.extend(cf)
    return texts, fell
