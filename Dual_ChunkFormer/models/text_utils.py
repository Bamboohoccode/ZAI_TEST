"""Text helpers shared by data building, calibration, GRPO rewards and inference."""
import json
import re


def normalize(text):
    """lowercase, drop punctuation (keep apostrophes/hyphens inside words), split into words."""
    text = re.sub(r"[^\w\s'\-]", " ", text.lower())
    return text.replace("_", " ").split()


def _align(hyp, ref):
    """Levenshtein path as a list of (h, r) points from (0,0) to (len(hyp), len(ref)); diagonal preferred."""
    n, m = len(hyp), len(ref)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j - 1] + (hyp[i - 1] != ref[j - 1]), d[i - 1][j] + 1, d[i][j - 1] + 1)
    i, j, path = n, m, [(n, m)]
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (hyp[i - 1] != ref[j - 1]):
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
        path.append((i, j))
    return path[::-1], d[n][m]


def wer(hyp, ref):
    """word error rate of two strings (normalized); 0 for two empty strings."""
    h, r = normalize(hyp), normalize(ref)
    if not r:
        return 0.0 if not h else 1.0
    return _align(h, r)[1] / len(r)


def ref_spans(hyp_words, ref_words, spans):
    """For each hyp span [i, j) (word indices) return the aligned reference words.
    Reference words inserted between two spans go to the following span."""
    path, _ = _align(hyp_words, ref_words)
    first = {}
    for h, r in path:
        first.setdefault(h, r)
    first[len(hyp_words)] = len(ref_words)
    return [" ".join(ref_words[first[i]:first[j]]) for i, j in spans]


def word_correct(hyp_words, ref_words):
    """list[bool]: hyp word i is matched to an identical reference word on the alignment path."""
    path, _ = _align(hyp_words, ref_words)
    ok = [False] * len(hyp_words)
    for (h0, r0), (h1, r1) in zip(path, path[1:]):
        if h1 == h0 + 1 and r1 == r0 + 1 and hyp_words[h0] == ref_words[r0]:
            ok[h0] = True
    return ok


def parse_choices(text):
    """{slot_id: str} from a model answer like {"SLOT_1": "abc", "2": "x"}; {} if unparseable."""
    m = re.search(r"\{.*\}", text, re.S)
    try:
        obj = json.loads(m.group(0)) if m else {}
    except ValueError:
        return {}
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            mk = re.search(r"\d+", str(k))
            if mk and isinstance(v, str):
                out[int(mk.group(0))] = v.strip()
    return out


def fill_slots(sentence, slots, choices):
    """Replace [SLOT_k] by the chosen text (fallback: the Vi base text of the slot)."""
    for s in slots:
        pick = choices.get(s["id"])
        sentence = sentence.replace(f"[SLOT_{s['id']}]", s["base"] if pick is None else pick)
    return " ".join(sentence.split())
