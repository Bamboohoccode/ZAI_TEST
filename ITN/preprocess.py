import datasets
import model
from transformers import PreTrainedTokenizerBase
from typing import Optional, Union, Any
from transformers.file_utils import PaddingStrategy
import re
import time
import random
from dataclasses import dataclass
import validators

tokenizer = None

@dataclass
class DataCollatorInvertTextNormalization:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        batch_src, batch_tgt = [], []
        for item in features:
            src_spans, tgt_spans = create_invert_text_norm(item['src'], item['tgt'])
            batch_src.append(src_spans)
            batch_tgt.append(tgt_spans)

        # print("Batch: ", batch_src, batch_tgt)

        features = preprocess_function({"src": batch_src, "tgt": batch_tgt})
        # print("Preprocess batch: ", features)

        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["outputs"] for feature in features] if "outputs" in features[0].keys() else None
        spoken_labels = [feature["spoken_label"] for feature in features] if "spoken_label" in features[0].keys() else None
        spoken_idx = [feature["src_spoken_idx"] for feature in features] if "src_spoken_idx" in features[0].keys() else None

        word_src_lengths = [feature["inputs_length"] for feature in features] if "inputs_length" in features[0].keys() else None
        word_tgt_lengths = [feature["outputs_length"] for feature in features] if "outputs_length" in features[0].keys() else None

        # print("Spoken labels: ", spoken_labels)
        
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            max_src_length = max(len(l) for l in spoken_labels)
            max_spoken_idx_length = max(len(l) for l in spoken_idx)
            max_word_src_length = max(len(l) for l in word_src_lengths)
            max_word_tgt_length = max(len(l) for l in word_tgt_lengths)

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["outputs"]))
                remainder_word_tgt_length = [0] * (max_word_tgt_length - len(feature["outputs_length"]))
                remainder_spoken = [self.label_pad_token_id] * (max_src_length - len(feature["spoken_label"]))
                remainder_spoken_idx = [self.label_pad_token_id] * (max_spoken_idx_length - len(feature["src_spoken_idx"]))
                remainder_word_src_length = [0] * (max_word_src_length - len(feature["inputs_length"]))

                feature["labels"] = (
                    feature["outputs"] + [
                        self.tokenizer.eos_token_id] + remainder if padding_side == "right" else remainder + feature[
                        "outputs"] + [self.tokenizer.eos_token_id]
                )

                feature["spoken_label"] = [self.label_pad_token_id] + feature["spoken_label"] + [self.label_pad_token_id]
                feature["spoken_label"] = feature["spoken_label"] + remainder_spoken if padding_side == "right" else remainder_spoken + feature["spoken_label"]
                feature["src_spoken_idx"] = feature["src_spoken_idx"] + remainder_spoken_idx

                feature['inputs_length'] = [1] + feature['inputs_length'] + [1]
                feature['outputs_length'] = feature['outputs_length'] + [1]

                feature["inputs_length"] = feature["inputs_length"] + remainder_word_src_length
                feature["outputs_length"] = feature["outputs_length"] + remainder_word_tgt_length

        # print("Features: ", features)

        features_inputs = [{
            "input_ids": [self.tokenizer.bos_token_id] + item["input_ids"] + [self.tokenizer.eos_token_id],
            "attention_mask": [self.tokenizer.pad_token_id] + item["attention_mask"] + [self.tokenizer.pad_token_id]
        } for item in features]
        features_inputs = self.tokenizer.pad(
            features_inputs,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        outputs = self.tokenizer.pad({"input_ids": [feature["labels"] for feature in features]},
                                     return_tensors=return_tensors)['input_ids']

        spoken_label = self.tokenizer.pad({"input_ids": [feature["spoken_label"] for feature in features]},
                                          return_tensors=return_tensors)['input_ids']

        spoken_idx = self.tokenizer.pad({"input_ids": [feature["src_spoken_idx"] for feature in features]},
                                        return_tensors=return_tensors)['input_ids'] + 1  # 1 for bos token

        word_src_lengths = self.tokenizer.pad({"input_ids": [feature["inputs_length"] for feature in features]},
                                              return_tensors=return_tensors)['input_ids']

        word_tgt_lengths = self.tokenizer.pad({"input_ids": [feature["outputs_length"] for feature in features]},
                                              return_tensors=return_tensors)['input_ids']

        features = {
            "input_ids": features_inputs["input_ids"],
            "spoken_label": spoken_label,
            "spoken_idx": spoken_idx,
            "word_src_lengths": word_src_lengths,
            "word_tgt_lengths": word_tgt_lengths,
            "attention_mask": features_inputs["attention_mask"],
            "labels": outputs,
        }

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features

def create_invert_text_norm(item_1, item_2):
    src_list, tgt_list = [], []
    for src, tgt in zip(item_1, item_2):
        if src == tgt:  # unchanged word: keep only 10% (was len(src) == len(tgt), which also dropped changed spans like 'năm xị' -> '500000')
            if random.random() < 0.1:
                src_list.append(src)
                tgt_list.append(tgt)
        else:
            src_list.append(src)
            tgt_list.append(tgt)
    
    return src_list, tgt_list

# sentence templates for the JSON spans: "{}" is one slot, filled by a sampled record's spoken/formatted form.
# Written for money/price spans (dataset_vietnam_money.json); pass --template_file for other kinds of spans.
TEMPLATES = [
    "Qua đây tôi bán cho anh cái xe máy {} với con laptop {}",
    "Chuyển khoản cho chị Lan {} vào tài khoản sáng nay",
    "Cái điện thoại này giá {} chứ không rẻ hơn đâu",
    "Hôm qua em mới mua cái tủ lạnh hết {}",
    "Tháng này tiền điện nhà mình mất {}",
    "Bạn ấy vừa nhận lương {} rồi đi ăn liền",
    "Cho tôi vay {} đến cuối tháng trả nhé",
    "Phòng trọ này thuê một tháng {} cả điện nước",
    "Ông chủ trả công {} cho mỗi ngày làm việc",
    "Đặt cho tôi cái bàn học giá khoảng {} thôi",
    "Ăn tối hôm nay tổng cộng hết {} đó",
    "Anh đưa em {} để đóng học phí cho con",
    "Chiếc xe đạp này bán lại được {} còn cái máy giặt thì {}",
    "Mình đã trả trước {} còn lại {} thì trả sau",
    "Vé máy bay ra Hà Nội đang giảm còn {}",
    "Nhớ nhắc tôi đóng tiền mạng {} vào cuối tuần",
    "Quán cà phê này một ly có {} thôi mà ngon lắm",
    "Sửa cái điều hòa tốn {} mà vẫn chưa chạy",
    "Hóa đơn tháng trước là {} còn tháng này là {}",
    "Gửi giúp mình {} cho bạn Nam qua ví điện tử",
]


# Money is always written "đ": no "vnđ", no "đồng", whatever the spoken form ("năm xị", "trăm hai", "việt nam đồng"...).
# VietAI writes the same amount as "N vnđ" / "N đồng" / "N đ" (about 1/3 each), so its targets are mapped to "đ" too,
# in the plain amount ("232 đồng" -> "232 đ") and in compound units ("725 đồng/ml" -> "725 đ/ml", "865 cm/vnđ" -> "865 cm/đ").
# "đô la" is untouched: VietAI already writes it "$" (120 plain + compound units, no other form).
# Safety: "đ" is never ADDED, and a span is only touched when its spoken form says it is money (đồng), so codes
# ("đờ ngang chín không không" -> "đ-900"), dates ("ngày 30/06/1150") and ids keep their written form.
MONEY_CUE_RE = re.compile(r"đồng", re.I)  # not "đê": it is also the letter d in spoken codes ("vê en đê gờ")
MONEY_END_RE = re.compile(r"^(.*?\d)\s*(?:vnđ|vnd|đồng|đ)$", re.I)  # amount + unit at the very end
GLUED_AMOUNT_RE = re.compile(r"^(-?[\d.,]+)đ$")  # "600đ": a bare number with đ glued on can only be an amount
MONEY_UNIT_RE = re.compile(r"(\d\s*|/)(?:vnđ|vnd|đồng)(?![\w])", re.I)  # unit inside a compound ("đồng/ml", "cm/vnđ")


def normalize_money(spoken, written, trusted=False):
    """Written form of a money span with its unit as 'đ'. `trusted`: the span is known to be money (the JSON file),
    otherwise the spoken form must contain a money cue. Anything else is returned unchanged."""
    g = GLUED_AMOUNT_RE.match(written)
    if g:
        return f"{g.group(1)} đ"
    if not trusted and not MONEY_CUE_RE.search(spoken):
        return written
    m = MONEY_END_RE.match(written)
    if m:
        return f"{m.group(1)} đ"
    return MONEY_UNIT_RE.sub(lambda m: m.group(1) + "đ", written)


def normalize_example(ex):
    """datasets.map fn: money targets of the changed spans -> 'đ'."""
    return {"tgt": [normalize_money(s, t) if t != s else t for s, t in zip(ex["src"], ex["tgt"])]}


def audit_money(name, rows):
    """Checks to pass before training. rows: iterable of (src list, tgt list). Raises if a money span still ends in vnđ/đồng."""
    n_money = n_leftover = 0
    left = []
    for src, tgt in rows:
        for s, t in zip(src, tgt):
            if s != t and MONEY_CUE_RE.search(s):
                n_money += 1
                if re.search(r"(\d\s*|/)(?:vnđ|vnd|đồng)(?![\w])", t, re.I):
                    n_leftover += 1
                    left.append((s, t))
    print(f"[money audit] {name}: {n_money} money spans (spoken has đồng), {n_leftover} still written vnđ/đồng")
    if n_leftover:
        raise ValueError(f"{name}: money targets not normalized, e.g. {left[:3]}")


def load_templates(path=None):
    """One template per line, '{}' marks a slot (at least one per line). Default: TEMPLATES."""
    if not path:
        return TEMPLATES
    with open(path, encoding="utf-8") as f:
        t = [l.strip() for l in f if l.strip() and "{}" in l]
    if not t:
        raise ValueError(f"no template with a '{{}}' slot in {path}")
    return t


def apply_templates(records, templates=None, repeat=1, bare_prob=0.2, seed=0):
    """records: [{'spoken','formatted'}] -> rows {'src': [...], 'tgt': [...]} in the VietAI layout:
    every template word is a (word, word) pair, every slot is one (spoken, formatted) pair.
    Each record fills the first slot of `repeat` sentences; other slots (and the template) are sampled, so each
    record is seen at least `repeat` times. With probability bare_prob the span is kept alone, as before."""
    rng = random.Random(seed)
    templates = templates or TEMPLATES
    pair = lambda rec: (rec["spoken"], normalize_money(rec["spoken"], rec["formatted"], trusted=True))  # noqa: E731
    rows = []
    for _ in range(repeat):
        for r in records:
            if rng.random() < bare_prob:
                sp, fm = pair(r)
                rows.append({"src": [sp], "tgt": [fm]})
                continue
            first, src, tgt = True, [], []
            for part in re.split(r"(\{\})", rng.choice(templates)):
                if part == "{}":
                    rec = r if first else records[rng.randrange(len(records))]
                    first = False
                    sp, fm = pair(rec)
                    src.append(sp)
                    tgt.append(fm)
                else:
                    src += part.split()
                    tgt += part.split()
            rows.append({"src": src, "tgt": tgt})
    return rows


# data init
def init_data(json_path=None, only_json=False, viet_frac=0.02, template=True, template_file=None, template_repeat=1,
              bare_prob=0.2):
    """Default: VietAI/spoken_norm_assignment (random viet_frac of train, 1.0 = all).
    json_path (list of {'spoken','formatted'} spans):
      only_json=True  -> train and valid are that file only
      only_json=False -> train = VietAI shard + the file (general + domain data), valid = VietAI valid
    """
    if json_path:
        import json
        with open(json_path, encoding="utf-8") as f:
            records = json.load(f)
        n_fix = sum(normalize_money(r["spoken"], r["formatted"], trusted=True) != r["formatted"] for r in records)
        print(f"JSON: {len(records)} spans, {n_fix} money targets rewritten to 'đ'")
        if template:  # put each span into a sampled sentence template (context words stay unchanged)
            rows = apply_templates(records, load_templates(template_file), template_repeat, bare_prob)
            for r in rows[:3]:
                print("Template sample:", " ".join(r["src"]), "=>", " ".join(r["tgt"]))
        else:  # each record is one span: spoken form -> written form
            rows = [{"src": [r["spoken"]], "tgt": [normalize_money(r["spoken"], r["formatted"], trusted=True)]} for r in records]
        extra = datasets.Dataset.from_list(rows)
    if json_path and only_json:
        # use every record for training; valid is the same data (only used for eval loss / best-checkpoint selection)
        dataset = datasets.DatasetDict({"train": extra, "valid": extra})
    else:
        dataset = datasets.load_dataset('VietAI/spoken_norm_assignment')
        dataset = dataset.map(normalize_example)  # VietAI money targets -> "N đ"
        train = dataset['train']
        if viet_frac < 1:  # subsample the big default set (random, fixed seed)
            train = train.shuffle(seed=0).select(range(max(1, int(len(train) * viet_frac))))
        if json_path:
            train = train.select_columns(["src", "tgt"]).cast(extra.features)
            train = datasets.concatenate_datasets([train, extra]).shuffle(seed=0)
            print(f"Train: {len(train) - len(extra)} VietAI + {len(extra)} from {json_path}")
        dataset = datasets.DatasetDict({"train": train, "valid": dataset['valid']})

    # nothing is trained before the money targets pass the audit
    for split in ("train", "valid"):
        if split == "valid" and dataset["valid"] is dataset["train"]:
            continue
        audit_money(split, zip(dataset[split]["src"], dataset[split]["tgt"]))
    print("Dataset: ", dataset)
    return dataset

def preprocess_function(batch):

    global tokenizer
    if tokenizer is None:
        tokenizer = model.init_tokenizer()

    features = []
    for src_words, tgt_words in zip(batch["src"], batch["tgt"]):
        src_ids, pad_ids, src_lengths, tgt_ids, tgt_lengths = [], [], [], [], []
        # 0: "O", 1: "B", 2: "I"
        src_spoken_label = []  

        src_spoken_idx = []
        tgt_spoken_ids = []

        for idx, (src, tgt) in enumerate(zip(src_words, tgt_words)):
            # print("Src, Tgt: ", src, tgt)
            
            is_remain = False
            if src == tgt:
                is_remain = True

            src_tokenized = tokenizer(src)
            # print("Src tokenized: ", src_tokenized)
            
            if len(src_tokenized['input_ids']) < 3:
                continue

            # hardcode fix tokenizer email
            if validators.email(tgt):
                tgt_tokenized = tokenizer(tgt.replace('@', ' @'))
            else:
                tgt_tokenized = tokenizer(tgt)
            
            if len(tgt_tokenized['input_ids']) < 3:
                continue
            
            src_ids.extend(src_tokenized["input_ids"][1:-1])
            
            if is_remain:
                src_spoken_label.extend([0 if random.random() < 0.5 else -100 for _ in range(len(src_tokenized["input_ids"][1:-1]))])
                if random.random() < 0.1:
                    # Random pick normal word for spoken norm
                    src_spoken_idx.append(idx)
                    tgt_spoken_ids.append(tgt_tokenized["input_ids"][1:-1])
            else:
                src_spoken_label.extend([1] + [2] * (len(src_tokenized["input_ids"][1:-1]) - 1))
                src_spoken_idx.append(idx)
                tgt_spoken_ids.append(tgt_tokenized["input_ids"][1:-1])

            pad_ids.extend(src_tokenized["attention_mask"][1:-1])
            src_lengths.append(len(src_tokenized["input_ids"]) - 2)
            tgt_ids.extend(tgt_tokenized["input_ids"][1:-1])
            tgt_lengths.append(len(tgt_tokenized["input_ids"]) - 2)
            
            if len(src_ids) > 80 or len(tgt_ids) > 80:
                # print("Ignore sample")
                break

        if len(src_ids) < 1 or len(tgt_ids) < 1:
            continue

        features.append({
            "input_ids": src_ids,
            "attention_mask": pad_ids,
            "spoken_label": src_spoken_label,
            "inputs_length": src_lengths,
            "outputs": tgt_ids,
            "outputs_length": tgt_lengths,
            "src_spoken_idx": src_spoken_idx,
            "tgt_spoken_ids": tgt_spoken_ids
        })

    return features


if __name__ == "__main__":
    split_datasets = init_data()

    # model, model_tokenizer = model_handling.init_model()
    # data_collator = DataCollatorForNormSeq2Seq(model_tokenizer, model=model)
    if tokenizer is None:
        tokenizer = model.init_tokenizer()
    data_collator = DataCollatorInvertTextNormalization(tokenizer=tokenizer)
    import time
    start = time.time()
    # batch = data_collator([split_datasets["train"][i] for i in [random.randint(0, 900) for _ in range(0, 64)]])
    # print(batch)
    print("Sample 0: ", split_datasets["train"][0])
    data_collator([split_datasets["train"][0]])
    # print("{}s".format(time.time() - start))
    # print(split_datasets)