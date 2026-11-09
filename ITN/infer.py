import argparse
import json
import os
import re
import torch
import model
from model import InvertTextNormalization

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def resolve_checkpoint_path(checkpoint_path):
    """
    Resolve checkpoint directory when checkpoint_path is either:
    1. A direct checkpoint folder (containing model.safetensors or pytorch_model.bin)
    2. An output folder of main.py (e.g., ./checkpoints) containing checkpoint-<step> subfolders
    """
    if not checkpoint_path:
        return None

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    if os.path.isfile(checkpoint_path):
        checkpoint_path = os.path.dirname(checkpoint_path)

    # 1. Direct checkpoint directory check
    has_weights = (
        os.path.isfile(os.path.join(checkpoint_path, "model.safetensors")) or
        os.path.isfile(os.path.join(checkpoint_path, "pytorch_model.bin"))
    )
    if has_weights:
        return checkpoint_path

    # 2. Check if checkpoint_path is an output folder from main.py
    best_checkpoint = None

    # Check trainer_state.json directly in checkpoint_path
    state_file = os.path.join(checkpoint_path, "trainer_state.json")
    if os.path.isfile(state_file):
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                state_data = json.load(f)
                best_checkpoint = state_data.get("best_model_checkpoint")
        except Exception:
            pass

    # Find all checkpoint-* subdirectories
    checkpoint_subdirs = []
    for item in os.listdir(checkpoint_path):
        item_path = os.path.join(checkpoint_path, item)
        if os.path.isdir(item_path):
            m = re.match(r"^checkpoint-(\d+)$", item)
            if m:
                checkpoint_subdirs.append((int(m.group(1)), item_path))

    checkpoint_subdirs.sort(key=lambda x: x[0], reverse=True)

    # Check trainer_state.json in checkpoint subdirectories if best_checkpoint was not found
    if not best_checkpoint and checkpoint_subdirs:
        for _, subdir in checkpoint_subdirs:
            sub_state_file = os.path.join(subdir, "trainer_state.json")
            if os.path.isfile(sub_state_file):
                try:
                    with open(sub_state_file, "r", encoding="utf-8") as f:
                        state_data = json.load(f)
                        best_checkpoint = state_data.get("best_model_checkpoint")
                        if best_checkpoint:
                            break
                except Exception:
                    pass

    # Verify best_checkpoint if found
    if best_checkpoint:
        if os.path.isdir(best_checkpoint):
            return best_checkpoint
        base_name = os.path.basename(best_checkpoint.rstrip("/\\"))
        candidate = os.path.join(checkpoint_path, base_name)
        if os.path.isdir(candidate):
            return candidate

    # Fallback to the latest checkpoint directory with model weights
    for _, subdir in checkpoint_subdirs:
        if (
            os.path.isfile(os.path.join(subdir, "model.safetensors")) or
            os.path.isfile(os.path.join(subdir, "pytorch_model.bin"))
        ):
            return subdir

    # Scan any subfolder containing model weights
    candidates = []
    for item in os.listdir(checkpoint_path):
        item_path = os.path.join(checkpoint_path, item)
        if os.path.isdir(item_path):
            if (
                os.path.isfile(os.path.join(item_path, "model.safetensors")) or
                os.path.isfile(os.path.join(item_path, "pytorch_model.bin"))
            ):
                candidates.append(item_path)
    if candidates:
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]

    raise FileNotFoundError(
        f"No valid checkpoint found in '{checkpoint_path}'. "
        f"Expected 'model.safetensors', 'pytorch_model.bin', or subdirectories named 'checkpoint-<step>'."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="fine-tuned checkpoint directory or main.py output folder (default: looks for ./checkpoints; omit for untrained base model)",
    )
    parser.add_argument("--input", nargs="+", default=None, help="one or more sentences to convert; omit to run the built-in examples")
    args = parser.parse_args()

    checkpoint_to_load = args.checkpoint_path
    if checkpoint_to_load is None:
        if os.path.isdir("./checkpoints"):
            print("No --checkpoint_path provided, but './checkpoints' exists. Using './checkpoints'...")
            checkpoint_to_load = "./checkpoints"

    if checkpoint_to_load is None or checkpoint_to_load.lower() in ("none", "null", ""):
        print("Using untrained base model...")
        roberta, tokenizer = model.init_model()
    else:
        ckpt_dir = resolve_checkpoint_path(checkpoint_to_load)
        print(f"Loading checkpoint from: {ckpt_dir}")
        tokenizer = model.init_tokenizer()
        # Not from_pretrained: transformers 5.x crashes in mark_tied_weights_as_initialized ("no attribute `cls`") for the custom decoder.
        from safetensors.torch import load_file
        from transformers import GenerationConfig
        from model_config import InvertTextNormalizationConfig

        roberta = InvertTextNormalization(InvertTextNormalizationConfig.from_pretrained(ckpt_dir))
        safetensors_path = os.path.join(ckpt_dir, "model.safetensors")
        bin_path = os.path.join(ckpt_dir, "pytorch_model.bin")

        if os.path.isfile(safetensors_path):
            print(roberta.load_state_dict(load_file(safetensors_path), strict=False))
        elif os.path.isfile(bin_path):
            print(roberta.load_state_dict(torch.load(bin_path, map_location="cpu"), strict=False))
        else:
            raise FileNotFoundError(f"Neither model.safetensors nor pytorch_model.bin found in {ckpt_dir}")

        roberta.tie_weights()  # lm_head is tied to the embeddings and absent from the checkpoint
        try:
            roberta.generation_config = GenerationConfig.from_pretrained(ckpt_dir)
        except OSError:
            roberta.generation_config.max_length = 200
            roberta.generation_config.no_repeat_ngram_size = 3
        roberta.generation_config.num_beams = 1

    roberta.eval()

    list_inputs = ['ngày chín tháng một năm một chín bốn lăm nạn đói bùng phát ở Việt Nam và ảnh hưởng tới sáu mươi phần trăm dân',
                'tôi làm việc ở Khoa Khoa học và Kỹ thuật Máy tính và hôm nay tôi đã chấm được hai ngàn không trăm hai mươi ba bài',
                'tám giờ chín phút ngày mười tám tháng năm năm hai nghìn không trăm hai mươi hai',
                'mã số quy đê tê tê đê hai tám chéo hai không không ba',
                'thể tích tám mét khối trọng lượng năm mươi ki lô gam',
                'ngày hai tám tháng tư cô vít bùng phát ở Việt Nam gây nhiễm bệnh mười phần trăm dân Việt Nam'
                ]

    """
    input  : ngày chín tháng một năm một chín bốn lăm nạn đói bùng phát ở Việt Nam và ảnh hưởng tới sáu mươi phần trăm dân
    output : ngày 9/1/1945 nạn đói bùng phát ở Việt Nam và ảnh hưởng tới 60 % dân
    """

    def generate_span(span):
        enc = tokenizer([span])
        outputs = roberta.generate(input_ids=torch.tensor(enc['input_ids']),
                                   attention_mask=torch.tensor(enc['attention_mask']),
                                   num_beams=1, num_return_sequences=1)
        text = tokenizer.decode(outputs[0].tolist(), skip_special_tokens=True)
        # a generation can stop right after a BPE piece ("6@@" + eos): drop the dangling continuation marker
        return re.sub(r"@@$", "", text.strip()).strip()

    for input_str in (args.input or list_inputs):
        words = input_str.split()
        # Build the input the way training does (preprocess_function): bos + each word's subwords + eos, so we know
        # which subword starts each word. The tagger labels SUBWORDS (B on the first, I on the rest), e.g. "hóa" is
        # h@@ ó@@ a, so tags must be read at each word's first subword, not at index = word position.
        input_ids, first_tok = [tokenizer.bos_token_id], []
        for w in words:
            first_tok.append(len(input_ids))
            input_ids.extend(tokenizer(w)['input_ids'][1:-1])
        input_ids.append(tokenizer.eos_token_id)

        with torch.no_grad():
            encoder_outputs = roberta.encoder(input_ids=torch.tensor([input_ids]),
                                              attention_mask=torch.ones(1, len(input_ids), dtype=torch.long))
            logits = roberta.spoken_tagging_classifier(roberta.dropout(encoder_outputs[0]))
        token_tags = torch.argmax(logits, dim=-1)[0].tolist()
        word_tags = [token_tags[p] for p in first_tok]  # 0: O (keep), 1: B (span start), 2: I (span continues)

        # group words: O words are kept as is, B/I words form spans sent to the decoder
        segments = []  # [is_span, [words]]
        for w, tag in zip(words, word_tags):
            if tag == 0:
                segments.append([False, [w]])
            elif tag == 2 and segments and segments[-1][0]:
                segments[-1][1].append(w)
            else:  # B, or an I with no open span
                segments.append([True, [w]])

        with torch.no_grad():
            pieces = [generate_span(" ".join(ws)) if is_span else " ".join(ws) for is_span, ws in segments]
        output_str = " ".join(p for p in pieces if p)

        print("Input:", input_str)
        print("Output:", output_str)
        print()
