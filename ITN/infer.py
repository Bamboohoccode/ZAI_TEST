import argparse
import torch
import model
from model import InvertTextNormalization
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", default=None, help="fine-tuned checkpoint; omit for the untrained base model")
parser.add_argument("--input", nargs="+", default=None, help="one or more sentences to convert; omit to run the built-in examples")
args = parser.parse_args()

if args.checkpoint_path is None:
    roberta, tokenizer = model.init_model()
else:
    tokenizer = model.init_tokenizer()
    # Not from_pretrained: transformers 5.x crashes in mark_tied_weights_as_initialized ("no attribute `cls`") for the custom decoder.
    from safetensors.torch import load_file
    from transformers import GenerationConfig
    from model_config import InvertTextNormalizationConfig
    roberta = InvertTextNormalization(InvertTextNormalizationConfig.from_pretrained(args.checkpoint_path))
    print(roberta.load_state_dict(load_file(os.path.join(args.checkpoint_path, "model.safetensors")), strict=False))
    roberta.tie_weights()  # lm_head is tied to the embeddings and absent from the checkpoint
    try:
        roberta.generation_config = GenerationConfig.from_pretrained(args.checkpoint_path)
    except OSError:
        roberta.generation_config.max_length = 200
        roberta.generation_config.no_repeat_ngram_size = 3
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

for input_str in (args.input or list_inputs):
    inputs = tokenizer([input_str])
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    inputs = {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
    }

    encoder_outputs = roberta.encoder(**inputs)
    spoken_tagging_output = roberta.spoken_tagging_classifier(roberta.dropout(encoder_outputs[0]))
    spoken_tagging_output = torch.argmax(spoken_tagging_output, dim=-1).tolist()[0][1:-1] # Skip <bos> and <eos> token label

    # print("Sequence Tagging:", spoken_tagging_output)

    temp = ""
    output_str = ""

    for i, ele in enumerate(input_str.split()):
        if spoken_tagging_output[i] in [0,1] or i == len(input_str.split()) - 1:
            if i == len(input_str.split()) - 1:
                temp += " " + ele
            if len(temp) > 0:
                inputs = tokenizer([temp])
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']

                inputs = {
                    "input_ids": torch.tensor(input_ids),
                    "attention_mask": torch.tensor(attention_mask),
                }

                outputs = roberta.generate(**inputs, output_attentions=True, num_beams=1, num_return_sequences=1)

                for output in outputs.cpu().detach().numpy().tolist():
                    decoded_output = tokenizer.decode(output, skip_special_tokens=True) 
                    output_str += decoded_output if len(output_str) == 0 else " " + decoded_output
            
            if spoken_tagging_output[i] == 0:
                output_str += ele if len(output_str) == 0 else " " + ele
                temp = ""
            else:
                temp = ele
        else:
            temp += ele if len(temp) == 0 else " " + ele

    print("Input:", input_str)
    print("Output:", output_str)
    print()
