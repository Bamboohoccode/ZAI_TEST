from transformers import Seq2SeqTrainingArguments
import argparse
import math
import os
import model
import preprocess
from my_trainer import InvertTextNormalizationTrainer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--input_json", default=None, help="JSON file (list of {'spoken','formatted'}), e.g. dataset_vietnam_money.json; mixed into VietAI/spoken_norm_assignment unless --only_json")
    parser.add_argument("--only_json", action="store_true", help="train on --input_json only; without it the JSON is mixed into the VietAI data")
    parser.add_argument("--viet_frac", type=float, default=0.02, help="fraction of the VietAI train set to use (1 = all); ignored with --only_json")
    parser.add_argument("--checkpoint_path", default=None, help="fine-tune from this checkpoint instead of the base bert2bert model")
    parser.add_argument("--skip_init_model", action="store_true", help="do not call init_model (no base-model init); load --checkpoint_path directly")
    args = parser.parse_args()

    if args.skip_init_model:
        if args.checkpoint_path is None:
            parser.error("--skip_init_model requires --checkpoint_path")
        roberta, tokenizer = model.load_checkpoint(args.checkpoint_path)
    else:
        roberta, tokenizer = model.init_model()
        if args.checkpoint_path is not None:
            roberta, tokenizer = model.load_checkpoint(args.checkpoint_path)

    if not 0 < args.viet_frac <= 1:
        parser.error("--viet_frac must be in (0, 1]")
    if args.only_json and not args.input_json:
        parser.error("--only_json requires --input_json")
    dataset = preprocess.init_data(args.input_json, args.only_json, args.viet_frac)

    data_collator = preprocess.DataCollatorInvertTextNormalization(tokenizer, model=roberta)

    num_epochs = args.epoch
    checkpoint_path = "./checkpoints"
    batch_size = args.batch_size
    train_dataset = dataset['train']  # init_data already subsampled / mixed it
    steps_per_epoch = math.ceil(len(train_dataset) / batch_size)
    """
    eval_accumulation_steps (int, optional): 
            Number of predictions steps to accumulate the output tensors for, before moving the results to the 
            CPU. If left unset, the whole predictions are accumulated on GPU/TPU before being moved to 
            the CPU (faster but requires more memory).

    save_steps (`int`, *optional*, defaults to 500):
            Number of updates steps before two checkpoint saves if `save_strategy="epoch"`.
    """
    training_args = Seq2SeqTrainingArguments(
        output_dir=checkpoint_path,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="steps",
        save_strategy="steps",  # must match eval_strategy for load_best_model_at_end
        save_steps=2 * steps_per_epoch,
        save_total_limit=1,  # keeps only the best checkpoint (the latest one may also be kept)
        load_best_model_at_end=True,  # default metric: eval_loss
        learning_rate=args.lr,
        gradient_accumulation_steps=1,
        predict_with_generate=True,
        do_train=True,
        do_eval=True,
        logging_steps=5000,
        eval_steps=2 * steps_per_epoch,  # eval every 2 epochs
        num_train_epochs=num_epochs,
        warmup_steps=1/num_epochs,
        eval_accumulation_steps=10,
        dataloader_num_workers=0,
        generation_max_length=50,
        fp16=True,
        ignore_data_skip=True
    )

    trainer = InvertTextNormalizationTrainer(
        model=roberta,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dataset['valid'],
        data_collator=data_collator,
        processing_class=tokenizer
    )
    trainer.train()
    # trainer.evaluate()
    # trainer.save_model(checkpoint_path)
    # trainer.evaluate()
