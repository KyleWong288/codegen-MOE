import argparse
import evaluate
import numpy as np
import os
import torch
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    DataCollatorWithPadding, 
    Trainer, 
    TrainingArguments
)

ID2LABEL = {
    0: "Amortized analysis",
    1: "Bit manipulation",
    2: "Complete search",
    3: "Data structures",
    4: "Dynamic programming",
    5: "Greedy algorithms",
    6: "Range queries",
    7: "Sorting",
}
LABEL2ID = {
    "Amortized analysis": 0,
    "Bit manipulation": 1,
    "Complete search": 2,
    "Data structures": 3,
    "Dynamic programming": 4,
    "Greedy algorithms": 5,
    "Range queries": 6,
    "Sorting": 7,
}


def preprocess(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True)


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)
    accuracy = evaluate.load("accuracy")
    return accuracy.compute(predictions=preds, references=labels)


def main(args):

    output_dir = "./finetuned_models"
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", trust_remote_code=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Load dataset:
    train_file = os.path.join(args.data_path, "train.jsonl")
    dev_file = os.path.join(args.data_path, "dev.jsonl")
    train_data = load_dataset("json", data_files=train_file, split="train", download_mode="force_redownload")
    eval_data = load_dataset("json", data_files=dev_file, split="train", download_mode="force_redownload")
    tokenized_train_data = train_data.map(lambda examples: preprocess(examples, tokenizer), batched=True)
    tokenized_eval_data = eval_data.map(lambda examples: preprocess(examples, tokenizer), batched=True)
    

    # Load model:
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=8, id2label=ID2LABEL, label2id=LABEL2ID
    )

    # Configure trainer:
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/{args.run_name}",
        logging_steps=args.log_interval,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.scheduler,
        weight_decay=args.weight_decay,
        report_to="wandb",
        run_name=args.run_name,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_data,
        eval_dataset=tokenized_eval_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Finetune:
    print("TRAINING BEGIN")
    torch.cuda.empty_cache()
    trainer.train()

    # save model
    model.save_pretrained(training_args.output_dir)
    print("MODEL SUCCESSFULLY SAVED")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--run_name", default='local-test', type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--base_model", required=True, type=str, choices=["roberta-base"])
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--max_seq_length", default=2048, type=int)
    parser.add_argument("--num_epochs", default=5, type=float)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=8, type=int)
    parser.add_argument("--evaluation_strategy", default='steps', type=str)  
    parser.add_argument("--eval_steps", default=100, type=float) 
    parser.add_argument("--warmup_steps", default=10, type=float)
    parser.add_argument("--log_interval", default=20, type=int)
    parser.add_argument("--scheduler", default='linear', type=str)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--from_remote", default=False, type=bool)
    parser.add_argument("--ds_config", default='./config_new.json', type=str)
    args = parser.parse_args()

    wandb.login()
    main(args)