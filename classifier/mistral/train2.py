import argparse
import functools
import os
import torch
import torch.nn.functional as F
import wandb
from sklearn.metrics import f1_score
from datasets import load_dataset
from peft import (
    LoraConfig,
    TaskType,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer

def preprocess(examples, tokenizer):
    tokenized_inputs = tokenizer(examples["text"], max_length=args.max_seq_length, padding="max_length", truncation=True)
    tokenized_inputs["labels"] = examples["labels"]
    return tokenized_inputs


def collate_fn(batch, tokenizer):
    dict_keys = ['input_ids', 'attention_mask', 'labels']
    d = {k: [dic[k] for dic in batch] for k in dict_keys}
    d['input_ids'] = torch.nn.utils.rnn.pad_sequence(
        d['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id
    )
    d['attention_mask'] = torch.nn.utils.rnn.pad_sequence(
        d['attention_mask'], batch_first=True, padding_value=0
    )
    d['labels'] = torch.stack(d['labels'])
    return d


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    f1_micro = f1_score(labels, preds > 0, average = 'micro')
    f1_macro = f1_score(labels, preds > 0, average = 'macro')
    f1_weighted = f1_score(labels, preds > 0, average = 'weighted')
    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }


# Custom trainer for computing multi-label loss
class MistralTrainer(SFTTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        
        labels = inputs.pop("labels")
        
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # compute custom loss
        loss = F.binary_cross_entropy_with_logits(logits, labels.to(torch.float32))
        return (loss, outputs) if return_outputs else loss


def main(args):
    
    output_dir = "./finetuned_models"

    # Load tokenizer:
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset:
    train_file = os.path.join(args.data_path, "train.jsonl")
    dev_file = os.path.join(args.data_path, "dev.jsonl")
    train_data = load_dataset("json", data_files=train_file, split="train", download_mode="force_redownload")
    eval_data = load_dataset("json", data_files=dev_file, split="train", download_mode="force_redownload")
    tokenized_train_data = train_data.map(lambda examples: preprocess(examples, tokenizer), batched=True).with_format("torch")
    tokenized_eval_data = eval_data.map(lambda examples: preprocess(examples, tokenizer), batched=True).with_format("torch")

    # Configure peft:
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        bias="none",
    )

    # Load model:
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        device_map="auto",
        num_labels=8,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

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

    trainer = MistralTrainer(
        model=model,
        args=training_args,
        max_seq_length=args.max_seq_length,
        train_dataset=tokenized_train_data,
        eval_dataset=tokenized_eval_data,
        tokenizer=tokenizer,
        dataset_text_field="text",
        data_collator=functools.partial(collate_fn, tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        peft_config=peft_config,
    )

    # Finetune:
    print("TRAINING BEGIN")
    torch.cuda.empty_cache()
    trainer.train()

    # Save model:
    model.save_pretrained(training_args.output_dir)
    print("MODEL SUCCESSFULLY SAVED")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--run_name", default='local-test', type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--base_model", required=True, type=str, choices=["mistralai/Mistral-7B-v0.1"])
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