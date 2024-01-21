import torch
import sys
import wandb
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import TaskType, LoraConfig
from utils import *      


def main(args):
        
    model_name = parse_model_name(args.base_model, args.from_remote)
    
    # load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        trust_remote_code=True,
        device_map="auto"
    )
    if args.local_rank == 0:
        print(model)
    print("MODEL LOADED")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # load data
    # dataset_dir = "./poc_data/" + args.dataset
    dataset_dir = "./../knowledge-of-knowledge/data/" + args.dataset
    train_dataset, dev_dataset = load_dataset(dataset_dir)
    print("DATASETS LOADED")
    
    
    training_args = TrainingArguments(
        output_dir=f'finetuned_models/{args.run_name}',
        logging_steps=args.log_interval,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.scheduler,
        save_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        fp16=True,
        evaluation_strategy=args.evaluation_strategy,
        remove_unused_columns=False,
        report_to='wandb',
        run_name=args.run_name
    )
    
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True
    model.model.config.use_cache = False
    
    # setup peft
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=lora_module_dict[args.base_model],
        bias='none',
    )

    # Train
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        max_seq_length=args.max_seq_length,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        peft_config=peft_config,
    )
    
    print("TRAINING BEGIN")
    torch.cuda.empty_cache()
    trainer.train()

    # save model
    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--run_name", default='local-test', type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--test_dataset", type=str)
    parser.add_argument("--base_model", required=True, type=str, choices=["codellama", "codellama_python", "llama2-7b"])
    parser.add_argument("--max_seq_length", default=1024, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--num_epochs", default=8, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--log_interval", default=20, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=8, type=int)
    parser.add_argument("--warmup_ratio", default=0.05, type=float)
    parser.add_argument("--scheduler", default='linear', type=str)
    parser.add_argument("--instruct_template", default='default')
    parser.add_argument("--evaluation_strategy", default='steps', type=str)    
    parser.add_argument("--eval_steps", default=0.2, type=float)    
    parser.add_argument("--from_remote", default=False, type=bool)    
    args = parser.parse_args()
    
    wandb.login()
    main(args)