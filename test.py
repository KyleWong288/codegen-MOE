import argparse
import os
import torch
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments
from trl import SFTTrainer
from peft import TaskType, LoraConfig


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--message", type=str)
    args = parser.parse_args()
    print(args.message)
