DATA_PATH="../../dsc_limit_data/all"
OUTPUT_PATH="./finetuned_models/all"
MODEL="dsc-6.7b-instruct"

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python finetune.py \
    --model_name_or_path $MODEL \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 8 \
    --model_max_length 2048 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy steps \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --warmup_steps 10 \
    --logging_steps 10 \
    --lr_scheduler_type cosine \
    --report_to wandb \
    --fp16 True \

# deepspeed finetune.py \
#     --model_name_or_path $MODEL \
#     --data_path $DATA_PATH \
#     --output_dir $OUTPUT_PATH \
#     --num_train_epochs 3 \
#     --model_max_length 1024 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 4 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 100 \
#     --save_total_limit 100 \
#     --learning_rate 2e-5 \
#     --warmup_steps 10 \
#     --logging_steps 1 \
#     --lr_scheduler_type "cosine" \
#     --gradient_checkpointing True \
#     --report_to "tensorboard" \
#     --deepspeed configs/ds_config_zero3.json \
#     --bf16 False