export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=0
RUN_NAME=sorting_bs4_e8
MODEL_NAME=codellama_python
DATASET=sorting

# deepspeed \
# --include localhost:0,1,2,3,4,5,6,7 \
# train.py \
#     --run_name $RUN_NAME \
#     --base_model $MODEL_NAME \
#     --dataset $DATASET \
#     --max_seq_length 2048 \
#     --batch_size 4 \
#     --gradient_accumulation_steps 2 \
#     --learning_rate 5e-5 \
#     --num_epochs 8 \
#     --log_interval 10 \
#     --warmup_ratio 0.03 \
#     --scheduler constant \
#     --evaluation_strategy steps \
#     --ds_config ds_config.json \

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python train.py \
    --run_name $RUN_NAME \
    --base_model $MODEL_NAME \
    --dataset $DATASET \
    --max_seq_length 2048 \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --num_epochs 8 \
    --log_interval 10 \
    --warmup_ratio 0.03 \
    --scheduler constant \
    --evaluation_strategy steps \
    --ds_config ds_config.json \