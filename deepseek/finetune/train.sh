export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=0
RUN_NAME="4000_dsc_data_structures"
MODEL_NAME=dsc-6.7b-instruct
DATA_PATH=../../dsc_limit_data/data_structures

CUDA_VISIBLE_DEVICES=1,2,3,4,5 python train.py \
    --run_name $RUN_NAME \
    --base_model $MODEL_NAME \
    --data_path $DATA_PATH \
    --max_seq_length 2048 \
    --num_epochs 6 \
    --learning_rate 2e-5 \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --warmup_steps 10 \
    --log_interval 10 \
    --scheduler cosine \
    --ds_config ds_config.json \
