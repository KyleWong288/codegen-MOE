export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=0
RUN_NAME="dsc_sorted_multians"
MODEL_NAME=dsc-6.7b-instruct
DATA_PATH=../../dsc_data_sorted_multians/all

CUDA_VISIBLE_DEVICES=6,7 python train.py \
    --run_name $RUN_NAME \
    --base_model $MODEL_NAME \
    --data_path $DATA_PATH \
    --max_seq_length 2048 \
    --num_epochs 1 \
    --learning_rate 2e-5 \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --warmup_steps 10 \
    --log_interval 10 \
    --scheduler cosine \
    --ds_config ds_config.json \
