export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=0
RUN_NAME="dsc_cc_base"
MODEL_NAME=dsc-6.7b-base
DATA_PATH=../../dsc_data_code_contests_base/

CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py \
    --run_name $RUN_NAME \
    --base_model $MODEL_NAME \
    --data_path $DATA_PATH \
    --max_seq_length 2048 \
    --num_epochs 2 \
    --learning_rate 1e-5 \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --warmup_steps 40 \
    --log_interval 10 \
    --scheduler cosine \
    --ds_config ds_config.json \
