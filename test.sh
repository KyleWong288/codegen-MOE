RUN_NAME=roberta_test
MODEL_NAME=roberta-base
DATA_PATH=./data

CUDA_VISIBLE_DEVICES=1 python test.py \
    --run_name $RUN_NAME \
    --data_path $DATA_PATH \
    --message "hello world" \
    --base_model "roberta-base" \
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