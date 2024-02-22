export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=0
RUN_NAME="roberta_test"
MODEL_NAME="roberta-base"
DATA_PATH="./data"

CUDA_VISIBLE_DEVICES=1,2,3,4 python train_roberta.py \
    --run_name $RUN_NAME \
    --base_model $MODEL_NAME \
    --data_path $DATA_PATH \
    --num_epochs 10 \
    --learning_rate 2e-5 \
    --batch_size 16 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --warmup_steps 10 \
    --log_interval 10 \
    --scheduler cosine \