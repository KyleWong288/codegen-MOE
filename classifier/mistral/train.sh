export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=0
RUN_NAME="mistral_test"
MODEL_NAME="mistralai/Mistral-7B-v0.1"
DATA_PATH="./data"

CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py \
    --run_name $RUN_NAME \
    --base_model $MODEL_NAME \
    --data_path $DATA_PATH \
    --num_epochs 2 \
    --learning_rate 2e-5 \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --warmup_steps 10 \
    --log_interval 5 \
    --scheduler cosine \
    --max_seq_length 1024 \