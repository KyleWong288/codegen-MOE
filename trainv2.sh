export TOKENIZERS_PARALLELISM=0

RUN_NAME=ft_test
MODEL_NAME=codellama_python
DATASET=kok-controversial
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python trainv2.py \
    --run_name $RUN_NAME \
    --base_model $MODEL_NAME \
    --dataset $DATASET \
    --max_seq_length 256 \
    --batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --num_epochs 1 \
    --log_interval 10 \
    --warmup_ratio 0.03 \
    --scheduler constant \
    --evaluation_strategy steps \