export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=0
RUN_NAME=cc_python13b
MODEL_NAME=codellama-python-13b
DATA_PATH=../../dsc_data_code_contests/

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
    --run_name $RUN_NAME \
    --base_model $MODEL_NAME \
    --data_path $DATA_PATH \
    --max_seq_length 2048 \
    --num_epochs 2 \
    --learning_rate 2e-5 \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --eval_stpes 100 \
    --warmup_steps 40 \
    --log_interval 10 \
    --scheduler constant \
    --ds_config ds_config.json \