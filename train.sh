MODEL_NAME=codellama/CodeLlama-7b-Python-hf
DATASET=train-sorting
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python train.py \
    --model_name $MODEL_NAME \
    --dataset_path /mnt/edward/data5/knw/moe-cf/poc_data/$DATASET \
    --seq_length 1024 \
    --load_in_8bit \
    --output_dir ./checkpoints/$DATASET \
    --log_with wandb \
    --wandb_project llama \
    --use_peft \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 10 \
    --learning_rate 5e-5 \
    # --n_train_pairs 256