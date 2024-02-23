RUN_NAME="mistral_test"
MODEL_PATH="./finetuned_models/mistral_test/checkpoint-1000"
CUDA_VISIBLE_DEVICES=1 python scratch.py \
    --run_name $RUN_NAME \
    --model_path $MODEL_PATH \