RUN_NAME="roberta_test"
MODEL_PATH="./finetuned_models/roberta_test/checkpoint-1000"
CUDA_VISIBLE_DEVICES=1 python evaluate_classifier.py \
    --run_name $RUN_NAME \
    --model_path $MODEL_PATH \