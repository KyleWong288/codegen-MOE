OUTPUT_PATH="./output/roberta_test.json"
CUDA_VISIBLE_DEVICES=1 python evaluate_output.py \
    --output_path $OUTPUT_PATH \
    --eval_type 3 \