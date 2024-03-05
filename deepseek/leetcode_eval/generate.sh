RUN_NAME="4000_dsc_all"
CHECKPOINT=checkpoint-1000
DATA_PATH="./test_data/20240121-Jul_50.jsonl"
CUDA_VISIBLE_DEVICES=1 python generate.py \
    --run_name $RUN_NAME \
    --checkpoint $CHECKPOINT \
    --model_name dsc-6.7b-base \
    --checkpoint_path "../finetuned_models/$RUN_NAME/$CHECKPOINT" \
    --data_path $DATA_PATH \
