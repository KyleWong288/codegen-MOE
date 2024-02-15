RUN_NAME="4000_dsc_data_structures"
CHECKPOINT=checkpoint-1800
SKILL="data_structures"
CUDA_VISIBLE_DEVICES=3,4 python generate.py \
    --run_name $RUN_NAME \
    --model_name dsc-6.7b-instruct \
    --checkpoint_path "./finetuned_models/$RUN_NAME/$CHECKPOINT" \
    --skill $SKILL \