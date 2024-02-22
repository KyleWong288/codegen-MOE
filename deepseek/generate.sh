RUN_NAME="4000_dsc_data_structures"
CHECKPOINT=checkpoint-1800
SKILL="complete_search"
CUDA_VISIBLE_DEVICES=1,2 python generate.py \
    --run_name $RUN_NAME \
    --model_name dsc-6.7b-instruct \
    --checkpoint_path "./finetuned_models/$RUN_NAME/$CHECKPOINT" \
    --skill $SKILL \
    --use_base_model true \