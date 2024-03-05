RUN_NAME="4000_dsc_all"
CHECKPOINT=checkpoint-1000
SKILL="all"
CUDA_VISIBLE_DEVICES=1 python generate.py \
    --run_name $RUN_NAME \
    --checkpoint $CHECKPOINT \
    --model_name dsc-6.7b-base \
    --checkpoint_path "./finetuned_models/$RUN_NAME/$CHECKPOINT" \
    --skill $SKILL \