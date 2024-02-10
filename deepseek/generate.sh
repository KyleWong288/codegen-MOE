RUN_NAME=dsc_all_cosine
CHECKPOINT=checkpoint-1000
SKILL=greedy
CUDA_VISIBLE_DEVICES=1,2,3 python generate.py \
    --run_name $RUN_NAME \
    --model_name dsc-6.7b-instruct \
    --checkpoint_path "./finetuned_models/$RUN_NAME/$CHECKPOINT" \
    --skill $SKILL \