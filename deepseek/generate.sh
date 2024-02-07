RUN_NAME=dsc_all_cosine
CHECKPOINT=checkpoint-1000
SKILL=all
CUDA_VISIBLE_DEVICES=1,2 python generate.py \
    --run_name $RUN_NAME \
    --model_name dsc-6.7b-instruct \
    --num_return_sequences 3 \
    --checkpoint_path "./finetuned_models/$RUN_NAME/$CHECKPOINT" \
    --skill $SKILL \