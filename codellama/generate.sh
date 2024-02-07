RUN_NAME=sorting_bs4_e8
SKILL=sorting
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python generate.py \
    --run_name $RUN_NAME \
    --model_name codellama_python \
    --checkpoint_path "./finetuned_models/$RUN_NAME/checkpoint-1476" \
    --skill $SKILL \