# Finetuned
# CUDA_VISIBLE_DEVICES=0,1,2,3 python generate.py \
#     --model_name finetuned \
#     --base_model_name meta-llama/Llama-2-7b-chat-hf \
#     --checkpoint_path "./../checkpoints/Llama-2-7b-chat-hf" \
#     --load_in_8bit \
#     --max_new_tokens 256 \


# LLAMA
RUN_NAME=ft_test_bs4_e10
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python generate.py \
    --run_name $RUN_NAME \
    --model_name codellama_python \
    --checkpoint_path "./finetuned_models/$RUN_NAME/checkpoint-2964" \