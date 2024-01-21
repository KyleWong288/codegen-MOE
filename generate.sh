# Finetuned
# CUDA_VISIBLE_DEVICES=0,1,2,3 python generate.py \
#     --model_name finetuned \
#     --base_model_name meta-llama/Llama-2-7b-chat-hf \
#     --checkpoint_path "./../checkpoints/Llama-2-7b-chat-hf" \
#     --load_in_8bit \
#     --max_new_tokens 256 \


# LLAMA
CUDA_VISIBLE_DEVICES=0,1 python generate.py \
    --model_name code-llama \
