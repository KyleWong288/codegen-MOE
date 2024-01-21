CUDA_VISIBLE_DEVICES=1,2 python pretokenize.py \
    --tokenizer_dir codellama/CodeLlama-7b-hf \
    --cache_dir ./ \
    --dataset_name codellama_tokenized  