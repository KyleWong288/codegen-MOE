RUN_NAME=dsc_clean_q
python evaluate_lc.py \
    --generation_dir output/dsc-6.7b-instruct/$RUN_NAME \
    --result_dir eval_results/dsc-6.7b-instruct/$RUN_NAME \
    --data_path ./test_data/20240121-Jul_50.jsonl