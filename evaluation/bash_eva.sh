python eva.py \
--model_1 codeparrot/codeparrot \
--model_2 codeparrot/codeparrot-small \
--top_k 40 \
--temperature 1.0 \
--seq_len 512 \
--gpu_id 0 \
--extract_n 1000 \
--N 20000 \
--extract_mode 'large-first' 