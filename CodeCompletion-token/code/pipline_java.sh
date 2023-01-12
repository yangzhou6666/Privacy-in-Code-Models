LANG=java                       # set python for py150
DATADIR=../dataset/javaCorpus/token_completion
LITFILE=../dataset/javaCorpus/literals.json
OUTPUTDIR=../save/javaCorpus
PRETRAINDIR=microsoft/CodeGPT-small-java-adaptedGPT2        # microsoft/CodeGPT-small-py for py150
LOGFILE=completion_javaCorpus.log
PER_NODE_GPU=1       # modify YOUR_GPU_NUM
SAMPLE_RATIO=0.2 # modify SAMPLE_RATIO

# 训练时，sample_ratio设置为0.1，模拟BOB知道部分数据
CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=$PER_NODE_GPU run_lm.py \
        --data_dir=$DATADIR \
        --lit_file=$LITFILE \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=gpt2 \
        --block_size=1024 \
        --do_train \
        --gpu_per_node $PER_NODE_GPU \
        --learning_rate=8e-5 \
        --weight_decay=0.01 \
        --evaluate_during_training \
        --per_gpu_train_batch_size=4 \
        --per_gpu_eval_batch_size=8 \
        --gradient_accumulation_steps=2 \
        --num_train_epochs=5 \
        --logging_steps=100 \
        --save_steps=500 \
        --seed=42 \
        --overwrite_output_dir \
        --not_pretrain \
        --sample_ratio $SAMPLE_RATIO \
        --save_sample 

export CUDA_VISIBLE_DEVICES=3
PRETRAINDIR=../save/javaCorpus/microsoft/CodeGPT-small-java-adaptedGPT2/20/checkpoint-last     # directory of your saved model
LOGFILE=completion_javaCorpus_eval.log
python -u run_lm.py \
        --data_dir=$DATADIR \
        --lit_file=$LITFILE \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=gpt2 \
        --block_size=1024 \
        --do_eval \
        --per_gpu_eval_batch_size=16 \
        --logging_steps=100 \
        --seed=42 