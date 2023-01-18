export CUDA_VISIBLE_DEVICES=3
MODEL=microsoft/CodeGPT-small-java-adaptedGPT2 #change model
SAMPLE_RATIO=10 # modify SAMPLE_RATIO
MASTER_PORT=94257 # modify
MODE=surrogate # modify


LANG=java                       # set python for py150


for SAMPLE_RATIO in {100..100..10}
do

DATADIR="../dataset/javaCorpus/${MODEL##*/}/${SAMPLE_RATIO}/"
LITFILE=../dataset/javaCorpus/literals.json
PRETRAINDIR="../../CodeCompletion-token/save/javaCorpus/${MODEL}/${SAMPLE_RATIO}/checkpoint-last"
LOGFILE=completion_javaCorpus_eval.log

python -u run_lm.py \
        --data_dir=$DATADIR \
        --lit_file=$LITFILE \
        --langs=$LANG \
        --output_dir=$DATADIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=gpt2 \
        --block_size=1024 \
        --eval_line \
        --logging_steps=100 \
        --seed=42 \
        --MASTER_PORT $MASTER_PORT \
        --per_gpu_eval_batch_size 128 \
        --mode $MODE 
done