export CUDA_VISIBLE_DEVICES=4
MODEL=microsoft/CodeGPT-small-java-adaptedGPT2 #change model
MASTER_PORT=92257 # modify
MODE=victim # modify
Percentage=0.01 # modify

LANG=java                       # set python for py150


for SAMPLE_RATIO in {10..30..10}
do

DATADIR="../dataset/javaCorpus/${Percentage}/${SAMPLE_RATIO}/"
LITFILE=../dataset/javaCorpus/literals.json
PRETRAINDIR="../../CodeCompletion-token/save/javaCorpus/${MODEL}/100/checkpoint-last"
LOGFILE="completion_javaCorpus_${MODEL##*/}_${MODE}_eval.log"

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