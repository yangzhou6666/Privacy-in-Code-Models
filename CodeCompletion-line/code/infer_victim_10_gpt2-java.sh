MODE=victim # modify surrogate


export CUDA_VISIBLE_DEVICES=1
MASTER_PORT=82257 # modify
SURROGATE_MODEL_NAME=CodeGPT-small-java
SAMPLE_RATIO=10

LANG=java                       # set python for py150


DATADIR="../dataset/javaCorpus/${SURROGATE_MODEL_NAME}/${SAMPLE_RATIO}/"
LITFILE=../dataset/javaCorpus/literals.json
PRETRAINDIR="../../CodeCompletion-token/save/javaCorpus/microsoft/CodeGPT-small-java-adaptedGPT2/100/checkpoint-last"
LOGFILE="completion_javaCorpus_eval_${SURROGATE_MODEL_NAME}_${SAMPLE_RATIO}_victim.log"

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
