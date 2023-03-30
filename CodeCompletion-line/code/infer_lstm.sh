export CUDA_VISIBLE_DEVICES=1 # modify
MODEL=rnn 
MASTER_PORT=77777 # modify
MODE=surrogate 
Percentage=0.01 

LANG=java                       


for SAMPLE_RATIO in {10..20..10}
do

DATADIR="../dataset/javaCorpus/${Percentage}/${SAMPLE_RATIO}/"
LITFILE=../dataset/javaCorpus/literals.json
PRETRAINDIR="../../CodeCompletion-token/save/javaCorpus/${MODEL}/${SAMPLE_RATIO}/checkpoint-epoch-4"
LOGFILE="completion_javaCorpus_${MODEL##*/}_${MODE}_eval.log"

python -u run.py \
        --data_dir=$DATADIR \
        --lit_file=$LITFILE \
        --langs=$LANG \
        --output_dir=$DATADIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=rnn \
        --block_size=1024 \
        --eval_line \
        --logging_steps=100 \
        --seed=42 \
        --MASTER_PORT $MASTER_PORT \
        --per_gpu_eval_batch_size 128 \
        --mode $MODE 
done