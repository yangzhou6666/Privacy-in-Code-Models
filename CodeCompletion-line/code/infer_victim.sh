export CUDA_VISIBLE_DEVICES=2  # modify
# MODEL=microsoft/CodeGPT-small-java-adaptedGPT2 
# MODEL=codeparrot/codeparrot-small
# MODEL=NinedayWang/PolyCoder-0.4B 
# MODEL=Salesforce/codegen-350M-multi
# MODEL=bigcode/santacoder
MODEL=bigcode/starcoderbase-1b
MASTER_PORT=92257
MODE=victim 
Percentage=0.01 

LANG=java                      


for SAMPLE_RATIO in {10..10..10}
do

DATADIR="../dataset/javaCorpus/${Percentage}/${SAMPLE_RATIO}/"
LITFILE=../dataset/javaCorpus/literals.json
PRETRAINDIR="../../CodeCompletion-token/save/javaCorpus/${MODEL}/100/checkpoint-epoch-3"
LOGFILE="logs/${MODEL##*/}_${MODE}_eval.log"

python -u run.py \
        --data_dir=$DATADIR \
        --lit_file=$LITFILE \
        --langs=$LANG \
        --output_dir=$DATADIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=polycoder \
        --block_size=1024 \
        --eval_line \
        --logging_steps=100 \
        --seed=42 \
        --MASTER_PORT $MASTER_PORT \
        --per_gpu_eval_batch_size 128 \
        --mode $MODE
done 