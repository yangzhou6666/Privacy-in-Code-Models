LANG=java                      

LITFILE=../dataset/javaCorpus/literals.json

PRETRAINDIR=t5-small  # t5-small or Salesforce/codet5-small
PER_NODE_GPU=1      

MASTER_PORT=55129 #modify this every time
export CUDA_VISIBLE_DEVICES=0 #specify GPU, each time you need to change

for SAMPLE_RATIO in {10..20..10}
do
LOGFILE="completion_javaCorpus_${PRETRAINDIR##*/}_${SAMPLE_RATIO}".log
DATADIR="../dataset/javaCorpus/token_completion/"
OUTPUTDIR="../save/javaCorpus/"
echo $LOGFILE
python -u run_lm.py \
        --data_dir=$DATADIR \
        --lit_file=$LITFILE \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=orignal_t5 \
        --block_size=1024 \
        --do_train \
        --gpu_per_node $PER_NODE_GPU \
        --learning_rate=8e-5 \
        --weight_decay=0.01 \
        --per_gpu_train_batch_size=2 \
        --per_gpu_eval_batch_size=2 \
        --gradient_accumulation_steps=8 \
        --num_train_epochs=5 \
        --logging_steps=100 \
        --save_steps=100 \
        --seed=42 \
        --overwrite_output_dir \
        --not_pretrain \
        --sample_ratio $SAMPLE_RATIO \
        --save_sample \
        --MASTER_PORT $MASTER_PORT


done