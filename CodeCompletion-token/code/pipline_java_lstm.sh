LANG=java                       # set python for py150

LITFILE=../dataset/javaCorpus/literals.json

PRETRAINDIR=gpt2  
PER_NODE_GPU=1      

MASTER_PORT=64129  #change every time
export CUDA_VISIBLE_DEVICES=3 #specify GPU, change every time

for SAMPLE_RATIO in {10..20..10}
do
LOGFILE="logs/LSTM_${SAMPLE_RATIO}".log
DATADIR="../dataset/javaCorpus/token_completion/"
OUTPUTDIR="../save/javaCorpus/"
echo $LOGFILE
python -u run_lm.py \
        --data_dir=$DATADIR \
        --lit_file=$LITFILE \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --tokenizer_dir=$PRETRAINDIR \
        --config_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=rnn \
        --block_size=1024 \
        --do_train \
        --gpu_per_node $PER_NODE_GPU \
        --learning_rate=8e-5 \
        --weight_decay=0.01 \
        --per_gpu_train_batch_size=4 \
        --per_gpu_eval_batch_size=4 \
        --gradient_accumulation_steps=4 \
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