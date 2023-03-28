LANG=java                       # set python for py150

LITFILE=../dataset/javaCorpus/literals.json

PRETRAINDIR=gpt2  # microsoft/CodeGPT-small-py for py150
PER_NODE_GPU=1       # modify YOUR_GPU_NUM

MASTER_PORT=64129 #每次跑这个都要改这个
export CUDA_VISIBLE_DEVICES=3 #指定GPU,每次都要改
Percentage=0.01

for SAMPLE_RATIO in 5
do
LOGFILE="completion_javaCorpus_lstm_${SAMPLE_RATIO}".log
DATADIR="../dataset/javaCorpus/token_completion/"
OUTPUTDIR="../save/javaCorpus/"
echo $LOGFILE
# 训练时，sample_ratio设置为0.1，模拟BOB知道部分数据
# python -m torch.distributed.launch --nproc_per_node=$PER_NODE_GPU run_lm.py \
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

# PRETRAINDIR="../save/javaCorpus/${PRETRAINDIR}/${SAMPLE_RATIO}/checkpoint-last"     # directory of your saved model
# LOGFILE="completion_javaCorpus_eval_${PRETRAINDIR##*/}_${SAMPLE_RATIO}".log
# python -u run_lm.py \
#         --data_dir=$DATADIR \
#         --lit_file=$LITFILE \
#         --langs=$LANG \
#         --output_dir=$OUTPUTDIR \
#         --pretrain_dir=$PRETRAINDIR \
#         --log_file=$LOGFILE \
#         --model_type=gpt2 \
#         --block_size=1024 \
#         --do_eval \
#         --per_gpu_eval_batch_size=16 \
#         --logging_steps=100 \
#         --seed=42 
done