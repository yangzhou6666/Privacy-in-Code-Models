LANG=java                       # set python for py150
DATADIR=../dataset/javaCorpus/token_completion
LITFILE=../dataset/javaCorpus/literals.json
OUTPUTDIR=../save/javaCorpus
PRETRAINDIR=microsoft/CodeGPT-small-java-adaptedGPT2  # microsoft/CodeGPT-small-py for py150
PER_NODE_GPU=1       # modify YOUR_GPU_NUM

MASTER_PORT=94027 #每次跑这个都要改这个
export CUDA_VISIBLE_DEVICES=3 #指定GPU,每次都要改


for SAMPLE_RATIO in {100..100..10}
do
temp=${PRETRAINDIR:0-9:7} #根据长度确定要不要改,microsoft/CodeGPT-small-py for py150
# temp=$PRETRAINDIR
LOGFILE="completion_javaCorpus_${temp}_${SAMPLE_RATIO}".log
echo $temp
echo $LOGFILE
# 训练时，sample_ratio设置为0.1，模拟BOB知道部分数据
python -m torch.distributed.launch --nproc_per_node=$PER_NODE_GPU run_lm.py \
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
        --per_gpu_train_batch_size=2 \
        --per_gpu_eval_batch_size=2 \
        --gradient_accumulation_steps=8 \
        --num_train_epochs=2 \
        --logging_steps=100 \
        --save_steps=1000 \
        --seed=42 \
        --overwrite_output_dir \
        --not_pretrain \
        --sample_ratio $SAMPLE_RATIO \
        --save_sample \
        --MASTER_PORT $MASTER_PORT

PRETRAINDIR="../save/javaCorpus/${PRETRAINDIR}/${SAMPLE_RATIO}/checkpoint-last"     # directory of your saved model
LOGFILE="completion_javaCorpus_eval_${temp}_${SAMPLE_RATIO}".log
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
done