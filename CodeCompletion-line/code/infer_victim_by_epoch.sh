# 定义所有要遍历的模型
MODELS=(
    # "microsoft/CodeGPT-small-java-adaptedGPT2"
    # "codeparrot/codeparrot-small"
    # "NinedayWang/PolyCoder-0.4B"
    # "Salesforce/codegen-350M-multi"
    "EleutherAI/gpt-neo-125m"
    "NinedayWang/PolyCoder-160M"
)

GPU_GROUPS["1"]="0"
GPU_GROUPS["2"]="1"
GPU_GROUPS["3"]="0"
GPU_GROUPS["4"]="1"

MASTER_PORT=92257
MODE=victim 
Percentage=0.01 
LANG=java                      

# 遍历每个模型
for EPOCH in {1..4}

do
    # 遍历 epoch 数字 0 到 4
    for MODEL in "${MODELS[@]}"
    do
        export CUDA_VISIBLE_DEVICES=${GPU_GROUPS[$EPOCH]}
        # 遍历 SAMPLE_RATIO，从 10 到 20，步长为 10
        for SAMPLE_RATIO in {10..20..10}
        do
            DATADIR="../dataset/javaCorpus/${Percentage}/${SAMPLE_RATIO}/"
            LITFILE="../dataset/javaCorpus/literals.json"
            PRETRAINDIR="../../CodeCompletion-token/save/javaCorpus/${MODEL}/100/checkpoint-epoch-${EPOCH}"
            LOGFILE="logs/${MODEL##*/}_${MODE}_eval_epoch${EPOCH}.log"
            
            echo "Running MODEL: $MODEL, EPOCH: $EPOCH, SAMPLE_RATIO: $SAMPLE_RATIO"
            
            python -u run.py \
                --data_dir="$DATADIR" \
                --lit_file="$LITFILE" \
                --langs="$LANG" \
                --output_dir="$DATADIR" \
                --pretrain_dir="$PRETRAINDIR" \
                --log_file="$LOGFILE" \
                --model_type=santacoder \
                --block_size=1024 \
                --eval_line \
                --logging_steps=100 \
                --seed=42 \
                --MASTER_PORT "$MASTER_PORT" \
                --per_gpu_eval_batch_size 128 \
                --mode "$MODE"
        done 
    done &
done