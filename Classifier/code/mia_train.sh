export MASTER_PORT=43422 # modify
# export SURROGATE_MODEL=microsoft/CodeGPT-small-java-adaptedGPT2 # modify
export SURROGATE_MODEL=transformer

# Define seed groups and their corresponding CUDA devices
declare -A SEED_GROUPS
SEED_GROUPS["2"]="30 42 58"
SEED_GROUPS["3"]="10 21 31"
SEED_GROUPS["1"]="50 54 98"
SEED_GROUPS["0"]="59 20 41"


# Iterate over each CUDA device and its seed values
for CUDA_DEVICE in "${!SEED_GROUPS[@]}"; do
    SEEDS=${SEED_GROUPS[$CUDA_DEVICE]}
    export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

    # Iterate over each seed value
    for SEED in $SEEDS; do
        export SEED
        echo "  Starting with SEED=$SEED on CUDA_VISIBLE_DEVICES=$CUDA_DEVICE"
        # Run the script in the background
        bash ./run_3_component.sh 
    done &
done



echo "All experiments completed."

