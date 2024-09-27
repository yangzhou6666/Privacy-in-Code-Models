export CUDA_VISIBLE_DEVICES=3 # modify
export MASTER_PORT=43422 # modify
export SURROGATE_MODEL=gpt2 # modify
# export VICTIM_MODEL=microsoft/CodeGPT-small-java-adaptedGPT2
# export VICTIM_MODEL=codeparrot/codeparrot-small

# Define an array of VICTIM_MODELs
VICTIM_MODELS=(
    # "codeparrot/codeparrot-small"
    "microsoft/CodeGPT-small-java-adaptedGPT2"
    "NinedayWang/PolyCoder-0.4B"
    "bigcode/starcoderbase-1b"
    "Salesforce/codegen-350M-multi"
    "bigcode/santacoder"
    "facebook/incoder-1B"
)

# Define seed groups and their corresponding CUDA devices
declare -A SEED_GROUPS
SEED_GROUPS["2"]="30 42 58"
SEED_GROUPS["3"]="10 21 31"
SEED_GROUPS["1"]="50 54 98"
SEED_GROUPS["0"]="59 20 41"


# Iterate over each VICTIM_MODEL
for MODEL in "${VICTIM_MODELS[@]}"; do
    export VICTIM_MODEL="$MODEL"
    echo "Running experiments for VICTIM_MODEL: $VICTIM_MODEL"

    # Iterate over each CUDA device and its seed values
    for CUDA_DEVICE in "${!SEED_GROUPS[@]}"; do
        SEEDS=${SEED_GROUPS[$CUDA_DEVICE]}
        export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

        # Iterate over each seed value
        for SEED in $SEEDS; do
            export SEED
            echo "  Starting with SEED=$SEED on CUDA_VISIBLE_DEVICES=$CUDA_DEVICE"
            # Run the script in the background
            bash ./mia_3_component.sh &
        done
    done

    # Wait for all background processes to finish before moving to the next model
    wait
    echo "Completed experiments for VICTIM_MODEL: $VICTIM_MODEL"
done

echo "All experiments completed."

# for SEED in 30 42 58  
# do
# export CUDA_VISIBLE_DEVICES=2
# export SEED=${SEED}
# # bash ./run_3_component.sh
# bash ./mia_3_component.sh
# done &

