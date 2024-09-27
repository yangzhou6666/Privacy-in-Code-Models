#!/bin/bash

export MASTER_PORT=43422 # modify

# Define arrays of models and epochs
VICTIM_MODELS=(
    "codeparrot/codeparrot-small"
    "microsoft/CodeGPT-small-java-adaptedGPT2"
    "NinedayWang/PolyCoder-0.4B"
    "Salesforce/codegen-350M-multi"
    "EleutherAI/gpt-neo-125m"
    "NinedayWang/PolyCoder-160M"
)

EPOCHS=(
    1
    2
    3
    4
)

SURROGATE_MODELS=(
    "transformer"
    "gpt2"
    "rnn"
    "microsoft/CodeGPT-small-java-adaptedGPT2"
)

# Define seed groups and their corresponding CUDA devices
declare -A SEED_GROUPS
SEED_GROUPS["2"]="30 42 58 10"
SEED_GROUPS["1"]="50 54 98 21"
SEED_GROUPS["0"]="59 20 41 31"


# Iterate over each CUDA device and its seed values
for CUDA_DEVICE in "${!SEED_GROUPS[@]}"; do
    SEEDS=${SEED_GROUPS[$CUDA_DEVICE]}
    export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE


    # Iterate over each SURROGATE_MODEL
    for SURROGATE_MODEL in "${SURROGATE_MODELS[@]}"; do
        export SURROGATE_MODEL="$SURROGATE_MODEL"
        echo "  Using SURROGATE_MODEL: $SURROGATE_MODEL"

        # Iterate over each EPOCH
        for EPOCH in "${EPOCHS[@]}"; do
            export EPOCH="$EPOCH"
            echo "    With EPOCH: $EPOCH"

            # Iterate over each VICTIM_MODEL
            for VICTIM_MODEL in "${VICTIM_MODELS[@]}"; do
                export VICTIM_MODEL="$VICTIM_MODEL"
                echo "Running experiments for VICTIM_MODEL: $VICTIM_MODEL"
                # Iterate over each seed value
                for SEED in $SEEDS; do
                    export SEED
                    echo "      Starting with SEED=$SEED on CUDA_VISIBLE_DEVICES=$CUDA_DEVICE"

                    # Run the script in the background
                    bash ./mia_3_component.sh 

                done
            done

            # Wait for all background processes to finish before moving to the next EPOCH
            wait
            echo "    Completed experiments for EPOCH: $EPOCH"

        done &

        echo "  Completed experiments for SURROGATE_MODEL: $SURROGATE_MODEL"

    done 

    echo "Completed experiments for VICTIM_MODEL: $VICTIM_MODEL"
done

echo "All experiments completed."