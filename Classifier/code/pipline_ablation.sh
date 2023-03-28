export CUDA_VISIBLE_DEVICES=2
export MASTER_PORT=66699 # modify
export SURROGATE_MODEL=microsoft/CodeGPT-small-java # modify
export VICTIM_MODEL=microsoft/CodeGPT-small-java-adaptedGPT2 # modify

export ablation_mode=no_code #no_title/no_text/no_code
export SAMPLE_RATIO=20

for SEED in 42 30 58
do
export SEED=${SEED}
bash ./run_3_component_ablation.sh
bash ./mia_3_component_ablation.sh
done