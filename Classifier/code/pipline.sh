export CUDA_VISIBLE_DEVICES=3
export MASTER_PORT=43422 # modify
export SURROGATE_MODEL=rnn # modify
export VICTIM_MODEL=microsoft/CodeGPT-small-java-adaptedGPT2 # modify

for SEED in 42 30 58
do
export SEED=${SEED}
# bash ./run_3_component.sh
bash ./mia_3_component.sh
done