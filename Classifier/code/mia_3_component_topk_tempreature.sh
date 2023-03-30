
export CUDA_VISIBLE_DEVICES=2
MASTER_PORT=94457 # modify
SURROGATE_MODEL=microsoft/CodeGPT-small-java 
VICTIM_MODEL=microsoft/CodeGPT-small-java-adaptedGPT2 
Percentage=0.01

for SEED in 30 42 58
do
for SAMPLE_RATIO in {20..20..10}
do
for TOPK in 10 50 100
do
for tempreature in 1.0
do


LANG=java    
CLASSIFIER_SAVE_DICT=../classifier_save/PTM3/javaCorpus/${SURROGATE_MODEL##*/}/${SAMPLE_RATIO}/${SEED}/
PREDICTION_DATA_FOLDER_PATH=../../CodeCompletion-line/dataset/javaCorpus/${Percentage}/${SAMPLE_RATIO}/
LITFILE=../dataset/javaCorpus/literals.json


python mia.py \
    --do_lower_case \
    --lang ${LANG} \
    --surrogate_model ${SURROGATE_MODEL} \
    --victim_model ${VICTIM_MODEL} \
    --sample_ratio ${SAMPLE_RATIO} \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --num_train_epochs 4 \
    --classifier_save_dir ${CLASSIFIER_SAVE_DICT} \
    --prediction_data_folder_path ${PREDICTION_DATA_FOLDER_PATH} \
    --lit_file ${LITFILE} \
    --classifier_model_path ${CLASSIFIER_SAVE_DICT} \
    --weight_decay=0.01 \
    --mode victim_K_${TOPK}_T_${tempreature} \
    --use_tree_component \
    --seed ${SEED} \
    --consider_topk_tempreature
    

done
done
done
done
