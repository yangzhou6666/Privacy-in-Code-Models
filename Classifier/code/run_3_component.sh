# 1. 把数据进行划分并保存 dataset/model/sample_ratio/
# 2. 在1:1上进行训练 (name: bert_${model}_${sample_ratio}.pth),一共保存4x3个
# 3. 在1:1上进行评测
export CUDA_VISIBLE_DEVICES=7
MODEL=microsoft/codebert-base #change model
MASTER_PORT=94287 # modify
SURROGATE_MODEL=rnn # modify



for SAMPLE_RATIO in {10..30..10}
do

LANG=java    
CLASSIFIER_SAVE_DICT=../classifier_save/PTM3/javaCorpus/${SURROGATE_MODEL##*/}/${SAMPLE_RATIO}/
PREDICTION_DATA_FOLDER_PATH=../../CodeCompletion-line/dataset/javaCorpus/${SURROGATE_MODEL##*/}/${SAMPLE_RATIO}/
LITFILE=../dataset/javaCorpus/literals.json

python run.py \
    --do_lower_case \
    --lang ${LANG} \
    --surrogate_model ${SURROGATE_MODEL} \
    --sample_ratio ${SAMPLE_RATIO} \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --num_train_epochs 4 \
    --classifier_save_dir ${CLASSIFIER_SAVE_DICT} \
    --prediction_data_folder_path ${PREDICTION_DATA_FOLDER_PATH} \
    --lit_file ${LITFILE} \
    --classifier_model_path ${MODEL} \
    --weight_decay=0.01 \
    --use_tree_component 

done

