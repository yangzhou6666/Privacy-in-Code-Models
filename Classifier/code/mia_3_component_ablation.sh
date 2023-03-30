
Percentage=0.01



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
    --mode victim \
    --use_tree_component \
    --seed ${SEED} \
    --ablation_mode ${ablation_mode} \


