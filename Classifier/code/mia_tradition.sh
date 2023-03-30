SURROGATE_MODEL=microsoft/CodeGPT-small-java
SAMPLE_RATIO=20
CLASSIFIER_SAVE_DICT=../classifier_save/PTM3/javaCorpus/CodeGPT-small-java/20/42

python naive_bayes.py \
    --classifier_save_dir ${CLASSIFIER_SAVE_DICT} \
    --prefix_path ../dataset/java/${SURROGATE_MODEL}/${SAMPLE_RATIO} \
    --mode victim

python decision_tree.py \
    --classifier_save_dir ${CLASSIFIER_SAVE_DICT} \
    --prefix_path ../dataset/java/${SURROGATE_MODEL}/${SAMPLE_RATIO} \
    --mode victim

python nearest_neighbors.py \
    --classifier_save_dir ${CLASSIFIER_SAVE_DICT} \
    --prefix_path ../dataset/java/${SURROGATE_MODEL}/${SAMPLE_RATIO} \
    --mode victim

python mlp.py \
    --classifier_save_dir ${CLASSIFIER_SAVE_DICT} \
    --prefix_path ../dataset/java/${SURROGATE_MODEL}/${SAMPLE_RATIO} \
    --mode victim

python dnn.py \
    --classifier_save_dir ${CLASSIFIER_SAVE_DICT} \
    --prefix_path ../dataset/java/${SURROGATE_MODEL}/${SAMPLE_RATIO} \
    --mode victim