from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
import json
from sklearn import naive_bayes
import os
import argparse
import logging
from args import victim_maps
VICTIM_MODE2MODEL_MAP = victim_maps()
logger = logging.getLogger(__name__)

def n_gram_feature_extract(n, text):
    # n: n-gram
    # text: string
    # return: list of n-gram
    text = text.lower()
    n_gram = []
    text = text.split(' ')
    for i in range(len(text)-n+1):
        n_gram.append(' '.join(text[i:i+n]))
    return n_gram

def get_data(path,file_type):
    logger.info(f'file_type: {file_type}')
    with open(os.path.join(path,f'{file_type}.json'), 'r') as f:
        train = f.readlines()

    total_feature = []
    total_label = []


    for j,t in enumerate(train):
        feature = []
        t = t.strip()
        t = json.loads(t)
        gt = t['gt']
        pred = t['prediction']
        for i in range(1, 5):
            gt_n_gram = n_gram_feature_extract(i, gt)
            pred_n_gram = n_gram_feature_extract(i, pred)
            n_gram_precision = 0
            for g in pred_n_gram:
                if g in gt_n_gram:
                    n_gram_precision += 1
            n_gram_precision = n_gram_precision / len(gt_n_gram)
            feature.append(n_gram_precision)
        sentence_bleu_score = sentence_bleu([gt.split(" ")], pred.split(" "),smoothing_function=SmoothingFunction().method1)
        feature.append(sentence_bleu_score)
        total_feature.append(feature[:])
        total_label.append(t['label'])
    return total_feature, total_label
def args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classifier_save_dir",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        '--prefix_path',
        type=str,
    )
    parser.add_argument(
        "--victim_model",
        type=str,
        default='micrsoft/CodeGPT-small-py-adaptedGPT2',
        choices=['gpt2','microsoft/CodeGPT-small-py','microsoft/CodeGPT-small-java','rnn','transformer',
                'micrsoft/CodeGPT-small-py-adaptedGPT2','microsoft/CodeGPT-small-java-adaptedGPT2']
    )
    parser.add_argument(
        "--mode",
        # choices=['surrogate','victim'],
        type=str,
    )
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = args()
    args.mode = VICTIM_MODE2MODEL_MAP[args.mode]
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') 
    if not os.path.exists(args.classifier_save_dir):
        os.makedirs(args.classifier_save_dir)
    log_file = os.path.join(args.classifier_save_dir,'log_eval.txt')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info("---------------------------------")
    model_name = args.victim_model.split('/')[-1]
    logger.info("[victim model]: "+ f"{model_name}_{args.mode}")
    logger.info("[classifier model]: naive_bayes")
    logger.info("---------------------------------")
    # prefix_path = '../dataset/java/gpt2/30'
    # prefix_path = '../dataset/java/rnn/30'
    # prefix_path = '../dataset/java/transformer/30'
    # prefix_path = '../dataset/java/microsoft/CodeGPT-small-java-adaptedGPT2/30'
    # prefix_path = '../dataset/java/microsoft/CodeGPT-small-java/30'
    train_feature, train_label = get_data(args.prefix_path, 'train')
    logger.info("[get training data]")
    clf = naive_bayes.GaussianNB()
    logger.info(f"[begin training],{len(train_feature)}")
    clf = clf.fit(train_feature, train_label)
    logger.info("[finish training]")
    val_feature, val_label = get_data(args.prefix_path, 'val')
    logger.info("[get val data]")
    logger.info(clf.score(val_feature, val_label))
    test_feature, test_label = get_data(args.prefix_path, f'test_{model_name}_{args.mode}')
    logger.info("[get testing data]")
    logger.info(clf.score(test_feature, test_label))
    logger.info("\n\n")


    


       