import argparse
import json
import logging
import os
import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertConfig

from args import victim_maps
from dataset import ClassificationDataset, prepare_data, keep_test_data, ClassificationDataset_collate_fn, \
    keep_sample_data, prepare_sample_java_data, prepare_sample_data, keep_normal_test_data
from model import TBertT, TBertTNoTitle, TBertTNoText, TBertTNoCode

VICTIM_MODE2MODEL_MAP = victim_maps()
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="random seed for initialization",
    )
    parser.add_argument(
        "--do_lower_case", 
        action='store_true',
        help="Set this flag if you are using an uncased model."
    )
    parser.add_argument(
        '--use_tree_component',
        action='store_true',
        help='whether to use tree component'
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default='../dataset',
        help="the data for classifier",
    )
    parser.add_argument(
        "--lang",
        type=str,
        choices=['python', 'java']
    ) 
    parser.add_argument(
        "--surrogate_model",
        type=str,
        choices=['gpt2','microsoft/CodeGPT-small-py','microsoft/CodeGPT-small-java','rnn','transformer',
                'micrsoft/CodeGPT-small-py-adaptedGPT2','microsoft/CodeGPT-small-java-adaptedGPT2', 'Salesforce/codegen-350M-multi']
    )
    parser.add_argument(
        "--victim_model",
        type=str,
        default='micrsoft/CodeGPT-small-py-adaptedGPT2',
        choices=['gpt2','microsoft/CodeGPT-small-py','microsoft/CodeGPT-small-java','rnn','transformer',
                'micrsoft/CodeGPT-small-py-adaptedGPT2','microsoft/CodeGPT-small-java-adaptedGPT2', "Salesforce/codegen-350M-multi",
                "bigcode/starcoderbase-1b",
                "NinedayWang/PolyCoder-0.4B",
                "codeparrot/codeparrot-small",
                "bigcode/santacoder"]
    )
    parser.add_argument(
        "--sample_ratio",
        type=str,
        choices=['5','10','20','30']
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--consider_epoch",
        action='store_true',
        help="whether to consider epoch"
    )
    parser.add_argument(
        "--num_train_epochs",
        default=2,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps.",
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight deay if we apply some.",
    )
    parser.add_argument(
        "--adam_epsilon", 
        default=1e-8, 
        type=float,
    help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--gradient_accumulation_steps", 
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--classifier_save_dir",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--prediction_data_folder_path",
        type=str,
    )
    parser.add_argument(
        "--lit_file",
        type=str,
    )
    parser.add_argument(
        '--total_length',
        type=int,
        default=512, 
    )
    parser.add_argument(
        '--input_length',
        type=int,
        default=400, #最长输入为512，还要拼接2个回答，和特殊字符，所以这里设置为400
    )
    parser.add_argument(
        '--prediction_length',
        type=int,
        default=16,
    )
    parser.add_argument(
        '--classifier_model_path',
        type=str,
    )
    parser.add_argument(
        "--evaluate_steps",
        type=int,
        default=100,
        help="evaluate every evaluate_steps steps"
    )
    parser.add_argument(
        "--mode",
        # choices=['surrogate','victim'],
        type=str,
    )
    parser.add_argument(
        '--save_results',
        action='store_true'
    )
    parser.add_argument(
        '--consider_topk_tempreature',
        action='store_true'
    )
    parser.add_argument(
        '--consider_sample_java',
        action='store_true'
    )
    parser.add_argument(
        '--consider_sample_all',
        action='store_true'
    )
    parser.add_argument(
        '--ablation_mode',
        type=str,
        choices=['no_title','no_text','no_code'],
        default=None
    )
    
    args = parser.parse_args()
    return args

def get_special_tokens(path):
    lits = json.load(open(path))
    tokens = ["<STR_LIT>", "<NUM_LIT>", "<CHAR_LIT>"]
    for lit in lits["str"]:
        tokens.append(f"<STR_LIT:{lit}>")
    for lit in lits["num"]:
        tokens.append(f"<NUM_LIT:{lit}>")
    for lit in lits["char"]:
        tokens.append(f"<CHAR_LIT:{lit}>")
    return tokens

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if args.n_gpu > 0:
    #     torch.cuda.manual_seed_all(args.seed)
def evaluate(args,model,dataloader,device):
    model.eval()
    eval_loss = 0
    acc = 0
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    all_prediction = []
    all_labels = []
    all_logits = []
    for batch in tqdm(dataloader, desc="Evaluating"):
        with torch.no_grad():
            if args.use_tree_component:
                inputs = {'title_ids': batch['input_ids'].to(device),
                          'title_attention_mask': batch['input_mask'].to(device),
                          'text_ids': batch['groundtruth_ids'].to(device),
                          'text_attention_mask': batch['groundtruth_mask'].to(device),
                          'code_ids': batch['prediction_ids'].to(device),
                          'code_attention_mask': batch['prediction_mask'].to(device),
                        }
                logits = model(**inputs)
            else:
                inputs = {'input_ids': batch['input_ids'].to(device),
                        'attention_mask': batch['input_mask'].to(device)}
                logits = model(**inputs)[0]
            labels = batch['labels'].to(device)
            loss_fct = torch.nn.CrossEntropyLoss()
            e_l = loss_fct(logits.view(-1, 2), labels.view(-1))
            eval_loss += e_l.mean().item()
            prediction = logits.detach().cpu().argmax(dim=1).numpy()
            logits = logits.detach().cpu().numpy()
            labels = labels.to('cpu').numpy()
            acc += np.sum(prediction == labels)
            tn += np.sum((prediction == 0) & (labels == 0))
            tp += np.sum((prediction == 1) & (labels == 1))
            fp += np.sum((prediction == 1) & (labels == 0))
            fn += np.sum((prediction == 0) & (labels == 1))
            all_prediction.extend(list(prediction))
            all_labels.extend(list(labels))
            all_logits.extend(list(logits[:,1]))
    return {
        'loss':eval_loss / len(dataloader), 
        'acc':acc / len(dataloader.dataset),
        'recall':tp / (tp + fn),
        'precision':tp / (tp + fp),
        'f1':2 * tp / (2 * tp + fp + fn),
        'TPR':tp / (tp + fn),
        'FPR':fp / (fp + tn),
        'AUC':roc_auc_score(all_labels,all_logits) if not args.consider_sample_java else 'not applicable',
    },all_prediction


            
def main():
    args = get_args()
    if args.consider_epoch:
        args.mode = VICTIM_MODE2MODEL_MAP[args.mode]
    
    
    # utils
    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    if args.seed:
        set_seed(args)
    # setting logger
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') 
    if args.ablation_mode is not None:
        args.classifier_save_dir = os.path.join(args.classifier_save_dir,args.ablation_mode)
        args.classifier_model_path = os.path.join(args.classifier_model_path,args.ablation_mode)
    if not os.path.exists(args.classifier_save_dir):
        os.makedirs(args.classifier_save_dir)
    log_file = os.path.join(args.classifier_save_dir,'log_eval.txt')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    special_token = get_special_tokens(args.lit_file)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base",do_lower_case=True,additional_special_tokens=special_token)
    
    if args.use_tree_component:
        if args.ablation_mode is None:
            model = TBertT(BertConfig(),'microsoft/codebert-base',num_class=2)
        elif args.ablation_mode == 'no_title': #对应无input_ids
            model = TBertTNoTitle(BertConfig(),'microsoft/codebert-base',num_class=2)
        elif args.ablation_mode == 'no_text': #对应无groundtruth_ids
            model = TBertTNoText(BertConfig(),'microsoft/codebert-base',num_class=2)
        elif args.ablation_mode == 'no_code': #对应无prediction_ids
            model = TBertTNoCode(BertConfig(),'microsoft/codebert-base',num_class=2)
        model.resize_token_embeddings(len(tokenizer))
        model.load_state_dict(torch.load(os.path.join(args.classifier_model_path,'pytorch_model.bin')),strict=True)
        # model.load_state_dict(torch.load(os.path.join(args.classifier_model_path, 'model.safetensors')), strict=True)
        logger.info("[load model]: "+ f"{args.classifier_model_path}")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.classifier_model_path,num_labels=2) #label=0/1
        model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    logger.info("---------------------------------")
    model_name = args.victim_model.split('/')[-1]
    logger.info(f"[surrogate model loaded]: {args.classifier_model_path}")
    logger.info("[victim model]: "+ f"{model_name}_{args.mode}")
    logger.info("[classifier model]: BERT-based")
    logger.info("[seed]: "+ f"{args.seed}")
    if args.ablation_mode is not None:
        logger.info("[ablation mode]: "+ f"{args.ablation_mode}")
    if args.use_tree_component:
        logger.info("[tree component]: used")
    logger.info("---------------------------------")
    
    # data prepare
    data_path = os.path.join(args.data_dir,args.lang,args.surrogate_model,args.sample_ratio)
    if args.consider_epoch or args.consider_topk_tempreature:
        logger.info('[USE EPOCH or TOPL-TEMPERATURE DATA]')
        if not os.path.exists(os.path.join(data_path,f'test_{model_name}_{args.mode}.json')):
            classifier_train, classifier_test = prepare_data(args)
            logger.info(f"[data prepared]: length :{len(classifier_train)} {len(classifier_test)}")
            keep_test_data(args,classifier_train,classifier_test)
        test_dataset = ClassificationDataset(args,file_type=f'test_{model_name}_{args.mode}',tokenizer=tokenizer)
    elif args.consider_sample_java:
        data_path = os.path.join(args.prediction_data_folder_path,'sample_java')
        classifier_test = prepare_sample_java_data(args)
        logger.info(f"[data prepared]: length : {len(classifier_test)}")
        keep_sample_data(args,classifier_test)
        test_dataset = ClassificationDataset(args,file_type=f'test_{model_name}_{args.mode}',tokenizer=tokenizer)
    elif args.consider_sample_all:
        data_path = os.path.join(args.prediction_data_folder_path,'sample_all')
        classifier_train, classifier_test = prepare_sample_data(args)
        logger.info(f"[data prepared]: length : {len(classifier_test)}")
        keep_test_data(args,classifier_train,classifier_test)
        test_dataset = ClassificationDataset(args,file_type=f'test_{model_name}_{args.mode}',tokenizer=tokenizer)
    else:
        if not os.path.exists(os.path.join(data_path,'test.json')):
            classifier_train, classifier_test = prepare_data(args)
            logger.info(f"[data prepared]: length :{len(classifier_train)} {len(classifier_test)}")
            keep_normal_test_data(args,classifier_train,classifier_test)
        test_dataset = ClassificationDataset(args,file_type='test',tokenizer=tokenizer)

    # 其中train.json和dev.json是用于训练和测试的classifier的数据，test.json是用于验证真实世界情况的数据
    batch_size = args.batch_size * args.gradient_accumulation_steps
    if not args.consider_sample_java and not args.consider_sample_all:
        val_dataset = ClassificationDataset(args,file_type='val',tokenizer=tokenizer)
        val_dataloader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,collate_fn=ClassificationDataset_collate_fn)
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,collate_fn=ClassificationDataset_collate_fn)
    
    
    # test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,collate_fn=ClassificationDataset_collate_fn)

    
    
    if not args.consider_sample_java and not args.consider_sample_all:
        res, _ = evaluate(args, model, val_dataloader, device=device)
        logger.info('[val_best_acc]: {}\n\n'.format(res))
    res, all_logits  =evaluate(args, model, test_dataloader, device=device)
    logger.info('[best_acc]: {}\n\n'.format(json.dumps(res, indent=2)))
    if args.save_results:
        logger.info('results saved \n')
        data_dir = os.path.join(args.data_dir,args.lang,args.surrogate_model,args.sample_ratio)
        save_dir = os.path.join(args.classifier_save_dir,f'res_{args.sample_ratio}_{args.seed}.json')
        with open(os.path.join(data_dir,'test'+'.json'),'r') as f:
            data = f.readlines()
        with open(save_dir,'w')as f:
            for i,d in enumerate(data):
                d = json.loads(d)
                d['predicition_label'] = int(all_logits[i])
                f.write(json.dumps(d))
                f.write('\n')
        logger.info(f'[results saved]: {save_dir} \n')


if __name__ == '__main__':
    main()