from dataset import ClassificationDataset,prepare_data,keep_test_data,ClassificationDataset_collate_fn
import os
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from torch.utils.data import DataLoader
import torch
import random
import numpy as np
from transformers import AdamW,get_linear_schedule_with_warmup
import logging
from tqdm import tqdm
import argparse
import json
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
        "--data_dir",
        type=str,
        default='../dataset',
        help="the data for classifier",
    )
    parser.add_argument(
        "--lang",
        type=str,
        choices=['python','java']
    ) 
    parser.add_argument(
        "--surrogate_model",
        type=str,
        choices=['gpt2','microsoft/CodeGPT-small-py','microsoft/CodeGPT-small-java',
                'micrsoft/CodeGPT-small-py-adaptedGPT2','microsoft/CodeGPT-small-java-adaptedGPT2']
    )
    parser.add_argument(
        "--sample_ratio",
        type=str,
        choices=['10','20','30']
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
        choices=['surrogate','victim'],
        type=str,
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
    for batch in tqdm(dataloader, desc="Evaluating"):
        with torch.no_grad():
            inputs = {'input_ids':      batch['input_ids'].to(device),
                      'attention_mask': batch['attention_mask'].to(device),
                      }
            labels = batch['label'].to(device)
            logits = model(**inputs)[0]
            loss_fct = torch.nn.CrossEntropyLoss()
            e_l = loss_fct(logits.view(-1, 2), labels.view(-1))
            eval_loss += e_l.mean().item()
            logits = logits.detach().cpu().argmax(dim=1).numpy()
            labels = labels.to('cpu').numpy()
            acc += np.sum(logits == labels)
    return eval_loss / len(dataloader), acc / len(dataloader.dataset)


            
def main():
    args = get_args()
    
    
    # utils
    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    if args.seed:
        set_seed(args)
    # setting logger
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

    special_token = get_special_tokens(args.lit_file)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base",do_lower_case=True,additional_special_tokens=special_token)
    model = AutoModelForSequenceClassification.from_pretrained(args.classifier_model_path,num_labels=2) #label=0/1
    logger.info(f"[model loaded]: {args.classifier_model_path}")
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    
    # data prepare
    data_path = os.path.join(args.data_dir,args.lang,args.surrogate_model,args.sample_ratio)
    if not os.path.exists(os.path.join(data_path,'test.json')):
        classifier_train, classifier_test = prepare_data(args)
        logger.info(f"[data prepared]: length :{len(classifier_train)} {len(classifier_test)}")
        keep_test_data(args,classifier_train,classifier_test)
    # 其中train.json和dev.json是用于训练和测试的classifier的数据，test.json是用于验证真实世界情况的数据
    test_dataset = ClassificationDataset(args,file_type='test',tokenizer=tokenizer)
    # test_dataset = ClassificationDataset(args,file_type='test',tokenizer=tokenizer)

    batch_size = args.batch_size * args.gradient_accumulation_steps
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,collate_fn=ClassificationDataset_collate_fn)
    # test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,collate_fn=ClassificationDataset_collate_fn)

    
    

    
    test_loss,val_acc  =evaluate(args, model, test_dataloader,device=device)
    
    
    logger.info('[best_acc]: {}'.format(val_acc))

if __name__ == '__main__':
    main()