from dataset import ClassificationDataset,prepare_data,divide_data,ClassificationDataset_collate_fn
from model import TBertT,TBertTNoTitle,TBertTNoText,TBertTNoCode
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
        choices=['gpt2','microsoft/CodeGPT-small-py','microsoft/CodeGPT-small-java','transformer',
                'micrsoft/CodeGPT-small-py-adaptedGPT2','microsoft/CodeGPT-small-java-adaptedGPT2','rnn']
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
        default='surrogate',
        type=str,
    )
    parser.add_argument(
        '--use_tree_component' ,
        action='store_true',
        help='whether to use tree component'
    )
    parser.add_argument(
        '--ablation_mode',
        type=str,
        choices=['no_title','no_text','no_code'],
        default=None
    )
    #下面这两个参数是mia用的，run.py不需要
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
        "--consider_epoch",
        action='store_true',
        help="whether to consider epoch"
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
            logits = logits.detach().cpu().argmax(dim=1).numpy()
            labels = labels.to('cpu').numpy()
            acc += np.sum(logits == labels)
    return eval_loss / len(dataloader), acc / len(dataloader.dataset)


            
def main():
    args = get_args()
    # args.mode = 'surrogate'
    
    
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
    if not os.path.exists(args.classifier_save_dir):
        os.makedirs(args.classifier_save_dir)
    log_file = os.path.join(args.classifier_save_dir,'log.txt')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    special_token = get_special_tokens(args.lit_file)
    tokenizer = AutoTokenizer.from_pretrained(args.classifier_model_path,do_lower_case=True,additional_special_tokens=special_token)
    model = AutoModelForSequenceClassification.from_pretrained(args.classifier_model_path,num_labels=2) #label=0/1
    model.resize_token_embeddings(len(tokenizer))
    if args.use_tree_component:
        if args.ablation_mode is None:
            model = TBertT(model.config,args.classifier_model_path,num_class=2)
        elif args.ablation_mode == 'no_title': #对应无input_ids
            model = TBertTNoTitle(model.config,args.classifier_model_path,num_class=2)
        elif args.ablation_mode == 'no_text': #对应无groundtruth_ids
            model = TBertTNoText(model.config,args.classifier_model_path,num_class=2)
        elif args.ablation_mode == 'no_code': #对应无prediction_ids
            model = TBertTNoCode(model.config,args.classifier_model_path,num_class=2)
        model.resize_token_embeddings(len(tokenizer))

    model.to(device)
    
    # data prepare
    data_path = os.path.join(args.data_dir,args.lang,args.surrogate_model,args.sample_ratio)
    if not os.path.exists(os.path.join(data_path,'train.json')) or not os.path.exists(os.path.join(data_path,'test.json')):
        classifier_train, classifier_test = prepare_data(args)
        divide_data(args,classifier_train,classifier_test)
    # 其中train.json和dev.json是用于训练和测试的classifier的数据，test.json是用于验证真实世界情况的数据
    train_dataset = ClassificationDataset(args,file_type='train',tokenizer=tokenizer)
    val_dataset = ClassificationDataset(args,file_type='val',tokenizer=tokenizer)
    # test_dataset = ClassificationDataset(args,file_type='test',tokenizer=tokenizer)

    batch_size = args.batch_size * args.gradient_accumulation_steps
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,collate_fn=ClassificationDataset_collate_fn)
    val_dataloader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,collate_fn=ClassificationDataset_collate_fn)
    # test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,collate_fn=ClassificationDataset_collate_fn)

    # Prepare optimizer and schedule (linear warmup and decay)
    if args.num_train_epochs > 0:
        t_total = len(train_dataset) // batch_size * args.num_train_epochs
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  surrogate_model = "+ args.surrogate_model)
    logger.info("  Num examples = %d", len(train_dataset)) 
    logger.info("  Num epoch = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  seed = %d", args.seed)
    if args.ablation_mode is not None:
        logger.info("  ablation_mode = %s", args.ablation_mode)
    best_acc = 0.0
    model.zero_grad()
    for epoch in range(args.num_train_epochs):
        model.train()
        total_loss = 0.0
        for step, batch in tqdm(enumerate(train_dataloader),total=len(train_dataloader)):
            if args.use_tree_component:
                inputs = {'title_ids': batch['input_ids'].to(device),
                          'title_attention_mask': batch['input_mask'].to(device),
                          'text_ids': batch['groundtruth_ids'].to(device),
                          'text_attention_mask': batch['groundtruth_mask'].to(device),
                          'code_ids': batch['prediction_ids'].to(device),
                          'code_attention_mask': batch['prediction_mask'].to(device),
                        }
                outputs = model(**inputs)
                targets = batch['labels'].to(device)
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(outputs, targets)
            else:
                inputs = {'input_ids': batch['input_ids'].to(device),
                        'attention_mask': batch['input_mask'].to(device),
                        'labels': batch['label'].to(device)}
                outputs = model(**inputs)
                loss = outputs[0]
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            total_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            # if (step % args.evaluate_steps == 0 and step != 0) or (step == len(train_dataloader) - 1):
            if (step % args.evaluate_steps == 0) or (step == len(train_dataloader) - 1):
                logger.info('epoch: {}, step: {}, loss: {}'.format(epoch, step, total_loss / args.evaluate_steps))
                total_loss = 0.0
                val_loss, val_acc = evaluate(args, model, val_dataloader,device=device)
                logger.info('epoch: {}, step: {}, val_loss: {}, val_acc: {}'.format(epoch, step, val_loss, val_acc))
                model.train()
                if val_acc > best_acc:
                    best_acc = val_acc
                    # save model
                    model_to_save = model.module if hasattr(model, 'module') else model
                    if not os.path.exists(args.classifier_save_dir):
                        os.makedirs(args.classifier_save_dir)
                    model_to_save.save_pretrained(args.classifier_save_dir)
                    logger.info("Saving model checkpoint to %s", args.classifier_save_dir)
    logger.info('[best_acc]: {}'.format(best_acc))

if __name__ == '__main__':
    main()