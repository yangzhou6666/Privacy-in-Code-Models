# 训练classifer

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import json
import torch
import numpy as np
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForNextSentencePrediction, BertTokenizer,
                          )
import logging
from dataset import NextSentencePredictionDataset
logger = logging.getLogger(__name__)
MODEL_CLASSES = {
    'bert': (BertConfig, BertForNextSentencePrediction, BertTokenizer),
}

def load_and_cache_examples(args, tokenizer, evaluate=False):
    dataset = NextSentencePredictionDataset(tokenizer, args, logger, file_type='dev' if evaluate else 'train', 
                                block_size=args.block_size)

    return dataset         

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

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

def main():
    parser = argparse.ArgumentParser()
    pass
    args = parser.parse_args()

    # Set seed
    set_seed(args) 

    # get special tokens
    special_tokens = get_special_tokens(args.lit_file)
    # Load pre-trained model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    pretrained = args.pretrain_dir
    if pretrained:
        tokenizer = tokenizer_class.from_pretrained(pretrained, do_lower_case=args.do_lower_case,  additional_special_tokens=special_tokens)
        if args.model_type == "rnn":
            model = model_class(len(tokenizer), 768, 768, 1)
            model_last = os.path.join(pretrained, 'model.pt')
            if os.path.exists(model_last):
                logger.warning(f"Loading model from {model_last}")
                model.load_state_dict(torch.load(model_last, map_location="cpu")) 
        else:
            model = model_class.from_pretrained(pretrained)
            model.resize_token_embeddings(len(tokenizer))
    else:
        raise

    model_parameters = model.parameters()
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info(f"Model has a total of {num_params} trainable parameters")


    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Only works on single GPU
    if args.do_eval:
        # dev_total, dev_cr = eval_acc(args, model, tokenizer, 'dev')
        # logger.info(f"Dev total tokens: {dev_total}, accuracy: {dev_cr/dev_total}")
        test_total, test_cr = eval_acc(args, model, tokenizer, 'test')
        logger.info(f"Test total tokens: {test_total}, accuracy: {test_cr/test_total}")
