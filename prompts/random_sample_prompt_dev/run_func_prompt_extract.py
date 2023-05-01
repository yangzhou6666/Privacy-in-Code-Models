"""
Generate samples with GPT-2 and filter out those that are likely to be
memorized samples from the training set.
"""

import logging
import hashlib
import time
import argparse
import numpy as np
import os
from pprint import pprint
import sys
import torch
import zlib
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from torch.nn.functional import softmax
import json
import random


def get_model_and_tokenizer(model_name):
    print("Loading model {} ...".format(model_name))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token
    if 'santacoder' in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    # it seems that santacode cannot be load in half precision

    print("Model {} is loaded.".format(model_name))
    return tokenizer, model

def save_samples(path_to_save: str, text:str, file_id):
    with open(os.path.join(path_to_save, str(file_id)), 'w') as f:
        f.write(text)
    

def main():
    sample_size = args.sample_number
    log_file_name = "./log/{}-{}-{}-{}-func_prompt.log".format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()),args.model.split('/')[-1],args.sample_mode,args.sample_number)
    logging.basicConfig(filename=log_file_name,
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger()
    # log the basic info
    logger.info("[BASE INFO]")
    logger.info("model_type: {}".format(args.model))
    logger.info("total_sample_number: {}".format(args.sample_number * args.sample_prompt_size if args.sample_mode == "total" else args.sample_number ))
    logger.info("sample_mode: {}".format(args.sample_mode))
    logger.info("Persampele: {}".format(args.sample_number // args.sample_prompt_size if args.sample_mode == "per_prompt" else args.sample_number))
    logger.info('prompt_size: {}'.format(args.sample_prompt_size))
    logger.info("\n\n")

    assert args.sample_mode in ["total","per_prompt"]


    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer, model = get_model_and_tokenizer(args.model)
    model.to(args.device)
    model.eval()


    with open("../prompts_paser_func.json","r") as f: #prompts_paser_func_selected.json / prompts_paser_func.json
        prompts_dict = json.load(f)
    contents = list(prompts_dict.values())
    if args.sample_prompt_size != -1:
        random.shuffle(contents)
        sample_contents = contents
    else:
        sample_contents = contents
        args.sample_prompt_size = len(sample_contents)
    
    already_sampled_size = 0
    pbar = tqdm(total=args.sample_prompt_size)
    while True:
        if len(sample_contents) == 0:
            logger.info("Already sampled all the prompts")
            break
        content = sample_contents.pop()
        if len(content.split(" ")) > 220:
            continue
        try:
            hash_value = generate(tokenizer,model,content)

            already_sampled_size += 1

            logger.info("content:\n{}".format(content))        
            logger.info("hash: {}".format(hash_value))
            logger.info("\n\n")

            pbar.update(1)
        except Exception as e:
            logger.info("Error: {}".format(e))
            continue
        if already_sampled_size >= args.sample_prompt_size:
            break
    pbar.close()




def generate(tokenizer,model,prompt):
    
    hash_value = hashlib.sha1(prompt.encode('utf-8')).hexdigest()
    path_to_save = None
    # set the prompts
    prompts_txt = [prompt]
    prompts = prompts_txt * args.batch_size
    inputs = tokenizer(prompts, return_tensors="pt")
    # save the prompts
    
    input_len = len(inputs['input_ids'][0])
    # number of tokens to generate
    seq_len = args.seq_len
    # sample from the top_k tokens output by the model
    top_k = args.top_k

    samples = []
    existing_count = 0 #这里写成0是因为，如果是从头开始生成，那么就没有重复的
    
    N = args.sample_number if args.sample_mode =='total' else args.sample_number // args.sample_prompt_size
    num_batches = int(np.ceil(N / args.batch_size))

    for i in tqdm(range(num_batches)):
        if args.temperature > 1.0:
            # use temperature decaying strategy
            start_temperature = 10.0
            end_temperature = 1.0
            decay_tokens = 20

            output_sequences = inputs['input_ids'].to(args.device)
            with torch.no_grad():
                for step in range(seq_len):
                    outputs = model(output_sequences)
                    logits = outputs.logits[:, -1, :]

                    if step < decay_tokens:
                        decay_ratio = step / decay_tokens
                        current_temperature = start_temperature - (start_temperature - end_temperature) * decay_ratio
                    else:
                        current_temperature = end_temperature

                    logits /= current_temperature
                    probabilities = softmax(logits, dim=-1)
                    next_token = torch.multinomial(probabilities, num_samples=1)

                    output_sequences = torch.cat((output_sequences, next_token), dim=-1)
        else: 
            # batch generation
            output_sequences = model.generate(
                input_ids=inputs['input_ids'].to(args.device),
                attention_mask=inputs['attention_mask'].to(args.device),
                max_length=input_len + seq_len,
                do_sample=True, 
                top_k=top_k, 
                top_p=1.0
            )

        texts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        
        # undo: check if we write at the end rather than every batch, wether we can save some time
        for text in texts:
            text = text[len(prompts_txt[0]):]
            existing_count += 1
            if path_to_save is None:
                path_to_save = 'results/{}-temp{}-len{}-k{}/internet/{}'.format(args.model, args.temperature, args.seq_len, args.top_k,hash_value)
                os.makedirs(path_to_save)
            save_samples(path_to_save, text, existing_count)
            # store the results
        # raise
    with open(os.path.join(path_to_save, 'prompts.txt'), 'w') as f:
        f.write(prompts_txt[0])
    return hash_value




def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="The model to load")
    # parser.add_argument('--N', type=int, default=20000, help="Number of samples to generate")
    parser.add_argument('--batch-size', type=int, default=20, help="Batch size for generation")
    # 1B Model, batch size 80, half-precision, consumes 12 GB of GPU memory
    # 2.7B Model, batch size 20, half-precision, consumes 18 GB of GPU memory
    # 6B Model, batch size 10, half-precision, consumes 20 GB of GPU memory
    parser.add_argument('--temperature', type=float, default=1.0, help="Start temperature")
    parser.add_argument('--seq_len', type=int, default=256, help="The length of extracted sequence")
    parser.add_argument('--top_k', type=int, default=40, help="sample from the top_k tokens output by the model")
    parser.add_argument('--gpu_id', type=str, default="1", help="specify the GPU id")
    # parser.add_argument('--internet-sampling', action='store_true', help="condition the generation on the internet")

    # parser.add_argument('--prompt_mode', type=str, default="single_md5",choices=["single_md5","direct_prompt"], help="The mode of the prompt to use for generation")
    # parser.add_argument('--prompt', type=str, default="", help="The prompt to use for generation(can also be the path to a file containing the prompt)")
    # parser.add_argument('--prompt_hash', type=str, default="", help="The hash of the prompt to use for generation")
    parser.add_argument('--sample_prompt_size', type=int, default=-1, help="The number of prompts to sample from the prompt file")
    parser.add_argument('--sample_number', type=int, default=-1, help="The number of samples to generate for  prompt")
    parser.add_argument('--sample_mode', type=str, default="total",choices=["total","per_prompt"], help="The mode of the prompt to use for generation")
    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main()