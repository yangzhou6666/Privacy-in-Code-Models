"""
Generate samples with GPT-2 and filter out those that are likely to be
memorized samples from the training set.
"""

import logging
logging.basicConfig(level='ERROR')

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
    model_name = args.model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer, model = get_model_and_tokenizer(model_name)
    model.to(device)
    model.eval()

    if not args.internet_sampling:
        # set the path to save the generated samples
        path_to_save = 'results/{}-temp{}-len{}-k{}/seperate'.format(model_name, args.temperature, args.seq_len, args.top_k)
        os.makedirs(path_to_save, exist_ok=True)
        # set the prompts
        prompts = [tokenizer.bos_token] * args.batch_size
        inputs = tokenizer(prompts, return_tensors="pt")
    else:
        # set the path to save the generated samples        
        if args.prompt_mode == 'single_md5':
            hash_value = args.prompt_hash
        elif args.prompt_mode == 'direct_prompt':
            hash_value = hashlib.sha1(args.prompt.encode('utf-8')).hexdigest()
        path_to_save = 'results/{}-temp{}-len{}-k{}/internet'.format(model_name, args.temperature, args.seq_len, args.top_k,hash_value)
        os.makedirs(path_to_save, exist_ok=True)
        # set the prompts
        prompts_txt = get_prompts(args)
        prompts = prompts_txt * args.batch_size
        inputs = tokenizer(prompts, return_tensors="pt")
        # save the prompts
        with open(os.path.join(path_to_save, 'prompts.txt'), 'w') as f:
            f.write(prompts_txt[0])
    print("The generated samples will be saved to {}...".format(path_to_save))
    input_len = len(inputs['input_ids'][0])
    # number of tokens to generate
    seq_len = args.seq_len
    # sample from the top_k tokens output by the model
    top_k = args.top_k

    
    
    
    samples = []
    existing_count = len(os.listdir(path_to_save))

    num_batches = int(np.ceil(args.N / args.batch_size))

    for i in tqdm(range(num_batches)):
        if args.temperature > 1.0:
            # use temperature decaying strategy
            start_temperature = 10.0
            end_temperature = 1.0
            decay_tokens = 20

            output_sequences = inputs['input_ids'].to(device)
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
                input_ids=inputs['input_ids'].to(device),
                attention_mask=inputs['attention_mask'].to(device),
                max_length=input_len + seq_len,
                do_sample=True, 
                top_k=top_k, 
                top_p=1.0
            )

        texts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        
        for text in texts:
            if args.internet_sampling:
                text = text[len(prompts_txt[0]):]
            existing_count += 1
            save_samples(path_to_save, text, existing_count)
            # store the results
        raise

def get_prompts(args):
    if args.prompt_mode == "single_md5":
        with open(args.prompt, 'r') as f:
            json_data = json.load(f)
            try:
                prompts = [json_data[args.prompt_hash]['prompt']]
            except:
                raise ValueError("The prompt file is not in the correct format or the hash is not correct")
            return prompts
    elif args.prompt_mode == 'direct_prompt':
        return [args.prompt]


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="The model to load")
    parser.add_argument('--N', type=int, default=20000, help="Number of samples to generate")
    parser.add_argument('--batch-size', type=int, default=20, help="Batch size for generation")
    # 1B Model, batch size 80, half-precision, consumes 12 GB of GPU memory
    # 2.7B Model, batch size 20, half-precision, consumes 18 GB of GPU memory
    # 6B Model, batch size 10, half-precision, consumes 20 GB of GPU memory
    parser.add_argument('--temperature', type=float, default=1.0, help="Start temperature")
    parser.add_argument('--seq_len', type=int, default=256, help="The length of extracted sequence")
    parser.add_argument('--top_k', type=int, default=40, help="sample from the top_k tokens output by the model")
    parser.add_argument('--gpu_id', type=str, default="1", help="specify the GPU id")
    parser.add_argument('--internet-sampling', action='store_true', help="condition the generation on the internet")

    parser.add_argument('--prompt_mode', type=str, default="single_md5",choices=["single_md5","direct_prompt"], help="The mode of the prompt to use for generation")
    parser.add_argument('--prompt', type=str, default="", help="The prompt to use for generation(can also be the path to a file containing the prompt)")
    parser.add_argument('--prompt_hash', type=str, default="", help="The hash of the prompt to use for generation")
    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main()