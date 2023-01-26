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





def get_model_and_tokenizer(model_name):
    print("Loading model {} ...".format(model_name))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

    print("Model {} is loaded.".format(model_name))
    return tokenizer, model

def save_samples(path_to_save: str, text:str):
    start_id = len(os.listdir(path_to_save))
    with open(os.path.join(path_to_save, str(start_id+1)), 'w') as f:
        f.write(text)
    

def main():
    model_name = args.model
    
    path_to_save = 'results/{}-temp{}-len{}-k{}'.format(model_name, args.temperature, args.seq_len, args.top_k)
    os.makedirs(path_to_save, exist_ok=True)
    

    # number of tokens to generate
    seq_len = args.seq_len
    # sample from the top_k tokens output by the model
    top_k = args.top_k

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer, model = get_model_and_tokenizer(model_name)
    model.to(device)
    model.eval()
    
    
    samples = []

    num_batches = int(np.ceil(args.N / args.batch_size))

    for i in tqdm(range(num_batches)):
        # encode the prompts
        if args.internet_sampling:
            raise NotImplementedError
        else:
            prompts = ["<|endoftext|>"] * args.batch_size
            input_len = 1
            inputs = tokenizer(prompts, return_tensors="pt")

        if args.temperature < 1.0:
            raise NotImplementedError
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
            save_samples(path_to_save, text)
            # store the results


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

    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main()