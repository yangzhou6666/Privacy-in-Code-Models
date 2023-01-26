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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def calculatePerplexity(sentence, model, tokenizer):
    """
    exp(loss)
    """
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return torch.exp(loss)

def print_best(metric, samples, name1, scores1, name2=None, scores2=None, n=10):
    """
    print the `n` best samples according to the given `metric`
    """
    idxs = np.argsort(metric)[::-1][:n]

    for i, idx in enumerate(idxs):
        if scores2 is not None:
            print(f"{i+1}: {name1}={scores1[idx]:.3f}, {name2}={scores2[idx]:.3f}, score={metric[idx]:.3f}")
        else:
            print(f"{i+1}: {name1}={scores1[idx]:.3f}, , score={metric[idx]:.3f}")

        print()
        #for line in samples[idx].split("\n"):
        #    print(f"\t {line.rstrip()}")
        pprint(samples[idx])
        print()
        print()
        

def parse_commoncrawl(wet_file):
    """
    Quick and ugly parsing of a WET file.
    Tested for the May 2021 crawl.
    """
    with open(wet_file) as f:
        lines = f.readlines() 
    
    start_idxs = [i for i in range(len(lines)) if "WARC/1.0" in lines[i]]
    
    all_eng = ""

    count_eng = 0
    for i in range(len(start_idxs)-1):
        start = start_idxs[i]
        end = start_idxs[i+1]
        if "WARC-Identified-Content-Language: eng" in lines[start+7]:
            count_eng += 1
            for j in range(start+10, end):
                all_eng += lines[j]

    return all_eng


def get_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    print("Model {} is loaded.".format(model_name))
    return tokenizer, model

def save_samples(path_to_save: str, text:str):
    start_id = len(os.listdir(path_to_save))
    with open(os.path.join(path_to_save, str(start_id+1)), 'w') as f:
        f.write(text)
    

def main():
    model_name = "facebook/incoder-6B"
    
    path_to_save = 'results/{}'.format(model_name)
    os.makedirs(path_to_save, exist_ok=True)
    
    
    print(f"using device: {device}")

    # number of tokens to generate
    seq_len = 256

    # sample from the top_k tokens output by the model
    top_k = 40

    
    tokenizer, model = get_model_and_tokenizer(model_name)
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

        if temperature_decaying:
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
    parser.add_argument('--N', type=int, default=20000, help="Number of samples to generate")
    parser.add_argument('--batch-size', type=int, default=20, help="Batch size for generation")
    parser.add_argument('--temperature', type=float, default=1.0, help="Start temperature")

    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main()