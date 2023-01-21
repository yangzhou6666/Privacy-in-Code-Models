"""
Generate samples with chatGPT and filter out those that are likely to be
memorized samples from the training set.
"""

import logging
import os
logging.basicConfig(level='ERROR')
import pandas as pd
import argparse
import numpy as np
from pprint import pprint
import sys
import torch
import zlib
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForCausalLM
from tqdm import tqdm

GPU = __file__[-4] # last character of the filename is the GPU number
print(f"using GPU: {GPU}")
os.environ['CUDA_VISIBLE_DEVICES'] = GPU
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

def sort_best(metric, samples, name1, scores1, name2=None, scores2=None, n=100):
    """
    store the `n` best samples according to the given `metric`
    """
    idxs = np.argsort(metric)[::-1][:n]
    for i, idx in enumerate(idxs):
        if scores2 is not None:
            print(f"{i+1}: {name1}={scores1[idx]:.3f}, {name2}={scores2[idx]:.3f}, score={metric[idx]:.3f}")
        else:
            print(f"{i+1}: {name1}={scores1[idx]:.3f}, , score={metric[idx]:.3f}")
        for line in samples[idx].split("\n"):
            print(f"\t {line.rstrip()}")


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


def main():
    print(f"using device: {device}")

    if args.internet_sampling:
        print("Loading common crawl...")
        cc = parse_commoncrawl(args.wet_file)
    model1_name = "facebook/incoder-1B"
    model2_name = "microsoft/CodeGPT-small-java"
    model_to_record = model1_name.split("/")[1]
    output_file = f"samples_{model_to_record}_{GPU}.csv"

    # number of tokens to generate
    seq_len = 256

    # sample from the top_k tokens output by the model
    top_k = 40

    print(f"Loading {model1_name} and {model2_name}...")
    tokenizer1 = AutoTokenizer.from_pretrained(model1_name)
    tokenizer1.padding_side = "left" 
    tokenizer1.pad_token = tokenizer1.eos_token
    print(tokenizer1.bos_token)
    tokenizer2 = AutoTokenizer.from_pretrained(model2_name)
    tokenizer2.padding_side = "left"
    tokenizer2.pad_token = tokenizer2.eos_token

    #model1 = AutoModelForCausalLM.from_pretrained(model1_name,revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
    model1 = AutoModelForCausalLM.from_pretrained(model1_name).to(device)
    #model1.config.pad_token_id = model1.config.eos_token_id
    model2 = AutoModelForCausalLM.from_pretrained(model2_name).to(device)
    model2.config.pad_token_id = model2.config.eos_token_id
    
    model1.eval()
    model2.eval()
    
    num_batches = int(np.ceil(args.N / args.batch_size))
    start_temperature = 0.1
    end_temperature = 1
    decay_token_num = 20
    temperatures = np. linspace(start_temperature, end_temperature, decay_token_num). tolist()
    # use 1 to fill the rest of the sequence
    temperatures += [1.0] * (1 + seq_len - decay_token_num)
    with tqdm(total=args.N) as pbar:
        for i in range(num_batches):
            # encode the prompts
            if args.internet_sampling:
                # pick a random 10-token prompt in common crawl 

                input_len = 10
                input_ids = []
                attention_mask = []

                while len(input_ids) < args.batch_size:
                    # take some random words in common crawl
                    r = np.random.randint(0, len(cc))
                    prompt = " ".join(cc[r:r+100].split(" ")[1:-1])

                    # make sure we get the same number of tokens for each prompt to enable batching
                    inputs = tokenizer1(prompt, return_tensors="pt", max_length=input_len, truncation=True)
                    if len(inputs['input_ids'][0]) == input_len:
                        input_ids.append(inputs['input_ids'][0])
                        attention_mask.append(inputs['attention_mask'][0])

                inputs = {'input_ids': torch.stack(input_ids), 
                          'attention_mask': torch.stack(attention_mask)}

                # the actual truncated prompts
                prompts = tokenizer1.batch_decode(inputs['input_ids'], skip_special_tokens=True)
            else:
                prompts = ["<|endoftext|>"] * args.batch_size
                input_len = 1
                inputs = tokenizer1(prompts, return_tensors="pt")

            # batch generation
            output_sequences = []
            for i, temp in enumerate(temperatures):
                output_sequence = model1.generate(
                    input_ids=inputs['input_ids'].to(device),
                    attention_mask=inputs['attention_mask'].to(device),
                    #max_new_tokens=1,
                    max_length= 1 + i,
                    do_sample=True,
                    temperature=temp,
                    top_k=top_k,
                    top_p=1.0
                )
                #inputs['input_ids'] = torch.cat((inputs['input_ids'].to(device), output_sequence), dim=1)
                inputs['input_ids'] = output_sequence
                inputs['attention_mask'] = torch.ones_like(output_sequence)
            output_sequences.append(output_sequence)
            texts = []
            for output_sequence in output_sequences:
                texts.extend(tokenizer1.batch_decode(output_sequence, skip_special_tokens=True))
            scores = {"TEXT": [], "XL": [], "S": [], "Lower": [], "zlib": []}
            for text in texts:
                # perplexity of GPT2-XL and GPT2-S
                p1 = calculatePerplexity(text, model1, tokenizer1)
                p2 = calculatePerplexity(text, model2, tokenizer2)

                # perplexity on lower-case sample
                p_lower = calculatePerplexity(text.lower(), model1, tokenizer1)

                # Zlib "entropy" of sample
                zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))

                scores["TEXT"].append(text)
                scores["XL"].append(p1)
                scores["S"].append(p2)
                scores["Lower"].append(p_lower)
                scores["zlib"].append(zlib_entropy)


            # Store
            for metrics in scores:
                for i in range(len(scores[metrics])):
                    if type(scores[metrics][i]) == torch.Tensor:
                        scores[metrics][i] = scores[metrics][i].cpu()

            # Record the samples into output_file as dataframe
            pbar.update(args.batch_size)
            df = pd.DataFrame({'TEXT': scores["TEXT"], 'XL': np.asarray(scores["XL"]), 'S': np.asarray(scores["S"]), 'Lower': np.asarray(scores["Lower"]), 'zlib': np.asarray(scores["zlib"])})
            df.to_csv(output_file, index=False, mode='a', header=False)
    return

    # Sort by perplexity
    metric = -np.log(scores["XL"])
    print(f"======== top sample by XL perplexity: ========")
    print_best(metric, samples, "PPL", scores["XL"])
    print()
    print()

    # Sort by ratio of log perplexities of S and XL models
    metric = np.log(scores["S"]) / np.log(scores["XL"])
    print(f"======== top sample by ratio of S and XL perplexities: ========")
    print_best(metric, samples, "PPL-XL", scores["XL"], "PPL-S", scores["S"])
    print()
    print()

    # Sort by ratio of log perplexities of lower-case and normal-case perplexities 
    metric = np.log(scores["Lower"]) / np.log(scores["XL"])
    print(f"======== top sample by ratio of lower-case and normal-case perplexities: ========")
    print_best(metric, samples, "PPL-XL", scores["XL"], "PPL-XL-Lower", scores["Lower"])
    print()
    print()

    # Sort by ratio of Zlib entropy and XL perplexity
    metric = scores["zlib"] / np.log(scores["XL"])
    print(f"======== top sample by ratio of Zlib entropy and XL perplexity: ========")
    print_best(metric, samples, "PPL-XL", scores["XL"], "Zlib", scores["zlib"])

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=20000, help="Number of samples to generate")
    parser.add_argument('--batch-size', type=int, default=1, help="Batch size for generation")
    parser.add_argument('--internet-sampling', action='store_true', help="condition the generation using commoncrawl")
    parser.add_argument('--wet-file', type=str, default=None, help="path to a commoncrawl WET file")
    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    while True:
        try:
            main()
            break
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            break
        except Exception as e:
            pass
