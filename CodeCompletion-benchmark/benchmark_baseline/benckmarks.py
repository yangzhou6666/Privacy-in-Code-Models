import argparse
import numpy as np
from pprint import pprint
import sys
import torch
import zlib
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import json
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device("cuda")

# Change the following to your own path
# Path to the dataset root folder
data_dir = "/mnt/hdd1/XXXX/XXXX/Privacy-in-Code-Models/Classifier/dataset/java"
victim_model_dir = "/mnt/hdd1/XXXX/XXXX/Privacy-in-Code-Models/CodeCompletion-token/save/javaCorpus/microsoft/CodeGPT-small-java-adaptedGPT2/100/checkpoint-last"
surrogate_model_dir = "/mnt/hdd1/XXXX/XXXX/Privacy-in-Code-Models/CodeCompletion-token/save/javaCorpus/microsoft/CodeGPT-small-java/20/checkpoint-last"
# Path to the root folder where you want to save the results, saving folder format will be the same as the input folder format
save_dir = "/mnt/hdd1/XXXX/LM_Memorization"

def calculate_perplexity(text, model, tokenizer):
    """
    exp(loss)
    """
    max_position_embeddings = model.config.max_position_embeddings
    inputs = tokenizer.encode(text)
    if len(text) > max_position_embeddings:
        inputs = inputs[-max_position_embeddings:]
    input_ids = torch.tensor(inputs).unsqueeze(0)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return torch.exp(loss)

def calculate_zlib(text):
    """
    zlib compression ratio
    """
    return len(zlib.compress(bytes(text, 'utf-8'))) / len(text)

def main():
    print(f"using device: {device}")
    
    # Load the json file
    file_path = data_dir + "/" + args.file_path
    # Read the file as a json object
    with open(file_path, 'r') as f:
        lines = f.readlines()
    json_objects = [json.loads(line) for line in lines]
    N = len(json_objects)
    print(f"first ID of this batch: {json_objects[0]['id']}")

    print("Loading CodeGPT...")
    tokenizer = GPT2Tokenizer.from_pretrained(victim_model_dir)
    #tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(victim_model_dir).to(device)
    model.eval()
    
    print("Loading CodeGPT surrogate...")
    tokenizer_surrogate = GPT2Tokenizer.from_pretrained(surrogate_model_dir)
    #tokenizer_surrogate.pad_token = tokenizer_surrogate.eos_token
    model_surrogate = GPT2LMHeadModel.from_pretrained(surrogate_model_dir).to(device)
    model_surrogate.eval()
    
    results = []

    num_batches = int(np.ceil(N / args.batch_size))
    print(f"Generating {N} samples in {num_batches} batches of size {args.batch_size}...")
    
    with tqdm(total=N) as pbar:
        for i in range(num_batches):
            # encode the prompts
            full_texts = [json_object["input"] for json_object in json_objects[i*args.batch_size:(i+1)*args.batch_size]]
            # full_texts = [(json_object["input"] + json_object["prediction"]) for json_object in json_objects[i*args.batch_size:(i+1)*args.batch_size]]
            ids = [json_object["id"] for json_object in json_objects[i*args.batch_size:(i+1)*args.batch_size]]
            labels = [json_object["label"] for json_object in json_objects[i*args.batch_size:(i+1)*args.batch_size]]
            #print(f"full_text example: {full_texts[0]}")
            #print(f'ids example: {ids[0]}')
            #print(f"input text length example: {len(full_texts[0])}")
            
            
            for id, text, label in zip(ids, full_texts, labels):
                # perplexity
                result = {}
                #print(f"TEXT {text}")
                perplexity = float(calculate_perplexity(text, model, tokenizer).cpu().numpy())
                zlib_entropy = calculate_zlib(text)
                #print(f"id: {id}, perplexity: {perplexity}, zlib: {zlib_entropy}, label: {label}")
                
                perplexity_compare = perplexity / float(calculate_perplexity(text, model_surrogate, tokenizer_surrogate).cpu().numpy())
                zlib_compare = np.log(perplexity) / zlib_entropy
                
                result["id"] = id
                #result["text"] = ""
                result["perplexity"] = perplexity
                result["zlib"] = zlib_entropy
                result["label"] = label
                
                # New features
                result["perplexity_compare"] = perplexity_compare
                result["zlib_compare"] = zlib_compare
                
                
                
                results.append(result)
            pbar.update(args.batch_size)
            
            # Save the result (per epoch)
            save_location = save_dir + "/" + args.file_path
            save_path = "/".join(save_location.split("/")[:-1]) + "/"
            print(f"save_location {save_location}")
            print(f"save_path {save_path}")
            
            # Check if file exists at save_path, if not create an empty file
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            if not os.path.exists(save_location):
                open(save_location, 'w').close()
            
            with open(save_location, "a") as f:
                # iterate over the list and write each object as a separate line
                for item in results:
                    # serialize the object to JSON string
                    json_string = json.dumps(item)
                    # write the JSON string to the file followed by a newline character
                    f.write(json_string + "\n")

            results = []

    
    

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, help="The path to the file to generate from")
    parser.add_argument('--batch_size', type=int, default=100, help="Batch size for generation")
    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main()

