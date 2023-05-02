
import logging
import hashlib
import time
import argparse
import numpy as np
import os
import sys
import torch
import zlib
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from torch.nn.functional import softmax
import json
import re
from torch.utils.data import DataLoader, TensorDataset
def print_best(memorized_content_index,metric, samples, name1, scores1, name2=None, scores2=None, n=10):
    """
    logger.info the `n` best samples according to the given `metric`
    """
    idxs = np.argsort(metric)[::-1][:n]
    memorized_content_num = 0
    top_n_output = []

    for i, idx in enumerate(idxs):
        if idx in memorized_content_index:
            logger.info(f'[!!!]: memorized content {idx+1} [!!!]') # list index starts from 0
            memorized_content_num += 1
        if scores2 is not None:
            logger.info(f"{i+1}: {name1}={scores1[idx]:.3f}, {name2}={scores2[idx]:.3f}, score={metric[idx]:.3f}")
        else:
            logger.info(f"{i+1}: {name1}={scores1[idx]:.3f}, , score={metric[idx]:.3f}")
        logger.info('\n')
        if i > args.output_n:
            continue
        logger.info(samples[idx])
        top_n_output.append((idx+1,samples[idx]))
        logger.info('\n')
        logger.info('\n')
    logger.info(f"[memorized content num]: {memorized_content_num}")
    logger.info(f"[memorized content ratio]: {memorized_content_num/n}")
    return top_n_output

def save_the_best(save_path,res,name):
    file_path = os.path.join(save_path,f'{name}.csv')
    idx = [i[0] for i in res]
    content = [i[1] for i in res]
    df = pd.DataFrame({'idx':idx,'content':content})
    df.to_csv(file_path,index=False)


def get_model_and_tokenizer(model_name):
    logger.info("Loading model {} ...".format(model_name))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token
    if 'santacoder' in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    # it seems that santacode cannot be load in half precision

    logger.info("Model {} is loaded.".format(model_name))
    return tokenizer, model

def get_memorized_flag(root_path):
    map_file_path = os.path.join(root_path, 'map.json')
    if not os.path.exists(map_file_path):
        raise
    with open(map_file_path, 'r') as f:
        map_data = json.load(f)
    labels = [int(k) for k in map_data.keys()]
    end_values = [v['end'] for v in map_data.values()]
    start_values = [v['start'] for v in map_data.values()]
    flag_values = []
    for i,label in enumerate(labels):
        flag_values.extend([label] * (end_values[i] - start_values[i] + 1))
    begin_list = []
    begin_end_list_file = os.path.join(root_path.replace('extract/results','log/save'),'analyze','begin_end_list.txt')
    with open(begin_end_list_file, 'r') as f:
        for line in f:
            begin,end = line.strip().split(' ')
            begin_list.append(int(begin))
    memorized_content_index = set()
    for b in tqdm(begin_list,total=len(begin_list)):
        b -= 1 # list index starts from 0
        flag = flag_values[b]
        memorized_content_index.add(flag-1) # list index starts from 0
    return list(memorized_content_index)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_1", type=str, required=True, help="The model to load")
    parser.add_argument("--model_2", type=str, required=True, help="The model to load")
    parser.add_argument('--N', type=int, default=20000, help="Number of samples to generate")

    parser.add_argument('--extract_n', type=int, default=100, help="Number of ranked samples to extract")
    parser.add_argument('--output_n', type=int, default=100, help="Number of ranked samples to output")
    parser.add_argument('--save_output', action='store_true', help="save the output to a file")
    parser.add_argument('--extract_mode', type=str, default="small-first", choices=["small-first","large-first"], help="The mode of the extraction")


    parser.add_argument('--temperature', type=float, default=1.0, help="Start temperature")
    parser.add_argument('--seq_len', type=int, default=256, help="The length of extracted sequence")
    parser.add_argument('--top_k', type=int, default=40, help="sample from the top_k tokens output by the model")
    parser.add_argument('--gpu_id', type=str, default="0", help="The gpu id to use, -1 for cpu")

    parser.add_argument('--internet-sampling', action='store_true', help="condition the generation on the internet")
    parser.add_argument('--prompt_mode', type=str, default="single_md5",choices=["single_md5","direct_prompt"], help="The mode of the prompt to use for generation")
    parser.add_argument('--prompt', type=str, default="", help="The prompt to use for generation(can also be the path to a file containing the prompt)")
    parser.add_argument('--prompt_hash', type=str, default="", help="The hash of the prompt to use for generation")
    return parser.parse_args(argv)

# def calculatePerplexity(sentence, model, tokenizer):
#     """
#     exp(loss)
#     """
#     input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
#     input_ids = input_ids.to(device)
#     with torch.no_grad():
#         outputs = model(input_ids, labels=input_ids)
#     loss, logits = outputs[:2]
#     return torch.exp(loss).cpu()


def calculatePerplexity_batch(sentences, model_1,model_2, tokenizer, batch_size=32):
    # get the encoded sentences with padding and attention mask

    encoded_sentences = tokenizer.batch_encode_plus(sentences, padding=True, return_tensors="pt")

    input_ids_tensor = encoded_sentences['input_ids']
    attention_mask_tensor = encoded_sentences['attention_mask']
    input_ids_tensor = input_ids_tensor.to(device)
    
    # create the dataset and dataloader
    dataset = TensorDataset(input_ids_tensor, attention_mask_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    total_loss_1 = 0
    total_loss_2 = 0
    for batch in dataloader:
        batch_input_ids = batch[0].to(device)
        batch_attention_mask = batch[1].to(device)
        with torch.no_grad():
            outputs_1 = model_1(batch_input_ids, labels=batch_input_ids, attention_mask=batch_attention_mask)
            outputs_2 = model_2(batch_input_ids, labels=batch_input_ids, attention_mask=batch_attention_mask)
        loss_1, logits_1 = outputs_1[:2]
        total_loss_1 += loss_1.item()
        loss_2, logits_2 = outputs_2[:2]
        total_loss_2 += loss_2.item()
    
    perplexity_1 = torch.exp(torch.tensor(total_loss_1 / len(sentences)))
    perplexity_2 = torch.exp(torch.tensor(total_loss_2 / len(sentences)))
    return perplexity_1.cpu(), perplexity_2.cpu()



def main():
    root_path = '../extract/results/{}-temp{}-len{}-k{}'.format(args.model_1, args.temperature, args.seq_len, args.top_k)
    if not args.internet_sampling:
        folder_path = os.path.join(root_path,'seperate')
    else:
        pass
    
    memorized_content_index = get_memorized_flag(root_path)

    logger.info("[Basic Info]")
    logger.info("Model 1: {}".format(args.model_1))
    logger.info("Model 2: {}".format(args.model_2))
    logger.info("Number of samples: {}".format(args.N))
    logger.info("Temperature: {}".format(args.temperature))
    logger.info("Sequence length: {}".format(args.seq_len))
    logger.info("Top k: {}".format(args.top_k))
    logger.info("Folder path: {}".format(folder_path))
    logger.info("Extract n: {}".format(args.extract_n))
    logger.info("Memorized content ratio: {}".format(len(memorized_content_index)/args.N))
    logger.info("Extract mode: {}".format(args.extract_mode))
    logger.info("\n\n")
    


    # 我们这里先默认model1，model2都是使用的tokenizer一样
    tokenizer,model1  = get_model_and_tokenizer(args.model_1)
    _, model2 = get_model_and_tokenizer(args.model_2)
    model1.to(device)
    model2.to(device)
    model1.eval()
    model2.eval()
    
    samples = []
    scores = {"model_1": [], "model_2": [],  "zlib": []}



    # num_batches = int(np.ceil(args.N / args.batch_size))
    with tqdm(total=args.N) as pbar:
        for i in range(1,args.N+1):
            file_path = os.path.join(folder_path, "{}".format(i))
            with open(file_path, 'r') as f:
                sample = f.readlines()
            sample = ''.join(sample)
            sample = re.sub('\n+', '\n', sample) #去除多个换行符的情况
            sample = sample.split('\n')
             # window size = 6
            range_size = len(sample)-6 if len(sample) > 6 else 1
            sample_windows = [sample[i:i+6] for i in range(range_size) ] # 6-gram,from 0 to len-6
            
            total_p1 = 0
            total_p2 = 0
            total_p_lower = 0
            total_zlib_entropy = 0
            # for s_w in sample_windows:
            #     text = '\n'.join(s_w)
            #     # perplexity of model1 and model2 on sample
               
            #     p1 = calculatePerplexity(text, model1, tokenizer)
                
            #     p2 = calculatePerplexity(text, model2, tokenizer)

            #     # perplexity on lower-case sample
            #     p_lower = calculatePerplexity(text.lower(), model1, tokenizer)

            #     # Zlib "entropy" of sample
            #     zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))
            #     total_p1 += p1
            #     total_p2 += p2
            #     total_p_lower += p_lower
            #     total_zlib_entropy += zlib_entropy

            sentence_batch = [ '\n'.join(s_w) for s_w in sample_windows]
            p1,p2 = calculatePerplexity_batch(sentence_batch, model1,model2, tokenizer)
            for text in sentence_batch:
                total_zlib_entropy += len(zlib.compress(bytes(text, 'utf-8')))


            samples.append('\n'.join(sample))
            # scores["model_1"].append(torch.mean(total_p1))
            # scores["model_2"].append(torch.mean(total_p2))
            # scores["Lower"].append(torch.mean(total_p_lower))
            # scores["zlib"].append(total_zlib_entropy / len(sample_windows))
            scores["model_1"].append(torch.mean(p1))
            scores["model_2"].append(torch.mean(p2))
            scores["zlib"].append(total_zlib_entropy / len(sample_windows))

            pbar.update(1)

    scores["model_1"] = np.asarray(scores["model_1"])
    scores["model_2"] = np.asarray(scores["model_2"])
    scores["zlib"] = np.asarray(scores["zlib"])

    # Sort by perplexity
    metric = -np.log(scores["model_1"])
    logger.info(f"======== top sample by {args.model_1} perplexity: ========")
    ppl_res = print_best(memorized_content_index,metric, samples, "PPL", scores["model_1"],n=args.extract_n)
    logger.info('\n')
    logger.info('\n')

    # Sort by ratio of log perplexities of S and XL models 
    # 要大除以小
    if args.extract_mode == 'small-first':
        metric = np.log(scores["model_1"]) / np.log(scores["model_2"])
    else:
        metric = np.log(scores["model_2"]) / np.log(scores["model_1"])
    logger.info(f"======== top sample by ratio of {args.model_1} and {args.model_2} perplexities: ========")
    comparing_ppl_res = print_best(memorized_content_index,metric, samples, f"PPL-{args.model_1}", scores["model_1"], f"PPL-{args.model_2}", scores["model_2"],n=args.extract_n)
    logger.info('\n')
    logger.info('\n')

    # Sort by ratio of Zlib entropy and XL perplexity
    metric = scores["zlib"] / np.log(scores["model_1"])
    logger.info(f"======== top sample by ratio of Zlib entropy and {args.model_1} perplexity: ========")
    zlib_res = print_best(memorized_content_index,metric, samples, f"PPL-{args.model_1}", scores["model_1"], "Zlib", scores["zlib"],n=args.extract_n)

    if args.save_output:
        save_res_path = 'results-window/{}-temp{}-len{}-k{}'.format(args.model_1, args.temperature, args.seq_len, args.top_k)
        os.makedirs(save_res_path, exist_ok=True)
        save_the_best(save_res_path, ppl_res, 'ppl')
        save_the_best(save_res_path, comparing_ppl_res, 'comparing_ppl')
        save_the_best(save_res_path, zlib_res, 'zlib')
        logger.info(f"======== save the results to {save_res_path} ========")

if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    # save the log to file, get the time as the log name
    if not os.path.exists("./log"):
        os.mkdir("./log") 
    log_file_name = "./log/evl-window-6-{}-{}-{}-temp{}-len{}-k{}.log".format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()), args.model_1.split('/')[-1], args.model_2.split('/')[-1], args.temperature, args.seq_len, args.top_k)
    logging.basicConfig(filename=log_file_name,
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    logger = logging.getLogger()
    # device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main()