import pandas as pd
import random
from tqdm import tqdm
import subprocess
import logging
import hashlib
import re
import time
import os
import ast
import re
import hashlib
import json
def paser_func(content):
    tree = ast.parse(content)
    lines = content.split('\n')
    for item in ast.walk(tree):
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)): 
            func_begin = item.lineno 
            first_stmt = item.body[0].lineno
            if lines[first_stmt-1].strip().startswith('"""'):
                curline = first_stmt
            else:
                curline = first_stmt-1 if first_stmt > 1 else 1 
            # # if function_body-1 is a comment line or blank, skip it and pull up. 
            # while curline >= 0 and (lines[curline-1].strip().startswith("#") or lines[curline-1].strip() == ""): 
            #     curline -= 1
            return '\n'.join(lines[:curline])

def remove_all_the_comments(content):
    content = re.sub(r"#.*", "", content, flags=re.MULTILINE)
    return content

def process_save_prompts():
    save_prompts_dict = {}
    # 读取数据
    df = pd.read_csv("prompt_data-train.csv") #selected_prompt.csv / prompt_data-train.csv
    contents = df['content'].tolist()
    # 记录失败：1.paser失败的 2.没有函数的，即没有def的
    failed_contents = []
    for content in tqdm(contents):
        content = re.sub('\n+', '\n', content) #去除多个换行符的情况
        try:
            content = paser_func(content)
        except:
            failed_contents.append(content)
            # print(content)
            continue
        if content is None: # 有的不存在函数def
            failed_contents.append(content)
            continue
        hash_value = hashlib.sha1(content.encode('utf-8')).hexdigest()
        save_prompts_dict[hash_value] = content
    with open("prompts_paser_func.json.json","w") as f: #prompts_paser_func_selected.json / prompts_paser_func.json
        json.dump(save_prompts_dict,f,indent=4)
    print(len(failed_contents))
        


def main(sample_size=1000,max_length=100,model_type=None,gpu_id="0",sample_number=30000,sample_mode="total"):
    if model_type is None:
        raise
     # save the log to file, get the time as the log name
    log_file_name = "./log/{}-{}-{}-{}-func_prompt.log".format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()),model_type.split('/')[-1],sample_size,max_length)
    logging.basicConfig(filename=log_file_name,
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    logger = logging.getLogger()
        
    
    # log the basic info
    logger.info("[BASE INFO]")
    logger.info("sample_size: {}".format(sample_size))
    logger.info("max_length: {}".format(max_length))
    logger.info("model_type: {}".format(model_type))
    logger.info("total_sample_number: {}".format(sample_number*sample_size if sample_mode == "per_prompt" else sample_number))
    logger.info("sample_mode: {}".format(sample_mode))
    logger.info("Persampele: {}".format(sample_number))
    logger.info("\n\n")

    assert sample_mode in ["total","per_prompt"]
    assert (sample_size == -1 and sample_mode =='total') or sample_size > 0

    # read the prompts
    with open("prompts_paser_func.json","r") as f: #prompts_paser_func_selected.json / prompts_paser_func.json
        prompts_dict = json.load(f)
    contents = list(prompts_dict.values())
    if sample_size != -1:
        random.shuffle(contents)
        sample_contents = contents
    else:
        sample_contents = contents
        sample_size = len(sample_contents)
    
    already_sampled_size = 0
    pbar = tqdm(total=sample_size)
    while True:
        if already_sampled_size >= len(sample_contents):
            logger.info("Already sampled all the prompts")
            break
        content = sample_contents[already_sampled_size]
        if max_length != -1:
            content = content[:max_length]
        content = re.sub('\n+', '\n', content) #去除多个换行符的情况
        hash_value = hashlib.sha1(content.encode('utf-8')).hexdigest()

        try:
            subprocess.call([
                        "python", "../extract/extract.py", 
                        "--model", model_type, 
                        "--N", str(sample_number),
                        "--batch-size","16",
                        "--seq_len","512", 
                        "--top_k", "40", 
                        "--temperature","1.0", 
                        "--gpu_id", gpu_id,
                        "--prompt_mode","direct_prompt",
                        "--prompt", content,
                        "--internet-sampling"
                        ]               
                        ,shell=False)
            already_sampled_size += 1
            logger.info("content:\n{}".format(content))        
            logger.info("hash: {}".format(hash_value))
            logger.info("\n\n")
            pbar.update(1)
        except Exception as e:
            logger.info("[Error]")
            logger.error(e)
            logger.info("\n\n")
            continue
        if already_sampled_size >= sample_size:
            break
        

if __name__ == "__main__":
    os.makedirs('log',exist_ok=True)
    process_save_prompts()

    # 如果规定了prompt，max_length最好为-1
    # main(sample_size=100,model_type="codeparrot/codeparrot-small",max_length=-1,gpu_id="1")
    # main(sample_size=-1,model_type="codeparrot/codeparrot",max_length=-1,gpu_id="1") # for selected_prompt.csv
    # main(sample_size=2000,model_type="codeparrot/codeparrot",max_length=-1,gpu_id="1",sample_mode = "per_prompt",sample_number=100) # for selected_prompt.csv

        
