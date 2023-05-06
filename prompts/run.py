import pandas as pd
import random
from tqdm import tqdm
import subprocess
import logging
import hashlib
import re
import time
import os
def main(sample_size=1000,max_length=100,model_type=None,gpu_id="0"):
    # if model_type is None, throw out the error
    if model_type is None:
        raise
     # save the log to file, get the time as the log name
    log_file_name = "./log/{}-{}-{}-{}.log".format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()),model_type.split('/')[-1],sample_size,max_length)
    logging.basicConfig(filename=log_file_name,
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    logger = logging.getLogger()
    
    # log the basic info
    logger.info("[BASE INFO]")
    logger.info("sample_size: {}".format(sample_size))
    logger.info("max_length: {}".format(max_length))
    logger.info("model_type: {}".format(model_type))
    logger.info("\n\n")

    df = pd.read_csv("prompt_data-train.csv")
    contents = df['content'].tolist()
    sample_contents = random.sample(contents, sample_size)
    for content in tqdm(sample_contents):
        content = content[:max_length]
        content = re.sub('\n+', '\n', content) #去除多个换行符的情况
        logger.info("content:\n{}".format(content))
        logger.info("hash: {}".format(hashlib.sha1(content.encode('utf-8')).hexdigest()))
        logger.info("\n\n")
        try:
            subprocess.call([
                        "python", "../extract/extract.py", 
                        "--model", model_type, 
                        "--N", "20000", 
                        "--batch-size","128",
                        "--seq_len","512", 
                        "--top_k", "40", 
                        "--temperature","1.0", 
                        "--gpu_id", gpu_id,
                        "--prompt_mode","direct_prompt",
                        "--prompt", content,
                        "--internet-sampling"
                        ]               
                        ,shell=False)
        except Exception as e:
            logger.info("[Error]")
            logger.error(e)
            logger.info("\n\n")
            continue

if __name__ == "__main__":
    os.makedirs('log',exist_ok=True)
    main(sample_size=100,model_type="codeparrot/codeparrot-small")


        
