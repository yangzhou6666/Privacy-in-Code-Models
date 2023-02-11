'''Cache huggingFace datasets'''

from datasets import load_dataset
import os
from tqdm import tqdm
import logging
import hashlib

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def remove_lines(content: str):
    lines = content.split('\n')
    new_content = ''
    for line in lines:
        if line == '':
            continue
        new_content += line + '\n'
    return new_content

def download_and_save_data(data_name, cache_dir=None, data_dir=None, split='train', save_dir='./save'):
    logger.info("Downloading/Reusing data {} from {}".format(data_name, cache_dir))

    if data_dir:
        ds = load_dataset(data_name, split=split, cache_dir=cache_dir)
    else:
        ds = load_dataset(data_name, data_dir=data_dir, split=split, cache_dir=cache_dir)
    
    logger.info("Processing {} examples".format(len(ds)))

    data_dir = data_dir if data_dir else ''
    save_dir = os.path.join(save_dir, data_name, data_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger.info("Saving data to {} ...".format(save_dir))

    batch_size = 100*1000 # 100k
    batches = int(len(ds) / batch_size)
    for i in tqdm(range(batches)):
        with open(os.path.join(save_dir, str(i)), 'w') as f:
            for j in range(i*batch_size,(i+1)*batch_size):
                content = remove_lines(ds[j]['content'])
                hash = hashlib.md5(content.encode('utf-8')).hexdigest()
                f.write(ds[j]['content'] + '\n' + hash + '\n')
    

if __name__ == '__main__':
    cache_dir = "./.data_cache" # Set your own cache directory here
    save_dir = "/home/zyang/privacy/save"

    download_and_save_data(
        data_name="codeparrot/codeparrot-clean",
        cache_dir=cache_dir,
        split="train",
        save_dir=save_dir
    )

    download_and_save_data(
        data_name="bigcode/the-stack",
        data_dir="data/java",
        cache_dir=cache_dir,
        split="train",
        save_dir=save_dir
    )

    download_and_save_data(
        data_name="bigcode/the-stack",
        data_dir="data/python",
        cache_dir=cache_dir,
        split="train",
        save_dir=save_dir
    )

    download_and_save_data(
        data_name="bigcode/the-stack",
        data_dir="data/javascript",
        cache_dir=cache_dir,
        split="train",
        save_dir=save_dir
    )

    exit()
    download_and_save_data(
        data_name="codeparrot/codeparrot-clean",
        cache_dir=cache_dir,
        split="train",
        save_dir=save_dir
    )