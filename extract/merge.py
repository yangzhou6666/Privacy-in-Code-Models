'''Merge generated code files into one and generate some statistical information'''
import os
import argparse
import logging
from tqdm import tqdm
import hashlib
import json
import logging
import numpy as np
import multiprocessing
import hashlib
# set up the logger
logger = logging.getLogger('user_actions')
logger.setLevel(logging.INFO)

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="The model to load")
    parser.add_argument('--temperature', type=float, default=1.0, help="Start temperature")
    parser.add_argument('--seq_len', type=int, default=256, help="The length of extracted sequence")
    parser.add_argument('--top_k', type=int, default=40, help="sample from the top_k tokens output by the model")

    parser.add_argument('--internet-sampling', action='store_true', help="condition the generation on the internet")
    parser.add_argument('--prompt_mode', type=str, default="single_md5",choices=["single_md5","direct_prompt"], help="The mode of the prompt to use for generation")
    parser.add_argument('--prompt', type=str, default="", help="The prompt to use for generation(can also be the path to a file containing the prompt)")
    parser.add_argument('--prompt_hash', type=str, default="", help="The hash of the prompt to use for generation")
    return parser.parse_args()

def store(result_path, result_save_path, files, start, end,args):
    # merge all the files into one
    assert start <= end, "start should be smaller than end"
    assert start < len(files) and end < len(files), "start and end should be smaller than the number of files"
    
    # 这个是因为extract.py中的代码中internet_sampling的时候，保存的错误，导致这里的文件名不是从0开始的（已修改但是没有resample）
    if start == 0:
        start += 1

    size = end - start + 1
    # path to store the merged file
    if args.internet_sampling:
        merged_file_path = os.path.join(result_path, 'all_{}-{}-{}'.format(args.prompt_hash, start, end))
    else:
        merged_file_path = os.path.join(result_path, 'all_{}-{}'.format(start, end))
    logger.info("Start merging files from {} to {}".format(start, end))
    
    curser = 0
    map = {}

    with open(merged_file_path, 'w') as f:
        logger.info("Start writing to {}".format(merged_file_path))
        
        for file in tqdm(range(start, end)):
            file = str(file + 1)
            # jsut to make sure that the file is stored in the right order
            with open(os.path.join(result_save_path, file), 'r') as f2:
                content = f2.readlines()
                new_content = ''
                for line in content:
                    if line == '\n':
                        continue
                    new_content += line
                content = new_content
                # compute MD5
                md5 = hashlib.md5(content.encode('utf-8')).hexdigest()
                to_write = content + '\nSubmission>>>>>>' + md5 + '>>>>>>Submission\n'
                num_of_line = len(to_write.split('\n'))
                f.write(to_write)

                # store information to map
                map[file] = {'md5': md5, 'start': curser + 1, 'end': curser + num_of_line - 1}

                # update curser
                curser = curser + num_of_line - 1

    # store map into json file
    with open(os.path.join(result_path, 'map_{}-{}.json'.format(start, end)), 'w') as f:
        json.dump(map, f)

if __name__ == '__main__':
    args = parse_arguments()
    result_path = 'results/{}-temp{}-len{}-k{}'.format(args.model, args.temperature, args.seq_len, args.top_k)
    if not args.internet_sampling:
        result_save_path = os.path.join(result_path, 'seperate')
        logger.info("Analyzing reuslts in {}".format(result_save_path))
    else:
        if args.prompt_mode == 'single_md5':
            hash_value = args.prompt_hash
        elif args.prompt_mode == 'direct_prompt':
            hash_value = hashlib.sha1(args.prompt.encode('utf-8')).hexdigest()
            args.prompt_hash = hash_value
        # result_save_path = result_save_path = os.path.join(result_path, 'internet',hash_value)
        result_save_path = result_save_path = os.path.join(result_path, 'internet')
    files = os.listdir(result_save_path)
    logger.info("Found {} files".format(len(files)))

    # store in chunks of 200k
    chunk_size = 200000
    if len(files) < chunk_size:
        chunk_size = len(files)-1
    num_of_chunks = int(np.ceil(len(files) / chunk_size)) - 1
    logger.info("Start storing {} chunks".format(num_of_chunks))
    mulprocess = True
    if mulprocess:
        processes = []
        for i in range(num_of_chunks):
            start = i * chunk_size
            end =(i + 1) * chunk_size
            p = multiprocessing.Process(target=store, args=(result_path, result_save_path, files, start, end,args))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        for i in range(num_of_chunks):
            start = i * chunk_size
            end =(i + 1) * chunk_size
            store(result_path, result_save_path, files, start, end,args)







