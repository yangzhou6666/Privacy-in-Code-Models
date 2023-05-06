'''Sample from generated outputs and save as csv file'''
import os
import argparse
from analyze import get_args
import csv
import re


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', 
                        type=str)
    parser.add_argument('--top_k',
                        type=int)
    parser.add_argument('--temperature',
                        type=float)
    parser.add_argument('--seq_len',
                        type=int)
    parser.add_argument('--num_files',
                        type=int,
                        default=53)
    parser.add_argument('--mode',
                        type=str,
                        choices=['analyze', 'extract_prompt','all'],
                        default='analyze'
    )
    parser.add_argument('--file_begin_number',
                        type=int,
                        default=1
    )
    parser.add_argument('--file_end_number',
                        type=int,
                        default=20000
    )
    parser.add_argument('--internet-sampling', action='store_true', help="condition the generation on the internet")
    parser.add_argument('--prompt_mode', type=str, default="single_md5",choices=["single_md5","direct_prompt"], help="The mode of the prompt to use for generation")
    parser.add_argument('--prompt', type=str, default="", help="The prompt to use for generation(can also be the path to a file containing the prompt)")
    parser.add_argument('--prompt_hash', type=str, default="", help="The hash of the prompt to use for generation")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args=get_args()
    folder_path = os.path.join('save', args.model+'-temp'+str(args.temperature)+'-len'+str(args.seq_len)+'-k'+str(args.top_k))
    if args.internet_sampling:
        if args.prompt_mode == 'single_md5':
            hash_value = args.prompt_hash
        elif args.prompt_mode == 'direct_prompt':
            if isinstance(args.prompt, str) and len(args.prompt) == 40 and re.match("^[a-f0-9]+$", args.prompt):
                args.prompt_hash = args.prompt
            else:
                hash_value = hashlib.sha1(args.prompt.encode('utf-8')).hexdigest()
                args.prompt_hash = hash_value
        folder_path = os.path.join(folder_path,args.prompt_hash)
    folder_path = os.path.join(folder_path, f'{args.file_begin_number}-{args.file_end_number}')
    folder_path = os.path.join(folder_path, 'analyze')
    assert os.path.exists(folder_path), f" {folder_path} does not exist"


    pattern = r'>>>>>>>>>>fingerprints (\w+) >>>>>>>>>>>>>(.*?)<<<<<<<<<<fingerprints \1 <<<<<<<<<<'

    # save as csv file in the same folder
    with open(os.path.join(folder_path, 'all.txt'), 'r') as f_all:
        with open(os.path.join(folder_path, 'sample.csv'), 'w') as f_sample:
            # process f_all
            content = f_all.read()
            memorizations = re.findall(pattern, content, re.DOTALL)
            # sample 384 randomly
            import random
            memorizations = random.sample(memorizations, 384)

            # write to csv file
            writer = csv.writer(f_sample)
            writer.writerow(['MD5', 'Code'])
            for memorization in memorizations:
                if '++++fingerprints' in memorization[1]:
                    code = memorization[1].split('++++fingerprints')[0]
                else:
                    code = memorization[1]
                writer.writerow([memorization[0], code])

            print('Save to {} file successfully!'.format(os.path.join(folder_path, 'sample.csv')))