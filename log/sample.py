'''Sample from generated outputs and save as csv file'''
import os
import argparse
from analyze import get_args
import csv
import re


if __name__ == '__main__':
    args=get_args()
    folder_path = os.path.join('save', args.model+'-temp'+str(args.temperature)+'-len'+str(args.seq_len)+'-k'+str(args.top_k), 'analyze')
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