'''Analyze memroization '''

import os
import logging
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
import argparse

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="The model to load")
    parser.add_argument('--temperature', type=float, default=1.0, help="Start temperature")
    parser.add_argument('--seq_len', type=int, default=256, help="The length of extracted sequence")
    parser.add_argument('--top_k', type=int, default=40, help="sample from the top_k tokens output by the model")

    return parser.parse_args()

def process_file(log_path):
    memorization = {}
    data = None
    with open(log_path, 'r') as f:
        logger.info("Analyzing {}".format(log_path))
        lines = f.readlines()
        for line in lines:
            if 'duplicate lines with fingerprint' in line:
                # store the previous data
                if data:
                    if data['extract'] > 0 and data['train'] > 0:
                        # only store memorized data
                        memorization[fingerprint] = data

                # update the information
                suffix = line.split('fingerprint ')[1]
                fingerprint = suffix.split(' in')[0]
                prefix = line.split(' duplicate')[0]
                len = int(prefix.split('Found ')[1])

                try:
                    data = memorization[fingerprint]
                except:
                    data = {'train': 0, 'extract': 0, "len": len}

            # analyzing the clone information
            if 'clone' in line:
                data['train'] += 1
            if 'extract' in line:
                data['extract'] += 1
    
    return memorization


def merge_memorizations(memorizations):
    memorization = {}
    for m in memorizations:
        for fingerprint in m:
            try:
                memorization[fingerprint]['train'] += m[fingerprint]['train']
                memorization[fingerprint]['extract'] += m[fingerprint]['extract']
            except:
                memorization[fingerprint] = m[fingerprint]
    return memorization

if __name__ == '__main__':
    root_dir = './'

    # from which model the results to be analyzed
    args = parse_arguments()
    config = {
        'model': args.model,
        'temp': args.temperature,
        'len': args.seq_len,
        'k': args.top_k
    }
    generated_folder = '{}-temp{}-len{}-k{}'.format(config['model'], config['temp'], config['len'], config['k'])

    step = 2 # * 100000
    start = 0
    end = 32 # * 100000
    # process in chunks

    memorizations = []
    previous_memorizations = {}
    
    # collect the memorization data for each chunk
    for size in range(start, end, step):
        _start = size * 100000
        _end = (size + step) * 100000
        log_dir = os.path.join(root_dir, 'log/save/', generated_folder, "{}-{}".format(_start, _end))

        stats_path = os.path.join(log_dir, 'stats')
        memorization_path = os.path.join(stats_path, 'memorization.json')

        os.makedirs(stats_path, exist_ok=True)

        logs = [os.path.join(log_dir, log) for log in os.listdir(log_dir) if log.endswith('.log')]
        
        # multiprocessing
        with multiprocessing.Pool() as pool:
            for result in pool.map(process_file, logs):
                memorizations.append(result)
    
        # merge the memorizations
        memorization = merge_memorizations(memorizations)
        # save as json
        logger.info("Saving memorization to {}".format(memorization_path))
        with open(memorization_path, 'w') as f:
            json.dump(memorization, f, indent=4)

        # merge with previous memorizations
        previous_memorizations = merge_memorizations([previous_memorizations, memorization])

        # count the number of unique fingerprints
        count = 0
        for fingerprint in previous_memorizations:
            count += 1
        logger.info("Number of unique fingerprints: {}".format(count))


    # store as json
    with open(memorization_path, 'w') as f:
        json.dump(memorization, f, indent=4)

    '''Analyze the memorization'''

    # length distribution
    lens = {}
    count = 0
    for fingerprint in memorization:
        count += 1
        data = memorization[fingerprint]
        try:
            lens[data['len']] += 1
        except:
            lens[data['len']] = 1

    # number of unique fingerprints
    logger.info("Number of unique fingerprints: {}".format(count))


    # draw the length distribution
    plt.bar(lens.keys(), lens.values())
    plt.xlabel('length')
    plt.ylabel('count')
    # save
    plt.savefig(os.path.join(stats_path, 'length_distribution.png'))


    # correlation between train and extract
    train = []
    extract = []
    for fingerprint in memorization:
        data = memorization[fingerprint]
        train.append(data['train'])
        extract.append(data['extract'])
    
    plt.scatter(train, extract)
    plt.xlabel('train')
    plt.ylabel('extract')
    # save
    plt.savefig(os.path.join(stats_path, 'correlation.png'))





