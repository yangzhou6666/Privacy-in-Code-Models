'''Analyze memroization '''

import os
import logging
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing


logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
    config = {'model': 'codeparrot/codeparrot-small', 'temp': 1.0, 'len': 512, 'k': 21}
    size = 200 * 1000
    generated_folder = '{}-temp{}-len{}-k{}'.format(config['model'], config['temp'], config['len'], config['k'])


    log_dir = os.path.join(root_dir, 'log/save/', generated_folder, str(size))
    stats_path = os.path.join(log_dir, 'stats')
    os.makedirs(stats_path, exist_ok=True)


    memorizations = []
    memorization_path = os.path.join(stats_path, 'memorization.json')

    logs = [os.path.join(log_dir, log) for log in os.listdir(log_dir) if log.endswith('.log')]
    
    # multiprocessing
    with multiprocessing.Pool() as pool:
        for result in pool.map(process_file, logs):
            memorizations.append(result)
    
    
    # merge the memorizations
    memorization = merge_memorizations(memorizations)

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





