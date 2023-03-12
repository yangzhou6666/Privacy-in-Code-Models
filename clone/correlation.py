'''Analyzing the correlation between train and extract'''

import os
import logging
import json
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    root_dir = './'

    # from which model the results to be analyzed
    config = {'model': 'codeparrot/codeparrot-small', 'temp': 1.0, 'len': 512, 'k': 21}
    generated_folder = '{}-temp{}-len{}-k{}'.format(config['model'], config['temp'], config['len'], config['k'])

    size = 100000 

    # analyze the memorization

    memorization_data = log_dir = os.path.join(root_dir, 'log/save/', generated_folder, "{}/stats".format(size), 'memorization.json')

    # read the memorization data
    with open(memorization_data, 'r') as f:
        memorization = json.load(f)
    
    # print a sample
    for fingerprint in memorization:
        print(fingerprint, memorization[fingerprint])
        break

    # analyze the correlation
    train = []
    extract = []
    fingerprints = []
    for fingerprint in memorization:
        data = memorization[fingerprint]
        train.append(data['train'])
        extract.append(data['extract'])
        fingerprints.append(fingerprint)

    max_index = np.argmax(train)
    print("The fingerprint with the most train data is:", fingerprints[max_index])
    print("The number of train data is:", train[max_index])
    print("The number of extract data is:", extract[max_index])
    print("The length of the fingerprint is:", memorization[fingerprints[max_index]]['len'])
    
    exit()

    corr_coef = np.corrcoef(train, extract)[0, 1]

    print("The Pearson correlation coefficient is:", corr_coef)

    plt.scatter(train, extract)

    m, b = np.polyfit(extract, train, 1)
    print(m)
    plt.plot(train, m*np.array(train) + b, color='red')

    plt.xlabel('train')
    plt.ylabel('extract')

    plt.savefig(os.path.join('_correlation.png'))
