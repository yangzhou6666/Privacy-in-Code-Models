'''Scripts to obtain results in RQ1'''

import json
import os

def number_of_unique_memorization(json_path):
    pass


def total_number_of_memorization(json_path):
    # the path should exist
    assert os.path.exists(json_path)
    # read the json file
    with open(json_path, 'r') as f:
        memorization = json.load(f)
    
    total_num = 0
    for fingerprint in memorization:
        data = memorization[fingerprint]
        total_num += data['extract']

    return total_num


if __name__ == '__main__':
    json_path = '../log/save/codeparrot/codeparrot-temp1.0-len512-k20/stats/memorization.json'

    print(total_number_of_memorization(json_path))
