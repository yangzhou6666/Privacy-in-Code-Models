'''analyze the tend of memorization'''

import os
import json
from analyze import merge_memorizations

if __name__ == '__main__':
    root_dir = './'

    # from which model the results to be analyzed
    configs = [
        {'model': 'codeparrot/codeparrot-small', 'temp': 1.0, 'len': 512, 'k': 21},
        {'model': 'codeparrot/codeparrot-small', 'temp': 1.0, 'len': 512, 'k': 22},
        {'model': 'codeparrot/codeparrot-small', 'temp': 1.0, 'len': 512, 'k': 23},
        {'model': 'codeparrot/codeparrot-small', 'temp': 1.0, 'len': 512, 'k': 24}
    ]

    memorization_paths = []
    for config in configs:

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
            try:
                assert os.path.exists(memorization_path), "memorization file not found: {}".format(memorization_path)
            except:
                continue
            memorization_paths.append(memorization_path)
            print(memorization_path)

    # merge the memorizations
    memorization_count = []
    number_samples = []
    count = 0
    all_memorization = {}
    for memorization_path in memorization_paths:
        with open(memorization_path, 'r') as f:
            memorization = json.load(f)

        all_memorization = merge_memorizations([all_memorization, memorization])
        print(count)
        print(len(all_memorization))
        memorization_count.append(len(all_memorization))
        number_samples.append(count * 200000)
        count += 1

    # visualize the memorization trend
    import matplotlib.pyplot as plt
    plt.plot(number_samples, memorization_count)
    plt.show()
    plt.savefig('memorization_trend.png')
        

    
