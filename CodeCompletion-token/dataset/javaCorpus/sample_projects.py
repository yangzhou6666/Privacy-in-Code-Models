import os
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def find_all_java_files(directory):
    """Find all files recursively in a directory"""
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.java'):
                files.append(os.path.join(root, filename))
    return files



if __name__=='__main__':
    # set seed for reproducible sampling
    set_seed(42)
    
    repos_dir = './JavaCorpusRaw/java_projects'
    all_repos = os.listdir(repos_dir)
    random.shuffle(all_repos) # randomly shuffle
    # take the first 11,000 as training, and the 1,000 as validation, and the rest as test
    train_repos = all_repos[:11000]
    dev_repos = all_repos[11000:12000]
    test_repos = all_repos[12000:]
    
    sampling_rates = [0.01, 0.02, 0.05, 0.10, 0.15, 0.2, 1.0]
    
    os.makedirs('./separation', exist_ok=True)
    os.makedirs('./token_completion', exist_ok=True)
    for rate in sampling_rates:
        # sampling
        train_repos_sampled = random.sample(train_repos, int(rate*len(train_repos)))
        dev_repos_sampled = random.sample(dev_repos, int(rate*len(dev_repos)))
        test_repos_sampled = random.sample(test_repos, int(rate*len(test_repos)))
        
        with open(f'./separation/trainJava-{rate}.txt', 'w') as f:
            for repo in train_repos_sampled:
                f.write(repo + '\n')
        
        with open(f'./separation/devJava-{rate}.txt', 'w') as f:
            for repo in dev_repos_sampled:
                f.write(repo + '\n')
        
        with open(f'./separation/testJava-{rate}.txt', 'w') as f:
            for repo in test_repos_sampled:
                f.write(repo + '\n')
        
        # processing
        os.makedirs(f'./token_completion/{rate}', exist_ok=True)
        train_data_path = f'./token_completion/{rate}/train.txt'
        dev_data_path = f'./token_completion/{rate}/dev.txt'
        test_data_path = f'./token_completion/{rate}/test.txt'
        
        # process files in a repo
        
        for repo_name in train_repos_sampled:
            repo_path = os.path.join(repos_dir, repo_name)
            # iterate all the files under the path
            java_files = find_all_java_files(repo_path)
            print(java_files)
            
            exit()
            