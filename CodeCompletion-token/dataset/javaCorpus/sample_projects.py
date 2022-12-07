import os
import random
import numpy as np
import javalang
from tqdm import tqdm, trange

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


def process_java_code(path):
    assert path.endswith('.java')
    
    new_data = []
    try:
        source_code = open(path).read()
        for tok in list(javalang.tokenizer.tokenize(source_code)):
            token = tok.value
            new_data.append(token)
    except Exception:
        return None
    if len(new_data) == 0:
        return None
    data = "<s> " + " ".join(new_data) + " </s>"
    return data

            

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
        dev_repos_sampled = random.sample(dev_repos, 100)
        test_repos_sampled = random.sample(test_repos, 100)
        
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
        with open(train_data_path, 'w') as f:
            processed_codes = []
            for repo_name in tqdm(train_repos_sampled, desc="Processing train data"):
                repo_path = os.path.join(repos_dir, repo_name)
                # iterate all the files under the path
                java_files = find_all_java_files(repo_path)
                for file_path in java_files:
                    processed_code = process_java_code(file_path)
                    if processed_code is not None:
                        processed_codes.append(processed_code)
            
            random.shuffle(processed_codes)
            for code in processed_codes:
                try:
                    f.write(code+"\n")
                except Exception:
                    continue
                
        with open(test_data_path, 'w') as f:
            processed_codes = []
            for repo_name in tqdm(test_repos_sampled, desc="Processing test data"):
                repo_path = os.path.join(repos_dir, repo_name)
                # iterate all the files under the path
                java_files = find_all_java_files(repo_path)
                for file_path in java_files:
                    processed_code = process_java_code(file_path)
                    if processed_code is not None:
                        processed_codes.append(processed_code)
            
            random.shuffle(processed_codes)
            for code in processed_codes:
                try:
                    f.write(code+"\n")
                except Exception:
                    continue
            
        with open(dev_data_path, 'w') as f:
            processed_codes = []
            for repo_name in tqdm(dev_repos_sampled, desc="Processing dev data"):
                repo_path = os.path.join(repos_dir, repo_name)
                # iterate all the files under the path
                java_files = find_all_java_files(repo_path)
                for file_path in java_files:
                    processed_code = process_java_code(file_path)
                    if processed_code is not None:
                        processed_codes.append(processed_code)
            
            random.shuffle(processed_codes)
            for code in processed_codes:
                try:
                    f.write(code+"\n")
                except Exception:
                    continue
            