'''Generate scripts to run the experiments'''
import os


def run_merge(model, temp, len, k):
    cmd = 'python3 merge.py --model {} --temperature {} --seq_len {} --top_k {} &'.format(model, temp, len, k)
    print(cmd)
    exit()
    os.system(cmd)
    

if __name__ == '__main__':
    configs = [
        {'model': 'codeparrot/codeparrot', 'temp': 1.0, 'len': 1024, 'k': 5},
        {'model': 'codeparrot/codeparrot', 'temp': 1.0, 'len': 1024, 'k': 10},
        {'model': 'codeparrot/codeparrot', 'temp': 1.0, 'len': 1024, 'k': 20},
        {'model': 'codeparrot/codeparrot', 'temp': 1.0, 'len': 1024, 'k': 40},
    ]
    
    for config in configs:
        run_merge(config['model'], config['temp'], config['len'], config['k'])