'''Generate scripts to run the experiments'''
import os


def run_merge(model, temp, len, k):
    cmd = 'python3 merge.py --model {} --temperature {} --seq_len {} --top_k {} &'.format(model, temp, len, k)
    print(cmd)
    os.system(cmd)
    

if __name__ == '__main__':
    configs = [
        {'model': 'Salesforce/codegen-350M-mono', 'temp': 1.0, 'len': 256, 'k': 40},
        {'model': 'Salesforce/codegen-350M-mono', 'temp': 1.0, 'len': 768, 'k': 40},
        {'model': 'Salesforce/codegen-350M-mono', 'temp': 1.0, 'len': 1024, 'k': 40},
        {'model': 'Salesforce/codegen-350M-mono', 'temp': 1.0, 'len': 512, 'k': 10},
        {'model': 'Salesforce/codegen-350M-mono', 'temp': 1.0, 'len': 512, 'k': 20},
        {'model': 'Salesforce/codegen-350M-mono', 'temp': 1.0, 'len': 512, 'k': 40},
        {'model': 'Salesforce/codegen-350M-multi', 'temp': 1.0, 'len': 256, 'k': 40},
        {'model': 'Salesforce/codegen-350M-multi', 'temp': 1.0, 'len': 768, 'k': 40},
        {'model': 'Salesforce/codegen-350M-multi', 'temp': 1.0, 'len': 1024, 'k': 40},
        {'model': 'Salesforce/codegen-350M-multi', 'temp': 1.0, 'len': 512, 'k': 5},
        {'model': 'Salesforce/codegen-350M-multi', 'temp': 1.0, 'len': 512, 'k': 10},
        {'model': 'Salesforce/codegen-350M-multi', 'temp': 1.0, 'len': 512, 'k': 20},
        {'model': 'Salesforce/codegen-350M-multi', 'temp': 1.0, 'len': 512, 'k': 40},
        {'model': 'Salesforce/codegen-350M-nl', 'temp': 1.0, 'len': 256, 'k': 40},
        {'model': 'Salesforce/codegen-350M-nl', 'temp': 1.0, 'len': 768, 'k': 40},
        {'model': 'Salesforce/codegen-350M-nl', 'temp': 1.0, 'len': 1024, 'k': 40},
        {'model': 'Salesforce/codegen-350M-nl', 'temp': 1.0, 'len': 512, 'k': 5},
        {'model': 'Salesforce/codegen-350M-nl', 'temp': 1.0, 'len': 512, 'k': 10},
        {'model': 'Salesforce/codegen-350M-nl', 'temp': 1.0, 'len': 512, 'k': 20},
        {'model': 'Salesforce/codegen-350M-nl', 'temp': 1.0, 'len': 512, 'k': 40},

    ]
    
    for config in configs:
        run_merge(config['model'], config['temp'], config['len'], config['k'])