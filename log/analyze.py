import argparse
import os
import re
import linecache
from tqdm import tqdm
import hashlib
import json
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', 
                        type=str)
    parser.add_argument('--top_k',
                        type=int)
    parser.add_argument('--temperature',
                        type=float)
    parser.add_argument('--seq_len',
                        type=int)
    parser.add_argument('--num_files',
                        type=int,
                        default=53)
    parser.add_argument('--mode',
                        type=str,
                        choices=['analyze', 'extract_prompt','all'],
                        default='analyze'
    )
    parser.add_argument('--file_begin_number',
                        type=int,
                        default=0
    )
    args = parser.parse_args()
    return args

def main(args):
    # folder_path = os.path.join('save', args.model+'-temp'+str(args.temperature)+'-len'+str(args.seq_len)+'-k'+str(args.top_k), f'{args.file_begin_number}-{200000+args.file_begin_number}')
    folder_path = os.path.join('save', args.model+'-temp'+str(args.temperature)+'-len'+str(args.seq_len)+'-k'+str(args.top_k))
    save_path = os.path.join(folder_path, 'analyze')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    all_result = {}

    for i in range(args.num_files):
        i = str(i)
        print("[now processing file]: ", i+".log")

        result = {}
        file_path = os.path.join(folder_path, i+'.log')

        GETLINE = False
        md5 = None
        lines= None
        path_all = False
        path_extract = False
        GETLINE = False
        WRITELINE = False

        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines,total=len(lines)):
                line = line.strip()
                if line.startswith('Found'):
                    try:
                        md5 = re.search(r'fingerprint\s+([a-fA-F\d]{32})', line).group(1)
                    except:
                        print(line) #ending with summary which is starting with Found

                    
                    path_all = False
                    path_extract = False
                    GETLINE = False
                    WRITELINE = False

                elif line.startswith('Between'): #start with between
                    if WRITELINE:
                        continue
                    
                    matches = re.search(r'Between lines (\d+) and (\d+) in (.+)', line)
                    try:
                        line_start = int(matches.group(1))
                        line_end = int(matches.group(2))
                        program_file_path = matches.group(3)
                    except:
                        print("error in line: ", line)
                        raise
                    if not GETLINE:
                        lines = linecache.getlines(program_file_path)[line_start-1:line_end]
                        lines = ''.join(lines)
                        GETLINE = True
                    
                    if 'all' in program_file_path.split('/')[-1]:
                        path_all = True
                    else:
                        path_extract = True
                    if path_all and path_extract:
                        if md5 not in result:
                            result[md5] = [lines]
                        else:
                            if lines not in result[md5]:
                                result[md5].append(lines)

                        WRITELINE = True
                        if md5 not in all_result:
                            all_result[md5] = [lines]
                        else:
                            if lines not in all_result[md5]:
                                all_result[md5].append(lines)
                else:
                    print(line)
        with open(os.path.join(save_path, i+'.txt'), 'w') as f:
            for k,lines in result.items():
                f.write(f">>>>>>>>>>fingerprints {k} >>>>>>>>>>>>>\n")
                for line in lines:
                    f.write(line)
                    if len(lines)>1:
                        f.write(f"++++fingerprints {k} ++++\n")
                f.write(f"<<<<<<<<<<fingerprints {k} <<<<<<<<<<\n")
                f.write("\n\n")
        print("===="*10)
        print("[NUM of duplicate]: ", len(result))
        print("===="*10)
    with open(os.path.join(save_path, 'all.txt'), 'w') as f:
        for k,lines in all_result.items():
            f.write(f">>>>>>>>>>fingerprints {k} >>>>>>>>>>>>>\n")
            for line in lines:
                f.write(line)
                if len(lines)>1:
                    f.write(f"++++fingerprints {k} ++++\n")
            f.write(f"<<<<<<<<<<fingerprints {k} <<<<<<<<<<\n")
            f.write("\n\n")
    print("===="*10)
    print("[NUM of duplicate]: ", len(all_result))
    print("===="*10)

def extract_prompt(args):
    '''
    {
        'hash_value': {
            'prompt': 'prompt',
            'occurrence': xxx,
            'fignerprints': []
        }
    }
    '''
    # folder_path = os.path.join('save', args.model+'-temp'+str(args.temperature)+'-len'+str(args.seq_len)+'-k'+str(args.top_k), f'{args.file_begin_number}-{200000+args.file_begin_number}')
    folder_path = os.path.join('./log/save', args.model+'-temp'+str(args.temperature)+'-len'+str(args.seq_len)+'-k'+str(args.top_k))
    saved_path = os.path.join(folder_path, 'analyze')

    extract_prompt = {}
    with open(os.path.join(saved_path, 'all.txt'), 'r') as f:
        lines = f.readlines()
        prompt = None
        md5 = None
        for line in lines:
            line = line.strip()
            if line.startswith('>>>>>>>>>>'):
                prompt = None
                try:
                    md5 = re.search(r'fingerprints\s+([a-fA-F\d]{32})', line).group(1)
                except:
                    print(line)
                    raise
            elif line.startswith('<<<<<<<<<<') or line.startswith('++++'):
                if prompt is None: #先+++再<<<<<
                    continue
                hash_value = hashlib.sha1(prompt.encode('utf-8')).hexdigest()
                if hash_value not in extract_prompt:
                    extract_prompt[hash_value] = {
                        'prompt': prompt,
                        'occurrence': 1,
                        'fingerprints': md5
                    }
                else:
                    extract_prompt[hash_value]['occurrence'] += 1
                prompt = None
            else:
                if prompt is None:
                    prompt = line
                else:
                    prompt += '\n'
                    prompt += line

    extract_prompt = dict(sorted(extract_prompt.items(), key=lambda x: x[1]["occurrence"],reverse=True))
    with open(os.path.join(saved_path, 'extract_prompt.json'), 'w') as f:
        json.dump(extract_prompt, f, indent=4)

if __name__ == '__main__':
    args = get_args()
    if args.mode == 'analyze':
        main(args)
    elif args.mode == 'extract_prompt':
        extract_prompt(args)
    elif args.mode == 'all':
        main(args)
        extract_prompt(args)
    else:
        raise ValueError("mode should be one of ['analyze', 'extract_prompt','all']")