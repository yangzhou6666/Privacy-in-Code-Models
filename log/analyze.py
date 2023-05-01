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
                        default=1
    )
    parser.add_argument('--file_end_number',
                        type=int,
                        default=20000
    )

    parser.add_argument('--internet-sampling', action='store_true', help="condition the generation on the internet")
    parser.add_argument('--prompt_mode', type=str, default="single_md5",choices=["single_md5","direct_prompt"], help="The mode of the prompt to use for generation")
    parser.add_argument('--prompt', type=str, default="", help="The prompt to use for generation(can also be the path to a file containing the prompt)")
    parser.add_argument('--prompt_hash', type=str, default="", help="The hash of the prompt to use for generation")


    parser.add_argument('--save_begin_end_list', action='store_true', help="save the begin and end list")

    args = parser.parse_args()
    return args

def main(args):
    folder_path = os.path.join('save', args.model+'-temp'+str(args.temperature)+'-len'+str(args.seq_len)+'-k'+str(args.top_k))
    # folder_path = os.path.join('save', args.model+'-temp'+str(args.temperature)+'-len'+str(args.seq_len)+'-k'+str(args.top_k))
    if args.internet_sampling:
        if args.prompt_mode == 'single_md5':
            hash_value = args.prompt_hash
        elif args.prompt_mode == 'direct_prompt':
            if isinstance(args.prompt, str) and len(args.prompt) == 40 and re.match("^[a-f0-9]+$", args.prompt):
                args.prompt_hash = args.prompt
            else:
                hash_value = hashlib.sha1(args.prompt.encode('utf-8')).hexdigest()
                args.prompt_hash = hash_value
        folder_path = os.path.join(folder_path,args.prompt_hash)
    # folder_path = os.path.join(folder_path, f'{args.file_begin_number}-{args.file_end_number}')
    save_path = os.path.join(folder_path, 'analyze')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    all_result = {}
    begin_end_list = []
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
        
        WRITBEGINEND = False
        temp_begin_end_list = []

        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines,total=len(lines)):
                line = line.strip()
                if line.startswith('Found'):
                    try:
                        md5 = re.search(r'fingerprint\s+([a-fA-F\d]{32})', line).group(1)
                    except:
                        print(line) #ending with summary which is starting with Found

                    if WRITBEGINEND:
                        begin_end_list.extend(temp_begin_end_list)

                    path_all = False
                    path_extract = False
                    GETLINE = False
                    WRITELINE = False
                    WRITBEGINEND = False
                    temp_begin_end_list = []

                elif line.startswith('Between'): #start with between
                    if WRITELINE and not args.save_begin_end_list: #如果不保存begin_end_list才能快进
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
                        temp_begin_end_list.append((line_start, line_end))
                    else:
                        path_extract = True
                    if path_all and path_extract:
                        WRITBEGINEND = True
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

    if args.save_begin_end_list:
        with open(os.path.join(save_path, 'begin_end_list.txt'), 'w') as f:
            for begin, end in begin_end_list:
                f.write(f"{begin} {end}\n")

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