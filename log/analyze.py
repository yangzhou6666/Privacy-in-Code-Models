import argparse
import os
import re
import linecache
from tqdm import tqdm
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
    args = parser.parse_args()
    return args

def main():
    args = get_args()
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
                    
                    if program_file_path.split('/')[-1] == 'all':
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

if __name__ == '__main__':
    main()






                    
    




    
