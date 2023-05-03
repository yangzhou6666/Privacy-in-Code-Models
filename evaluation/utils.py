import os
import json
from tqdm import tqdm
def get_memorized_flag(root_path):
    map_file_path = os.path.join(root_path, 'map.json')
    if not os.path.exists(map_file_path):
        raise
    with open(map_file_path, 'r') as f:
        map_data = json.load(f)
    labels = [int(k) for k in map_data.keys()]
    end_values = [v['end'] for v in map_data.values()]
    start_values = [v['start'] for v in map_data.values()]
    flag_values = []
    for i,label in enumerate(labels):
        flag_values.extend([label] * (end_values[i] - start_values[i] + 1))
    begin_list = []
    begin_end_list_file = os.path.join(root_path.replace('extract/results','log/save'),'analyze','begin_end_list.txt')
    with open(begin_end_list_file, 'r') as f:
        for line in f:
            begin,end = line.strip().split(' ')
            begin_list.append(int(begin))
    memorized_content_index = set()
    for b in tqdm(begin_list,total=len(begin_list)):
        b -= 1 # list index starts from 0
        flag = flag_values[b]
        memorized_content_index.add(flag-1) # list index starts from 0
    return list(memorized_content_index)

def get_exact_memorized_content_dict(root_path):
    map_file_path = os.path.join(root_path, 'map.json')
    if not os.path.exists(map_file_path):
        raise
    with open(map_file_path, 'r') as f:
        map_data = json.load(f)
    labels = [int(k) for k in map_data.keys()]
    end_values = [v['end'] for v in map_data.values()]
    start_values = [v['start'] for v in map_data.values()]
    memorized_flag_values = [0]*end_values[-1]

    begin_list = []
    end_list = []
    begin_end_list_file = os.path.join(root_path.replace('extract/results','log/save'),'analyze','begin_end_list.txt')
    with open(begin_end_list_file, 'r') as f:
        for line in f:
            begin,end = line.strip().split(' ')
            begin_list.append(int(begin))
            end_list.append(int(end))

    for b,e in tqdm(zip(begin_list,end_list),total=len(begin_list)):
        b -= 1 # list index starts from 0
        e -= 1 # list index starts from 0
        memorized_flag_values[b:e+1] = [1]*(e-b+1)

    print('root_path:',root_path)
    all_content_path = os.path.join(root_path, 'all')
    with open(all_content_path, 'r') as f:
        all_content = f.readlines()

    now_end_index = 0
    idx = 0
    temp_content = ''
    memorized_content_dict = {}


    print('len(all_content):',len(all_content))
    print('max end_values:',end_values[-1])
    for i,c in tqdm(enumerate(all_content),total=len(all_content)):
        if i == now_end_index:
            now_end_index = end_values.pop(0)           
            memorized_content_dict[idx] = temp_content
            idx += 1
            temp_content = ''
        if memorized_flag_values[i] == 1:
            temp_content += c
        else:
            temp_content += '=*=*=*=*=*=*=*=\n'
    memorized_content_dict[idx] = temp_content
    return memorized_content_dict
        
            
    
    