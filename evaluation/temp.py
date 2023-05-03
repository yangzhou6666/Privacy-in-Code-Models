import re
from utils import get_exact_memorized_content_dict
import os
import subprocess
import json
from tqdm import tqdm
def get_the_avg_from_log():
    '''
    1./mnt/hdd1/zyang/Privacy-in-Code-Models/evaluation/log/evl-window-6-2023-05-01-18-27-05-codeparrot-small-codeparrot-temp1.0-len256-k40.log
    2./mnt/hdd1/zyang/Privacy-in-Code-Models/evaluation/log/evl-window-6-2023-05-01-18-26-46-codeparrot-small-codeparrot-temp1.0-len512-k40.log
    3. /mnt/hdd1/zyang/Privacy-in-Code-Models/evaluation/log/evl-window-6-2023-05-01-19-52-10-codeparrot-codeparrot-small-temp1.0-len256-k40.log
    4. /mnt/hdd1/zyang/Privacy-in-Code-Models/evaluation/log/evl-window-6-2023-05-01-20-32-53-codeparrot-codeparrot-small-temp1.0-len512-k40.log
    '''
    with open('/mnt/hdd1/zyang/Privacy-in-Code-Models/evaluation/log/evl-window-6-2023-05-01-20-32-53-codeparrot-codeparrot-small-temp1.0-len512-k40.log','r')as f:
        lines = f.readlines()
    res = []
    total = 0
    ADD_FLAG = False
    memorized_above = False
    for line in lines:
        if 'INFO ========' in line:
            res.append(total)
            total = 0
            ADD_FLAG = True
            memorized_above = False
        if not ADD_FLAG:
            memorized_above = False
            continue
        elif '[!!!]: memorized content' in line:
            if ADD_FLAG:
                total += 1
                memorized_above = True
        else:
            match1 = re.search(r"INFO (\d+): PPL", line)
            if match1:
                num1 = int(match1.group(1))
                if num1>100:
                    ADD_FLAG = False
                    if memorized_above:
                        total -= 1
            memorized_above = False
    res.append(total)
    print(res)

def test_get_exact_memorized_content_dict():
    # function in utils.py
    root_path = '/mnt/hdd1/zyang/Privacy-in-Code-Models/extract/results/codeparrot/codeparrot-small-temp1.0-len256-k40'
    memorized_content_dict = get_exact_memorized_content_dict(root_path)
    values = list(memorized_content_dict.values())
    print(len(values))
    print(values[1])
    print(len(values[1].split('\n')))

    print(values[-1])
    print(len(values[-1].split('\n')))

def analyze_sensitive_information():
    files = os.listdir('sensitive')
    # train_data_folder = '../clone/save/codeparrot/codeparrot-clean'
    train_data_folder = '/mnt/hdd1/zyang/Privacy-in-Code-Models/clone/save/codeparrot/codeparrot-clean'
    # files =['results-base-512-nonprompt-all-merged.txt'] #12
    files =['results-base-512-promptConditional-all-merged-ffff.txt'] #11
    result = {}
    for txt_file in files:
        file_path = os.path.join('sensitive',txt_file)
        with open(file_path,'r') as f:
            lines = f.readlines()
        for line in tqdm(lines,total=len(lines)):
            data = json.loads(line)
            for d in data:
                search_line = d['value'].strip()
                if search_line not in result:
                    result[search_line] = {}
                    result[search_line]['tag'] = d['tag']
        print(len(result))
        for search_line in tqdm(result,total=len(result)):
            # search_command = 'grep -r {} {}'.format(search_line,train_data_folder)
            # print(search_command)
            # try:
            #     output = subprocess.check_output(search_command,shell=True,universal_newlines=True)
            #     result[search_line]['count'] = len(output.split('\n'))
            # except:
            #     result[search_line]['count'] = 0

            search_command = ['grep', '-r','-F', search_line, train_data_folder]
            output = subprocess.run(search_command, capture_output=True, text=True)
            if output.returncode == 0:
                result[search_line]['count'] = len(output.stdout.split('\n'))
            else:
                result[search_line]['count'] = 0

            
        with open(f'./sensitive-result-{txt_file.replace(".txt","")}.json','w') as f:
            json.dump(result,f,indent=4)
    



if __name__ == '__main__':
    # get_the_avg_from_log()
    # test_get_exact_memorized_content_dict()
    analyze_sensitive_information()