# 1.把整个测试集的gt补全
# 2.把训练集（in/out）随机划分成test.json的形式，尽可能让长度一致（？且一句可能划为多句 ===> in_train.json/out_train.json
# 3. 按20%-80%把in的训练集
import random
import json
from tqdm import tqdm
import os
def get_the_ground_truth_for_testing(path,lang,original_data_path,sep_token):
    g_t_length = 0
    if lang == "python":
        break_ids = [sep_token]
        contain_ids = []
    else:
        break_ids = [';', '}', '{']
        contain_ids = [';', '}']

    with open(path,'r') as f:
        temp_test_data = f.readlines()
    test_data = []
    for temp_d in temp_test_data:
        t_d = json.loads(temp_d)
        test_data.append(t_d)
    gt_data = []
    with open(original_data_path,'r') as f:
        origin_data = f.readlines()
    for t_d in test_data:
        o_d = origin_data[t_d['id']-1] #这里-1是因为test的4755对应原始4755行，但实际上是列表序号是4744
        
        n_d = o_d.replace(t_d['input'],'')
        try:
            assert len(o_d) == len(n_d)+len(t_d['input']) # 刚好是前序
        except:
            t_d['id'] = t_d['id'] + 1 #id:334,对应第335行，序号334
            o_d = origin_data[t_d['id']-1] 
            n_d = o_d.replace(t_d['input'],'') 
            
        try:
            assert len(o_d) == len(n_d)+len(t_d['input']) # 刚好是前序
        except:
            print(t_d['id'])
            raise

        n_d = n_d.split(" ")
        for i,n in enumerate(n_d):
            if n in break_ids:
                if n in contain_ids:
                    idx = i
                else:
                    idx = i-1
                break
        g_t_length += idx
        gt_data.append({
            "id":t_d['id'],
            "input":t_d['input'],
            "gt":" ".join(n_d[:idx+1])
        })
    with open('test_gt_origin_partial.json','w') as f:
        for g_d in gt_data:
            f.write(json.dumps(g_d))
            f.write('\n')
    print(g_t_length/len(gt_data))   

def split_train_surrogate(data_file,lang,sep_token,file_type,save_path):
    new_dict = []
    apperaed_id = list()
    if lang == "python":
        break_ids = [sep_token]
        contain_ids = []
        gt_token_length_range=[3,11]
    else:
        break_ids = [';', '}', '{']
        contain_ids = [';', '}']
        gt_token_length_range=[4,12]
    with open(data_file,'r') as f:
        data = f.readlines()
    total =0
    total_origin_length=0
    total_gt_token_length = 0
    count_data =0
    for d in tqdm(data,total=len(data)):
        d = json.loads(d)

        new_temp = {}
        new_temp['id'] = d['id']
        apperaed_id.append(d['id'])

        program = d['input']
        program = program.split(' ')
        program_length = len(program)
        total_origin_length += program_length
    
        count =0
        continue_Flag = False
        while True: #先确定input长度，再确定gt长度，满足要求就break，如果不满足就重新随机，直到count>1000
            count+=1
            if count > 1000:
                continue_Flag = True
                break
            input_token_length = random.randint(min(400,int(program_length/2)),min(program_length-1,1200)) #尝试下来让input和gt的长度都落在原始分布的值
            end_index = 0
            for i in range(input_token_length,program_length):
                if  program[i] in break_ids:
                    end_index = i
                    if program[i]  in contain_ids:
                        end_index +=1
                    break
            if gt_token_length_range[0] <end_index  - input_token_length <gt_token_length_range[1]:
                gt_token_length = end_index - input_token_length
                break
        if continue_Flag:
            continue
        count_data +=1
        total += input_token_length
        total_gt_token_length += gt_token_length
        new_gt = program[end_index-gt_token_length:end_index]
        new_input = program[:end_index-gt_token_length]
        new_temp['input'] = ' '.join(new_input)
        new_temp['gt'] = ' '.join(new_gt)
        new_temp['input_token_length'] =input_token_length
        new_temp['gt_token_length'] = gt_token_length
        new_dict.append(new_temp)
    with open(os.path.join(save_path,f'{file_type}_surrogate.json'),'w') as f:
        for n_d in new_dict:
            f.write(json.dumps(n_d))
            f.write('\n')
    print(total/count_data)
    print(total_origin_length/count_data)
    print(total_gt_token_length/count_data)
    print(count_data,len(data))
    return count_data,apperaed_id

def split_testing(data_file,lang,sep_token,save_path,surrogate_train_num):
    new_dict = []
    if lang == "python":
        break_ids = [sep_token]
        contain_ids = []
        gt_token_length_range=[3,11]
    else:
        break_ids = [';', '}', '{']
        contain_ids = [';', '}']
        gt_token_length_range=[4,12]
    with open(data_file,'r') as f:
        data = f.readlines()
    total =0
    total_origin_length=0
    total_gt_token_length = 0
    count_data =0
    for id, d in tqdm(enumerate(data),total=len(data)):
        # if id ==40:
        #     print(d.split(' ')[:40])
        #     raise
        new_temp = {}
        new_temp['id'] = id
        program = d
        program = program.split(' ')
        program_length = len(program)
        total_origin_length += program_length
    
        count =0
        continue_Flag = False
        while True:
            count+=1
            if count > 1000:
                continue_Flag = True
                break
            input_token_length = random.randint(min(400,int(program_length/2)),min(program_length-1,1200)) #尝试下来让input和gt的长度都落在原始分布的值
            end_index = 0
            for i in range(input_token_length,program_length):
                if  program[i] in break_ids:
                    end_index = i
                    if program[i]  in contain_ids:
                        end_index +=1
                    break
            if gt_token_length_range[0] <end_index  - input_token_length <gt_token_length_range[1]:
                gt_token_length = end_index - input_token_length
                break
        if continue_Flag:
            continue
        count_data +=1
        total += input_token_length
        total_gt_token_length += gt_token_length
        new_gt = program[end_index-gt_token_length:end_index]
        new_input = program[:end_index-gt_token_length]
        new_temp['input'] = ' '.join(new_input)
        new_temp['gt'] = ' '.join(new_gt)
        new_temp['input_token_length'] =input_token_length
        new_temp['gt_token_length'] = gt_token_length
        new_dict.append(new_temp)

    random.shuffle(new_dict)
    with open(os.path.join(save_path,'test_surrogate.json'),'w') as f:
        for n_d in new_dict[:surrogate_train_num]:
            f.write(json.dumps(n_d))
            f.write('\n')
    with open(os.path.join(save_path,'test_victim.json'),'w') as f:
        for n_d in new_dict[surrogate_train_num:]:
            f.write(json.dumps(n_d))
            f.write('\n')
    print(total/count_data)
    print(total_origin_length/count_data)
    print(total_gt_token_length/count_data)
    print(count_data,len(data))

    return len(new_dict[surrogate_train_num:])

def split_train_victim(data_file,lang,sep_token,file_type,save_path,apperaed_id,victim_test_num):
    new_dict = []
    if lang == "python":
        break_ids = [sep_token]
        contain_ids = []
        gt_token_length_range=[3,11]
    else:
        break_ids = [';', '}', '{']
        contain_ids = [';', '}']
        gt_token_length_range=[4,12]
    with open(data_file,'r') as f:
        data = f.readlines()
    total =0
    total_origin_length=0
    total_gt_token_length = 0
    count_data =0
    for idx,d in tqdm(enumerate(data),total=len(data)):
        if idx in apperaed_id:
            continue

        new_temp = {}
        new_temp['id'] = idx
        program = d
        program = program.split(' ')
        program_length = len(program)
        total_origin_length += program_length
    
        count =0
        continue_Flag = False
        while True: #先确定input长度，再确定gt长度，满足要求就break，如果不满足就重新随机，直到count>1000
            count+=1
            if count > 1000:
                continue_Flag = True
                break
            input_token_length = random.randint(min(400,int(program_length/2)),min(program_length-1,1200)) #尝试下来让input和gt的长度都落在原始分布的值
            end_index = 0
            for i in range(input_token_length,program_length):
                if  program[i] in break_ids:
                    end_index = i
                    if program[i]  in contain_ids:
                        end_index +=1
                    break
            if gt_token_length_range[0] <end_index  - input_token_length <gt_token_length_range[1]:
                gt_token_length = end_index - input_token_length
                break
        if continue_Flag:
            continue
        count_data +=1
        total += input_token_length
        total_gt_token_length += gt_token_length
        new_gt = program[end_index-gt_token_length:end_index]
        new_input = program[:end_index-gt_token_length]
        new_temp['input'] = ' '.join(new_input)
        new_temp['gt'] = ' '.join(new_gt)
        new_temp['input_token_length'] =input_token_length
        new_temp['gt_token_length'] = gt_token_length
        new_dict.append(new_temp)

    random.shuffle(new_dict)
    with open(os.path.join(save_path,f'{file_type}_victim.json'),'w') as f:
        for n_d in new_dict[:victim_test_num]:
            f.write(json.dumps(n_d))
            f.write('\n')
    print(total/count_data)
    print(total_origin_length/count_data)
    print(total_gt_token_length/count_data)
    print(count_data,len(data))
    print('[saved data num]',len(new_dict[:victim_test_num]))
    return count_data

  

if __name__ =='__main__':
    percentatge = 0.01
    token_data_dir = f"../../../CodeCompletion-token/dataset/javaCorpus/token_completion/"
    # model_name = 'gpt2'
    # model_name = model_name.split('/')[-1]
    SAMPLE_RATIO=[10,20]
    for s in SAMPLE_RATIO:
        if not os.path.exists(os.path.join(str(percentatge),str(s))):
            os.makedirs(os.path.join(str(percentatge),str(s)))
        saved_path = os.path.join(str(percentatge),str(s))
        path = os.path.join(token_data_dir,f'train_{str(s)}.txt')
        surrogate_train_num,apperaed_id = split_train_surrogate(path,'java',None,'train',saved_path)

        path = os.path.join(token_data_dir,f'test.txt')
        victim_test_num =split_testing(path,'java',None,saved_path,surrogate_train_num)
        split_train_victim(os.path.join(token_data_dir,'train.txt'),'java',None,'train',saved_path,apperaed_id,victim_test_num)
    # get_the_ground_truth_for_testing('/workspace/CodeCompletion-line/dataset/javaCorpus/line_completion/test.json','java','/workspace/CodeCompletion-token/dataset/javaCorpus/token_completion/test.txt',None)
    
    