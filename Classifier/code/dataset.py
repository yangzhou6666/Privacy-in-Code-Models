from torch.utils.data import  Dataset,DataLoader
import os
import json
import random
import pickle   
import torch
from torch.nn.utils.rnn import pad_sequence
def get_data(path,file):
    with open(os.path.join(path,file), 'r') as f:
        data = f.readlines()
    return data

# 把数据中的gt和prediction提取出来
def prepare_data(args):
    model_name = args.surrogate_model.split('/')[-1]
    if args.mode == 'victim':
        model_name = "CodeGPT-small-java-adaptedGPT2"
    predictions_train = get_data(args.prediction_data_folder_path,f'train_{model_name}_{args.mode}_infer.txt')
    predictions_test = get_data(args.prediction_data_folder_path,f'test_{model_name}_{args.mode}_infer.txt')
    train = get_data(args.prediction_data_folder_path,f'train_{args.mode}.json')
    test = get_data(args.prediction_data_folder_path,f'test_{args.mode}.json')

    assert len(predictions_train) == len(train)
    assert len(predictions_test) == len(test)
    classifier_train = []
    for i in range(len(train)):
        train_data = json.loads(train[i])
        classifier_train.append({
            'id':train_data['id'],
            "input":train_data['input'],
            "gt":train_data['gt'],
            "prediction":predictions_train[i]
         })
    classifier_test = []
    for i in range(len(test)): 
        test_data = json.loads(test[i])
        classifier_test.append({
            'id':test_data['id'], 
            "input":test_data['input'],
            "gt":test_data['gt'],
            "prediction":predictions_test[i]
         })

    return classifier_train, classifier_test

# 将data分为训练集和验证集
def divide_data(args,classifier_train,classifier_test,scale=0.8):

    training_data_num = int(len(classifier_train) *scale)
    
    train_false_data = classifier_test[:training_data_num]
    train_true_data = classifier_train[:training_data_num]

    val_false_data = classifier_test[training_data_num:]
    val_true_data = classifier_train[training_data_num:]
    
    assert len(train_false_data) == len(train_true_data)
    assert len(val_false_data) == len(val_true_data)
    
    saved_dir = os.path.join(args.data_dir,args.lang,args.surrogate_model,args.sample_ratio)
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    with open(os.path.join(saved_dir,'train.json'),'w') as f:
        for i in range(len(train_false_data)):
            train_false_data[i]['label'] = 0
            train_true_data[i]['label'] = 1
            f.write(json.dumps(train_false_data[i]))
            f.write('\n')
            f.write(json.dumps(train_true_data[i]))
            f.write('\n')
    with open(os.path.join(saved_dir,'val.json'),'w') as f:
        for i in range(len(val_false_data)):
            val_false_data[i]['label'] = 0
            val_true_data[i]['label'] = 1
            f.write(json.dumps(val_false_data[i]))
            f.write('\n')
            f.write(json.dumps(val_true_data[i]))
            f.write('\n')

def keep_test_data(args,classifier_train,classifier_test):
    saved_dir = os.path.join(args.data_dir,args.lang,args.surrogate_model,args.sample_ratio)
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    with open(os.path.join(saved_dir,'test.json'),'w') as f:
        for i in range(len(classifier_train)):
            classifier_test[i]['label'] = 0
            classifier_train[i]['label'] = 1
            f.write(json.dumps(classifier_test[i]))
            f.write('\n')
            f.write(json.dumps(classifier_train[i]))
            f.write('\n')


class ClassificationDataset(Dataset):
    def __init__(self,args,file_type,tokenizer) -> None:
        super().__init__()
        saved_dir = os.path.join(args.data_dir,args.lang,args.surrogate_model,args.sample_ratio)
        if os.path.exists(os.path.join(saved_dir,file_type+'.pickle')):
            with open(os.path.join(saved_dir,file_type+'.pickle'),'rb') as f:
                self.data = pickle.load(f)
        else:
            with open(os.path.join(saved_dir,file_type+'.json'),'r') as f:
                data = f.readlines()
            self.data = []
            for d in data:
                d = json.loads(d)

                inputs = d['input'][1:] # 去掉开头的<s>

                inputs = tokenizer.encode(inputs,add_special_tokens=False)
                inputs = inputs[-args.input_length:]

                gt = d['gt']
                gt = tokenizer.encode(gt,add_special_tokens=False)
                gt = gt[:args.prediction_length]

                prediction = d['prediction']
                prediction = tokenizer.encode(prediction,add_special_tokens=False)
                prediction = prediction[:args.prediction_length]

                self.data.append({
                    'input':[tokenizer.cls_token_id]+inputs+[tokenizer.sep_token_id]+gt+[tokenizer.sep_token_id]+prediction+[tokenizer.sep_token_id]+[tokenizer.pad_token_id]*(args.total_length-len(inputs)-len(gt)-len(prediction)-4),
                    'label':d['label']
                }
                )
            with open(os.path.join(saved_dir,file_type+'.pickle'),'wb') as f:
                pickle.dump(self.data,f)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]['input'],self.data[item]['label']

def ClassificationDataset_collate_fn(batch,padding_value=0):
    inputs = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    return {
        "input_ids":torch.tensor(inputs).long(),
        "label":torch.tensor(labels).long(),
        "attention_mask":(torch.tensor(inputs)!=padding_value).long(),
    }