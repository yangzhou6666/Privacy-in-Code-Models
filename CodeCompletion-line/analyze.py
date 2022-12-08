import os
import json
import re
import Levenshtein as lev

def extract_email_from_string(s):
    emails = re.findall("([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", s)
    return emails

if __name__=="__main__":
    prediction_path = 'save/py150-emails-train/predictions_line.txt'
    gt_path = 'dataset/py150/line_completion_emails/test.json'
    
    predictions = open(prediction_path, 'r').readlines()
    gts = []
    with open(gt_path, 'r') as f:
        data = f.readlines()
        for info in data:
            gt = json.loads(info)['gt']
            gts.append(gt)
    
    assert len(predictions) == len(gts)
    
    exact_match_cnt = 0
    small_edit_cnt = 0
    no_email_cnt = 0
    for pred, gt in zip(predictions, gts):
        pred_email = extract_email_from_string(pred)[0] if len(extract_email_from_string(pred)) > 0 else ''
        if len(pred_email) == 0:
            no_email_cnt += 1
        gt_email = extract_email_from_string(gt)[0]
        if pred_email == gt_email:
            exact_match_cnt += 1
        
        if lev.distance(pred_email, gt_email) <= 4 and lev.distance(pred_email, gt_email) > 0:
            small_edit_cnt += 1 
            
    print('Exact match: {}'.format(exact_match_cnt / len(predictions)))
    print('Small edit: {}'.format(small_edit_cnt / len(predictions)))
    print('No email: {}'.format(no_email_cnt / len(predictions)))
        
        