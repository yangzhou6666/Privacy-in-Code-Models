import os
import re
from email_extractor import extract_email_from_string
import json
def find_email_pos(code):
    p = re.compile("([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")
    pos = []
    for m in p.finditer(code):
        pos.append((m.start(), m.end()))
    return pos
    

if __name__ == '__main__':
    dir = './token_completion-original'
    save_dir = '../../../CodeCompletion-line/dataset/py150/line_completion_emails'
    os.makedirs(save_dir, exist_ok=True)
    file_types = ['train']
    for type in file_types:
        data = []
        path = os.path.join(dir, type+'.txt') # path to the source
        save_path = os.path.join(save_dir, type+'.json') # path to the target
        outfile = open(save_path, 'w')
        with open(path, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                pos = find_email_pos(line)
                if len(pos) > 0:
                    for p in pos:
                        start = p[0]
                        end = p[1]
                        input = line[:start]
                        gt = line[start:end]
                        outfile.write(
                            json.dumps({'id': idx, 'input': input, 'gt': gt})
                        )
                        outfile.write('\n')
        exit()