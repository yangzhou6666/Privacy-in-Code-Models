import os
import re
import argparse
from tokenize import tokenize
from io import BytesIO


def extract_email_from_string(s):
    emails = re.findall("([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", s)
    return emails



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="py150_files", type=str, 
                        help="The downloaded data path")
    args = parser.parse_args()
    
    train_paths = open(os.path.join(args.base_dir, "python100k_train.txt")).readlines()[:-5000]
    dev_paths = open(os.path.join(args.base_dir, "python100k_train.txt")).readlines()[-5000:]
    
    cnt_code_with_emails = 0
    cnt_email_occurrences = 0

    email_list = []
    for path in train_paths:
        try:
            code = open(os.path.join(args.base_dir, path.strip())).read()
        except UnicodeDecodeError:
            print("UnicodeDecodeError at path: ", path)
            continue
        emails = extract_email_from_string(code)
        
        if len(emails) > 0:
            cnt_code_with_emails += 1
            cnt_email_occurrences += len(emails)
            email_list.extend(emails)
    
    unique_emails = set(email_list)        
    print("%d out of %d codes have emails" % (cnt_code_with_emails, len(train_paths)))
    print("%d email occurences" % cnt_email_occurrences)
    print("%d unique emails" % len(unique_emails))
    
                

    