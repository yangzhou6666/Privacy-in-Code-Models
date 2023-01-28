from pii_detection import scan_pii_batch
from tqdm import tqdm

if __name__ == '__main__':
    result_path = "/Users/zhouyang/Downloads/all"
    with open(result_path, 'r') as f:
        contents = f.readlines()
        code = ""
        example = {'content': []}
        for line in tqdm(contents):
            if "Submission>>>>>>" in line:
                example['content'].append(code)
                code = ""
            else:
                code += line

    results = scan_pii_batch(example, key_detector="regex")

    counter = 0
    for s in results['secrets']:
        if s != '[]':
            print(s)
            counter += 1

    print(counter)