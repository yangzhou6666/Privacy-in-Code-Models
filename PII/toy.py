import argparse
from pii_detection import scan_pii_batch
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('--result_path', type=str, required=True,
                        help='path to result file')
    parser.add_argument('--output_path', type=str, required=True,
                        help='path to output file')

    args = parser.parse_args()
    result_path = args.result_path
    output_path = args.output_path

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
    with open(output_path, 'w') as f:
        for s in results['secrets']:
            if s != '[]':
                f.write(s + '\n')
                counter += 1

    print("Results stored in:", output_path)
    print("Total secrets found:", counter)
