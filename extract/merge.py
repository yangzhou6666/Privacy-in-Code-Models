'''Merge generated code files into one and generate some statistical information'''
import os
import argparse
import logging
from tqdm import tqdm
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="The model to load")
    parser.add_argument('--temperature', type=float, default=1.0, help="Start temperature")
    parser.add_argument('--seq_len', type=int, default=256, help="The length of extracted sequence")
    parser.add_argument('--top_k', type=int, default=40, help="sample from the top_k tokens output by the model")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    result_path = 'results/{}-temp{}-len{}-k{}'.format(args.model, args.temperature, args.seq_len, args.top_k)
    seperate_path = os.path.join(result_path, 'seperate')

    logger.info("Analyzing reuslts in {}".format(seperate_path))
    files = os.listdir(seperate_path)
    logger.info("Found {} files".format(len(files)))

    # merge all the files into one
    with open(os.path.join(result_path, 'all'), 'w') as f:
        for file in tqdm(files):
            with open(os.path.join(seperate_path, file), 'r') as f2:
                f.write(f2.read())

    logger.info("Merged file is saved to {}".format(os.path.join(result_path, 'all')))





