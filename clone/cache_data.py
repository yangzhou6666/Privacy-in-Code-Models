'''Cache huggingFace datasets'''

from datasets import load_dataset
import os
from tqdm import tqdm


if __name__ == '__main__':
    cache_dir = "./.data_cache" # Set your own cache directory here
    ds_parrot = load_dataset("codeparrot/codeparrot-clean", split="train",cache_dir=cache_dir)
    ds_java = load_dataset("bigcode/the-stack", data_dir="data/java", split="train",cache_dir=cache_dir)
    ds_python = load_dataset("bigcode/the-stack", data_dir="data/python", split="train",cache_dir=cache_dir)
    ds_js = load_dataset("bigcode/the-stack", data_dir="data/javascript", split="train",cache_dir=cache_dir)
    ds_parrot = load_dataset("codeparrot/codeparrot-clean", split="train",cache_dir=cache_dir)