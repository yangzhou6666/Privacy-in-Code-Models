# PII detection and redaction for Emails, IP adresses and Secret keys

We provide code to detect Emails, IP addresses and API/SSH keys in text datasets (in particular datasets of source code). We use regexes for emails and IP addresses (they are adapted from [BigScience PII pipeline](https://github.com/bigscience-workshop/data-preparation/tree/main/preprocessing/training/02_pii)). And we use [detect-secrets](https://github.com/Yelp/detect-secrets) for finding secrets keys. We additionally implement some filters on top to reduce the number of false positives. There is also some evaluation code to test the pipeline on a PII benchamrk we annotated.

## Usage
```
pip install -r requirements.txt
```
Also make sure to have `git lfs` installed, and login to your `huggingface-hub` account with
````
huggingface-cli login
````
* `main.py` is the main script to run the pipeline. It takes as input a dataset and outputs a new dataset with the PII removed and some additional column containing the secrets found and their statistics.

For example, you can use the following command to run the pipeline on the python subset of the-stack-smol while saving manual shards (to push directly to hub use `--save_mode hub` and to use random replacements use `--load_replacements False`):
```
python main.py --dataset_name bigcode/the-stack-smol --subset data/python --batch_size 1000 --num_proc 64 --target_dataset stack-smol-python-pii --load_replacements True --save_mode_checks manual_shards --save_mode manual_shards
```

Make sure you have the `gibberish_data` folder in the same directory as the script. It contains a [gibberish-detector](https://github.com/domanchi/gibberish-detector) that we use for the filters for keys.

* `pii_detection.py` contains the code to perform PII detection.
* `pii_redaction.py` contains the code to redact the PII.
*  `utils/evaluation.py` contains the code to evaluate the PII detection on our annotated benchmark, with `tests` containing some test cases. (TODO: add script for automatic evaluation on the benchmark)

## Notebooks
* `example.ipynb` is an example notebook to show how to use the pipeline.
* there are several notebooks in `notebooks` folder with some of our experiments.


## How we utilize this utility
1. PII instance detect & record
   1. Under the *PII* folder, run `python3 toy.py --result_path [path_to_dataset_file] --output_path [path_to_output_file]`. We used `base-512-nonprompt-all-merged.txt` and `base-512-promptConditional-all-merged-ffff.txt` as the dataset. Then the PII information will be detected and stored as output file. The dataset files and output files that we used and generated are inside the *PII/output* folder.
2. PII matching
   1. @zzp