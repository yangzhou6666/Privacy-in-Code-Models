# Overview

This repository 💻 is for our ASE submission 📚 that explores 🔍 the memorization 🧠 in code models.


**The outline of our repositry:**
- [Environment Configuration](#environment-configuration)
- [Non-Conditional Sampling](#non-conditional-sampling)
- [Memorization Analysis](#memorization-analysis)
- [Two-step Sampling](#two-step-sampling)
- [Conditonal Sampling](./prompts/README.md)
- [Memorized Information Inference ](./evaluation/README.md)
- [Personal Information Identification](./PII/README.md)




# Environment Configuration

## Conda

The following code will create a conda environment `sample`. You can use `-n environment_name` to specify a new name.
```Shell
conda env create -f environment.yml
```
<span style="background-color: yellow">You may need to update the pytorch installation command accordingly based on your CUDA version.</span>


## Docker

```Shell
To-Do
```
# Non-conditional Sampling
## Sample Outputs from code models

First, `cd extract`. Then, execute the following command:

```bash

python extract.py \
    --model codeparrot/codeparrot-small \ # the model to be sampled
    --N 20000 \ # number of sampled outputs
    --batch-size 64 \  # batch size, depend on your GPU memory
    --seq_len 512 \  # token number of each output
    --top_k 40 \ # randomly sample from top k tokens
    --temperature 1.0 \ # temperature when sampling
    --gpu_id 3 # GPU to use
```

This command will result in a directory `extract/results/codeparrot/codeparrot-small-temp1.0-len512-k40`, which contains a directory `seperate`. This subdirectory will have many files, `1`, `2,` ..., `20000`; each file is an output from the model. 

<span style="background-color: yellow">Note: you can repeated execute the command if you want to sample more files. New file IDs will start from `20001`.</span>

Then, you can execute the following cammand:

```shell
python merge.py \
    --model codeparrot/codeparrot-small \
    --top_k 40 \
    --temperature 1.0 \
    --seq_len 512
```

It will merge these outputs into a big file `all` and generate a file `map.json`. The json file records:

* File ID
* MD5 of the file
* the lines of this file in the merged `all`



# Memorization Analysis

## Download the training data

Execute the following command to download the `codeparrot/codeparrot-clean` dataset, the training data for `codeparrot/codeparrot` models.
```
python cache_data.py 2>&1 | tee download.log
```

This will generate a folder `clone/save/codeparrot/codeparrot-clean`, where the dataset is split into 52 subfiles.
This allow us to analyze memorizations in parallel. 

<span style="background-color: yellow">Note: The dataset is over 50GB, so this process may takes a while, depending on your network status.</span>

## Finding Memorization

In the root directory of this repository, run the following command:

```shell
python clone/scripts.py \
    --model codeparrot/codeparrot-small \
    --top_k 40 \
    --temperature 1.0 \
    --seq_len 512
```
<span style="background-color: yellow">Note: 🚨 Please note that initiating this step will spawn 52 🔄 processes and may consume up to 400 💾 GB of memory. If your 💻 computational resources are limited, we recommend 🔧 modifying the code in clone/scripts.py to reduce the number of processes running in parallel.</span>


This command will analyze the code clones between the `all` file (all the outputs we sampled) and each subfiles of the training data we obtained in the previous step.
It will store the results into `log/save/codeparrot/codeparrot-small-temp1.0-len512-k40`.
This folder contains many log files, `0.log`, `1.log`, ..., `52.log`.
Each log file stores the memorization analyze results.
It could contain something like

```javascript
Found 6 duplicate lines with fingerprint 1176c28f7138b31961b65e38b6f7159b in the following files:
 Between lines 187773 and 187778 in <your-folder>/extract/results/codeparrot/codeparrot-small-temp1.0-len512-k40/all
 Between lines 17575538 and 17575543 in <your-folder>/clone/save/codeparrot/codeparrot-clean/0
 Between lines 4699049 and 4699054 in <your-folder>/clone/save/codeparrot/codeparrot-clean/0
 Between lines 2834883 and 2834888 in <your-folder>/clone/save/codeparrot/codeparrot-clean/0
 Between lines 733896 and 733901 in <your-folder>/clone/save/codeparrot/codeparrot-clean/0
```

It means that:
1. the identified clone has 6 lines.
2. The MD5 of the clone is `1176c28f7138b31961b65e38b6f7159b`.
3. The clone is found in multiple places, including both the `all` file (i.e., model outputs) and part of the  training data.

In other word, the code model **memorizes contents from the training data!**

## Analyze memorization

Then, we run the following command to analyze the memorization:

```shell
python clone/analyze.py \
    --model codeparrot/codeparrot-small \
    --top_k 40 \
    --temperature 1.0 \
    --seq_len 512
```

This command analyzes each log file.

1. extracts memorized contents (i.e., clones appearing in both `all` and subfile of training data) from each log file
2. merge memorized contents in each subfile of training data (using fingerprints), and save to `log/save/codeparrot/codeparrot-small-temp1.0-len512-k40/stats/memorization.json`
3. analyze the memorized contents.


The saved `memorization.json` contains:

```json
{
    "3a2ebcaa1123523fe878de0460533174": {
        "train": 3289,
        "extract": 330,
        "len": 6
    },
    ...
}
```

The key is the fingerprint of the memorized content. `"train": 3289` means it appears 3289 times in the training data and `"extract": 330` means that it appears 330 times in the model outputs. `"len": 6` means the length of the memorized content is 6 lines.

Then, we run the following command to get the memorization content:
```shell
python log/analyze.py \
    --model codeparrot/codeparrot-small \
    --top_k 40 \
    --temperature 1.0 \
    --seq_len 512
```
This command analyzes each log file.

1. extracts memorized contents (i.e., clones appearing in both `all` and subfile of training data) from each log file
2. store the memorized contents to `log/save/codeparrot/codeparrot-small-temp1.0-len512-k40/analyze/`

Each `x.txt` is corresponding to `x.log` and `all.txt` merges all the reslusts in `x.txt`.

the `txt` contains
```txt
>>>>>>>>>>fingerprints dc928385dd77b24d74cbf823d2ad9305 >>>>>>>>>>>>>
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
]
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
++++fingerprints dc928385dd77b24d74cbf823d2ad9305 ++++
   'sphinx.ext.todo',
   'sphinx.ext.coverage'
]
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
++++fingerprints dc928385dd77b24d74cbf823d2ad9305 ++++
<<<<<<<<<<fingerprints dc928385dd77b24d74cbf823d2ad9305 <<<<<<<<<<
```
where the 

1. `>>>>>>>>>>fingerprints dc928385dd77b24d74cbf823d2ad9305 >>>>>>>>>>>>>` is the beigining of the memorized contents and  `dc928385dd77b24d74cbf823d2ad9305` is the md5
2. if the files with the same md5 have more than one memorized content, we use  `++++fingerprints dc928385dd77b24d74cbf823d2ad9305 ++++` to split each memorized contents
3. `<<<<<<<<<<fingerprints dc928385dd77b24d74cbf823d2ad9305 <<<<<<<<<<` is the end of memorized contents.



# Two step sampling
## Sample with prompts
We have two kinds of prompts:

1. specify by humans
2. gets from samples

### specify by humans
if we want to specify the prompts by humans, we can run:
```shell
python extract/extract.py \
    --model codeparrot/codeparrot-small \ # the model to be sampled
    --N 20000 \ # number of sampled outputs
    --batch-size 64 \  # batch size, depend on your GPU memory
    --seq_len 512 \  # token number of each output
    --top_k 40 \ # randomly sample from top k tokens
    --temperature 1.0 \ # temperature when sampling
    --gpu_id 3 \ # GPU to use
    --prompt_mode direct_prompt \
    --prompt xxx \ # The prompt to use for generation
    --internet-sampling
```
### gets from samples
if we want to gets the prompt from the samples, we can run:
```shell
python log/analyze.py \
    --model codeparrot/codeparrot-small \
    --top_k 40 \
    --temperature 1.0 \
    --seq_len 512 \
    --mode extract_prompt
```
it will generate `extract_prompt.json` in `log/save/codeparrot/codeparrot-small-temp1.0-len512-k40/analyze/` and it contains:
```json
{
    "eb17e3d78803b21a4f6144d2f8b639edf3ebf5e3": {
        "prompt": "Licensed under the Apache License, Version 2.0 (the \"License\");\nyou may not use this file except in compliance with the License.\nYou may obtain a copy of the License at\nhttp://www.apache.org/licenses/LICENSE-2.0\nUnless required by applicable law or agreed to in writing, software\ndistributed under the License is distributed on an \"AS IS\" BASIS,\nWITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\nSee the License for the specific language governing permissions and\nlimitations under the License.\n\"\"\"\nimport os\nimport sys",
        "occurrence": 10,
        "fingerprints": "27b096e8c460efa8edde535e9fdbb28d"
    },
    ...
}
```
where the `eb17e3d78803b21a4f6144d2f8b639edf3ebf5e3` is the `sha1` of the `prompt` and the value of `prompt` is getting from `all.txt`, and `occurrence` is the times that the prompts shown in `all.txt`.

Then we run:
```shell
python extract/extract.py \
    --model codeparrot/codeparrot-small \ # the model to be sampled
    --N 20000 \ # number of sampled outputs
    --batch-size 64 \  # batch size, depend on your GPU memory
    --seq_len 512 \  # token number of each output
    --top_k 40 \ # randomly sample from top k tokens
    --temperature 1.0 \ # temperature when sampling
    --gpu_id 3 \ # GPU to use
    --prompt_mode single_md5 \
    --prompt log/save/codeparrot/codeparrot-small-temp1.0-len512-k40/analyze/extract_prompt.json \ # The the path to a file containing the prompt(i.e, the path of extract_prompt.json)
    ----prompt_hash  eb17e3d78803b21a4f6144d2f8b639edf3ebf5e3 \ #The prompt  sha1_hash in the file
    --internet-sampling
```

### the results of sampling with prompts
This command will result in a directory `extract/results/codeparrot/codeparrot-small-temp1.0-len512-k40/eb17e3d78803b21a4f6144d2f8b639edf3ebf5e3`, which contains a directory `internet`. This subdirectory will have many files, `1`, `2`, ..., `20000`; each file is an output from the model and `prompts.txt` contains the used prompts.

For the humans defined prompts, the `eb17e3d78803b21a4f6144d2f8b639edf3ebf5e3` in the dictionary path will be replaced by the sha hash caculted from the  `args.prompt`

🛎️: The outputs  remove the prompts text so if you want to do further analyze, you need to add them back to the top if you need. 