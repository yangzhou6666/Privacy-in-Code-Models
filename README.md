# Overview

This repository 💻 is for our ASE submission 📚 that explores 🔍 the memorization 🧠 in code models.



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

# Sample Outputs from code models

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

```
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

Execute the following command to download the `codeparrot/codeparrot-clean` dataset, the training data for `codeparrot/codeparrot` models.
```
python cache_data.py 2>&1 | tee download.log
```

<span style="background-color: yellow">Note: The dataset is over 50GB, so this process may takes a while, depending on your network status.</span>




