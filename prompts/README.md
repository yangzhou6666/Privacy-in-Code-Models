**This folder is used for conditional sampling.**
- [Download the dataset](#download-the-dataset)
- [Parse the prompts](#parse-the-prompts)
- [Conditional sampling](#conditional-sampling)
- [Analyze the results](#analyze-the-results)

# Download the dataset
We use the validation dataset of `codeparrot` to simulate real programming process. We can download the dataset using the following code:
```python
# download and save the dataset
from datasets import load_dataset
import pandas as pd
raw_datasets = load_dataset("codeparrot/codeparrot-valid-v2-near-dedup")
df = pd.DataFrame(raw_datasets)
df.to_csv('prompt_data-train.csv', index=False)

```
# Parse the prompts
Once we get the dataset, we assume that the code till the first function defenation appears is the prompt we should use. We get such prompt by running:
```shell
python run_func_prompts.py
```
it will generate the `prompts_paser_func.json`, where the `key` is the sha-1 hash value of the prompts and the `value` is the prompt.
# Conditional sampling
We use the prompts to do the conditional sampling using following commands:
```shell
$ cd random_sample_prompt_dev
$ python run_func_prompt_extract.py \
--model codeparrot/codeparrot  \ # the model used for conditional sampling
--sample_number 20000 \ # the toatl sample number
--batch-size 32     \ # the batch size
--seq_len 512     \ # the sequence length of sampling outputs
--top_k 40     \
--temperature 1.0     \
--gpu_id 1 \ # the gpu id
--sample_prompt_size 200 \ # the prompts number we use, i.e., for each prompt, the sampling number is sample_number/sample_prompt_size
--sample_mode per_prompt
```
# Analyze the results
After sampling, we will get `#sample_prompt_size` folders and we need to combine them together for further analyze.
We run:
```shell
python combine.py
```
to combine them all togther under the `ffffffffffffffffffffffffffffffffffffffff` folder.
Simiar to the [two step sampling](../README.md#two-step-sampling), we run the following commands to analyze the results. The detailed explanations can be referred [here](../README.md)
```shell
#merge
python ../../extract/merge.py 
    --model codeparrot/codeparrot \
    --top_k 40 \
    --temperature 1.0 \
    --seq_len 512\
    --prompt_mode direct_prompt \
    --prompt ffffffffffffffffffffffffffffffffffffffff \
    --internet-sampling 

#get cloned log
python ../../clone/scripts.py 
    --model codeparrot/codeparrot \
    --top_k 40 \
    --temperature 1.0 \
    --seq_len 512 \
    --tool_path ../../clone/simian-2.5.10.jar \
    --prompt_mode direct_prompt \
    --prompt ffffffffffffffffffffffffffffffffffffffff \
    --internet-sampling 

#analyze
cd log 
python analyze.py \
    --model codeparrot/codeparrot \
    --top_k 40 \
    --temperature 1.0 \
    --seq_len 512 \
    --prompt_mode direct_prompt \
    --prompt ffffffffffffffffffffffffffffffffffffffff \
    --internet-sampling \
    --file_end_number 20000 \
    --file_begin_number 1

# sample for labelling
python sample.py \
    --model codeparrot/codeparrot \
    --top_k 40 \
    --temperature 1.0 \
    --seq_len 512 \
    --prompt_mode direct_prompt \
    --prompt ffffffffffffffffffffffffffffffffffffffff \
    --internet-sampling \
    --file_end_number 20000 \
    --file_begin_number 1

```