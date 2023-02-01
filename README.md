# Overview



# Environment Configuration

## Conda

```Shell
pip install tqdm pandas
pip install transformers==3.0.2
python -m pip install huggingface_hub
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.1 -c pytorch
pip install accelerate
# You may need to update the pytorch installation command accordingly based on your CUDA version.
```




## Build Docker Image

```
docker build -f Dockerfile -t privacy-code .
```


## Create Docker Container

```
docker run --name=privacy-code --gpus all -it -v YOU_LOCAL_REPO_PATH:/Privacy-in-Code-Models privacy-code:latest
```

Example: 
```
docker run --name=privacy-code --gpus all -it -v /mnt/hdd1/zyang/Privacy-in-Code-Models:/Privacy-in-Code-Models privacy-code:latest
```

# Run


## Models

* facebook/incoder-6B
* facebook/incoder-1B
* Salesforce/codegen-350M-multi
* Salesforce/codegen-350M-nl
* Salesforce/codegen-350M-mono
* Salesforce/codegen-2B-multi
* Salesforce/codegen-2B-nl
* Salesforce/codegen-2B-mono
* Salesforce/codegen-6B-multi
* Salesforce/codegen-6B-nl
* Salesforce/codegen-6B-mono
* codeparrot/codeparrot-small
* codeparrot/codeparrot

You can `cd extract` and run `python cache_models.py` to download necessary models first.
You can also skip this step and download models as needed.


## Sample from Code Models

```bash
cd extract

python extract.py \
    --model Salesforce/codegen-350M-multi \
    --N 50000 \
    --batch-size 100 \
    --seq_len 512 \
    --top_k 5 \
    --gpu_id 4 &


python extract.py \
    --model Salesforce/codegen-350M-mono \
    --N 50000 \
    --batch-size 80 \
    --seq_len 768 \
    --top_k 40 \
    --gpu_id 7 &
```

```
python merge.py \
    --model Salesforce/codegen-350M-mono \
    --seq_len 256 &
```


## Memorization Analysis

```
python cache_data.py 2>&1 | tee download.log
```

## Personally Identifiable Information Analysis



