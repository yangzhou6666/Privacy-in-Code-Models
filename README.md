# Overview
This repository is the replication package for the submission for ICSE 2024: **Title: Gotcha! This Model Uses My Code! Evaluating Membership Leakage Risks in Code Models**.

Overall, this repository consists of the following steps:

0. Configurating the environment
1. Training CodeGPT for code completion
2. Evaluating CodeGPT to obtain the model output
3. Training and Evaluating the membership inference Classifiers.

# Declaration

> <span style="background-color: #ffcccc;">The replication package is intended for academic and research purposes only. We do not condone or support the use of the replication package for malicious purposes, e.g., operating membership inference attacks on other code models.</span>



# Environment Configuration

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
## Update and install the dependencies
Inside the Docker container, please run the following scripts to install necessary dependencies.
```
apt-get update
apt-get install wget
```

# Training CodeGPT for code completion

Refer to `CodeCompletion-token/README.md` for instructions.

# Evaluating CodeGPT to obtain model output

Refer to `CodeCompletion-line/README.md` for instructions.

# Train Membership Inference Attack Classifier 

Refer to `Classifier/README.md` for instructions.
