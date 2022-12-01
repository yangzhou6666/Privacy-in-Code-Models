# Overview



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


# Code Completion on Tokens 

Refer to `CodeCompletion-token/README.md` for instructions.

# Code Completion on Lines 

Refer to `CodeCompletion-line/README.md` for instructions.