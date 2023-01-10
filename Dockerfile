FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
RUN apt-get update
RUN apt-get install wget
RUN pip install transformers==4.0.0
RUN pip install fuzzywuzzy
RUN pip install javalang==0.13.0