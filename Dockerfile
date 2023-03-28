FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel
RUN pip install transformers==4.27.2
RUN pip install fuzzywuzzy
RUN pip install javalang==0.13.0
