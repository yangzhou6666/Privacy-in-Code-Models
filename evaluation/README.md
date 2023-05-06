We try to answers in RQ3:*How to infer whether an output contains memorized
information?*

We can use the following commands to get the `table 4`
```shell
bash bash_eva.sh
bash bash_eva_window.sh
```
here are the explanations of the parameters:
```python
'''
model_1: the main model used to calculate the ppl/ppl-zlib ratio/average ppl
model_2: the model used to calculate the ppl-ppl ration with model_1
extract_n: the top-n to print in the log
output_n: the top-n to save in the files
extract_mode: ['small-first',large-first']: the model_1 is the small version or large verision
save_output: whether to save the output_n 
save_cleaned: whether to just save the memorized part of the output_n 
'''
```