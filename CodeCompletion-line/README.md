# Evaluating CodeGPT to obtain model output

Here is the introduction and pipeline for line level code completion task.


### Prepare dataset
```
$ cd /workspace/CodeCompletion-line/dataset/javaCorpus
$ python process.py
```

### Infer the results
Infer the results of surrogate models:
```
$ cd /workspace/CodeCompletion-line/code
$ bash infer.sh # can change `MODEL` in [microsoft/CodeGPT-small-java,gpt2] 
```
The result is saved in `CodeCompletion-line/dataset/javaCorpus/0.01/10/test_CodeGPT-small-java-adaptedGPT2_surrogate_infer.txt`

```
$ bash infer_transformer.sh 
$ bash infer_lstm.sh
```

Infer the results of CodeGPT with top-k decoding methods:
```
$ bash infer_victim_fix_K.sh # get the results with different K in top-k
$ bash infer_victim_fix_T.sh # get the results with different temperature in top-k
```

Infer the results of victim models:
```
$ bash infer_victim.sh
```