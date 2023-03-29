# CodeXGLUE -- Code Completion (line level)

Here is the introduction and pipeline for line level code completion task.

## Task Definition

Complete the unfinished line given previous context. Models are evaluated by exact match and edit similarity.

We propose line completion task to test model's ability to autocomplete a line. Majority code completion systems behave well in token level completion, but fail in completing an unfinished line like a method call with specific parameters, a function signature, a loop condition, a variable definition and so on. When a software develop finish one or more tokens of the current line, the line level completion model is expected to generate the entire line of syntactically correct code.

## Dataset

Line level code completion task shares the train/dev dataset with token level completion. After training a model on CodeCompletion-token, you could directly use it to test on line-level completion. 



### Github Java Corpus line completion test set

We create test set from Github Java Corpus token level code comepltion test set. In the same way as for Python, we randomly cut a file as two parts. The former part is the input context, outputs is the code sequence in the latter part until the first ; or \{ and \} token (including ; or \} token, but excluding \{ token).

Test set is already at `dataset/javaCorpus/line_completion/test.json`.

### Data Format

Data is saved in json lines format files. Each line is a json object. To be consistent with token level code completion, codes have been tokenized. Here is an example of one line:
```
{
  "input": "<s> from __future__ import absolute_import , division , print_function <EOL> from . _ithreads import AlreadyQuit <EOL> class Quit ( object ) : <EOL>",
  "gt": "def __init__ ( self ) :"
}
```


### Prepare dataset
```
$ cd /workspace/CodeCompletion-line/dataset/javaCorpus
$ python process.py
```

### Infer the results
```
$ cd /workspace/CodeCompletion-line/code
$ bash infer.sh 
$ bash infer_victim.sh
```