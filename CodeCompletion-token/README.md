# Training CodeGPT for code completion




## Task Definition

Predict next code token given context of previous tokens. Models are evaluated by token level accuracy.

Code completion is a one of the most widely used features in software development through IDEs. An effective code completion tool could improve software developers' productivity. We provide code completion evaluation tasks in two granularities -- token level and line level. Here we introduce token level code completion. Token level task is analogous to language modeling. Models should have be able to predict the next token in arbitary types.


## Dataset

### Github Java Corpus

We use java corpus dataset mined by Allamanis and Sutton, in their MSR 2013 paper [Mining Source Code Repositories at Massive Scale using Language Modeling](https://homepages.inf.ed.ac.uk/csutton/publications/msr2013.pdf). We follow the same split and preprocessing in Karampatsis's ICSE 2020 paper [Big Code != Big Vocabulary: Open-Vocabulary Models for Source Code](http://homepages.inf.ed.ac.uk/s1467463/documents/icse20-main-1325.pdf).

To download the preprocessed dataset, navigate to `dataset/javaCorpus` directory, and run
```shell
bash download.sh
python preprocess.py --base_dir=token_completion --output_dir=token_completion
```

### Data Preprocessing

- Tokenization is applied since we focus the token-level completion. 
- We normalize uncommon literals for better user experience. Developers sometimes leave their names, IP address, phone numbers in their codes, and we don't encourage models to focus on these string or numeric literals. So we normalized these literals by some special tokens. Considering that frequently used literals may contain useful information, e.g. "\_\_main\_\_" or "utf-8", we preserve the 200 most frequent string and 30 most frequent numeric literals. These literals will be normalized by tokens in `"<STR_LIT:utf-8>"` format, while uncommon literals are replaced by `<STR_LIT>` or `<NUM_LIT>`. 
- We add `<s>` and `</s>` to indicate the start and the end of one piece of code. 
- `<EOL>` is added in python corpus to mark the ending of a line since in python there is no `;` or `}` to mark the ending of a statement like in java.

### Data Format

Code corpus are saved in txt format files. one line is a tokenized code snippets:
```
<s> from __future__ import unicode_literals <EOL> from django . db import models , migrations <EOL> class Migration ( migrations . Migration ) : <EOL> dependencies = [ <EOL> ] <EOL> operations = [ <EOL> migrations . CreateModel ( <EOL> name = '<STR_LIT>' , <EOL> fields = [ <EOL> ( '<STR_LIT:id>' , models . AutoField ( verbose_name = '<STR_LIT>' , serialize = False , auto_created = True , primary_key = True ) ) , <EOL> ( '<STR_LIT:name>' , models . CharField ( help_text = b'<STR_LIT>' , max_length = <NUM_LIT> ) ) , <EOL> ( '<STR_LIT:image>' , models . ImageField ( help_text = b'<STR_LIT>' , null = True , upload_to = b'<STR_LIT>' , blank = True ) ) , <EOL> ] , <EOL> options = { <EOL> '<STR_LIT>' : ( '<STR_LIT:name>' , ) , <EOL> '<STR_LIT>' : '<STR_LIT>' , <EOL> } , <EOL> bases = ( models . Model , ) , <EOL> ) , <EOL> ] </s>
```









## Pipeline

Let's first train the victim model. 
```
$ bash pipline_java_victim.sh # get the victim model
```
The victim model is trained on the whole dataset (100%).
You can change the `PRETRAINDIR=bigcode/santacoder` to other models, e.g., `bigcode/starcoderbase-1b` to use different models as the base model.


Then, we train the surrogate model on part of the training data (10% and 20%). We have three types of surrogate models: (1) GPT-2 based, including GPT-2 and microsoft/CodeGPT-small-java, (2) non-pretrained transformer, (3) LSTM. The following three files correspond to these types. 
```
$ cd /workspace/CodeCompletion-token/code
$ bash pipline_java.sh # can change `PRETRAINDIR` in [microsoft/CodeGPT-small-java,gpt2] to train different surrogate model
$ bash pipline_java_transformer.sh
$ bash pipline_java_lstm.sh
```


