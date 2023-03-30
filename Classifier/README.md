# pipline
train and evaluate the `Gotcha`
```
$ cd ./code
$ bash pipline.sh # modify `SURROGATE_MODEL` in [microsoft/CodeGPT-small-java,microsoft/CodeGPT-small-java-adaptedGPT2,gpt2,rnn,transformer]
```
train and evaluate the ablation study of  `Gotcha`
```
$ bash pipline_ablation.sh # modify `ablation_mode` in [#no_title,no_text,no_code]
# no_title: w/o input
# no_text: w/o ground truth
#no_code:  w/o ground prediction
```
evaluate `Gotcha` with top-k decoding methods
```
$ bash mia_3_component_topk_tempreature.sh 
```
evaluate tradition methods(i.e., Naive Bayes/Decision Tree/Nearest Neighbor/Multi-layer Perceptron/Deep Neural Network)
```
$ bash mia_tradition.sh
```
