# POS-tagging project

This repository involves a part-of-speech (POS) tagging task, i.e., the words in a given text are automatically labeled 
(using a classifier) as a specific POS (e.g. adjective, verb,...) based on the word's definition and its context.  The solution implemented
is partly based on the solution proposed in [1], which uses a bidirectional long short-term memory (bi-LSTM) neural network for the classification.

### 1. Data 

This task uses for training and evaluation the [Universal Dependencies v1.2 Treebank for English](https://github.com/ufal/rh_nntagging/tree/master/data/ud-1.2/en) [2] This data corpus includes training, development and evaluation sets, which are in conllu format.

A pre-trained model of word embedding vector model is also used for word representation: [Glove 6B (tokens) 100d (vectors)](https://github.com/stanfordnlp/GloVe) [3]. 

### 2. Word representations (feature extraction)

For input to the model the data was converted into word vectors, using the available pre-trained model.
Then word vectors were set to a fixed length (in the ran experiment: 100), and were zero-padded if needed.

### 3. Model

A bi-LSTM neural network is used for the classification.

#### 3.1 Training

In the experiment ran, training involved 20 runs (epochs), and using batch sizes of 32 sentences at each time. The optimization algorithm was (mini-batch) stochastic gradient descent (SGD), and 
the loss function was  categorical cross-entropy.
 
All the settings of the experiment are in the file config.ini.

#### 3.2 Evaluation

Accuracy was the metric used for evaluation. The results obtained in the experiment were:

```bash
Test set evaluation:
accuracy: 0.9314010739326477
```

### 2. Usage

* For training the model, run: 

```bash
$ ./run_train_postagging.py
```
* For testing the model, run:
 ```bash
$ ./run_test_postagging.py
```
In case you only want to run the test step, without prior training, the repository includes files extracted from the training step (e.g. the trained model), and which are needed during the test step.  

Note : The settings of the task can be changed in the configuration file (config.ini).

### 3. Pre-requisites

This repository requires Python 3.6 or greater. The required Python libraries can be installed from the provided file **requirements.txt** via terminal as:
```bash
$ pip install -r requirements.txt
```
NOTE: GPU computing support not included in these pre-requisites.

---
#### References:
[1] B. Plank, A. Søgaard, and Y. Goldberg. "Multilingual part-of-speech tagging with bidirectional long 
short-term memory models and auxiliary loss." In Proc. of Association for Computational Linguistics 
(ACL), pp. 412-418, 2016.

[2] N. Silveira, T. Dozat, M.C. De Marneffe, S. Bowman, M. Connor, J. Bauer and C.D. Manning. "A Gold Standard Dependency Corpus for English." In Proc. of Conference on Language
    Resources and Evaluation (LREC), pp. 2897-2904, 2014.
    
[3] J. Pennington, R. Socher, and C.D. Manning. "Glove: Global vectors for word representation." Proc. of Empirical Methods in Natural Language Processing (EMNLP), 2014.
