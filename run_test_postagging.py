#!/usr/bin/env python
# coding: utf-8

import os
import configparser
import tensorflow as tf
import pyconll
from keras.utils import np_utils
from keras.models import load_model

import functions
########################################################################################################################

# Import settings
config = configparser.ConfigParser()
config.read('config.ini')

# bypass GPU, and use CPU
if config.getboolean('computation','use_gpu') is False:
    print("Using CPU for computations.")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    print("Using GPU for computations.")
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
########################################################################################################################

# Load pre-trained model for the embeddings
word2vec_model = functions.load_word2vecModel(config.get('embeddings','word2vec_file'))
embedding_dim = len(word2vec_model['the'])

# Load conllu-formated data (train, validation and test)
data_test = pyconll.load_from_file(config.get('data','conllu_test_file'))

# Load tag dictionary and extra_embeddings
tag_dict = functions.load_obj(config.get('embeddings','tag_dict_file'))
extra_embeddings = functions.load_obj(config.get('embeddings','extra_embeddings_file'))

## POS-tag model (feature extraction (encoder) + classification (decoder))

# Encode conllu-formatted data into embeddings
[X_test, y_test] = functions.word2vec_data_encoding(data_test, word2vec_model, config.getint('model','sequence_len'), embedding_dim,
                                                    extra_embeddings, tag_dict)

# Load trained model
model_file = config.get('model','filename') + ".h5"
model = load_model(model_file)

# Evaluation:
num_tags = len(tag_dict)
scores = model.evaluate(X_test, np_utils.to_categorical(y_test, num_tags))

print("Test set evaluation:")
for idx,score in enumerate(scores[1:]):
    print(f"{model.metrics_names[idx+1]}: {score}")
