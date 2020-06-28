#!/usr/bin/env python
# coding: utf-8

import os
import configparser
import json
import pyconll
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import InputLayer, Dense, LSTM, Bidirectional
import tensorflow as tf

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
data_train = pyconll.load_from_file(config.get('data','conllu_train_file'))
data_val = pyconll.load_from_file(config.get('data','conllu_val_file'))

# Create a dictionary to encode tags into integers
tag_dict = functions.tag_encoding_dictionary(config.get('data','conllu_train_file'))

# Generate random vectors as embeddings for tags 'EOS', 'PAD', and 'OOV'
# list with elements: [oov_vec, eos_vec, pad_vec]
sequence_len = config.getint('model','sequence_len')
extra_embeddings = functions.generate_extra_embedding_vecs(sequence_len)

#save tag_dict and extra_embedding to use at test time
functions.save_obj(tag_dict, config.get('embeddings','tag_dict_file'))
functions.save_obj(extra_embeddings, config.get('embeddings','extra_embeddings_file'))

## POS-tag model (feature extraction (encoder) + classification (decoder))

# Encode conllu-formatted data into embeddings
[X_train, y_train] = functions.word2vec_data_encoding(data_train, word2vec_model, sequence_len, embedding_dim,
                                                      extra_embeddings, tag_dict)
[X_val, y_val] = functions.word2vec_data_encoding(data_val, word2vec_model, sequence_len, embedding_dim,
                                                  extra_embeddings, tag_dict)

# Model definition:
print("Setting up model...")
model = Sequential()

num_tags = len(tag_dict)
model.add(InputLayer(input_shape=(sequence_len, embedding_dim)))
model.add(Bidirectional(LSTM(config.getint('model','hidden_units'), return_sequences=True)))
model.add(Dense(num_tags, activation='softmax'))  # Dense can handle 3D input too now
print("Done.")

print("Compiling...")
model.compile(loss=config.get("training","loss"),
              optimizer=config.get("training","optimizer"),
              metrics=json.loads(config.get("evaluation","metrics")))
print("Done.")
model.summary()

# Model training:
print("Training model...")
model.fit(X_train, np_utils.to_categorical(y_train, num_tags), batch_size=config.getint('training','batch_size'),
          epochs=config.getint('training','epochs'),
          validation_data=(X_val, np_utils.to_categorical(y_val, num_tags)))
print("Done.")

#Save model
model_file = config.get('model','filename') + ".h5"
print("Model saved as {:s}".format(model_file))
model.save(model_file)
