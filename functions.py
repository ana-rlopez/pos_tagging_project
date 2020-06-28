"""
This module includes NLP-related functions.
"""

import numpy as np
import pickle
import os

def save_obj(obj,name):
    """
    Save a variable (obj) in file given by name

    Args:
        obj : variable to save
        name [string]: filename where to save

    Returns
        None
    """
    os.makedirs(os.path.dirname(name), exist_ok=True)
    print("Saving object {:s}.pkl...".format(name))
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    print("Done.")
    return

def load_obj(name):
    """
    Load a saved variable in file given by name, and return it

    Args:
        name [string]: filename where variable is saved

    Returns
        obj: variable loaded
    """
    print("Loading object {:s}.pkl...".format(name))
    with open(name + '.pkl', 'rb') as f:
        obj = pickle.load(f)
        print("Done")
        return obj

def load_word2vecModel(word2vec_file):
    """
    Load a pretrained word2vec model and return it as variable.

    Args:
        word2vec_file (string): filename (with full path) of textfile containing the word2vec model

    Returns:
        w2v_model (dictionary): dictionary with words as keys, and embeddings as values
    """
    print("Loading word2vec model {:s}...".format(word2vec_file))

    w2v_model = {}
    with open(word2vec_file) as f:
        for line in f:
            word, wordVector = line.split(maxsplit=1)
            wordVector = np.fromstring(wordVector, 'f', sep=' ')
            w2v_model[word] = wordVector

    print("Done.")
    return w2v_model


def generate_extra_embedding_vecs(EMBEDDING_DIM,seed_state=42):
    """
    Generate random embedding vectors for the tags 'EOS' (end of sentence),
    'PAD' (for zero-padding) and 'OOV' (out of vocabulary).

    Args:
        EMBEDDING_DIM (int)
        (seed_state) (int): for the random generator, to have predictable values [default: 42]

    Returns:
        rand_embed_vecs (list of 3 elements, each one a random vector (with embedding size) for the
        tags: [oov, eos, pad])
    """

    np.random.seed(seed_state)
    oov_vec = np.random.normal(size=EMBEDDING_DIM)
    #oov_vec.shape

    eos_vec = np.random.normal(size=EMBEDDING_DIM)
    #eos_vec.shape

    pad_vec = np.random.normal(size=EMBEDDING_DIM)
    #pad_vec.shape

    return [oov_vec, eos_vec, pad_vec]

def tag_encoding_dictionary(conllu_file):
    """
    Create dictionary for encoding all the unique tags found in the conllu-formatted file into integers

    Args:
        conllu_file (string): filename (with whole path) of conllu-formatted file that we use

    Returns:
        tag_dict (dictionary): tag dictionary used for encoding
    """

      #get unique list of tags from file, bash (didn't manage from parser Pyconll),
    import subprocess

    tag_list = subprocess.check_output("awk '{ print $4 }' " +  conllu_file + " | sort | uniq", shell=True)
    tag_list = tag_list.decode().splitlines()
    tag_list[:] = [x for x in tag_list if x] #remove empty strings

    tag_dict = {}
    for int_code, tag in enumerate(tag_list):
        tag_dict[tag] = int_code

        tag_dict['EOS'] = int_code+1
        tag_dict['PAD'] = int_code+2

    return tag_dict

def word2vec_data_encoding(data,word2vec_model, MAX_SEQUENCE_LEN, EMBEDDING_DIM, extra_embeddings, tag_dict):
    """
    Encode input data into: 1) arrays of word2vec embeddings and 2) corresponding labels ('tags')

    Args:
        data (PyConll object, with N sentences): the data to be encoded
        (https://pyconll.readthedocs.io/en/stable/index.html)
        word2vec_model (dictionary, keys=words, values=embeddings): pretrained word2vec model used in encoding
        MAX_SEQUENNCE_LEN (int): maximum length of sentence (of training data)
        EMBEDDING_DIM (int): length of the embedding vectors of word2vec_model
        extra_embeddings (list of 3 embedding vectors): this list corresponds to embeddings of
       [OOV, EOS, PAD], respectively.

    Returns:
        sentences_X (numpy array of size (N x MAX_SEQUENCE_LEN x EMBEDDING_DIM): input data for the classifier
        tags_y (numpy array of size (N x MAX_SEQUENCE_LEN): output data (labels) for the classifier
    """
    try:
        len(word2vec_model['the']) == EMBEDDING_DIM
    except:
        raise Exception('Error: mismatch between EMBEDDING_DIM and dimension of word2vec_model vectors')
    print("Encoding data into embeddings...")

    oov_vec= extra_embeddings[0]
    eos_vec= extra_embeddings[1]
    pad_vec= extra_embeddings[2]

    N = len(data)

    sentences_X = np.empty((N, MAX_SEQUENCE_LEN,EMBEDDING_DIM))
    tags_y = np.empty((N, MAX_SEQUENCE_LEN))

    for idx_sentence,sentence in enumerate(data):
        #print(sentence)

        idx_eos = len(sentence)
        for idx_word, token in enumerate(sentence):

            if idx_word < (MAX_SEQUENCE_LEN-1):
                token_fixed = token.form.lower()

                if token_fixed in word2vec_model:
                    #print('in:')
                    #print(token_fixed)
                    sentences_X[idx_sentence,idx_word,:] = word2vec_model[token_fixed]
                    tags_y[idx_sentence,idx_word] = tag_dict[token.upos]
                else:
                    #print('OOV:')
                    #print(token_fixed)
                    sentences_X[idx_sentence,idx_word,:] = oov_vec
                    tags_y[idx_sentence,idx_word] = tag_dict[token.upos]
            else:
                idx_eos = idx_word
                break
        #print('EOS')
        sentences_X[idx_sentence,idx_eos] = eos_vec
        tags_y[idx_sentence,idx_eos] = tag_dict['EOS']

        #add zero-padding if necessary
        if idx_eos < MAX_SEQUENCE_LEN:
            sentences_X[(idx_sentence,range(idx_eos+1,MAX_SEQUENCE_LEN))]= pad_vec
            tags_y[(idx_sentence,range(idx_eos+1,MAX_SEQUENCE_LEN))] = tag_dict['PAD']
    print("Done.")
    return [sentences_X, tags_y]
