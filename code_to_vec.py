"""
Author: Xiangyi Luo

This module runs  Word2Vec to learn word embeddings for code corpus.

Reference:
https://radimrehurek.com/gensim/models/word2vec.html

"""

import glob
import json

from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec

from code.data_preprocessing import CodePreprocessor

path = './../dataset/OCR/**/*.json'

# find all json files
json_file_list = glob.glob(path, recursive=False)

# build corpus
corpus = []

# code_preprocessing
code_pre = CodePreprocessor()

# load json files

for json_file in json_file_list:
    with open(json_file) as f:
        data = json.load(f)
        lines = data['lines']
        for line in lines:
            corpus.append(line['text'])

# ############ testing #######################
# file = './../dataset/OCR/8_102/00008.json'
# with open(file) as f:
#     data = json.load(f)
#     lines = data['lines']
#     for line in lines:
#         corpus.append(line['text'])
# #############################################

# convert to list of list of words

corpus = code_pre.preprocessing(corpus)
corpus = [x for x in corpus if x != []]  # drop empty list


# Initialize a model

model = Word2Vec(corpus, size=13, window=5, min_count=1, workers=4)

# Saving the word embeddings
model.save("word2vec.model")



