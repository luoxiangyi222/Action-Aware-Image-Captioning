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

from code.code_data_preprocessing import CodePreprocessor

path = './../dataset/OCR/**/*.json'

# find all json files
json_file_list = glob.glob(path, recursive=False)

# build corpus
corpus = []
# code_preprocessing
# load json files
code_pre = CodePreprocessor()
for json_file in json_file_list:
    with open(json_file) as f:
        data = json.load(f)
        lines = data['lines']
        for line in lines:
            corpus.append(line['text'])

# convert to list of list of words
corpus = code_pre.__call__(corpus)


# Initialize a model
path = get_tmpfile("word2vec.model")
model = Word2Vec(common_texts, size=13, window=5, min_count=1, workers=4)
model.save("word2vec.model")

# Train model
model = Word2Vec.load("word2vec.model")
model.train(corpus, total_examples=len(corpus), epochs=100)

# Saving the word embeddings
model.wv.save("wordvectors.kv")


