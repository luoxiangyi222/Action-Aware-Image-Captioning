"""
Author: Xiangyi Luo

This module load saved date for future analysis.
"""
import numpy as np
from gensim.models import KeyedVectors

# Loading from saved word embeddings
code_vectors = KeyedVectors.load("wordvectors.kv", mmap='r')
print(code_vectors['computer'])


# Loading one hot coding of actions
# first column is second, all rest are one-hot encoding of action at that particular second
actions_encoding = np.load('./../dataset/one_hot_action/8_0.npy')
print(actions_encoding)

