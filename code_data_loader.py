"""
Author: Xiangyi Luo
"""
from gensim.models import KeyedVectors

# Loading from saved word embeddings
wv = KeyedVectors.load("wordvectors.kv", mmap='r')
print(wv['computer'])