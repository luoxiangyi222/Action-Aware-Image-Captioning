"""
Author: Xiangyi Luo

This module load saved date for future analysis.
"""
import numpy as np
import glob
from gensim.models import KeyedVectors


class DataLoader(object):
    def __init__(self):
        self.code_vectors = None
        self.actions_dict = {}

    def load_code_vectors(self):
        # Loading from saved word embeddings
        self.code_vectors = KeyedVectors.load("wordvectors.kv", mmap='r')

    def load_action_one_hot(self):

        # Loading one hot coding of actions
        # first column is second, all rest are one-hot encoding of action at that particular second

        npy_path = './../dataset/one_hot_action/*.npy'

        # find all .npy files
        npy_file_list = glob.glob(npy_path)

        for npy_file in npy_file_list:

            file_num_str = npy_file.split('/')[-1][:-4]
            actions_encoding = np.load(npy_file)
            self.actions_dict[file_num_str] = actions_encoding

