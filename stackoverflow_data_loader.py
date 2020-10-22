"""
Author: Xiangyi Luo
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import numpy as np
import math
from collections import defaultdict, OrderedDict

from data_preprocessing import CodePreprocessor


class SOFDataLoader(object):
    def __init__(self):
        self.vectorizer = CountVectorizer()

        self.doc_count = 2
        self.parsed_data_path = './../dataset/stackoverflow/parsed_stackoverflow.csv'
        self.sof_df = None
        self.corpus = ['xiangyi went to dance byte byte bytes', 'are your a doctor here byte']
        self.vectorized_corpus = None
        self.tokens_list = None

        self.load_raw_corpus()

        self.prepro = CodePreprocessor()

        # for bm25
        self.doc_freq = None
        self.avg_doc_length = None
        self.doc_lengths = None

        self.get_vector_representation()


    def load_raw_corpus(self):
        self.sof_df = pd.read_csv(self.parsed_data_path)
        self.sof_df = self.sof_df.dropna(subset=['text']).reset_index(drop=True)

        for text in self.sof_df['text']:

            self.corpus.append(text)

        self.doc_count = len(self.corpus)

    def get_vector_representation(self):
        """
        This function converts the documents to tf-idf vectors and returns a sparse matrix representation of the data.
        You can change any of the settings of CountVectorizer.
        """

        self.corpus = self.prepro.preprocessing(self.corpus)
        self.corpus = [' '.join(line) for line in self.corpus]

        self.vectorizer.fit(self.corpus)
        self.vectorized_corpus = self.vectorizer.transform(self.corpus)

        doc_freq = self.vectorized_corpus.toarray().copy()
        doc_freq[doc_freq > 0] = 1
        self.doc_freq = doc_freq.sum(0)
        self.avg_doc_length = self.vectorized_corpus.sum(1).mean()
        self.doc_lengths = self.vectorized_corpus.sum(1).A1
        self.tokens_list = self.vectorizer.get_feature_names()

    def token_to_index(self, token: str):
        return self.tokens_list.index(token)

    def compute_bm25(self, query: str, b=0.75, k1=1.6, top_k=10):
        query = self.prepro.preprocessing([query])[0]
        bm25_scores = defaultdict(lambda: 0)

        for i, qi in enumerate(query):
            f_qi = self.doc_freq[0]
            IDF_qi = math.log(1 + (self.doc_count - f_qi + 0.5) / (f_qi + 0.5))

            for doc_num in range(self.doc_count):
                q_doc_freq = self.vectorized_corpus[(doc_num, self.token_to_index(qi))]
                doc_len = self.doc_lengths[doc_num]
                numerator = q_doc_freq * (k1 + 1)
                denominator = q_doc_freq + k1 * (1 - b + b * (doc_len/self.avg_doc_length))
                score = IDF_qi * numerator / denominator
                bm25_scores[doc_num] += score

        bm25_scores = {k: v for k, v in sorted(bm25_scores.items(), key=lambda item: item[1], reverse=True)}
        rank_indice = list(bm25_scores.keys())[0:top_k]
        rank_top_k_df = self.sof_df.iloc[rank_indice, :]
        return rank_top_k_df


ddd = SOFDataLoader()
top_k = ddd.compute_bm25('byte dance')
print(top_k)
