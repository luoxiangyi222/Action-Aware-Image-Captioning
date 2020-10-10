"""
Author: Xiangyi Luo

Code or text pre-processing
"""

import nltk
from functools import lru_cache
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from code.data_loader import DataLoader
import tensorflow as tf


class CaptionTokenizer:
    def __init__(self):
        self.top_k = 5000
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.top_k,
                                                               oov_token="<unk>",
                                                               filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')

        self.data_loader = DataLoader()
        self.corpus = []

        # get all captions
        self.get_all_captions()

    def get_all_captions(self):
        self.data_loader.load_ocr_subtitle_timestamp()
        caption_dict = self.data_loader.subtitle_dict.copy()
        for video_num, v_dict in enumerate(caption_dict):
            video_captions = v_dict.values()
            self.corpus.extend(video_captions)

    def preprocessing(self):

        pass


cc = CaptionTokenizer()
cc.get_all_captions()


class CodePreprocessor:
    def __init__(self):
        # Stemming is the most time-consuming part of the indexing process, we attach a lru_cache to the stermmer
        # which will store upto 100000 stemmed forms and reuse them when possible instead of applying the
        # stemming algorithm.
        self.tokenize = nltk.tokenize.WordPunctTokenizer().tokenize
        self.stemmer = lru_cache(maxsize=100000)(SnowballStemmer('english').stem)

    def preprocessing(self, lines):

        corpus = []

        for i in range(len(lines)):
            # 1. tokenization
            tokens = self.tokenize(lines[i])

            # 2. Normalization: remove punctuation, to lower case
            # remove punctuation
            tokens = [token for token in tokens if token.isalnum()]

            # to lower
            tokens = [token.lower() for token in tokens]

            # 3. remove stop words

            stop_words = set(stopwords.words('english'))
            tokens = [w for w in tokens if w not in stop_words]

            # 4 stemmming
            tokens = [self.stemmer(token) for token in tokens]

            corpus.append(tokens)

        return corpus
