import nltk
from functools import lru_cache
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


class CodePreprocessor:
    def __init__(self):
        # Stemming is the most time-consuming part of the indexing process, we attach a lru_cache to the stermmer
        # which will store upto 100000 stemmed forms and reuse them when possible instead of applying the
        # stemming algorithm.
        self.tokenize = nltk.tokenize.WordPunctTokenizer().tokenize
        self.stemmer = lru_cache(maxsize=100000)(SnowballStemmer('english').stem)

    def __call__(self, lines):

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
