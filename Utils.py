import numpy as np
import pandas as pd
import datetime

import copy
import csv
import os
import time
import pickle
import itertools

from pandas import DataFrame

import nltk
from nltk import skipgrams
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.util import ngrams
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
from string import punctuation
from nltk.corpus import stopwords

import gc
from random import random
import time

from collections import Counter
import heapq

import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

from functools import partial
from tqdm import tqdm
tqdm = partial(tqdm, position=0, leave=True)

import warnings
warnings.filterwarnings("ignore")

from itertools import chain
import copy

import argparse

tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
idx = 0
temp = dict()
offset = 65
for tag in tags:
    if offset + idx == 90:
        offset = 72
    temp[tag] = chr(offset + idx)
    idx += 1
tags = temp

to_char = lambda x: tags[x] if x in tags else x
token_and_tag = lambda text: [tup[1] for tup in pos_tag(tokenize(text))]
token_tag_join = lambda text: ''.join([to_char(tag) for tag in token_and_tag(text)])

def tag(texts):
    return [token_tag_join(text) for text in texts]

def countSkip(skipgram, texts):

    total = 0
    m = len(skipgram)

    for text in texts:

        n = len(text)

        mat = [[0 for i in range(n + 1)] for j in range(m + 1)]
        for j in range(n + 1):
            mat[0][j] = 1

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                mat[i][j] = mat[i][j - 1]

                if skipgram[i - 1] == text[j - 1]:
                    mat[i][j] += mat[i - 1][j - 1]

        total += mat[m][n]

    return total

def get_skipgrams(text, n, k):
    if n > 1:
        ans = [skipgram for skipgram in skipgrams(text, n, k)]
    else:
        ans = ngrams(text, n)
    return ans

def return_best_pos_n_grams(n, L, pos_texts):
    n_grams = ngrams(pos_texts, n)

    data = dict(Counter(n_grams))
    list_ngrams = heapq.nlargest(L, data.keys(), key=lambda k: data[k])
    return list_ngrams

def return_best_word_n_grams(n, L, tokens):

    all_ngrams = ngrams(tokens, n)

    data = dict(Counter(all_ngrams))
    list_ngrams = heapq.nlargest(L, data.keys(), key=lambda k: data[k])
    return list_ngrams

def return_best_n_grams(n, L, text):

    n_grams = ngrams(text, n)

    data = dict(Counter(n_grams))
    list_ngrams = heapq.nlargest(L, data.keys(), key=lambda k: data[k])
    return list_ngrams

def ngram_rep(text, pos_text, features):

    to_ret = []
    ret_idx = 0

    for idx in range(len(features[0])):
        num_ngrams = len(Counter(ngrams(text, len(features[0][idx][0]))))

        for n_gram in features[0][idx]:
            to_ret.append(text.count(''.join(n_gram)) / num_ngrams if num_ngrams != 0 else 0)

    for idx in range(len(features[1])):
        num_pos_ngrams = len(Counter(ngrams(pos_text, len(features[1][idx][0]))))

        for pos_n_gram in features[1][idx]:
            to_ret.append(pos_text.count(''.join(pos_n_gram)) / num_pos_ngrams if num_pos_ngrams != 0 else 0)

    words = tokenize(text)
    spaced_text = ' '.join(words)
    for idx in range(len(features[2])):
        num_word_ngrams = len(Counter(ngrams(words, len(features[2][idx][0]))))

        for word_ngram in features[2][idx]:
            to_ret.append(spaced_text.count(' '.join(word_ngram)) / num_word_ngrams if num_word_ngrams != 0 else 0)

    return to_ret


def tokenize(text):
    ret = []
    for sent in sent_tokenize(text):
        ret.extend(word_tokenize(sent))
    return ret
