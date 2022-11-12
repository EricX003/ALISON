import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
import gc

from nltk import pos_tag
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

from random import random
import torch
'''
from transformers import BartForConditionalGeneration, BartTokenizer
'''
import csv
import time

from nltk.util import ngrams
from collections import Counter
import heapq

from string import punctuation
from nltk.corpus import stopwords

import sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import preprocessing
import xgboost as xgb

#Math stuff
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

#Torch stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as tf
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import ReduceLROnPlateau

import copy
import csv

from torchvision import transforms
from sklearn import metrics
from tqdm import trange
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import normalized_mutual_info_score

import os
import random
from torch.utils.data import Dataset

import warnings
warnings.filterwarnings("ignore")

tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
cost = lambda x, t: (2*((x-t)/(x+t)))**2

idx = 0
temp = dict()
offset = 65
for tag in tags:
    if offset + idx == 90:
        offset = 72
    temp[tag] = chr(offset + idx)
    idx += 1
tags = temp

def return_best_pos_n_grams(n, L, text):

    words = word_tokenize(text)
    pos_tokens = pos_tag(words)

    text = [tup[1] for tup in pos_tokens]
    text = ' '.join(text)

    n_grams = ngrams(text, n)

    data = dict(Counter(n_grams))

    print('Number n-grams: ', len(data))

    list_ngrams = heapq.nlargest(L, data.keys(), key=lambda k: data[k])
    return list_ngrams


def return_best_n_grams(n, L, text):
    bigrams = ngrams(text, n)

    data = dict(Counter(bigrams))
    list_ngrams = heapq.nlargest(L, data.keys(), key=lambda k: data[k])
    return list_ngrams

def find_freq_n_gram_in_txt(text, pos_text, n_gram_lengths, n_grams, pos_n_gram_lengths, pos_n_grams):

    to_ret = []

    for idx in range(len(n_grams)):
        num_ngrams = len(Counter(ngrams(text, n_gram_lengths[idx])))

        for n_gram in n_grams[idx]:
            to_ret.append(text.count(''.join(n_gram))/num_ngrams)

    for idx in range(len(pos_n_grams)):
        num_pos_ngrams = len(Counter(ngrams(pos_text, pos_n_gram_lengths[idx])))

        for pos_n_gram in pos_n_grams[idx]:
            to_ret.append(pos_text.count(''.join(pos_n_gram))/num_pos_ngrams)

    return to_ret

def get_top_idx(freqs, k):

    freqs = [(idx, freqs[idx]) for idx in range(len(freqs))]
    freqs.sort(key = lambda val: val[1], reverse = True)

    return [freqs[idx][0] for idx in range(k)]

def tag_data(data):
    to_char = lambda x: tags[x] if x in tags else x
    token_and_tag = lambda text: [tup[1] for tup in pos_tag(word_tokenize(text))]
    token_tag_join = lambda text: ''.join([to_char(tag) for tag in token_and_tag(text)])

    data['POS Tagged'] = data['Generation'].apply(token_tag_join)

    return data

def preprocess_data(data, num_authors, num_total, tag = True):

    print('------------', '\n', 'Tagging...')
    data = tag_data(data) if tag else data

    print('------------', '\n', 'Counting and aggregating texts...')
    number_texts = [0 for idx in range(num_total)]

    texts = ['' for idx in range(num_total)]
    pos_texts = ['' for idx in range(num_total)]

    stop_words = set(stopwords.words('english'))

    for index, row in data.iterrows():
        number_texts[int(row[0])] += 1
        filtered_sentence = row[1]
        filtered_pos_sentence = row[2]
        '''
        filtered_sentence = ' '.join([w for w in filtered_sentence.split() if not w in stop_words])
        filtered_sentence = ''.join([w for w in filtered_sentence if w not in set(punctuation)])
        '''
        filtered_sentence = filtered_sentence.strip()
        texts[int(row[0])] += ' ' + filtered_sentence
        pos_texts[int(row[0])] += filtered_pos_sentence

    print(number_texts)

    top_idxs = get_top_idx(number_texts, num_authors)

    class_weights = [number_texts[author] for author in top_idxs]

    total  = [texts[idx] for idx in top_idxs]
    total = ' '.join(total)
    pos_total = [pos_texts[idx] for idx in top_idxs]
    pos_total = ''.join(pos_total)

    temp = dict()
    for idx in range(len(top_idxs)):
        temp[top_idxs[idx]] = idx

    top_idxs = temp

    print('------------', '\n', 'Preprocessing complete!')

    return top_idxs, total, pos_total

def gen_data(data, lengths, pos_lengths, top_idxs, total, pos_total):

    print('------------', '\n', 'Generating n-grams...')

    n_grams = [return_best_n_grams(n, 150, total) for n in lengths]

    print('------------', '\n', 'Generating POS n-grams...')

    pos_n_grams = [return_best_n_grams(n, 150, pos_total) for n in pos_lengths]

    print('------------', '\n', 'Generating data...')
    X = []
    y = []
    for index, row in data.iterrows():
        if int(row[0]) in top_idxs:
            y.append(top_idxs[int(row[0])])
            X.append(find_freq_n_gram_in_txt(row[1], row[2], lengths, n_grams, pos_lengths, pos_n_grams))

    X = np.array(X)
    y = np.array(y)

    return X, y #sklearn.model_selection.train_test_split(X, y, test_size = 0.15)

from sklearn.metrics import f1_score
from sklearn.

def train_and_test(X_train, X_test, y_train, y_test):
    print('------------', '\n', 'Training...')
    #model = xgb.XGBClassifier().fit(X_train, y_train)
    model = LogisticRegression(random_state=0, max_iter = 10000).fit(X_train, y_train)
    print('------------', '\n', 'Validating')
    y_pred = model.predict(X_test)
    print('------------')
    print('Accuracy: ', accuracy_score(y_test, y_pred))
    #print('F1 Score', f1_score(y_test, y_pred))
    print('------------', '\n', 'Complete!', '\n', '------------')
    return y_test, y_pred

gc.collect()

train = pd.read_csv('/content/gdrive/MyDrive/PSU REU/Data/TuringBench/ProcessedAA/train.csv')
test = pd.read_csv('/content/gdrive/MyDrive/PSU REU/Data/TuringBench/ProcessedAA/test.csv')
val = pd.read_csv('/content/gdrive/MyDrive/PSU REU/Data/TuringBench/ProcessedAA/val.csv')

train = train.drop(['Unnamed: 0'], axis = 1)
test = test.drop(['Unnamed: 0'], axis = 1)
val = val.drop(['Unnamed: 0'], axis = 1)


top_idxs, total, pos_total = preprocess_data(train, 20, 20, tag = False)

print(top_idxs)

train = pd.concat([train, val])

X_train, y_train = gen_data(train, [1, 2, 3, 4], [1, 2, 3, 4], top_idxs, total, pos_total)
X_test, y_test = gen_data(test, [1, 2, 3, 4], [1, 2, 3, 4], top_idxs, total, pos_total)

y_test, y_pred = train_and_test(X_train, X_test, y_train, y_test)
