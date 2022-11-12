import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
import gc

from nltk import pos_tag
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

from random import random
import torch
!pip install transformers -q

import sklearn

from sklearn.metrics import accuracy_score

#Math stuff
import numpy as np

#Torch stuff
import torch
import torch.distributed as dist
from torch import Tensor
from torch.multiprocessing import Process

from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity

import os
import random
from torch.utils.data import Dataset
import time

import pickle
import itertools
from pandas import DataFrame

import torch

# Parallelize apply on Pandas
!pip install pandarallel -q
from pandarallel import pandarallel
pandarallel.initialize()

!pip install captum -q
import captum
from captum.attr import IntegratedGradients

from transformers import BertTokenizer, BertForMaskedLM

!pip install evaluate -q
import evaluate

!pip install sentencepiece -q

from sklearn.model_selection import train_test_split

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from absl import logging

import tensorflow as tf

import tensorflow_hub as hub
import matplotlib.pyplot as plt
import re
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity as cs

from captum.attr import IntegratedGradients
import torch.distributed as dist
from torch import Tensor
from torch.multiprocessing import Process

import operator

meteor = evaluate.load('meteor')

from transformers import AutoModelForSequenceClassification, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gc.collect()

parser = argparse.ArgumentParser()

parser.add_argument("--obs", "-o", help="Path to Texts for Obfuscation", default = './train.csv')
parser.add_argument("--features", "-t", help="Path to Testing Data", default = './test.csv')
parser.add_argument("--L", "-L", help="L, The Number of POS-ngrams to Replace", default = 20)
parser.add_argument("--c", "-c", help="c, The Length Scaling Constant", default = 0.35)
parser.add_argument("--min_length", "-ml", help="Number of Training Epochs", default = 0)
parser.add_argument("--IG_batch_size", "-igbs", help="Number of Samples to Process at", default = 128)
parser.add_argument("--IG_number_steps", "-igns", help="Number of Steps in IG Calculation", default = 500)
parser.add_argument("--step", "-s", help="Scheduler Step Size", default = 8)
parser.add_argument("--gamma", "-g", help="Scheduler Gamma Constant", default = 0.30)
parser.add_argument("--authors_total", "-at", help="Number of Total Authors in Corpus", default = 20)
parser.add_argument("--authors_to_keep", "-atk", help="Number of Authors to Retain (In Order of Document Frequency)", default = 20)
parser.add_argument("--trial_name", "tm", help="The Current Trial's Name (E.g. Dataset Name)")

parse.parse_args()

features, split = find_features([1, 2, 3, 4], [1, 2, 3, 4], total, pos_total)

rev_tags = dict()
for tag in tags:
    rev_tags[tags[tag]] = tag

obs_set = torch.utils.data.DataLoader(validation_Loader, batch_size = args.batch_size, shuffle = False)

model = NeuralNetwork(len(X_train[0]), num_authors)
model.load_state_dict(torch.load('./models//model.pt'))
model = model.cuda()
ig = IntegratedGradients(model)

get_POS = lambda char: rev_tags[char] if char in rev_tags else char
convert_POS = lambda n_gram, index: ' '.join([get_POS(char) for char in n_gram]) if index >= split else n_gram

obs_idx = 0

start = time.time()
all = []
torch.cuda.empty_cache()
for data, label in obs_set:

    attributions = ig.attribute(data.cuda(), target = label.to(torch.int64).cuda(), n_steps = 250)
    attributions = attributions.tolist()

    for attribution in attributions:
        all.append(attribution)

    torch.cuda.empty_cache()
    del attributions


X_ref = obs_df.drop(['label'], axis = 1)
y_ref = obs_df['label']

_, X_ref, _, y_ref = train_test_split(X_ref, y_ref, stratify = y_ref, test_size = 0.0066921862 * 4 / 10, random_state = 1)

tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
model = BertForMaskedLM.from_pretrained("bert-large-uncased").to(device)

features = np.load('/content/gdrive/MyDrive/PSU_REU/Data/TuringBench/Blind_Black/features.npy')
split = 961

tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
to_compressed = lambda tag: tags[tag] if tag in tags else tag
idx = 0
temp = dict()
offset = 65
for tag in tags:
    if offset + idx == 90:
        offset = 72
    temp[tag] = chr(offset + idx)
    idx += 1
tags = temp

def run_trial(NUM_NGRAMS_TO_REPLACE, MIN_LENGTH, constant):

    obfuscated_texts = []
    change_rates = []

    for index in range(len(X)):

        [text, pos_tags, ranked_indexes] = X.iloc[index]
        label = y.iloc[index]

        ranked_indexes = ranked_indexes.strip('][').split(', ')

        ranked_indexes = [float(elem) for elem in ranked_indexes]

        #ranked_indexes = np.add(ranked_indexes, [np.min(ranked_indexes)] * len(ranked_indexes))
        mult = [constant ** len(feature) for feature in features]
        ranked_indexes = np.multiply(ranked_indexes, mult)

        ranked_indexes = np.argsort(np.array(ranked_indexes))
        ranked_indexes = [elem for elem in ranked_indexes if elem >= split]
        to_replace = [features[elem] for elem in ranked_indexes]

        to_replace = [replace for replace in to_replace if len(replace) > MIN_LENGTH]
        to_replace = [to_replace[idx] for idx in range(NUM_NGRAMS_TO_REPLACE)]
        to_replace.reverse()
        '''
        ranked_indexes = [int(elem) for elem in ranked_indexes]
        ranked_indexes = [elem for elem in ranked_indexes if elem >= split]
        to_replace = [features[elem] for elem in ranked_indexes]

        to_replace = [replace for replace in to_replace if len(replace) > 2]
        to_replace = [to_replace[idx] for idx in range(NUM_NGRAMS_TO_REPLACE)]

        to_replace.reverse()
        '''

        sentences = sent_tokenize(text)

        obfuscated_text = ""

        for sentence in sentences:

            words = word_tokenize(sentence)
            ref = words

            retagged = pos_tag(words)
            retagged = [to_compressed(tup[1]) for tup in retagged]

            intervals = []

            for replace in to_replace:

                starts = [i for i in range(len(retagged) - len(replace)) if replace == "".join(retagged[i:i + len(replace)])]

                for start in starts:
                    intervals.append([start, start + len(replace)])

            changed = [False] * len(words)

            for interval in intervals:
                if not any(changed[interval[0] : interval[1]]):
                    words[interval[0] : interval[1]] = ["[MASK]"] * (interval[1] - interval[0])

                    inputs = tokenizer(" ".join(words), return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
                    token_ids = tokenizer.encode(" ".join(words), return_tensors="pt", padding="max_length", truncation=True, max_length=512)
                    num_tokens = len(token_ids.numpy()[0])

                    masked_positions = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero()
                    masked_positions = [mask.item() for mask in masked_positions]

                    outputs = model(**inputs)
                    del inputs
                    predictions = outputs[0]
                    sorted_preds, sorted_idx = predictions[0].sort(dim=-1, descending=True)

                    predicted_index = [torch.argmax(predictions[0, i]).item() for i in range(len(predictions[0]))]
                    predicted_token = [tokenizer.convert_ids_to_tokens([predicted_index[x]])[0] for x in range(len(predictions[0]))]
                    predicted_tokens = [predicted_token[pos] for pos in masked_positions]

                    rep_idx = 0
                    for word_idx in range(len(words)):
                        if(words[word_idx] == "[MASK]"):
                            words[word_idx] = predicted_tokens[rep_idx]# if words[word_idx] not in '!.?' and predicted_tokens[rep_idx] not in '?.!' else ''
                            rep_idx += 1
                            if(rep_idx >= len(predicted_tokens)):
                                break

                    words[:] = [word if word != "[MASK]" else "" for word in words]
                    changed[interval[0] : interval[1]] = [True] * (interval[1] - interval[0])

            change_rates.append(sum(changed) / len(changed))
            obfuscated_text += " ".join(words)

        torch.cuda.empty_cache()
        obfuscated_texts.append(re.sub(r'\s([?.!"\s](?:\s|$))', r'\1', obfuscated_text))
        #print(obfuscated_text)

    return obfuscated_texts, np.mean(change_rates)


def genPreds(X, adv_tokenizer, adv_classifier):

    preds = []

    for elem in X:
        elem = adv_tokenizer(elem, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            logits = adv_classifier(**elem).logits

        preds.append(logits.argmax().item())

    return preds

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
USE = hub.load(module_url)
def embed(input):
  return USE(input)

logging.set_verbosity(logging.ERROR)

!pip install bert_score -q
from bert_score import score
