import argparse
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
import gc

import Extraction

from nltk import pos_tag
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

from random import random
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

#Torch stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import copy

from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import normalized_mutual_info_score

import os

import pickle
import itertools
from pandas import DataFrame

from pandarallel import pandarallel
pandarallel.initialize()

import tqdm

import captum
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument("--train", "-T", help = "Path to Training Data", './train.csv')
parser.add_argument("--test", "-t", help = "Path to Testing Data", default = './test.csv')
parser.add_argument("--batch_size", "-bs", help = "Batch Size", default = 512)
parser.add_argument("--t", "-t", help="t, The Number of Top Character and POS-ngrams to Retain", default = 512)
parser.add_argument("--epochs", "-e", help="Number of Training Epochs", default = 70)
parser.add_argument("--weight_decay", "-wd", help="Weight Decay Constant", default = 0.00)
parser.add_argument("--momentum", "-m", help="Momentum Constant", default = 0.90)
parser.add_argument("--step", "-s", help="Scheduler Step Size", default = 8)
parser.add_argument("--gamma", "-g", help="Scheduler Gamma Constant", default = 0.30)
parser.add_argument("--authors_total", "-at", help="Number of Total Authors in Corpus", default = 20)
parser.add_argument("--authors_to_keep", "-atk", help="Number of Authors to Retain (In Order of Document Frequency)", default = 20)
parser.add_argument("--trial_name", "tm", help="The Current Trial's Name (E.g. Dataset Name)")
parser.add_argument("--text_index", "ti", help="The Name of the Column Housing the Text")

parse.parse_args()

print("Device: ", device)
gc.collect()

#UPenn Treebank Tags: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']

class Loader(Dataset):

    def __init__(self, x, y, top_idxs, Scaler):

        self.x = torch.Tensor(Scaler.transform(x))
        self.y = torch.Tensor([top_idxs[val] for val in y])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class NeuralNetwork(nn.Module):
    def __init__(self, in_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.act = nn.ReLU()
        self.dp = 0.25
        self.width = 1750
        self.num_classes = num_classes
        self.linear_relu_stack = nn.Sequential(
            nn.Dropout(p = self.dp),
            nn.Linear(in_size, self.width),
            self.act,
            nn.Dropout(p = self.dp),
            nn.Linear(self.width, self.width),
            self.act,
            nn.Linear(self.width, self.width),
            self.act,
            nn.Linear(self.width, self.width),
            self.act,
            nn.Dropout(p = self.dp),
            nn.Linear(self.width, self.width),
            self.act,
            nn.Dropout(p = self.dp),
            nn.Linear(self.width, self.width),
            self.act,
            nn.Dropout(p = self.dp),
            nn.Linear(self.width, self.width),
            self.act,
            nn.Dropout(p = self.dp),
            nn.Linear(self.width, self.width),
            self.act,
            nn.Dropout(p = self.dp),
            nn.Linear(self.width, self.width),
            self.act,
            nn.Dropout(p = self.dp),
            nn.Linear(self.width, self.width),
            self.act,
            nn.Dropout(p = self.dp),
            nn.Linear(self.width, self.width),
            self.act,
            nn.Dropout(p = self.dp),
            nn.Linear(self.width, 256),
            self.act,
            nn.Dropout(p = self.dp),
            nn.Linear(256, self.num_classes)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)

def train_and_eval(model, EPOCHS, training_set, validation_set, loss_function, optimizer, scheduler):

    model.to(device)

    for epoch in range(1, EPOCHS + 1):
        print('------------', '\n', 'Epoch #:', epoch, '\n', 'Training:')

        with torch.set_grad_enabled(True):

            total = 0
            correct = 0

            model.train()

            labels = []
            predictions = []
            all_preds = []
            all_labels = []

            with tqdm(training_set, unit = "batch", bar_format = '{l_bar}{bar:20}{r_bar}{bar:-20b}') as tqdm_train:
                for training_data, labels in tqdm_train:
                    tqdm_train.set_description(f'Epoch: {epoch}')

                    training_data = training_data.to(device)
                    labels = torch.tensor(labels, dtype = torch.long)
                    #print(labels)
                    labels = labels.to(device)
                    #print(correct_prediction)

                    optimizer.zero_grad()

                    prediction = model(training_data).squeeze(1)

                    loss = loss_function(prediction, labels)

                    loss.backward()
                    optimizer.step()

                    for idx, i in enumerate(prediction):
                        if torch.argmax(i) == labels[idx]:
                            correct += 1
                        total += 1

                    tqdm_train.set_postfix(loss=loss.item(), accuracy=100.*correct/total)
                    time.sleep(0.1)

            training_accuracy = round(correct/total, 5)
            print('Training Accuracy: ', training_accuracy)

            scheduler.step()


        print('------------', '\n', 'Validation:')
        model.eval()
        gc.collect()

        total = 0
        correct = 0

        with torch.no_grad():

            with tqdm(validation_set, unit="batch", bar_format = '{l_bar}{bar:20}{r_bar}{bar:-20b}') as tqdm_val:
                tqdm_val.set_description(f'Epoch: {epoch}')
                for val_data, labels in tqdm_val:

                    val_data = val_data.to(device)

                    labels = torch.tensor(labels, dtype = torch.long)
                    labels = labels.to(device)

                    prediction = model(val_data).squeeze(1)

                    for idx, i in enumerate(prediction):
                        if torch.argmax(i) == labels[idx]:
                            correct += 1
                        total += 1

                    tqdm_val.set_postfix(accuracy=100.*correct/total)
                    time.sleep(0.1)

            validation_accuracy = round(correct/total, 5)
            print('Validation Accuracy: ', validation_accuracy)
            gc.collect()

    return model

idx = 0
temp = dict()
offset = 65
for tag in tags:
    if offset + idx == 90:
        offset = 72
    temp[tag] = chr(offset + idx)
    idx += 1
tags = temp

gc.collect()

train = pd.read_csv(args.train)
test = pd.read_csv(args.test)

top_idxs, total, pos_total = Extraction.preprocess_data(train, args.authors_total, args.authors_to_keep, tag = True)

print('------------', '\n', 'Generating n-grams...')

n_grams = [Extraction.return_best_n_grams(n, args.t, total) for n in [1, 2, 3, 4]]

print('------------', '\n', 'Generating POS n-grams...')

pos_n_grams = [Extraction.return_best_n_grams(n, args.t, pos_total) for n in [1, 2, 3, 4]]

print('------------', '\n', 'Generating data...')

X_train, y_train = Extraction.gen_data(train, [], n_grams, [1, 2, 3, 4], pos_n_grams, top_idxs, total, pos_total)
X_test, y_test = Extraction.gen_data(test, [], n_grams, [1, 2, 3, 4], pos_n_grams, top_idxs, total, pos_total)

print('------------', '\n', 'Scaling, Loading, and Shuffling Data')
Scaler = sklearn.preprocessing.StandardScaler().fit(X_train)
training_Loader = Loader(X_train, y_train, top_idxs, Scaler)
validation_Loader = Loader(X_test, y_test, top_idxs, Scaler)

training_set = torch.utils.data.DataLoader(training_Loader, batch_size = args.batch_size, shuffle = True)
validation_set = torch.utils.data.DataLoader(validation_Loader, batch_size = args.batch_size, shuffle = False)

model = NeuralNetwork(len(X_train[0]), num_authors)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = LR, weight_decay = args.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = args.step_size, gamma = args.gamma)

model = train_and_eval(model, EPOCHS, training_set, validation_set, loss_function, optimizer, scheduler)

torch.save(model.state_dict(), f'./models/{args.trial_name}/model.pt')
n_grams = [elem for sub in n_grams for elem in sub]
pos_n_grams = [elem for sub in pos_n_grams for elem in sub]
np.save(f'./models/{args.trial_name}/n_grams.npy', n_grams)
np.save(f'./models/{args.trial_name}/pos_ngrams.npy', pos_n_grams)
