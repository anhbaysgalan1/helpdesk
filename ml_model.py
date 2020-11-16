import random
import copy
import time
import gc
import torch
import pandas as pd

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from sklearn.metrics import f1_score
import os 

from sklearn.preprocessing import StandardScaler
from multiprocessing import  Pool
from functools import partial
from sklearn.decomposition import PCA
import torch as t
import torch.nn.functional as F

import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
import pickle

import warnings
warnings.filterwarnings('ignore')
import re

def clean_text(x):
    pattern = r'[^a-zA-z0-9\s]'
    text    = re.sub(pattern, ' ', x)
    return text

def clean_numbers(x):
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
        x = re.sub('[0-9]{1}', '', x)
    return x


from .mapping import *


def replace_norms(text):
    pattern = re.compile(r'\b(' + '|'.join(mapping.keys()) + r')\b')
    return pattern.sub(lambda x: mapping[x.group()], text)


with open('./helpdesk/ml_model/tes_preprocessing.pickle', 'rb') as handle:
    data = pickle.load(handle)
    
with open('./helpdesk/ml_model/encoder.pickle', 'rb') as handle:
    le = pickle.load(handle)
    
with open('./helpdesk/ml_model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
with open('./helpdesk/ml_model/fasttext.pickle', 'rb') as handle:
    embedding_matrix = pickle.load(handle)
    
    
embed_size = 300
batch_size = 32
max_features = 12000
hidden_size = 256
n_layers = 1
drop_prob = 0.4
n_classes = len(le.classes_)
    
class RNNLSTM(nn.Module):
    
    def __init__(self):
        super(RNNLSTM, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        
        # RNN Layer
        self.rnn= nn.LSTM(embed_size, hidden_size, n_layers, bidirectional= True, dropout=0.1)
        
        # hidden layer linear transformation
        self.fc = nn.Linear(hidden_size*4 , 64)   
        # fungsi aktivasi
        self.relu = nn.ReLU()
        
        # Droupout layer
        self.dropout = nn.Dropout(drop_prob)
        
        # Output layer
        self.out = nn.Linear(64, n_classes)


    def forward(self, x):
        out_embedding = self.embedding(x)
        out_rnn, _ = self.rnn(out_embedding)
        # pooling operation
        avg_pool = torch.mean(out_rnn, 1)
        max_pool,_ = torch.max(out_rnn, 1)
        conc = torch.cat((avg_pool, max_pool), 1)
        
        conc = self.relu(self.fc(conc))
        conc = self.dropout(conc)
        out = self.out(conc)
        
        # return final output
        return out
    
model = RNNLSTM()

model.load_state_dict(torch.load('./helpdesk/ml_model/lstmmodel.pt'))

def predict_single(x):    
    # lower the text
    x = x.lower()
    # Clean the text
    x =  clean_text(x)
    # Clean numbers
    x =  clean_numbers(x)
    # Clean Contractions
    x = replace_norms(x)
    # print(x)
    # tokenize
    x = tokenizer.texts_to_sequences([x])
    # pad
    x = pad_sequences(x, maxlen=150)
    # create dataset
    x = torch.tensor(x, dtype=torch.long)

    pred = model(x).detach()
    pred = F.softmax(pred).cpu().numpy()

    pred = pred.argmax(axis=1)

    pred = le.classes_[pred]

    return pred[0]