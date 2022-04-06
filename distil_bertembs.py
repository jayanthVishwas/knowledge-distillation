from __future__ import unicode_literals, print_function
from pdb import set_trace as dp

from sentence_transformers import SentenceTransformer


from scipy import spatial
import numpy as np
import pandas as pd
import torch
from torch.utils.data import SequentialSampler, DataLoader
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer

from experiments.sst2.settings import distillation_settings, TRAIN_FILE, ROOT_DATA_PATH
from knowledge_distillation.bert_data import df_to_dataset
from knowledge_distillation.bert_trainer import batch_to_inputs
from knowledge_distillation.lstm_trainer import LSTMDistilled
from knowledge_distillation.utils import get_logger, device, set_seed

# model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

modelPath = '/home1/thalari/CSCI566/distiller/knowledge-distillation/models/sbert'

# model.save(modelPath)

model = SentenceTransformer(modelPath)

#example
sentences = ['What a lovely day','what a beautiful day',
    'what a bad day']
    

sentence2=['Sentences are passed as a list of string.',
    'The quick brown fox jumps over the lazy dog.']

embeddings1 = model.encode(sentences)
embeddings2 = model.encode(sentence2)



# result = 1 - spatial.distance.cosine(embeddings1,embeddings2)

# print(result)


logger = get_logger()

train_df = pd.read_csv(TRAIN_FILE, encoding='utf-8', sep='\t')

X_train = train_df['sentence'].values
y_train = model.encode(train_df['sentence'])
y_real = train_df['label'].values

distiller = LSTMDistilled(distillation_settings, logger)

# 4. train
dist_model, vocab = distiller.train(X_train, y_train, y_real, ROOT_DATA_PATH)