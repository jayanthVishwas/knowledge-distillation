# coding: utf-8

from __future__ import unicode_literals, print_function

import pandas as pd
from transformers import BertTokenizer,AutoTokenizer

from experiments.sst2.settings import bert_settings, ROOT_DATA_PATH, TRAIN_FILE
from knowledge_distillation.bert_data import df_to_dataset
from knowledge_distillation.bert_trainer import BertTrainer
from knowledge_distillation.utils import get_logger, set_seed

if __name__ == '__main__':

    logger = get_logger()

    set_seed(3)

    train_df = pd.read_csv(TRAIN_FILE, encoding='utf-8', sep='\t')
    bert_base_uncased_path="/home1/thalari/CSCI566/distiller/knowledge-distillation/models/bert-base-uncased"

    bert_tokenizer = BertTokenizer.from_pretrained(bert_base_uncased_path,local_files_only=True)
    train_dataset = df_to_dataset(train_df, bert_tokenizer, bert_settings['max_seq_length'])

    trainer = BertTrainer(bert_settings, logger)
    model = trainer.train(train_dataset, bert_tokenizer, ROOT_DATA_PATH)

