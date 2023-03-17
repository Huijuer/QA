import torch
import sys
import random
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import os
import unicodedata
import re
# import time
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from transformers import *


class DatasetPruning(Dataset):
    def __init__(self, data, rel2idx, idx2rel):
        self.data = data
        self.rel2idx = rel2idx
        self.idx2rel = idx2rel
        self.tokenizer_class = RobertaTokenizer
        # self.pretrained_weights = 'hfl/chinese-roberta-wwm-ext'
        # self.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext', cache_dir='.')
        # self.pretrained_weights = 'hfl/chinese-roberta-wwm-ext'
        # self.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext', cache_dir='.')
        self.pretrained_weights = 'hfl/chinese-macbert-base'
        self.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-macbert-base', cache_dir='.')

    def __len__(self):
        return len(self.data)

    def pad_sequence(self, arr, max_len=128):
        if len(arr) >= max_len:
            return arr[:max_len]
        else:
            num_to_add = max_len - len(arr)
            arr = arr + ['<pad>'] * num_to_add
            return arr

    def toOneHot(self, indices):
        # print(indices)
        # sys.exit(0)
        indices = torch.randint(low=0, high=6, size=(6,))
        vec_len = len(self.rel2idx)
        one_hot = torch.FloatTensor(vec_len)
        one_hot.zero_()
        # print(indices)
        one_hot.scatter_(0, indices, 1)
        # print(one_hot, "one_hot")
        return one_hot

    def tokenize_question(self, question):
        # print(question)
        # question = re.sub(u"\\（.*?）|\\{.*?}|\\[.*?]|\\【.*?】|\\(.*?\\)", "", question)
        # question = "<s> " + question + " </s>"
        question = question.replace("[","").replace("]","")
        # print(question)
        # exit()
        question_tokenized = self.tokenizer.tokenize(question)
        question_tokenized = self.pad_sequence(question_tokenized, 32)
        question_tokenized = torch.tensor(self.tokenizer.encode(question_tokenized, add_special_tokens=False))
        # if question_tokenized.__len__() != 64:
        #     print(question,"不等于64",question_tokenized.__len__())
        #     sys.exit()
        attention_mask = []
        for q in question_tokenized:
            # 1 means padding token
            if q == 100:
                attention_mask.append(0)
            else:
                attention_mask.append(1)
        return question_tokenized, torch.tensor(attention_mask, dtype=torch.long)

    def __getitem__(self, index):
        data_point = self.data[index]
        question_text = data_point[0]
        question_tokenized, attention_mask = self.tokenize_question(question_text)
        rel_ids = data_point[1]
        rel_onehot = self.toOneHot(rel_ids)
        # print(question_tokenized.shape, attention_mask.shape, rel_onehot.shape)
        return question_tokenized, attention_mask, rel_onehot


def _collate_fn(batch):
    question_tokenized = batch[0]
    attention_mask = batch[1]
    rel_onehot = batch[2]
    print(len(batch))
    question_tokenized = torch.stack(question_tokenized, dim=0)
    attention_mask = torch.stack(attention_mask, dim=0)
    print(question_tokenized.shape,attention_mask.shape,rel_onehot.shape)
    return question_tokenized, attention_mask, rel_onehot




class DataLoaderPruning(DataLoader):
    def __init__(self, *args, **kwargs):
        super(DataLoaderPruning, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn
