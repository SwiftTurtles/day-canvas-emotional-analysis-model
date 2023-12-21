# -*- coding: utf-8 -*-
"""emotion_classifier_tokenizer.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wj5QfhjCQHlNvTV3m07rx7KTW1RgqCIt
"""

# !pip install mxnet
# !pip install gluonnlp pandas tqdm
# !pip install torch

import torch
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, vocab=vocab, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))