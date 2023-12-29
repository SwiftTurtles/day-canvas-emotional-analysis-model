from bert_dataset_tokenizer import tokenizer

import torch
from torch import nn
import gluonnlp as nlp
import numpy as np
from kobert_tokenizer import KoBERTTokenizer

kobert_pretrained_tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
vocab = nlp.vocab.BERTVocab.from_sentencepiece(kobert_pretrained_tokenizer.vocab_file, padding_token='[PAD]')
device = torch.device('cpu')
tok = kobert_pretrained_tokenizer.tokenize

# 감정 분석 모델
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes = 6,   
                 dr_rate = None,
                 params = None,
                 tok=None,
                 vocab=None,
                 max_len=0,
                 batch_size=0
                 ):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p = dr_rate)

        self.tok = tok
        self.vocab = vocab
        self.max_len = max_len
        self.batch_size = batch_size

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device),return_dict = False)
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)


    # 주어진 문장 감정 분석
    # return: 감정 확률 리스트
    def predict(self, predict_sentence):
        predict_data = [predict_sentence, '0']
        predict_dataset = [predict_data]

        another_test = tokenizer.BERTDataset(predict_dataset, 0, 1, self.tok, self.vocab, self.max_len, True, False)
        test_dataloader = torch.utils.data.DataLoader(another_test, batch_size = self.batch_size, num_workers = 0)

        out = None
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
            token_ids = token_ids.long().to(device)
            valid_length = valid_length
            segment_ids = segment_ids.long().to(device)

            out = self.forward(token_ids, valid_length, segment_ids)

        return out