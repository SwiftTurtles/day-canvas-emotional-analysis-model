from bert_dataset_tokenizer import tokenizer
from bert_classification_model import model

import torch
import numpy as np

from transformers import BertModel
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
device = torch.device('cpu')

#kobert 감정분석 후 한글 감정 반환
def predict(model_path, sentence):

    classification_model = model.BERTClassifier(bertmodel, dr_rate = 0.5, tok = model.tok, vocab = model.vocab, max_len = 128, batch_size = 32).to(device)
    classification_model.load_state_dict(torch.load(model_path, map_location=device))
    classification_model.eval()

    out = classification_model.predict(sentence)
    logits = out.detach().cpu().numpy()

    emotion = []
    if np.argmax(logits) == 0:
        emotion.append("기쁨")
    elif np.argmax(logits) == 1:
        emotion.append("당황")
    elif np.argmax(logits) == 2:
        emotion.append("분노")
    elif np.argmax(logits) == 3:
        emotion.append("불안")
    elif np.argmax(logits) == 4:
        emotion.append("상처")
    elif np.argmax(logits) == 5:
        emotion.append("슬픔")

    return emotion[0]