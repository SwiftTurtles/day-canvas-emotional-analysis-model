import kobert_emotion_classifier as kec

sentence = "너 도대체 왜그래?"

PATH = './model.pt'

print(kec.predict(PATH, sentence))