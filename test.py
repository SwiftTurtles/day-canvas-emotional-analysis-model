import kobert_emotion_classifier as kec

#테스트 문자열 작성
sentence = ""

PATH = './model.pt'

print(kec.predict(PATH, sentence))