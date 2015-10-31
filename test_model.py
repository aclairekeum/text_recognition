import cv2
from sklearn.externals import joblib
from sys import argv

model = joblib.load('model/model.pkl')

for arg in argv[1:]:
    im = cv2.imread(arg, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    im /= 16.0
    im.round()
    resized = cv2.resize(im, (8, 8))
    data = resized.reshape((1,64))

    digit = model.predict(data)[0]
    confidence = model.predict_proba(data).max() # digit guessed has maximum probability
    print arg, digit, confidence
