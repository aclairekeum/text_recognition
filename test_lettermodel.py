import cv2
from sklearn.externals import joblib
from sys import argv
import string 

model = joblib.load('lettermodel/lettermodel.pkl')

for arg in argv[1:]:
    im = cv2.imread(arg, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    # im /= 16.0
    # im.round()
    resized = cv2.resize(im, (8, 8))
    data = resized.ravel()

    predictedletter = model.predict_proba(data)
    # string.ascii_uppercase[]
    print arg, predictedletter, model.predict(data)

    #string.ascii_uppercase[predictedletter.argmax()] #string.ascii_uppercase[max(int(predictedletter[0])]
