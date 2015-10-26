from sklearn.datasets import load_digits
from sys import argv
import matplotlib.pyplot as plt
import cv2

digits = load_digits()

fig = plt.figure()
for i in range(10):
    subplot = fig.add_subplot(5,2,i+1)
    subplot.matshow(digits.data[i].reshape(8,8), cmap='gray')

fig2 = plt.figure()
i = 0
for arg in argv[1:]:
    subplot = fig2.add_subplot(5,2,i+1)
    im = cv2.imread(arg, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    im /= 16.0
    im.round()
    subplot.matshow(cv2.resize(im, (8, 8), interpolation=cv2.INTER_AREA), cmap='gray')
    i += 1

plt.show()